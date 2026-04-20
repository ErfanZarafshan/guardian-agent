"""Core agent loop: one perceive-reason-act iteration.

`run_cycle(profile, ...)` is the entire agent in one function. It:
  1. PERCEIVEs by calling the weather aggregator for the profile's location.
  2. REASONs by running the RiskEngine (which internally calls the ML
     classifier and the Bayesian network).
  3. ACTs by letting the planner produce Actions and the Dispatcher execute
     them — subject to dedup against the previous cycle.
  4. LOGs one JSONL line per cycle to `data/cycles.jsonl` and returns a
     CycleReport object.

State (last-cycle fingerprint, for dedup) is threaded through explicitly
rather than living in a class. Each cycle is a pure function of its inputs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .logging_setup import get_logger
from .output.dispatch import Dispatcher, DispatchReport
from .planning import Action, plan_actions
from .profile import UserProfile
from .risk.risk_engine import RiskEngine
from .weather.aggregator import observe
from .weather.observation import WeatherObservation


log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Fingerprint + state for dedup
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CycleFingerprint:
    """Compact hash of what the agent saw last cycle.

    If the fingerprint matches between two cycles, we skip re-dispatching
    non-console actions — no reason to SMS your spouse every 5 minutes during
    an hour-long flash flood warning. Console actions always dispatch so the
    user sees continuous confirmation the agent is alive.
    """
    risk_argmax: str
    active_alert_events: tuple[str, ...]   # sorted tuple of event strings

    @classmethod
    def from_report(
        cls, risk_argmax: str, observation: WeatherObservation,
    ) -> "CycleFingerprint":
        alerts = tuple(sorted(
            a.event for a in observation.alerts if a.is_active(observation.observed_at)
        ))
        return cls(risk_argmax=risk_argmax, active_alert_events=alerts)


@dataclass
class AgentState:
    """Mutable per-profile state that persists across cycles."""
    last_fingerprint: CycleFingerprint | None = None


# ---------------------------------------------------------------------------
# Cycle report
# ---------------------------------------------------------------------------

@dataclass
class CycleReport:
    """Everything that happened in one cycle, for logging + post-hoc audit."""
    cycle_started_at: datetime
    cycle_finished_at: datetime
    profile_user_id: str
    observation_sources: list[str]
    has_active_alerts: bool
    risk_posterior: dict[str, float]
    risk_argmax: str
    fingerprint_matched_previous: bool
    actions_planned: int
    actions_dispatched: int
    actions_suppressed_by_dedup: int
    dispatch_summary: str
    error: str | None = None

    # Not included in JSON: the full Action objects (they have dict metadata
    # with non-JSON-serializable fields in corner cases).
    actions: list[Action] = field(default_factory=list, repr=False)
    observation: WeatherObservation | None = field(default=None, repr=False)
    dispatch_report: DispatchReport | None = field(default=None, repr=False)

    def to_json_dict(self) -> dict[str, Any]:
        """JSON-safe dict for writing to cycles.jsonl."""
        d = asdict(self)
        # Drop non-serializable helpers and datetimes -> isoformat
        d.pop("actions", None)
        d.pop("observation", None)
        d.pop("dispatch_report", None)
        d["cycle_started_at"] = self.cycle_started_at.isoformat()
        d["cycle_finished_at"] = self.cycle_finished_at.isoformat()
        return d


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------

def _filter_duplicate_actions(
    actions: list[Action], matched_previous: bool,
) -> tuple[list[Action], int]:
    """If the fingerprint matched last cycle, drop non-console actions.

    Console actions are preserved so the user keeps seeing a heartbeat.
    Returns (filtered_actions, suppressed_count).
    """
    if not matched_previous:
        return actions, 0
    from .planning.actions import ActionChannel
    kept = [a for a in actions if a.channel is ActionChannel.CONSOLE]
    return kept, len(actions) - len(kept)


# ---------------------------------------------------------------------------
# One cycle
# ---------------------------------------------------------------------------

def run_cycle(
    profile: UserProfile,
    engine: RiskEngine,
    dispatcher: Dispatcher,
    state: AgentState,
    *,
    cycle_log_path: Path | None = None,
    skip_owm: bool = False,
) -> CycleReport:
    """One perceive-reason-act cycle.

    Parameters
    ----------
    profile
        The user's profile, containing location for perception and vulnerability
        factors for reasoning.
    engine
        A RiskEngine constructed once at agent startup (holds the network
        inference object and the optional classifier).
    dispatcher
        The Phase 7 output coordinator. Reused across cycles.
    state
        Mutable state holding the previous cycle's fingerprint. This function
        updates `state.last_fingerprint` before returning.
    cycle_log_path
        If provided, append one JSON line per cycle here. None = no logging.
    skip_owm
        Pass through to the weather aggregator for environments without an
        OWM key.
    """
    started = datetime.now(timezone.utc)
    error: str | None = None

    # ---- PERCEIVE ----
    try:
        obs = observe(
            latitude=profile.location.latitude,
            longitude=profile.location.longitude,
            nws_zone_id=profile.location.nws_zone_id,
            skip_owm=skip_owm,
        )
    except Exception as e:  # noqa: BLE001 — observation failure is a "soft" event
        log.error("Observation failed: %s", e)
        error = f"observation_failed: {e}"
        finished = datetime.now(timezone.utc)
        report = CycleReport(
            cycle_started_at=started,
            cycle_finished_at=finished,
            profile_user_id=profile.user_id,
            observation_sources=[],
            has_active_alerts=False,
            risk_posterior={},
            risk_argmax="Unknown",
            fingerprint_matched_previous=False,
            actions_planned=0,
            actions_dispatched=0,
            actions_suppressed_by_dedup=0,
            dispatch_summary="(skipped; observation failed)",
            error=error,
        )
        _append_log(cycle_log_path, report)
        return report

    # ---- REASON ----
    engine_result = engine.assess(obs, profile)
    assessment = engine_result.assessment

    fingerprint = CycleFingerprint.from_report(assessment.argmax, obs)
    matched = state.last_fingerprint == fingerprint
    state.last_fingerprint = fingerprint

    # ---- ACT ----
    actions = plan_actions(assessment.argmax, profile, obs)
    filtered, suppressed = _filter_duplicate_actions(actions, matched)
    dispatch_report = dispatcher.dispatch(filtered)

    finished = datetime.now(timezone.utc)
    report = CycleReport(
        cycle_started_at=started,
        cycle_finished_at=finished,
        profile_user_id=profile.user_id,
        observation_sources=list(obs.sources),
        has_active_alerts=obs.has_active_alerts,
        risk_posterior=dict(assessment.posterior),
        risk_argmax=assessment.argmax,
        fingerprint_matched_previous=matched,
        actions_planned=len(actions),
        actions_dispatched=dispatch_report.console.attempted
                           + dispatch_report.sms.attempted
                           + dispatch_report.smart_home.attempted,
        actions_suppressed_by_dedup=suppressed,
        dispatch_summary=dispatch_report.summary(),
        actions=filtered,
        observation=obs,
        dispatch_report=dispatch_report,
    )
    _append_log(cycle_log_path, report)
    return report


# ---------------------------------------------------------------------------
# Log writer
# ---------------------------------------------------------------------------

def _append_log(path: Path | None, report: CycleReport) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(report.to_json_dict()) + "\n")
    except Exception as e:  # noqa: BLE001
        log.warning("Failed to append cycle log at %s: %s", path, e)


__all__ = [
    "AgentState",
    "CycleFingerprint",
    "CycleReport",
    "run_cycle",
]
