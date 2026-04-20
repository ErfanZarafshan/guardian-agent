"""Smart-home output dispatcher (mock implementation).

For the prototype we cannot complete a full Alexa Skills Kit deployment (that
requires Amazon developer account verification, account linking, and a public
HTTPS endpoint — all out of scope for a semester project). So this dispatcher
is a *mock* that logs intended actions to the console.

Importantly, the interface — `SmartHomeDispatcher.dispatch(actions)` — is
identical to what a real Alexa/Google Home integration would implement. When
Guardian Agent is deployed to a real smart home, swapping the mock for an
`AlexaSmartHomeDispatcher` requires zero changes in the planner or agent.

Supported actions:
  - ACTIVATE_FLOOD_LIGHTS      → outdoor lighting ON
  - SOUND_ALARM                → audible alarm ON
  - ADJUST_THERMOSTAT          → setpoint per metadata['setpoint_f']

Unknown smart-home actions are logged but not dispatched (defensive).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..logging_setup import get_logger
from ..planning.actions import Action, ActionChannel, ActionKind


log = get_logger(__name__)


@dataclass(frozen=True)
class SmartHomeResult:
    """One smart-home action attempt."""
    kind: ActionKind
    mode: str                   # "mocked" | "unsupported"
    dispatched_at: datetime
    detail: str                 # human-readable summary


@dataclass(frozen=True)
class SmartHomeDispatchReport:
    attempted: int
    mocked: int
    unsupported: int
    results: tuple[SmartHomeResult, ...] = field(default_factory=tuple)


# Which kinds this dispatcher handles. Anything outside this set gets
# "unsupported" — a defensive fallback, not an error.
_SUPPORTED: frozenset[ActionKind] = frozenset({
    ActionKind.ACTIVATE_FLOOD_LIGHTS,
    ActionKind.SOUND_ALARM,
    ActionKind.ADJUST_THERMOSTAT,
})


class SmartHomeDispatcher:
    """Mock smart-home dispatcher.

    The class is deliberately stateless so it can be shared across agent
    cycles without reset logic.
    """

    def __init__(self, device_label: str = "mock-home") -> None:
        self.device_label = device_label

    # ------------------------------------------------------------------

    def _dispatch_one(self, action: Action) -> SmartHomeResult:
        now = datetime.now(timezone.utc)
        if action.kind not in _SUPPORTED:
            detail = f"Unsupported kind for smart-home dispatcher: {action.kind.value}"
            log.warning(detail)
            return SmartHomeResult(
                kind=action.kind,
                mode="unsupported",
                dispatched_at=now,
                detail=detail,
            )

        # Build a human-readable summary per action kind.
        if action.kind is ActionKind.ACTIVATE_FLOOD_LIGHTS:
            detail = f"[{self.device_label}] exterior flood lights ON"
        elif action.kind is ActionKind.SOUND_ALARM:
            detail = f"[{self.device_label}] audible alarm ON"
        elif action.kind is ActionKind.ADJUST_THERMOSTAT:
            setpoint = action.metadata.get("setpoint_f", "?")
            detail = f"[{self.device_label}] thermostat setpoint -> {setpoint}F"
        else:  # pragma: no cover — guarded by _SUPPORTED above
            detail = f"[{self.device_label}] {action.kind.value}"

        log.info("[SmartHome mock] %s", detail)
        return SmartHomeResult(
            kind=action.kind,
            mode="mocked",
            dispatched_at=now,
            detail=detail,
        )

    # ------------------------------------------------------------------

    def dispatch(self, actions: list[Action]) -> SmartHomeDispatchReport:
        """Dispatch all smart-home-channel actions in `actions`."""
        sh_actions = [a for a in actions if a.channel is ActionChannel.SMART_HOME]
        results = [self._dispatch_one(a) for a in sh_actions]
        mocked = sum(1 for r in results if r.mode == "mocked")
        unsupported = sum(1 for r in results if r.mode == "unsupported")
        return SmartHomeDispatchReport(
            attempted=len(results),
            mocked=mocked,
            unsupported=unsupported,
            results=tuple(results),
        )


__all__ = ["SmartHomeDispatcher", "SmartHomeDispatchReport", "SmartHomeResult"]
