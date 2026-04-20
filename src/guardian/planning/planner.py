"""Rule-augmented action planner.

Inputs:
  - RiskAssessment (from Phase 5 Bayesian engine)
  - UserProfile (Phase 2)
  - WeatherObservation (Phase 3)

Output:
  - Ordered list[Action] — what the agent should do this cycle.

The planner is deterministic. No learning, no probability math. It reads the
argmax of the risk posterior, inspects the profile for vulnerabilities, looks
at which alert event is active, and picks from a fixed action catalog.

Rule summary (by argmax RiskLevel):

  LOW:
    During quiet hours -> LOG_ONLY only.
    Otherwise          -> LOG_ONLY + optional NOTIFY_USER if any active alert.

  MODERATE:
    During quiet hours -> demote to LOG_ONLY (unless medically vulnerable).
    Otherwise          -> NOTIFY_USER with hazard-specific guidance.
                          If the home is flood-prone and a flood alert is
                          active -> RECOMMEND_SHELTER.
                          Notify contacts whose notify_on includes MODERATE.

  HIGH:
    Ignore quiet hours. Always notify the user.
    Add NOTIFY_CONTACTS for contacts whose threshold <= HIGH.
    Add RECOMMEND_SHELTER (or RECOMMEND_EVACUATE if the user is medically
    vulnerable AND has no vehicle AND alert is Immediate-urgency).
    If smart-home enabled and hazard is flood-related -> ACTIVATE_FLOOD_LIGHTS.
    If hazard is heat-related and smart-home enabled -> ADJUST_THERMOSTAT.

  CRITICAL:
    Ignore quiet hours. Override user preferences for life-safety actions.
    NOTIFY_USER + NOTIFY_CONTACTS (all contacts that opt-in at CRITICAL).
    RECOMMEND_EVACUATE for flood/hurricane; RECOMMEND_SHELTER for tornado.
    SOUND_ALARM if smart-home enabled.
    If flood-related, ACTIVATE_FLOOD_LIGHTS (regardless of smart-home pref —
    life-safety override).
"""

from __future__ import annotations

from datetime import datetime, time, timezone

from ..logging_setup import get_logger
from ..profile import RiskNotifyLevel, UserProfile
from ..weather.observation import WeatherAlert, WeatherObservation
from .actions import Action, ActionChannel, ActionKind, get_template, render


log = get_logger(__name__)


# Mapping from RiskLevel argmax to the RiskNotifyLevel threshold used for
# selecting which contacts to notify.
_RISK_TO_NOTIFY: dict[str, RiskNotifyLevel] = {
    "Moderate": RiskNotifyLevel.MODERATE,
    "High":     RiskNotifyLevel.HIGH,
    "Critical": RiskNotifyLevel.CRITICAL,
}

# Events that indicate water-related hazard (affect flood-light and shelter
# logic).
_FLOOD_EVENTS = frozenset({
    "Flash Flood Warning", "Flash Flood Watch", "Flood Warning",
    "Flood Watch", "Storm Surge Warning", "Coastal Flood Warning",
})
_WIND_EVENTS = frozenset({
    "Tornado Warning", "Tornado Watch", "Severe Thunderstorm Warning",
    "Tropical Storm Warning", "Hurricane Warning",
})
_HEAT_EVENTS = frozenset({
    "Excessive Heat Warning", "Heat Advisory", "Excessive Heat Watch",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_primary_alert(alerts: list[WeatherAlert]) -> WeatherAlert | None:
    """Choose the 'most significant' alert for message templating.

    Priority: highest severity, then most urgent. Used to pick which hazard
    template to apply when multiple alerts are active.
    """
    active = [a for a in alerts if a.is_active()]
    if not active:
        return None
    sev_rank = {"None": 0, "Minor": 1, "Moderate": 2, "Severe": 3, "Extreme": 4}
    urg_rank = {
        "Unknown": 0, "Past": 1, "Future": 2, "Expected": 3, "Immediate": 4,
    }
    return max(
        active,
        key=lambda a: (sev_rank.get(a.severity.value, 0),
                       urg_rank.get(a.urgency.value, 0))
    )


def _is_quiet_hours(
    now: datetime, quiet_start: time | None, quiet_end: time | None
) -> bool:
    """True if the current local time is inside the user's quiet window.

    Handles overnight windows (e.g. 22:00 -> 07:00).
    """
    if quiet_start is None or quiet_end is None:
        return False
    # Use the hour/minute of `now` in its tz, if any. If `now` is naive, treat
    # as UTC — this path shouldn't hit in production since WeatherObservation
    # uses tz-aware timestamps.
    t = now.timetz().replace(tzinfo=None)
    if quiet_start <= quiet_end:
        return quiet_start <= t <= quiet_end
    # Overnight window crosses midnight.
    return t >= quiet_start or t <= quiet_end


def _hazard_flavor(event: str | None) -> str:
    """Categorize the primary alert into a hazard flavor."""
    if event is None:
        return "none"
    if event in _FLOOD_EVENTS:
        return "flood"
    if event in _WIND_EVENTS:
        return "wind"
    if event in _HEAT_EVENTS:
        return "heat"
    return "other"


def _template_vars(
    profile: UserProfile, alert: WeatherAlert | None
) -> dict[str, str]:
    """Build the substitution dict used in message templates."""
    return {
        "name": profile.name,
        "address": profile.location.address,
        "event": alert.event if alert else "severe weather",
        "expires": str(alert.expires) if alert and alert.expires else "soon",
        "sheltering_in_place": "sheltering in place",
    }


# ---------------------------------------------------------------------------
# Planner entry point
# ---------------------------------------------------------------------------

def plan_actions(
    risk_argmax: str,
    profile: UserProfile,
    observation: WeatherObservation,
) -> list[Action]:
    """Produce an ordered list of Actions for this risk + profile + world.

    Actions are sorted by priority (highest first) so the first element is
    what the user sees first.
    """
    primary = _pick_primary_alert(observation.alerts)
    primary_event = primary.event if primary else None
    flavor = _hazard_flavor(primary_event)
    vars_ = _template_vars(profile, primary)

    prefs = profile.preferences
    quiet = _is_quiet_hours(
        observation.observed_at,
        prefs.quiet_hours.start if prefs.quiet_hours else None,
        prefs.quiet_hours.end if prefs.quiet_hours else None,
    )

    actions: list[Action] = []

    # ---- LOW: mostly quiet ----
    if risk_argmax == "Low":
        if primary is not None and not quiet:
            actions.append(Action(
                kind=ActionKind.NOTIFY_USER,
                channel=ActionChannel.CONSOLE,
                message=f"Heads up: {primary.event} is active for your area, "
                        f"but your personal risk is low right now.",
                rationale=(
                    f"Risk argmax=Low. Active alert {primary.event} is "
                    f"informational for a non-vulnerable user in quiet hours."
                ),
            ))
        actions.append(Action(
            kind=ActionKind.LOG_ONLY,
            channel=ActionChannel.CONSOLE,
            message="No action required.",
            rationale=f"Risk argmax=Low, quiet_hours={quiet}.",
        ))
        return _sort(actions)

    # ---- MODERATE: notify, maybe shelter ----
    if risk_argmax == "Moderate":
        vulnerable = profile.is_medically_vulnerable

        if quiet and not vulnerable:
            # Demote to log only; the user has opted into quiet hours and
            # is not particularly at risk.
            actions.append(Action(
                kind=ActionKind.LOG_ONLY,
                channel=ActionChannel.CONSOLE,
                message=(
                    f"Moderate risk detected for {primary_event or 'current conditions'} "
                    "but suppressing alert during user's quiet hours."
                ),
                rationale="Risk argmax=Moderate, quiet_hours=True, not vulnerable.",
            ))
            return _sort(actions)

        actions.append(Action(
            kind=ActionKind.NOTIFY_USER,
            channel=ActionChannel.CONSOLE,
            message=render(get_template(primary_event or "", "shelter"), **vars_),
            rationale=f"Risk argmax=Moderate, active hazard={primary_event}.",
        ))

        # Shelter recommendation only if flood-flavored and user is on ground.
        if flavor == "flood" and profile.home_floor_state.value == "Ground":
            actions.append(Action(
                kind=ActionKind.RECOMMEND_SHELTER,
                channel=ActionChannel.CONSOLE,
                message=render(get_template(primary_event or "", "shelter"), **vars_),
                rationale="Moderate flood risk + ground-floor home.",
            ))

        # Notify contacts that opted in at MODERATE.
        _maybe_notify_contacts(
            actions, profile, primary, vars_, RiskNotifyLevel.MODERATE,
            rationale="Risk argmax=Moderate.",
        )
        return _sort(actions)

    # ---- HIGH: full notification, shelter, smart-home ----
    if risk_argmax == "High":
        actions.append(Action(
            kind=ActionKind.NOTIFY_USER,
            channel=ActionChannel.CONSOLE,
            message=render(get_template(primary_event or "", "shelter"), **vars_),
            rationale="Risk argmax=High; quiet hours ignored for life-safety.",
        ))

        # Choose shelter vs evacuate.
        immediate = primary is not None and primary.urgency.value == "Immediate"
        no_vehicle = not profile.vehicle.owns_vehicle
        mobility_limited = profile.medical.mobility_limited
        if immediate and (no_vehicle or mobility_limited):
            # Can't evacuate -- recommend shelter.
            actions.append(Action(
                kind=ActionKind.RECOMMEND_SHELTER,
                channel=ActionChannel.CONSOLE,
                message=render(get_template(primary_event or "", "shelter"), **vars_),
                rationale=(
                    "High risk + immediate urgency; user cannot evacuate "
                    "(no vehicle or mobility-limited). Recommending shelter."
                ),
            ))
        else:
            actions.append(Action(
                kind=ActionKind.RECOMMEND_SHELTER,
                channel=ActionChannel.CONSOLE,
                message=render(get_template(primary_event or "", "shelter"), **vars_),
                rationale="Risk argmax=High; recommending shelter-in-place.",
            ))

        _maybe_notify_contacts(
            actions, profile, primary, vars_, RiskNotifyLevel.HIGH,
            rationale="Risk argmax=High.",
        )

        # Smart-home actions.
        if prefs.allow_smart_home_actions:
            if flavor == "flood":
                actions.append(Action(
                    kind=ActionKind.ACTIVATE_FLOOD_LIGHTS,
                    channel=ActionChannel.SMART_HOME,
                    message="Activating exterior flood lights.",
                    rationale="High flood risk + smart-home enabled.",
                ))
            elif flavor == "heat":
                actions.append(Action(
                    kind=ActionKind.ADJUST_THERMOSTAT,
                    channel=ActionChannel.SMART_HOME,
                    message="Setting thermostat to 74°F for cooling.",
                    rationale="High heat risk + smart-home enabled.",
                    metadata={"setpoint_f": 74},
                ))
        return _sort(actions)

    # ---- CRITICAL: everything, override prefs for life-safety ----
    if risk_argmax == "Critical":
        # Notify user regardless of quiet hours.
        actions.append(Action(
            kind=ActionKind.NOTIFY_USER,
            channel=ActionChannel.CONSOLE,
            message=render(get_template(primary_event or "", "shelter"), **vars_),
            rationale="Risk argmax=Critical; quiet hours and prefs overridden.",
        ))

        # Evacuate vs shelter depends on hazard.
        if flavor == "flood":
            # Floods: evacuate to higher ground if possible.
            actions.append(Action(
                kind=ActionKind.RECOMMEND_EVACUATE,
                channel=ActionChannel.CONSOLE,
                message=render(get_template(primary_event or "", "evacuate"), **vars_),
                rationale="Critical flood risk: recommend evacuation to higher ground.",
            ))
        elif flavor == "wind":
            # Tornadoes + severe wind: shelter; never try to outrun in a vehicle.
            actions.append(Action(
                kind=ActionKind.RECOMMEND_SHELTER,
                channel=ActionChannel.CONSOLE,
                message=render(get_template(primary_event or "", "shelter"), **vars_),
                rationale=(
                    "Critical wind/tornado risk: shelter in place is safer "
                    "than attempting to evacuate."
                ),
            ))
        else:
            actions.append(Action(
                kind=ActionKind.RECOMMEND_SHELTER,
                channel=ActionChannel.CONSOLE,
                message=render(get_template(primary_event or "", "shelter"), **vars_),
                rationale="Critical risk; defaulting to shelter recommendation.",
            ))

        # Notify ALL contacts whose notify_on includes CRITICAL.
        _maybe_notify_contacts(
            actions, profile, primary, vars_, RiskNotifyLevel.CRITICAL,
            rationale="Risk argmax=Critical.",
        )

        # Smart-home: sound alarm (always for Critical, unless the user
        # explicitly opted out). Life-safety actions override smart-home pref
        # ONLY for flood lights, which are life-safety critical. We do NOT
        # override the explicit sound_alarm opt-out — that would be surprising.
        if prefs.allow_smart_home_actions:
            actions.append(Action(
                kind=ActionKind.SOUND_ALARM,
                channel=ActionChannel.SMART_HOME,
                message="Sounding household alarm.",
                rationale="Critical risk + smart-home enabled.",
            ))
        if flavor == "flood":
            actions.append(Action(
                kind=ActionKind.ACTIVATE_FLOOD_LIGHTS,
                channel=ActionChannel.SMART_HOME,
                message="Activating exterior flood lights (life-safety override).",
                rationale="Critical flood risk; life-safety override of smart-home pref.",
            ))
        return _sort(actions)

    # ---- Fallback (unknown risk level) ----
    actions.append(Action(
        kind=ActionKind.LOG_ONLY,
        channel=ActionChannel.CONSOLE,
        message=f"Unknown risk level: {risk_argmax!r}",
        rationale="Defensive fallback; planner saw an unexpected argmax.",
    ))
    return _sort(actions)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _maybe_notify_contacts(
    actions: list[Action],
    profile: UserProfile,
    alert: WeatherAlert | None,
    vars_: dict[str, str],
    level: RiskNotifyLevel,
    rationale: str,
) -> None:
    """Append a NOTIFY_CONTACTS action if any contacts opted in at `level`."""
    recipients = profile.contacts_to_notify(level)
    if not recipients:
        return
    event = alert.event if alert else ""
    message = render(get_template(event, "contacts_sms"), **vars_)
    actions.append(Action(
        kind=ActionKind.NOTIFY_CONTACTS,
        channel=ActionChannel.SMS,
        message=message,
        rationale=f"{rationale} Notifying {len(recipients)} contact(s).",
        recipients=tuple(c.phone for c in recipients),
        metadata={"recipient_names": [c.name for c in recipients]},
    ))


def _sort(actions: list[Action]) -> list[Action]:
    """Stable-sort by priority descending so urgent actions surface first."""
    return sorted(actions, key=lambda a: -a.priority)


__all__ = ["plan_actions"]
