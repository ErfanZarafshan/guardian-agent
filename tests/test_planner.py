"""Tests for the rule-augmented planner (Phase 6)."""

from __future__ import annotations

from datetime import datetime, time, timezone

import pytest

from guardian.planning import Action, ActionChannel, ActionKind, plan_actions
from guardian.planning.actions import HAZARD_TEMPLATES, get_template, render
from guardian.planning.planner import (
    _hazard_flavor,
    _is_quiet_hours,
    _pick_primary_alert,
)
from guardian.profile import (
    EmergencyContact,
    Home,
    HomeType,
    Location,
    Medical,
    Preferences,
    QuietHours,
    RiskNotifyLevel,
    UserProfile,
    Vehicle,
    VehicleClearance,
)
from guardian.weather.observation import (
    CertaintyLevel,
    SeverityLevel,
    UrgencyLevel,
    WeatherAlert,
    WeatherObservation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _location() -> Location:
    return Location(
        address="100 Demo St, Baton Rouge, LA 70803",
        latitude=30.4, longitude=-91.2,
        nws_zone_id="LAZ036", county_fips="22033",
    )


def baseline_profile(
    *,
    quiet: QuietHours | None = None,
    smart_home: bool = True,
) -> UserProfile:
    return UserProfile(
        user_id="baseline", name="Alex",
        location=_location(),
        home=Home(type=HomeType.APARTMENT, floor_level=3),
        vehicle=Vehicle(owns_vehicle=True, clearance=VehicleClearance.HIGH),
        medical=Medical(),
        emergency_contacts=[EmergencyContact(
            name="Spouse", relationship="spouse", phone="+15555550100",
            notify_on=[RiskNotifyLevel.HIGH, RiskNotifyLevel.CRITICAL],
        )],
        preferences=Preferences(quiet_hours=quiet, allow_smart_home_actions=smart_home),
    )


def vulnerable_profile(
    *,
    quiet: QuietHours | None = None,
    smart_home: bool = True,
) -> UserProfile:
    return UserProfile(
        user_id="vulnerable", name="Sam",
        location=_location(),
        home=Home(type=HomeType.APARTMENT, floor_level=1, flood_zone="AE"),
        vehicle=Vehicle(owns_vehicle=False, clearance=VehicleClearance.NONE),
        medical=Medical(mobility_limited=True),
        emergency_contacts=[
            EmergencyContact(
                name="Sibling", relationship="sibling", phone="+15555550101",
                notify_on=[RiskNotifyLevel.MODERATE, RiskNotifyLevel.HIGH,
                           RiskNotifyLevel.CRITICAL],
            ),
            EmergencyContact(
                name="Parent", relationship="parent", phone="+15555550102",
                notify_on=[RiskNotifyLevel.CRITICAL],
            ),
        ],
        preferences=Preferences(quiet_hours=quiet, allow_smart_home_actions=smart_home),
    )


def _obs_now() -> datetime:
    return datetime(2026, 7, 15, 14, 0, 0, tzinfo=timezone.utc)


def _obs_3am() -> datetime:
    return datetime(2026, 7, 15, 3, 0, 0, tzinfo=timezone.utc)


def _calm_observation(when: datetime = None) -> WeatherObservation:
    return WeatherObservation(
        observed_at=when or _obs_now(),
        latitude=30.4, longitude=-91.2,
        wind_speed_mph=5.0, precip_rate_in_hr=0.0,
        sources=["nws", "owm"], alerts=[],
    )


def _alert(event: str, severity: SeverityLevel, urgency: UrgencyLevel) -> WeatherAlert:
    return WeatherAlert(
        source="nws", event=event, severity=severity, urgency=urgency,
        certainty=CertaintyLevel.LIKELY,
    )


def _obs_with_alert(event: str, severity: SeverityLevel, urgency: UrgencyLevel,
                    when: datetime = None) -> WeatherObservation:
    return WeatherObservation(
        observed_at=when or _obs_now(),
        latitude=30.4, longitude=-91.2,
        wind_speed_mph=20.0, precip_rate_in_hr=0.5,
        sources=["nws", "owm"],
        alerts=[_alert(event, severity, urgency)],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kinds(actions: list[Action]) -> list[ActionKind]:
    return [a.kind for a in actions]


# ---------------------------------------------------------------------------
# Action catalog tests
# ---------------------------------------------------------------------------

def test_template_lookup_default() -> None:
    """Unknown event falls back to default template."""
    t = get_template("Some Never-Before-Seen Event", "shelter")
    assert t == HAZARD_TEMPLATES["__default__"]["shelter"]


def test_template_lookup_specific() -> None:
    t = get_template("Tornado Warning", "shelter")
    assert "TORNADO WARNING" in t


def test_render_fills_variables() -> None:
    template = "Hello {name}, at {address}"
    out = render(template, name="Alex", address="Baton Rouge")
    assert out == "Hello Alex, at Baton Rouge"


def test_render_missing_vars_become_empty() -> None:
    template = "Hello {name}, expires {expires}"
    out = render(template, name="Alex")
    assert out == "Hello Alex, expires "


def test_action_priority_ordering() -> None:
    """Evacuate > shelter > alarm > notify > log_only."""
    from guardian.planning.actions import _PRIORITY
    assert _PRIORITY[ActionKind.RECOMMEND_EVACUATE] > _PRIORITY[ActionKind.RECOMMEND_SHELTER]
    assert _PRIORITY[ActionKind.RECOMMEND_SHELTER] > _PRIORITY[ActionKind.SOUND_ALARM]
    assert _PRIORITY[ActionKind.NOTIFY_CONTACTS] > _PRIORITY[ActionKind.NOTIFY_USER]
    assert _PRIORITY[ActionKind.NOTIFY_USER] > _PRIORITY[ActionKind.LOG_ONLY]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def test_pick_primary_alert_prefers_higher_severity() -> None:
    alerts = [
        _alert("Heat Advisory", SeverityLevel.MINOR, UrgencyLevel.EXPECTED),
        _alert("Tornado Warning", SeverityLevel.EXTREME, UrgencyLevel.IMMEDIATE),
    ]
    primary = _pick_primary_alert(alerts)
    assert primary is not None and primary.event == "Tornado Warning"


def test_pick_primary_alert_none_when_empty() -> None:
    assert _pick_primary_alert([]) is None


def test_quiet_hours_same_day_window() -> None:
    """Daytime window: 13:00-15:00"""
    qs, qe = time(13, 0), time(15, 0)
    assert _is_quiet_hours(datetime(2026, 1, 1, 14, 0), qs, qe) is True
    assert _is_quiet_hours(datetime(2026, 1, 1, 10, 0), qs, qe) is False


def test_quiet_hours_overnight_window() -> None:
    """Overnight window: 22:00-07:00"""
    qs, qe = time(22, 0), time(7, 0)
    assert _is_quiet_hours(datetime(2026, 1, 1, 3, 0), qs, qe) is True
    assert _is_quiet_hours(datetime(2026, 1, 1, 23, 0), qs, qe) is True
    assert _is_quiet_hours(datetime(2026, 1, 1, 12, 0), qs, qe) is False
    assert _is_quiet_hours(datetime(2026, 1, 1, 7, 0), qs, qe) is True


def test_quiet_hours_none_disables() -> None:
    assert _is_quiet_hours(datetime(2026, 1, 1, 3, 0), None, None) is False


def test_hazard_flavor_classification() -> None:
    assert _hazard_flavor("Flash Flood Warning") == "flood"
    assert _hazard_flavor("Tornado Warning") == "wind"
    assert _hazard_flavor("Excessive Heat Warning") == "heat"
    assert _hazard_flavor("Dense Fog Advisory") == "other"
    assert _hazard_flavor(None) == "none"


# ---------------------------------------------------------------------------
# LOW risk
# ---------------------------------------------------------------------------

def test_low_risk_no_alerts_logs_only() -> None:
    actions = plan_actions("Low", baseline_profile(), _calm_observation())
    assert _kinds(actions) == [ActionKind.LOG_ONLY]


def test_low_risk_with_alert_notifies_heads_up() -> None:
    obs = _obs_with_alert("Heat Advisory", SeverityLevel.MINOR,
                          UrgencyLevel.EXPECTED)
    actions = plan_actions("Low", baseline_profile(), obs)
    kinds = _kinds(actions)
    assert ActionKind.NOTIFY_USER in kinds
    assert ActionKind.LOG_ONLY in kinds


def test_low_risk_in_quiet_hours_stays_quiet() -> None:
    quiet = QuietHours(start=time(22, 0), end=time(7, 0))
    obs = _obs_with_alert("Heat Advisory", SeverityLevel.MINOR,
                          UrgencyLevel.EXPECTED, when=_obs_3am())
    actions = plan_actions("Low", baseline_profile(quiet=quiet), obs)
    assert _kinds(actions) == [ActionKind.LOG_ONLY]


# ---------------------------------------------------------------------------
# MODERATE risk
# ---------------------------------------------------------------------------

def test_moderate_risk_notifies_user() -> None:
    obs = _obs_with_alert("Flash Flood Watch", SeverityLevel.MODERATE,
                          UrgencyLevel.EXPECTED)
    actions = plan_actions("Moderate", baseline_profile(), obs)
    assert ActionKind.NOTIFY_USER in _kinds(actions)


def test_moderate_risk_quiet_hours_demotes_for_non_vulnerable() -> None:
    quiet = QuietHours(start=time(22, 0), end=time(7, 0))
    obs = _obs_with_alert("Flash Flood Watch", SeverityLevel.MODERATE,
                          UrgencyLevel.EXPECTED, when=_obs_3am())
    actions = plan_actions("Moderate", baseline_profile(quiet=quiet), obs)
    assert _kinds(actions) == [ActionKind.LOG_ONLY]


def test_moderate_risk_quiet_hours_still_acts_for_vulnerable() -> None:
    quiet = QuietHours(start=time(22, 0), end=time(7, 0))
    obs = _obs_with_alert("Flash Flood Watch", SeverityLevel.MODERATE,
                          UrgencyLevel.EXPECTED, when=_obs_3am())
    actions = plan_actions("Moderate", vulnerable_profile(quiet=quiet), obs)
    assert ActionKind.NOTIFY_USER in _kinds(actions)


def test_moderate_risk_flood_groundfloor_recommends_shelter() -> None:
    obs = _obs_with_alert("Flash Flood Watch", SeverityLevel.MODERATE,
                          UrgencyLevel.EXPECTED)
    actions = plan_actions("Moderate", vulnerable_profile(), obs)
    assert ActionKind.RECOMMEND_SHELTER in _kinds(actions)


def test_moderate_notifies_contacts_that_opted_in() -> None:
    obs = _obs_with_alert("Flash Flood Watch", SeverityLevel.MODERATE,
                          UrgencyLevel.EXPECTED)
    # Sibling in vulnerable profile opts in at MODERATE, Parent at CRITICAL only.
    actions = plan_actions("Moderate", vulnerable_profile(), obs)
    notify = next(a for a in actions if a.kind == ActionKind.NOTIFY_CONTACTS)
    assert notify.recipients == ("+15555550101",)  # just sibling


# ---------------------------------------------------------------------------
# HIGH risk
# ---------------------------------------------------------------------------

def test_high_risk_ignores_quiet_hours() -> None:
    quiet = QuietHours(start=time(22, 0), end=time(7, 0))
    obs = _obs_with_alert("Flash Flood Warning", SeverityLevel.SEVERE,
                          UrgencyLevel.IMMEDIATE, when=_obs_3am())
    actions = plan_actions("High", baseline_profile(quiet=quiet), obs)
    assert ActionKind.NOTIFY_USER in _kinds(actions)
    assert ActionKind.RECOMMEND_SHELTER in _kinds(actions)


def test_high_risk_flood_triggers_flood_lights_when_smart_home_on() -> None:
    obs = _obs_with_alert("Flash Flood Warning", SeverityLevel.SEVERE,
                          UrgencyLevel.IMMEDIATE)
    actions = plan_actions("High", baseline_profile(smart_home=True), obs)
    assert ActionKind.ACTIVATE_FLOOD_LIGHTS in _kinds(actions)


def test_high_risk_flood_respects_smart_home_opt_out() -> None:
    obs = _obs_with_alert("Flash Flood Warning", SeverityLevel.SEVERE,
                          UrgencyLevel.IMMEDIATE)
    actions = plan_actions("High", baseline_profile(smart_home=False), obs)
    assert ActionKind.ACTIVATE_FLOOD_LIGHTS not in _kinds(actions)


def test_high_risk_heat_triggers_thermostat() -> None:
    obs = _obs_with_alert("Excessive Heat Warning", SeverityLevel.SEVERE,
                          UrgencyLevel.EXPECTED)
    actions = plan_actions("High", baseline_profile(smart_home=True), obs)
    therm = [a for a in actions if a.kind == ActionKind.ADJUST_THERMOSTAT]
    assert len(therm) == 1
    assert "setpoint_f" in therm[0].metadata


def test_high_risk_notifies_high_level_contacts() -> None:
    obs = _obs_with_alert("Flash Flood Warning", SeverityLevel.SEVERE,
                          UrgencyLevel.IMMEDIATE)
    actions = plan_actions("High", baseline_profile(), obs)
    notify = next(a for a in actions if a.kind == ActionKind.NOTIFY_CONTACTS)
    assert notify.recipients == ("+15555550100",)  # spouse


# ---------------------------------------------------------------------------
# CRITICAL risk
# ---------------------------------------------------------------------------

def test_critical_flood_recommends_evacuation() -> None:
    obs = _obs_with_alert("Flash Flood Warning", SeverityLevel.EXTREME,
                          UrgencyLevel.IMMEDIATE)
    actions = plan_actions("Critical", baseline_profile(), obs)
    kinds = _kinds(actions)
    assert ActionKind.RECOMMEND_EVACUATE in kinds
    assert ActionKind.RECOMMEND_SHELTER not in kinds


def test_critical_tornado_recommends_shelter_not_evacuate() -> None:
    """Tornados: shelter, don't try to outrun."""
    obs = _obs_with_alert("Tornado Warning", SeverityLevel.EXTREME,
                          UrgencyLevel.IMMEDIATE)
    actions = plan_actions("Critical", baseline_profile(), obs)
    kinds = _kinds(actions)
    assert ActionKind.RECOMMEND_SHELTER in kinds
    assert ActionKind.RECOMMEND_EVACUATE not in kinds


def test_critical_notifies_all_critical_contacts() -> None:
    obs = _obs_with_alert("Tornado Warning", SeverityLevel.EXTREME,
                          UrgencyLevel.IMMEDIATE)
    actions = plan_actions("Critical", vulnerable_profile(), obs)
    notify = next(a for a in actions if a.kind == ActionKind.NOTIFY_CONTACTS)
    # Both sibling (threshold=MODERATE, so CRITICAL triggers) and parent
    # (threshold=CRITICAL) should be contacted.
    assert set(notify.recipients) == {"+15555550101", "+15555550102"}


def test_critical_flood_life_safety_overrides_smart_home_pref() -> None:
    """Even if the user opted out of smart-home, flood lights still activate
    for a Critical flood (life-safety override)."""
    obs = _obs_with_alert("Flash Flood Warning", SeverityLevel.EXTREME,
                          UrgencyLevel.IMMEDIATE)
    actions = plan_actions("Critical", baseline_profile(smart_home=False), obs)
    assert ActionKind.ACTIVATE_FLOOD_LIGHTS in _kinds(actions)


def test_critical_alarm_respects_smart_home_opt_out() -> None:
    """The audible alarm is not a life-safety override — if the user opted
    out, we don't sound it (we still notify them via console + SMS)."""
    obs = _obs_with_alert("Tornado Warning", SeverityLevel.EXTREME,
                          UrgencyLevel.IMMEDIATE)
    actions = plan_actions("Critical", baseline_profile(smart_home=False), obs)
    assert ActionKind.SOUND_ALARM not in _kinds(actions)


def test_critical_ignores_quiet_hours() -> None:
    quiet = QuietHours(start=time(22, 0), end=time(7, 0))
    obs = _obs_with_alert("Tornado Warning", SeverityLevel.EXTREME,
                          UrgencyLevel.IMMEDIATE, when=_obs_3am())
    actions = plan_actions("Critical", baseline_profile(quiet=quiet), obs)
    assert ActionKind.NOTIFY_USER in _kinds(actions)
    assert ActionKind.RECOMMEND_SHELTER in _kinds(actions)


# ---------------------------------------------------------------------------
# Sorting / output shape
# ---------------------------------------------------------------------------

def test_actions_sorted_by_priority_descending() -> None:
    obs = _obs_with_alert("Tornado Warning", SeverityLevel.EXTREME,
                          UrgencyLevel.IMMEDIATE)
    actions = plan_actions("Critical", vulnerable_profile(), obs)
    priorities = [a.priority for a in actions]
    assert priorities == sorted(priorities, reverse=True)


def test_every_action_has_rationale() -> None:
    """Explainability: no action should ship without a reason attached."""
    obs = _obs_with_alert("Tornado Warning", SeverityLevel.EXTREME,
                          UrgencyLevel.IMMEDIATE)
    actions = plan_actions("Critical", vulnerable_profile(), obs)
    assert all(a.rationale for a in actions)


def test_contacts_action_is_sms_channel() -> None:
    obs = _obs_with_alert("Tornado Warning", SeverityLevel.EXTREME,
                          UrgencyLevel.IMMEDIATE)
    actions = plan_actions("Critical", vulnerable_profile(), obs)
    contacts_action = next(
        a for a in actions if a.kind == ActionKind.NOTIFY_CONTACTS
    )
    assert contacts_action.channel is ActionChannel.SMS
    assert len(contacts_action.recipients) > 0


def test_smart_home_actions_are_smart_home_channel() -> None:
    obs = _obs_with_alert("Flash Flood Warning", SeverityLevel.EXTREME,
                          UrgencyLevel.IMMEDIATE)
    actions = plan_actions("Critical", baseline_profile(smart_home=True), obs)
    smart_home_actions = [
        a for a in actions
        if a.kind in (ActionKind.ACTIVATE_FLOOD_LIGHTS, ActionKind.SOUND_ALARM,
                      ActionKind.ADJUST_THERMOSTAT)
    ]
    assert len(smart_home_actions) >= 1
    for a in smart_home_actions:
        assert a.channel is ActionChannel.SMART_HOME


# ---------------------------------------------------------------------------
# Personalization
# ---------------------------------------------------------------------------

def test_vulnerable_user_gets_more_actions_than_baseline_at_moderate() -> None:
    """At Moderate, the vulnerable user gets shelter recs and more contacts
    pinged than the baseline user for the same hazard."""
    obs = _obs_with_alert("Flash Flood Watch", SeverityLevel.MODERATE,
                          UrgencyLevel.EXPECTED)
    base_actions = plan_actions("Moderate", baseline_profile(), obs)
    vuln_actions = plan_actions("Moderate", vulnerable_profile(), obs)
    assert len(vuln_actions) > len(base_actions)


def test_baseline_tornado_critical_urgent_gets_shelter_not_evacuate() -> None:
    """Even baseline user (with vehicle) gets SHELTER for tornado, never
    EVACUATE — this is a life-safety rule, not a personalization."""
    obs = _obs_with_alert("Tornado Warning", SeverityLevel.EXTREME,
                          UrgencyLevel.IMMEDIATE)
    actions = plan_actions("Critical", baseline_profile(), obs)
    assert ActionKind.RECOMMEND_SHELTER in _kinds(actions)
    assert ActionKind.RECOMMEND_EVACUATE not in _kinds(actions)


# ---------------------------------------------------------------------------
# Unknown risk argmax (defensive)
# ---------------------------------------------------------------------------

def test_unknown_risk_level_falls_back_to_log_only() -> None:
    actions = plan_actions("UNKNOWN_STATE", baseline_profile(), _calm_observation())
    assert _kinds(actions) == [ActionKind.LOG_ONLY]
