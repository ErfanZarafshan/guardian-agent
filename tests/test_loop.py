"""Tests for the Phase 8 agent loop.

These tests exercise `run_cycle` end-to-end using mocked weather clients
(so no real HTTP), a real Bayesian network, and mocked dispatchers (so no
real SMS / smart-home calls).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from guardian.loop import AgentState, CycleFingerprint, run_cycle
from guardian.output.console import ConsoleDispatcher
from guardian.output.dispatch import Dispatcher
from guardian.output.smart_home import SmartHomeDispatcher
from guardian.output.sms import SMSDispatcher
from guardian.profile import (
    EmergencyContact,
    Home,
    HomeType,
    Location,
    Medical,
    Preferences,
    RiskNotifyLevel,
    UserProfile,
    Vehicle,
    VehicleClearance,
)
from guardian.risk.risk_engine import RiskEngine
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


@pytest.fixture
def baseline_profile() -> UserProfile:
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
        preferences=Preferences(allow_smart_home_actions=True),
    )


@pytest.fixture(scope="module")
def shared_engine() -> RiskEngine:
    """Build one RiskEngine per module — Bayesian network construction is slow."""
    return RiskEngine()


@pytest.fixture
def dry_run_dispatcher(monkeypatch) -> Dispatcher:
    """Dispatcher with all three channels live but SMS in dry-run mode."""
    from guardian import config as config_module
    monkeypatch.setattr(config_module, "load_dotenv", lambda *a, **k: None)
    for var, val in {
        "SMS_DRY_RUN": "true",
        "SMS_MAX_PER_RUN": "10",
        "TWILIO_ACCOUNT_SID": "",
        "TWILIO_AUTH_TOKEN": "",
        "TWILIO_FROM_NUMBER": "",
    }.items():
        monkeypatch.setenv(var, val)
    config_module.get_config.cache_clear()
    try:
        yield Dispatcher(
            console=ConsoleDispatcher(),
            sms=SMSDispatcher(client=MagicMock()),
            smart_home=SmartHomeDispatcher(),
        )
    finally:
        config_module.get_config.cache_clear()


def _calm_obs() -> WeatherObservation:
    return WeatherObservation(
        observed_at=datetime(2026, 7, 15, 18, 0, 0, tzinfo=timezone.utc),
        latitude=30.4, longitude=-91.2,
        wind_speed_mph=5.0, precip_rate_in_hr=0.0,
        sources=["nws", "owm"], alerts=[],
    )


def _tornado_obs() -> WeatherObservation:
    return WeatherObservation(
        observed_at=datetime(2026, 7, 15, 18, 0, 0, tzinfo=timezone.utc),
        latitude=30.4, longitude=-91.2,
        wind_speed_mph=42.0, wind_gust_mph=70.0, precip_rate_in_hr=0.6,
        sources=["nws", "owm"],
        alerts=[WeatherAlert(
            source="nws", event="Tornado Warning",
            severity=SeverityLevel.EXTREME, urgency=UrgencyLevel.IMMEDIATE,
            certainty=CertaintyLevel.OBSERVED,
        )],
    )


# ---------------------------------------------------------------------------
# CycleFingerprint
# ---------------------------------------------------------------------------

def test_fingerprint_equal_for_same_world() -> None:
    fp1 = CycleFingerprint.from_report("Low", _calm_obs())
    fp2 = CycleFingerprint.from_report("Low", _calm_obs())
    assert fp1 == fp2


def test_fingerprint_changes_when_risk_changes() -> None:
    fp1 = CycleFingerprint.from_report("Low", _calm_obs())
    fp2 = CycleFingerprint.from_report("High", _calm_obs())
    assert fp1 != fp2


def test_fingerprint_changes_when_alerts_change() -> None:
    fp1 = CycleFingerprint.from_report("Low", _calm_obs())
    fp2 = CycleFingerprint.from_report("Low", _tornado_obs())
    assert fp1 != fp2


# ---------------------------------------------------------------------------
# run_cycle happy paths
# ---------------------------------------------------------------------------

def _mock_observe(obs: WeatherObservation):
    """Return a patch target for guardian.loop.observe."""
    return patch("guardian.loop.observe", return_value=obs)


def test_cycle_calm_day_produces_low_risk(
    baseline_profile, shared_engine, dry_run_dispatcher
) -> None:
    state = AgentState()
    with _mock_observe(_calm_obs()):
        report = run_cycle(baseline_profile, shared_engine, dry_run_dispatcher, state)
    assert report.risk_argmax == "Low"
    assert report.actions_planned > 0
    assert report.error is None
    assert report.has_active_alerts is False


def test_cycle_tornado_produces_critical_risk(
    baseline_profile, shared_engine, dry_run_dispatcher
) -> None:
    state = AgentState()
    with _mock_observe(_tornado_obs()):
        report = run_cycle(baseline_profile, shared_engine, dry_run_dispatcher, state)
    assert report.risk_argmax in ("High", "Critical")
    assert report.has_active_alerts is True
    assert report.actions_planned >= 2  # at minimum: shelter + notify user


def test_cycle_updates_state_fingerprint(
    baseline_profile, shared_engine, dry_run_dispatcher
) -> None:
    state = AgentState()
    assert state.last_fingerprint is None
    with _mock_observe(_calm_obs()):
        run_cycle(baseline_profile, shared_engine, dry_run_dispatcher, state)
    assert state.last_fingerprint is not None


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def test_second_identical_cycle_flags_fingerprint_match(
    baseline_profile, shared_engine, dry_run_dispatcher
) -> None:
    state = AgentState()
    with _mock_observe(_tornado_obs()):
        report1 = run_cycle(baseline_profile, shared_engine, dry_run_dispatcher, state)
        report2 = run_cycle(baseline_profile, shared_engine, dry_run_dispatcher, state)
    assert report1.fingerprint_matched_previous is False
    assert report2.fingerprint_matched_previous is True


def test_dedup_suppresses_sms_and_smart_home_on_repeat(
    baseline_profile, shared_engine, dry_run_dispatcher
) -> None:
    """Second identical cycle -> SMS + smart-home actions dropped, console kept."""
    state = AgentState()
    with _mock_observe(_tornado_obs()):
        report1 = run_cycle(baseline_profile, shared_engine, dry_run_dispatcher, state)
        report2 = run_cycle(baseline_profile, shared_engine, dry_run_dispatcher, state)
    # Cycle 2 should have strictly fewer dispatched actions than cycle 1.
    assert report2.actions_suppressed_by_dedup > 0
    # But at least one console action still goes through (the heartbeat).
    assert report2.actions_dispatched >= 1


def test_dedup_does_not_fire_when_world_changes(
    baseline_profile, shared_engine, dry_run_dispatcher
) -> None:
    """Calm -> tornado: fingerprint changes, no suppression."""
    state = AgentState()
    with _mock_observe(_calm_obs()):
        run_cycle(baseline_profile, shared_engine, dry_run_dispatcher, state)
    with _mock_observe(_tornado_obs()):
        report2 = run_cycle(baseline_profile, shared_engine, dry_run_dispatcher, state)
    assert report2.fingerprint_matched_previous is False
    assert report2.actions_suppressed_by_dedup == 0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_cycle_handles_observation_failure_without_crashing(
    baseline_profile, shared_engine, dry_run_dispatcher
) -> None:
    state = AgentState()
    with patch("guardian.loop.observe", side_effect=RuntimeError("apis down")):
        report = run_cycle(baseline_profile, shared_engine, dry_run_dispatcher, state)
    assert report.error is not None
    assert "apis down" in report.error
    assert report.risk_argmax == "Unknown"
    # Dispatcher should not have been called at all.
    assert report.actions_planned == 0


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def test_cycle_appends_to_jsonl_log(
    baseline_profile, shared_engine, dry_run_dispatcher, tmp_path: Path,
) -> None:
    log_path = tmp_path / "cycles.jsonl"
    state = AgentState()
    with _mock_observe(_calm_obs()):
        run_cycle(
            baseline_profile, shared_engine, dry_run_dispatcher, state,
            cycle_log_path=log_path,
        )
        run_cycle(
            baseline_profile, shared_engine, dry_run_dispatcher, state,
            cycle_log_path=log_path,
        )
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2
    # Each line is a valid JSON object with the expected fields.
    for line in lines:
        record = json.loads(line)
        assert "risk_argmax" in record
        assert "risk_posterior" in record
        assert "profile_user_id" in record
        assert record["profile_user_id"] == "baseline"


def test_cycle_log_contains_error_on_failure(
    baseline_profile, shared_engine, dry_run_dispatcher, tmp_path: Path,
) -> None:
    log_path = tmp_path / "cycles.jsonl"
    state = AgentState()
    with patch("guardian.loop.observe", side_effect=RuntimeError("nope")):
        run_cycle(
            baseline_profile, shared_engine, dry_run_dispatcher, state,
            cycle_log_path=log_path,
        )
    record = json.loads(log_path.read_text().strip())
    assert record["error"] and "nope" in record["error"]


def test_cycle_works_without_log_path(
    baseline_profile, shared_engine, dry_run_dispatcher,
) -> None:
    """Passing cycle_log_path=None should not error."""
    state = AgentState()
    with _mock_observe(_calm_obs()):
        report = run_cycle(
            baseline_profile, shared_engine, dry_run_dispatcher, state,
            cycle_log_path=None,
        )
    assert report.error is None


# ---------------------------------------------------------------------------
# Report serialization
# ---------------------------------------------------------------------------

def test_cycle_report_to_json_dict_is_serializable(
    baseline_profile, shared_engine, dry_run_dispatcher,
) -> None:
    state = AgentState()
    with _mock_observe(_tornado_obs()):
        report = run_cycle(baseline_profile, shared_engine, dry_run_dispatcher, state)
    d = report.to_json_dict()
    # Should round-trip through json.
    s = json.dumps(d)
    assert "risk_posterior" in json.loads(s)
    # Non-serializable fields should have been stripped.
    assert "actions" not in d
    assert "observation" not in d
    assert "dispatch_report" not in d


# ---------------------------------------------------------------------------
# Dispatcher wiring through loop
# ---------------------------------------------------------------------------

def test_cycle_calls_dispatcher_once_per_cycle(
    baseline_profile, shared_engine,
) -> None:
    """The loop should call dispatcher.dispatch exactly once per cycle."""
    mock_dispatcher = MagicMock(spec=Dispatcher)
    # Configure the mock to return something shaped like a real DispatchReport.
    from guardian.output.console import ConsoleDispatchReport
    from guardian.output.dispatch import DispatchReport
    from guardian.output.smart_home import SmartHomeDispatchReport
    from guardian.output.sms import SMSDispatchReport
    mock_dispatcher.dispatch.return_value = DispatchReport(
        console=ConsoleDispatchReport(attempted=0),
        sms=SMSDispatchReport(attempted=0, sent=0, dry_run=0, skipped=0, failed=0),
        smart_home=SmartHomeDispatchReport(attempted=0, mocked=0, unsupported=0),
    )
    state = AgentState()
    with _mock_observe(_calm_obs()):
        run_cycle(baseline_profile, shared_engine, mock_dispatcher, state)
    mock_dispatcher.dispatch.assert_called_once()
