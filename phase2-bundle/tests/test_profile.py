"""Tests for the user profile schema and storage."""

from __future__ import annotations

import json
from datetime import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from guardian.profile import (
    EmergencyContact,
    Home,
    HomeFloorState,
    HomeType,
    Location,
    Medical,
    Preferences,
    QuietHours,
    RiskNotifyLevel,
    UserProfile,
    Vehicle,
    VehicleClearance,
    load_profile,
    save_profile,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PROFILE = REPO_ROOT / "config" / "example_profile.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _minimal_profile_kwargs() -> dict:
    """Smallest valid profile — used as a base that tests tweak."""
    return {
        "user_id": "test-user",
        "name": "Test User",
        "location": Location(
            address="1 Test St, Baton Rouge, LA",
            latitude=30.4,
            longitude=-91.2,
        ),
        "home": Home(type=HomeType.APARTMENT, floor_level=1),
        "vehicle": Vehicle(owns_vehicle=True, clearance=VehicleClearance.LOW),
        "medical": Medical(),
        "emergency_contacts": [],
    }


@pytest.fixture
def minimal_profile() -> UserProfile:
    return UserProfile(**_minimal_profile_kwargs())


# ---------------------------------------------------------------------------
# Loading the shipped example
# ---------------------------------------------------------------------------

def test_example_profile_loads_and_validates() -> None:
    """The JSON we ship as a template must validate against our schema."""
    profile = load_profile(EXAMPLE_PROFILE)
    assert profile.user_id == "demo-user-001"
    assert profile.location.latitude == pytest.approx(30.4133)
    assert len(profile.emergency_contacts) == 2
    assert profile.home_floor_state is HomeFloorState.GROUND


# ---------------------------------------------------------------------------
# Schema validation — happy paths
# ---------------------------------------------------------------------------

def test_minimal_profile_valid(minimal_profile: UserProfile) -> None:
    assert minimal_profile.name == "Test User"
    # Defaults should have filled in preferences
    assert minimal_profile.preferences.language == "en"
    assert minimal_profile.preferences.allow_smart_home_actions is True


def test_location_bounds_enforced() -> None:
    with pytest.raises(ValidationError):
        Location(address="x", latitude=95.0, longitude=0.0)
    with pytest.raises(ValidationError):
        Location(address="x", latitude=0.0, longitude=-181.0)


def test_user_id_pattern_enforced() -> None:
    kwargs = _minimal_profile_kwargs()
    kwargs["user_id"] = "has spaces!"
    with pytest.raises(ValidationError):
        UserProfile(**kwargs)


def test_nws_zone_pattern_enforced() -> None:
    with pytest.raises(ValidationError):
        Location(address="x", latitude=30.0, longitude=-91.0, nws_zone_id="lowercase")


def test_county_fips_pattern_enforced() -> None:
    with pytest.raises(ValidationError):
        Location(address="x", latitude=30.0, longitude=-91.0, county_fips="ABCDE")


def test_phone_must_be_e164() -> None:
    with pytest.raises(ValidationError):
        EmergencyContact(name="x", relationship="friend", phone="555-1234")
    with pytest.raises(ValidationError):
        EmergencyContact(name="x", relationship="friend", phone="15551234567")  # no +


def test_phone_accepts_e164() -> None:
    c = EmergencyContact(name="x", relationship="friend", phone="+15551234567")
    assert c.phone == "+15551234567"


def test_vehicle_normalizes_when_not_owned() -> None:
    # If user toggles owns_vehicle off, clearance is forced to NONE.
    v = Vehicle(owns_vehicle=False, clearance=VehicleClearance.HIGH, four_wheel_drive=True)
    assert v.clearance is VehicleClearance.NONE
    assert v.four_wheel_drive is False


def test_floor_level_lower_bound() -> None:
    with pytest.raises(ValidationError):
        Home(type=HomeType.HOUSE, floor_level=0)


# ---------------------------------------------------------------------------
# Derived helpers
# ---------------------------------------------------------------------------

def test_home_floor_state_ground(minimal_profile: UserProfile) -> None:
    assert minimal_profile.home_floor_state is HomeFloorState.GROUND


def test_home_floor_state_upper() -> None:
    kwargs = _minimal_profile_kwargs()
    kwargs["home"] = Home(type=HomeType.APARTMENT, floor_level=3)
    p = UserProfile(**kwargs)
    assert p.home_floor_state is HomeFloorState.UPPER


def test_home_floor_state_elevated_even_on_floor_one() -> None:
    """A pier-and-beam house on 'floor 1' is still elevated."""
    kwargs = _minimal_profile_kwargs()
    kwargs["home"] = Home(type=HomeType.HOUSE, floor_level=1, elevated=True)
    p = UserProfile(**kwargs)
    assert p.home_floor_state is HomeFloorState.ELEVATED


def test_medical_vulnerability_false_by_default(minimal_profile: UserProfile) -> None:
    assert minimal_profile.is_medically_vulnerable is False


@pytest.mark.parametrize(
    "field",
    ["mobility_limited", "oxygen_dependent", "refrigerated_medication"],
)
def test_medical_vulnerability_triggered_by_any_flag(field: str) -> None:
    kwargs = _minimal_profile_kwargs()
    kwargs["medical"] = Medical(**{field: True})
    p = UserProfile(**kwargs)
    assert p.is_medically_vulnerable is True


def test_medical_vulnerability_from_conditions_list() -> None:
    kwargs = _minimal_profile_kwargs()
    kwargs["medical"] = Medical(chronic_conditions=["asthma"])
    p = UserProfile(**kwargs)
    assert p.is_medically_vulnerable is True


# ---------------------------------------------------------------------------
# contacts_to_notify: ranking logic
# ---------------------------------------------------------------------------

def _profile_with_contacts() -> UserProfile:
    kwargs = _minimal_profile_kwargs()
    kwargs["emergency_contacts"] = [
        EmergencyContact(
            name="Mod",
            relationship="friend",
            phone="+15550000001",
            notify_on=[RiskNotifyLevel.MODERATE],
        ),
        EmergencyContact(
            name="High",
            relationship="friend",
            phone="+15550000002",
            notify_on=[RiskNotifyLevel.HIGH],
        ),
        EmergencyContact(
            name="Crit",
            relationship="friend",
            phone="+15550000003",
            notify_on=[RiskNotifyLevel.CRITICAL],
        ),
    ]
    return UserProfile(**kwargs)


def test_contacts_to_notify_moderate_matches_only_moderate() -> None:
    p = _profile_with_contacts()
    names = [c.name for c in p.contacts_to_notify(RiskNotifyLevel.MODERATE)]
    assert names == ["Mod"]


def test_contacts_to_notify_high_matches_moderate_and_high() -> None:
    p = _profile_with_contacts()
    names = sorted(c.name for c in p.contacts_to_notify(RiskNotifyLevel.HIGH))
    assert names == ["High", "Mod"]


def test_contacts_to_notify_critical_matches_all() -> None:
    p = _profile_with_contacts()
    names = sorted(c.name for c in p.contacts_to_notify(RiskNotifyLevel.CRITICAL))
    assert names == ["Crit", "High", "Mod"]


# ---------------------------------------------------------------------------
# Round-trip save/load
# ---------------------------------------------------------------------------

def test_round_trip_save_and_load(tmp_path: Path, minimal_profile: UserProfile) -> None:
    out = tmp_path / "profile.json"
    save_profile(minimal_profile, out)
    assert out.exists()

    loaded = load_profile(out)
    assert loaded.model_dump() == minimal_profile.model_dump()


def test_save_creates_parent_dirs(tmp_path: Path, minimal_profile: UserProfile) -> None:
    nested = tmp_path / "deeply" / "nested" / "profile.json"
    save_profile(minimal_profile, nested)
    assert nested.exists()


def test_load_nonexistent_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_profile(tmp_path / "does_not_exist.json")


def test_load_malformed_json_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{ this is not json ")
    with pytest.raises(json.JSONDecodeError):
        load_profile(bad)


def test_load_valid_json_but_invalid_schema_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"user_id": "x", "name": "y"}')  # missing required fields
    with pytest.raises(ValidationError):
        load_profile(bad)


# ---------------------------------------------------------------------------
# Quiet hours serialization
# ---------------------------------------------------------------------------

def test_quiet_hours_round_trip(tmp_path: Path) -> None:
    kwargs = _minimal_profile_kwargs()
    kwargs["preferences"] = Preferences(
        quiet_hours=QuietHours(start=time(22, 0), end=time(7, 0))
    )
    p = UserProfile(**kwargs)
    out = tmp_path / "p.json"
    save_profile(p, out)
    loaded = load_profile(out)
    assert loaded.preferences.quiet_hours is not None
    assert loaded.preferences.quiet_hours.start == time(22, 0)
    assert loaded.preferences.quiet_hours.end == time(7, 0)
