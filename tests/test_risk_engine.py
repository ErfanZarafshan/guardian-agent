"""Integration tests for RiskEngine: WeatherObservation + UserProfile -> RiskAssessment."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from guardian.profile import (
    EmergencyContact,
    Home,
    HomeType,
    Location,
    Medical,
    UserProfile,
    Vehicle,
    VehicleClearance,
)
from guardian.risk.bayesian import RiskInference
from guardian.risk.classifier import (
    DEFAULT_BUCKET_THRESHOLDS,
    ThreatBucket,
    ThreatScore,
)
from guardian.risk.risk_engine import RiskEngine, _extract_state_name
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

@pytest.fixture(scope="module")
def shared_inference() -> RiskInference:
    """Build the network once across the module."""
    return RiskInference()


def _location() -> Location:
    return Location(
        address="100 Demo St, Baton Rouge, LA 70803",
        latitude=30.4133, longitude=-91.18,
        nws_zone_id="LAZ036", county_fips="22033",
    )


def baseline_profile() -> UserProfile:
    return UserProfile(
        user_id="baseline",
        name="Baseline",
        location=_location(),
        home=Home(type=HomeType.APARTMENT, floor_level=3),
        vehicle=Vehicle(owns_vehicle=True, clearance=VehicleClearance.HIGH),
        medical=Medical(),
        emergency_contacts=[EmergencyContact(
            name="x", relationship="y", phone="+15555550100"
        )],
    )


def vulnerable_profile() -> UserProfile:
    return UserProfile(
        user_id="vulnerable",
        name="Vulnerable",
        location=_location(),
        home=Home(type=HomeType.APARTMENT, floor_level=1, flood_zone="AE"),
        vehicle=Vehicle(owns_vehicle=False, clearance=VehicleClearance.NONE),
        medical=Medical(mobility_limited=True),
        emergency_contacts=[EmergencyContact(
            name="x", relationship="y", phone="+15555550101"
        )],
    )


def _now() -> datetime:
    return datetime(2026, 7, 15, 18, 0, 0, tzinfo=timezone.utc)


def _calm_observation() -> WeatherObservation:
    return WeatherObservation(
        observed_at=_now(), latitude=30.4133, longitude=-91.18,
        temperature_f=80.0, wind_speed_mph=5.0, precip_rate_in_hr=0.0,
        sources=["nws", "owm"], alerts=[],
    )


def _tornado_observation() -> WeatherObservation:
    return WeatherObservation(
        observed_at=_now(), latitude=30.4133, longitude=-91.18,
        wind_speed_mph=42.0, wind_gust_mph=70.0, precip_rate_in_hr=0.6,
        sources=["nws", "owm"],
        alerts=[WeatherAlert(
            source="nws", event="Tornado Warning",
            severity=SeverityLevel.EXTREME, urgency=UrgencyLevel.IMMEDIATE,
            certainty=CertaintyLevel.OBSERVED,
            onset=_now(),
        )],
    )


def _flash_flood_observation() -> WeatherObservation:
    return WeatherObservation(
        observed_at=_now(), latitude=30.4133, longitude=-91.18,
        wind_speed_mph=15.0, precip_rate_in_hr=1.5,
        sources=["nws", "owm"],
        alerts=[WeatherAlert(
            source="nws", event="Flash Flood Warning",
            severity=SeverityLevel.SEVERE, urgency=UrgencyLevel.IMMEDIATE,
            certainty=CertaintyLevel.OBSERVED,
            onset=_now(),
        )],
    )


# ---------------------------------------------------------------------------
# Engine smoke
# ---------------------------------------------------------------------------

def test_engine_calm_returns_low(shared_inference: RiskInference) -> None:
    eng = RiskEngine(inference=shared_inference)
    res = eng.assess(_calm_observation(), baseline_profile())
    assert res.assessment.argmax == "Low"


def test_engine_tornado_returns_high_or_critical(shared_inference: RiskInference) -> None:
    eng = RiskEngine(inference=shared_inference)
    res = eng.assess(_tornado_observation(), baseline_profile())
    assert res.assessment.argmax in ("High", "Critical")


def test_engine_vulnerable_user_higher_than_baseline(shared_inference: RiskInference) -> None:
    eng = RiskEngine(inference=shared_inference)
    obs = _flash_flood_observation()
    base = eng.assess(obs, baseline_profile())
    vuln = eng.assess(obs, vulnerable_profile())
    base_top = base.assessment.posterior["High"] + base.assessment.posterior["Critical"]
    vuln_top = vuln.assessment.posterior["High"] + vuln.assessment.posterior["Critical"]
    assert vuln_top > base_top + 0.05


def test_engine_summary_lines_includes_evidence(shared_inference: RiskInference) -> None:
    eng = RiskEngine(inference=shared_inference)
    res = eng.assess(_tornado_observation(), baseline_profile())
    text = "\n".join(res.summary_lines())
    assert "HazardSeverity" in text
    assert "RiskLevel" in text
    assert "Argmax" in text


def test_engine_without_classifier_uses_low_threat(shared_inference: RiskInference) -> None:
    """No classifier provided => EmergingThreat=Low always."""
    eng = RiskEngine(classifier=None, inference=shared_inference)
    res = eng.assess(_calm_observation(), baseline_profile())
    assert res.threat_score is None
    assert res.assessment.evidence["EmergingThreat"] == "Low"


# ---------------------------------------------------------------------------
# Classifier integration (using a stubbed classifier so tests are fast)
# ---------------------------------------------------------------------------

class _StubClassifier:
    """Minimal classifier shim for engine integration tests."""

    def __init__(self, bucket: ThreatBucket = ThreatBucket.HIGH) -> None:
        self._bucket = bucket
        self.pipeline = "stub"  # truthy

    def score_blank(self, state, county_fips, observed_at):
        return ThreatScore(
            probability=0.7, bucket=self._bucket, feature_summary={"state": state},
        )


def test_engine_uses_classifier_threat_bucket(shared_inference: RiskInference) -> None:
    eng = RiskEngine(classifier=_StubClassifier(ThreatBucket.HIGH),
                     inference=shared_inference)
    res = eng.assess(_calm_observation(), baseline_profile())
    assert res.assessment.evidence["EmergingThreat"] == "High"
    assert res.threat_score is not None
    assert res.threat_score.bucket is ThreatBucket.HIGH


def test_engine_classifier_failure_falls_back_to_low(shared_inference: RiskInference) -> None:
    """If the classifier raises, the engine should not crash; falls back to Low."""
    class _Boom:
        pipeline = "stub"
        def score_blank(self, **_):  # type: ignore[no-untyped-def]
            raise RuntimeError("intentional")
    eng = RiskEngine(classifier=_Boom(), inference=shared_inference)
    res = eng.assess(_calm_observation(), baseline_profile())
    assert res.assessment.evidence["EmergingThreat"] == "Low"


def test_engine_no_county_fips_skips_classifier(shared_inference: RiskInference) -> None:
    """A profile without county_fips can't be scored by the classifier."""
    profile = baseline_profile()
    profile_no_fips = profile.model_copy(update={
        "location": profile.location.model_copy(update={"county_fips": None})
    })
    eng = RiskEngine(classifier=_StubClassifier(ThreatBucket.HIGH),
                     inference=shared_inference)
    res = eng.assess(_calm_observation(), profile_no_fips)
    assert res.threat_score is None


# ---------------------------------------------------------------------------
# State name extraction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "addr, expected",
    [
        ("123 Main St, Baton Rouge, LA 70803", "LOUISIANA"),
        ("10 Beach Rd, Pensacola, FL 32501", "FLORIDA"),
        ("500 Elm, Houston, Texas 77001", "TEXAS"),
        ("apt 2, Mobile, AL 36601", "ALABAMA"),
        ("Unknown locale", None),
        ("", None),
    ],
)
def test_extract_state_name(addr: str, expected: str | None) -> None:
    assert _extract_state_name(addr) == expected
