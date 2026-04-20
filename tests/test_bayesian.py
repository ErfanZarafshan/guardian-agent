"""Tests for the Bayesian risk network.

Three tiers of coverage:

  1. Structural — the network has the right nodes/edges and pgmpy validates.
  2. CPT properties — every CPT column sums to 1.0; formula functions
     produce normalized distributions; monotonicity holds.
  3. Behavioral — when evidence changes from "calm" to "tornado warning,"
     the posterior shifts toward higher risk, and a vulnerable user gets a
     higher score than a baseline user under identical evidence.
"""

from __future__ import annotations

import math

import pytest

from guardian.risk.bayesian import (
    EMERGING_THREAT_STATES,
    HAZARD_IMPACT_STATES,
    HAZARD_SEVERITY_STATES,
    HOME_FLOOR_STATES,
    MOBILITY_STATES,
    PRECIP_STATES,
    RISK_LEVEL_STATES,
    URGENCY_STATES,
    VEHICLE_CLEARANCE_STATES,
    WIND_STATES,
    RiskInference,
    build_network,
    encode_evidence,
)


# ---------------------------------------------------------------------------
# Tier 1: structure
# ---------------------------------------------------------------------------

def test_network_builds_and_validates() -> None:
    bn = build_network()
    assert bn.check_model() is True


def test_network_has_expected_nodes() -> None:
    bn = build_network()
    expected = {
        "HazardSeverity", "Urgency", "WindCategory", "PrecipCategory",
        "EmergingThreat", "HomeFloor", "VehicleClearance", "MobilityLimited",
        "HazardImpact", "RiskLevel",
    }
    assert set(bn.nodes()) == expected


def test_network_edges_correct() -> None:
    bn = build_network()
    expected_edges = {
        ("HazardSeverity", "HazardImpact"),
        ("Urgency", "HazardImpact"),
        ("WindCategory", "HazardImpact"),
        ("PrecipCategory", "HazardImpact"),
        ("EmergingThreat", "HazardImpact"),
        ("HazardImpact", "RiskLevel"),
        ("HomeFloor", "RiskLevel"),
        ("VehicleClearance", "RiskLevel"),
        ("MobilityLimited", "RiskLevel"),
    }
    assert set(bn.edges()) == expected_edges


def test_state_name_lengths() -> None:
    """Sanity: enum-style state lists are consistent."""
    assert len(HAZARD_SEVERITY_STATES) == 5
    assert len(URGENCY_STATES) == 5
    assert len(WIND_STATES) == 4
    assert len(PRECIP_STATES) == 5
    assert len(EMERGING_THREAT_STATES) == 3
    assert len(HOME_FLOOR_STATES) == 3
    assert len(VEHICLE_CLEARANCE_STATES) == 4
    assert len(MOBILITY_STATES) == 2
    assert len(HAZARD_IMPACT_STATES) == 5
    assert len(RISK_LEVEL_STATES) == 4


# ---------------------------------------------------------------------------
# Tier 2: CPT properties
# ---------------------------------------------------------------------------

def test_all_cpts_columns_sum_to_one() -> None:
    bn = build_network()
    for cpd in bn.get_cpds():
        values = cpd.values
        # values shape: (variable_card, *evidence_cards). Sum along axis 0.
        sums = values.sum(axis=0)
        max_dev = float(abs(sums - 1.0).max())
        assert max_dev < 1e-9, (
            f"CPD {cpd.variable!r} has columns summing to {sums.min()}-{sums.max()}, "
            f"max deviation {max_dev}"
        )


def test_hazard_impact_distribution_monotone() -> None:
    """As score increases, P(Extreme) should never decrease,
    and P(None) should never increase."""
    from guardian.risk.bayesian import _hazard_impact_distribution
    scores = [0.0, 0.05, 0.15, 0.25, 0.35, 0.55, 0.80, 1.0]
    p_none, p_extreme = [], []
    for s in scores:
        d = _hazard_impact_distribution(s)
        assert math.isclose(sum(d), 1.0, abs_tol=1e-9)
        p_none.append(d[0])
        p_extreme.append(d[-1])
    for i in range(len(scores) - 1):
        assert p_none[i + 1] <= p_none[i] + 1e-9
        assert p_extreme[i + 1] >= p_extreme[i] - 1e-9


def test_risk_distribution_monotone() -> None:
    from guardian.risk.bayesian import _risk_distribution
    scores = [0.0, 0.10, 0.20, 0.40, 0.55, 0.70, 0.90, 1.0]
    p_low, p_critical = [], []
    for s in scores:
        d = _risk_distribution(s)
        assert math.isclose(sum(d), 1.0, abs_tol=1e-9)
        p_low.append(d[0])
        p_critical.append(d[-1])
    for i in range(len(scores) - 1):
        assert p_low[i + 1] <= p_low[i] + 1e-9
        assert p_critical[i + 1] >= p_critical[i] - 1e-9


def test_hazard_impact_score_max_combines_structural() -> None:
    """Wind alone vs precip alone vs all together: combination uses MAX."""
    from guardian.risk.bayesian import _hazard_impact_score
    s_wind = _hazard_impact_score("None", "Unknown", "Damaging", "None", "Low")
    s_precip = _hazard_impact_score("None", "Unknown", "Calm", "Extreme", "Low")
    s_both = _hazard_impact_score("None", "Unknown", "Damaging", "Extreme", "Low")
    assert s_both == pytest.approx(max(s_wind, s_precip), abs=1e-9)


def test_risk_score_vulnerability_increases_risk() -> None:
    """Same impact, higher vulnerability -> higher risk score."""
    from guardian.risk.bayesian import _risk_score
    base = _risk_score("Severe", "Upper", "High", "False")
    vulnerable = _risk_score("Severe", "Ground", "None", "True")
    protected = _risk_score("Severe", "Elevated", "High", "False")
    assert vulnerable > base > protected


# ---------------------------------------------------------------------------
# Tier 3: behavioral / inference
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def infer() -> RiskInference:
    """Build the network once per test module — pgmpy construction is slow."""
    return RiskInference()


def _query(infer: RiskInference, **overrides) -> dict[str, float]:
    """Convenience: query with sensible defaults plus any overrides."""
    base = {
        "severity": "None",
        "urgency": "Unknown",
        "wind": "Calm",
        "precip": "None",
        "threat": "Low",
        "home_floor": "Upper",
        "vehicle_clearance": "Medium",
        "mobility_limited": False,
    }
    base.update(overrides)
    evidence = encode_evidence(**base)
    return infer.assess(evidence).posterior


def test_inference_calm_is_low_risk(infer: RiskInference) -> None:
    p = _query(infer)
    assert p["Low"] > 0.5, p
    assert p["Critical"] < 0.05, p


def test_inference_tornado_warning_drives_higher_risk(infer: RiskInference) -> None:
    calm = _query(infer)
    tornado = _query(
        infer,
        severity="Extreme",
        urgency="Immediate",
        wind="Damaging",
    )
    # Mass should shift up: P(High)+P(Critical) much larger.
    high_calm = calm["High"] + calm["Critical"]
    high_tornado = tornado["High"] + tornado["Critical"]
    assert high_tornado > high_calm + 0.4, (
        f"Expected major shift; got {high_calm:.3f} -> {high_tornado:.3f}"
    )


def test_inference_vulnerable_user_higher_than_baseline(infer: RiskInference) -> None:
    """Same hazard, two users — vulnerable should outrank baseline."""
    severe_args = dict(severity="Severe", urgency="Expected",
                       wind="Strong", precip="Heavy")
    baseline = _query(infer, **severe_args, home_floor="Upper",
                      vehicle_clearance="High", mobility_limited=False)
    vulnerable = _query(infer, **severe_args, home_floor="Ground",
                        vehicle_clearance="None", mobility_limited=True)
    base_risk = baseline["High"] + baseline["Critical"]
    vuln_risk = vulnerable["High"] + vulnerable["Critical"]
    assert vuln_risk > base_risk + 0.05, (
        f"Vulnerable user risk should clearly exceed baseline; "
        f"got {base_risk:.3f} vs {vuln_risk:.3f}"
    )


def test_inference_emerging_threat_alone_lifts_risk(infer: RiskInference) -> None:
    """ML classifier saying 'High threat' alone (no NWS alert) should still
    push posterior away from pure Low."""
    quiet = _query(infer, threat="Low")
    threat_only = _query(infer, threat="High")
    # The shift won't be huge (no concrete hazard), but P(Low) should drop.
    assert threat_only["Low"] < quiet["Low"] - 0.02


def test_inference_flash_flood_ground_floor_critical_likely(infer: RiskInference) -> None:
    """Flash flood + ground floor + mobility limited should make Critical
    a real possibility."""
    p = _query(
        infer,
        severity="Severe",
        urgency="Immediate",
        precip="Extreme",
        home_floor="Ground",
        vehicle_clearance="None",
        mobility_limited=True,
    )
    assert p["Critical"] > 0.10, f"Expected meaningful Critical mass; got {p}"


def test_inference_posterior_sums_to_one(infer: RiskInference) -> None:
    p = _query(infer, severity="Moderate", urgency="Expected",
               wind="Strong", precip="Heavy")
    assert math.isclose(sum(p.values()), 1.0, abs_tol=1e-9)


def test_inference_argmax_consistent_with_posterior(infer: RiskInference) -> None:
    evidence = encode_evidence(
        severity="Severe", urgency="Immediate", wind="Damaging", precip="Heavy",
        threat="High", home_floor="Ground", vehicle_clearance="None",
        mobility_limited=True,
    )
    a = RiskInference().assess(evidence)
    assert max(a.posterior.values()) == a.posterior[a.argmax]
    assert a.argmax_prob == a.posterior[a.argmax]


def test_inference_tornado_vs_tropical_stay_distinguishable(infer: RiskInference) -> None:
    """Regression test: two severe-but-distinct scenarios should produce
    different posteriors even when the classifier pushes EmergingThreat=High.

    Earlier versions of _hazard_impact_distribution had too few buckets at
    the high end; scores of 0.87 (tropical + high threat) and 1.00 (tornado
    + high threat) both mapped to the identical distribution, causing the
    demo's tornado and tropical scenarios to show byte-for-byte identical
    posteriors.
    """
    base = dict(home_floor="Upper", vehicle_clearance="High",
                mobility_limited=False, threat="High")
    tornado = _query(infer, severity="Extreme", urgency="Immediate",
                     wind="Damaging", precip="Heavy", **base)
    tropical = _query(infer, severity="Severe", urgency="Expected",
                      wind="Damaging", precip="Extreme", **base)
    # Tornado should have more Critical mass than tropical.
    assert tornado["Critical"] > tropical["Critical"] + 0.03, (
        f"Expected tornado Critical > tropical Critical; got "
        f"tornado={tornado['Critical']:.3f}, tropical={tropical['Critical']:.3f}"
    )


# ---------------------------------------------------------------------------
# encode_evidence translation
# ---------------------------------------------------------------------------

def test_encode_evidence_normalizes_lowercase_vehicle() -> None:
    """profile.Vehicle.clearance.value is lowercase ("low"), so encode_evidence
    must translate to the network's Title-Cased state names."""
    e = encode_evidence(
        severity="None", urgency="Unknown", wind="Calm", precip="None",
        threat="Low", home_floor="Upper", vehicle_clearance="low",
        mobility_limited=False,
    )
    assert e["VehicleClearance"] == "Low"


def test_encode_evidence_passes_through_titlecase() -> None:
    e = encode_evidence(
        severity="None", urgency="Unknown", wind="Calm", precip="None",
        threat="Low", home_floor="Upper", vehicle_clearance="High",
        mobility_limited=False,
    )
    assert e["VehicleClearance"] == "High"


def test_encode_evidence_mobility_to_string() -> None:
    e = encode_evidence(
        severity="None", urgency="Unknown", wind="Calm", precip="None",
        threat="Low", home_floor="Upper", vehicle_clearance="Medium",
        mobility_limited=True,
    )
    assert e["MobilityLimited"] == "True"
    e2 = encode_evidence(
        severity="None", urgency="Unknown", wind="Calm", precip="None",
        threat="Low", home_floor="Upper", vehicle_clearance="Medium",
        mobility_limited=False,
    )
    assert e2["MobilityLimited"] == "False"


def test_encode_evidence_invalid_vehicle_raises() -> None:
    with pytest.raises(ValueError):
        encode_evidence(
            severity="None", urgency="Unknown", wind="Calm", precip="None",
            threat="Low", home_floor="Upper", vehicle_clearance="banana",
            mobility_limited=False,
        )
