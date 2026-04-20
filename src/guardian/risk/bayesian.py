"""Bayesian Risk Assessment Engine (Phase 5).

Core of Guardian Agent's reasoning. Builds a Bayesian network (pgmpy) over:

  Observable evidence nodes (set from a WeatherObservation):
    - HazardSeverity   ∈ {None, Minor, Moderate, Severe, Extreme}
    - Urgency          ∈ {Unknown, Past, Future, Expected, Immediate}
    - WindCategory     ∈ {Calm, Breezy, Strong, Damaging}
    - PrecipCategory   ∈ {None, Light, Moderate, Heavy, Extreme}
    - EmergingThreat   ∈ {Low, Medium, High}      (from ML classifier)

  User profile nodes (set as evidence from UserProfile):
    - HomeFloor        ∈ {Ground, Upper, Elevated}
    - VehicleClearance ∈ {None, Low, Medium, High}
    - MobilityLimited  ∈ {True, False}

  Latent + query nodes:
    - HazardImpact     ∈ {None, Minor, Moderate, Severe, Extreme}  (latent)
    - RiskLevel        ∈ {Low, Moderate, High, Critical}           (query)

Network structure (DAG):

    HazardSeverity ─┐
    Urgency        ─┤
    WindCategory   ─┼─► HazardImpact ──► RiskLevel ◄── HomeFloor
    PrecipCategory ─┤                                  ◄── VehicleClearance
    EmergingThreat ─┘                                  ◄── MobilityLimited

The two-layer factoring (HazardImpact → RiskLevel) shrinks CPT sizes by an
order of magnitude versus collapsing both into a single 8-parent RiskLevel
node, AND yields an interpretable middle layer: HazardImpact is "what is the
*world* doing right now", independent of the user; RiskLevel is "what does
that mean for *this* user."

CPTs are not enumerated cell-by-cell. Instead, formula-driven helpers
(_hazard_impact_distribution, _risk_distribution) compute distributions from
domain-knowledge severity/vulnerability scoring. Every CPT cell is auditable
by re-running the formula. See docs/ARCHITECTURE.md for the rationale.

Inference: exact, via pgmpy.inference.VariableElimination (AIMA Ch. 13.4.1).
The network is a polytree, so VE is fast (sub-millisecond per query).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork


# ---------------------------------------------------------------------------
# State name catalogs
#
# These MUST match the enum values in guardian.weather.observation and
# guardian.profile, because that's where evidence comes from at runtime.
# ---------------------------------------------------------------------------

HAZARD_SEVERITY_STATES = ("None", "Minor", "Moderate", "Severe", "Extreme")
URGENCY_STATES = ("Unknown", "Past", "Future", "Expected", "Immediate")
WIND_STATES = ("Calm", "Breezy", "Strong", "Damaging")
PRECIP_STATES = ("None", "Light", "Moderate", "Heavy", "Extreme")
EMERGING_THREAT_STATES = ("Low", "Medium", "High")
HOME_FLOOR_STATES = ("Ground", "Upper", "Elevated")
VEHICLE_CLEARANCE_STATES = ("None", "Low", "Medium", "High")
MOBILITY_STATES = ("False", "True")  # pgmpy needs string state names

HAZARD_IMPACT_STATES = ("None", "Minor", "Moderate", "Severe", "Extreme")
RISK_LEVEL_STATES = ("Low", "Moderate", "High", "Critical")


# ---------------------------------------------------------------------------
# Domain-knowledge scoring
# ---------------------------------------------------------------------------

# Severity weight per evidence value, in [0, 1].
_SEVERITY_WEIGHTS: dict[str, float] = {
    "None": 0.00, "Minor": 0.15, "Moderate": 0.40, "Severe": 0.70, "Extreme": 1.00,
}
_URGENCY_WEIGHTS: dict[str, float] = {
    "Unknown": 0.0, "Past": 0.0, "Future": 0.10, "Expected": 0.30, "Immediate": 0.50,
}
_WIND_WEIGHTS: dict[str, float] = {
    "Calm": 0.0, "Breezy": 0.10, "Strong": 0.40, "Damaging": 0.80,
}
_PRECIP_WEIGHTS: dict[str, float] = {
    "None": 0.0, "Light": 0.05, "Moderate": 0.20, "Heavy": 0.50, "Extreme": 0.90,
}
_THREAT_WEIGHTS: dict[str, float] = {
    "Low": 0.0, "Medium": 0.20, "High": 0.50,
}


def _hazard_impact_score(
    severity: str, urgency: str, wind: str, precip: str, threat: str
) -> float:
    """Combine evidence weights into a single 0–1 hazard-impact score.

    Design rationale:
      - The structural threats (severity / wind / precip) combine via MAX —
        a tornado warning isn't more dangerous because it's also raining;
        the worst signal dominates.
      - Urgency multiplies (Immediate makes everything more dangerous, Past
        makes everything less so). Range [0.5, 1.0] to never zero out.
      - The ML classifier's emerging-threat signal adds a bonus (independent
        evidence about what's coming, not what's here).
    """
    structural = max(
        _SEVERITY_WEIGHTS[severity],
        _WIND_WEIGHTS[wind],
        _PRECIP_WEIGHTS[precip],
    )
    urgency_mult = 0.5 + _URGENCY_WEIGHTS[urgency]
    threat_bonus = _THREAT_WEIGHTS[threat] * 0.3
    return float(min(1.0, structural * urgency_mult + threat_bonus))


def _hazard_impact_distribution(score: float) -> tuple[float, float, float, float, float]:
    """Map a 0–1 score to P(HazardImpact ∈ {None, Minor, Moderate, Severe, Extreme}).

    Rows are designed to (a) sum to 1, (b) put most mass on the matching
    category, (c) spread some mass to neighbors to encode epistemic
    uncertainty about the deterministic mapping.
    """
    if score < 0.10:
        return (0.85, 0.10, 0.04, 0.01, 0.00)
    if score < 0.30:
        return (0.30, 0.50, 0.15, 0.04, 0.01)
    if score < 0.50:
        return (0.10, 0.30, 0.45, 0.13, 0.02)
    if score < 0.75:
        return (0.03, 0.10, 0.30, 0.45, 0.12)
    return (0.00, 0.03, 0.10, 0.40, 0.47)


# ---------- RiskLevel half ----------

_HAZARD_BASE_SEVERITY: dict[str, float] = {
    "None": 0.0, "Minor": 0.20, "Moderate": 0.45, "Severe": 0.70, "Extreme": 0.90,
}
_HOME_FLOOR_VULN: dict[str, float] = {
    "Ground": 1.30, "Upper": 1.00, "Elevated": 0.85,
}
_VEHICLE_VULN: dict[str, float] = {
    "None": 1.40, "Low": 1.20, "Medium": 1.00, "High": 0.90,
}
_MOBILITY_VULN: dict[str, float] = {
    "True": 1.50, "False": 1.00,
}


def _risk_score(impact: str, home: str, vehicle: str, mobility: str) -> float:
    """Compute a 0–1 risk score from impact + user vulnerability.

    Design: vulnerability acts as a multiplier on top of base hazard severity.
    A user with all baseline vulnerability (1.0×1.0×1.0=1.0) gets the raw
    impact severity. A maximally vulnerable user (1.3×1.4×1.5=2.73, capped
    at 2.5) gets a substantial uplift. A maximally protected user (0.85×0.9×1.0
    =0.765) gets the impact dampened.
    """
    base = _HAZARD_BASE_SEVERITY[impact]
    vuln = _HOME_FLOOR_VULN[home] * _VEHICLE_VULN[vehicle] * _MOBILITY_VULN[mobility]
    vuln = min(2.5, vuln)
    # r = base + (vuln - 1) * base * 0.5  → vuln=1 means r=base, vuln=2 means r=1.5*base
    risk = base + (vuln - 1.0) * base * 0.5
    return float(min(1.0, max(0.0, risk)))


def _risk_distribution(score: float) -> tuple[float, float, float, float]:
    """Map a 0–1 score to P(RiskLevel ∈ {Low, Moderate, High, Critical})."""
    if score < 0.15:
        return (0.90, 0.08, 0.02, 0.00)
    if score < 0.35:
        return (0.40, 0.45, 0.13, 0.02)
    if score < 0.60:
        return (0.10, 0.45, 0.35, 0.10)
    if score < 0.85:
        return (0.02, 0.15, 0.50, 0.33)
    return (0.00, 0.05, 0.30, 0.65)


# ---------------------------------------------------------------------------
# CPT construction helpers
# ---------------------------------------------------------------------------

def _uniform_cpd(name: str, states: tuple[str, ...]) -> TabularCPD:
    """Build a uniform-prior CPD for a no-parent evidence node.

    These priors are uninformative; at inference time we always observe these
    nodes, so the prior gets overridden. Using uniform keeps the network
    well-formed without making unsupported claims about base rates.
    """
    n = len(states)
    return TabularCPD(
        variable=name,
        variable_card=n,
        values=[[1.0 / n]] * n,
        state_names={name: list(states)},
    )


def _build_hazard_impact_cpd() -> TabularCPD:
    """Build the HazardImpact CPT by evaluating the impact formula
    over the cross-product of its parents.

    Parent column ordering follows pgmpy convention: the first listed
    parent varies slowest, the last varies fastest.
    """
    parents = [
        ("HazardSeverity", HAZARD_SEVERITY_STATES),
        ("Urgency", URGENCY_STATES),
        ("WindCategory", WIND_STATES),
        ("PrecipCategory", PRECIP_STATES),
        ("EmergingThreat", EMERGING_THREAT_STATES),
    ]
    parent_state_lists = [s for _, s in parents]
    n_combos = int(np.prod([len(s) for s in parent_state_lists]))

    # Shape (n_outputs, n_combos)
    values = np.zeros((len(HAZARD_IMPACT_STATES), n_combos))
    for col, combo in enumerate(product(*parent_state_lists)):
        sev, urg, wind, precip, threat = combo
        score = _hazard_impact_score(sev, urg, wind, precip, threat)
        dist = _hazard_impact_distribution(score)
        values[:, col] = dist

    return TabularCPD(
        variable="HazardImpact",
        variable_card=len(HAZARD_IMPACT_STATES),
        values=values.tolist(),
        evidence=[name for name, _ in parents],
        evidence_card=[len(s) for _, s in parents],
        state_names={
            "HazardImpact": list(HAZARD_IMPACT_STATES),
            **{name: list(states) for name, states in parents},
        },
    )


def _build_risk_level_cpd() -> TabularCPD:
    parents = [
        ("HazardImpact", HAZARD_IMPACT_STATES),
        ("HomeFloor", HOME_FLOOR_STATES),
        ("VehicleClearance", VEHICLE_CLEARANCE_STATES),
        ("MobilityLimited", MOBILITY_STATES),
    ]
    parent_state_lists = [s for _, s in parents]
    n_combos = int(np.prod([len(s) for s in parent_state_lists]))

    values = np.zeros((len(RISK_LEVEL_STATES), n_combos))
    for col, combo in enumerate(product(*parent_state_lists)):
        impact, home, vehicle, mobility = combo
        score = _risk_score(impact, home, vehicle, mobility)
        dist = _risk_distribution(score)
        values[:, col] = dist

    return TabularCPD(
        variable="RiskLevel",
        variable_card=len(RISK_LEVEL_STATES),
        values=values.tolist(),
        evidence=[name for name, _ in parents],
        evidence_card=[len(s) for _, s in parents],
        state_names={
            "RiskLevel": list(RISK_LEVEL_STATES),
            **{name: list(states) for name, states in parents},
        },
    )


# ---------------------------------------------------------------------------
# Network builder
# ---------------------------------------------------------------------------

def build_network() -> DiscreteBayesianNetwork:
    """Construct the full Bayesian network with CPTs attached and validated."""
    edges = [
        ("HazardSeverity", "HazardImpact"),
        ("Urgency", "HazardImpact"),
        ("WindCategory", "HazardImpact"),
        ("PrecipCategory", "HazardImpact"),
        ("EmergingThreat", "HazardImpact"),
        ("HazardImpact", "RiskLevel"),
        ("HomeFloor", "RiskLevel"),
        ("VehicleClearance", "RiskLevel"),
        ("MobilityLimited", "RiskLevel"),
    ]
    bn = DiscreteBayesianNetwork(edges)

    bn.add_cpds(
        _uniform_cpd("HazardSeverity", HAZARD_SEVERITY_STATES),
        _uniform_cpd("Urgency", URGENCY_STATES),
        _uniform_cpd("WindCategory", WIND_STATES),
        _uniform_cpd("PrecipCategory", PRECIP_STATES),
        _uniform_cpd("EmergingThreat", EMERGING_THREAT_STATES),
        _uniform_cpd("HomeFloor", HOME_FLOOR_STATES),
        _uniform_cpd("VehicleClearance", VEHICLE_CLEARANCE_STATES),
        _uniform_cpd("MobilityLimited", MOBILITY_STATES),
        _build_hazard_impact_cpd(),
        _build_risk_level_cpd(),
    )

    if not bn.check_model():
        raise RuntimeError("Bayesian network failed pgmpy validation.")
    return bn


# ---------------------------------------------------------------------------
# Inference wrapper
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskAssessment:
    """Result of one inference query.

    posterior:  state name -> probability (sums to 1.0)
    argmax:     the most-likely state
    argmax_prob: probability of the most-likely state (calibration check)
    evidence:   the evidence dict that produced this assessment (for audit)
    """
    posterior: dict[str, float]
    argmax: str
    argmax_prob: float
    evidence: dict[str, str]

    def as_table(self) -> str:
        lines = [f"  {state:>10s}  {p:6.1%}" for state, p in self.posterior.items()]
        return "\n".join(lines)


# Mapping from observation/profile enum string values to the Bayesian network's
# state name strings. Most are identical, but VehicleClearance is lowercase in
# the profile (low/medium/high) and Title-Cased here, and MobilityLimited is a
# bool that we serialize to "True"/"False".
_VEHICLE_CLEARANCE_TO_STATE = {
    "none": "None", "low": "Low", "medium": "Medium", "high": "High",
}


def encode_evidence(
    *,
    severity: str,
    urgency: str,
    wind: str,
    precip: str,
    threat: str,
    home_floor: str,
    vehicle_clearance: str,
    mobility_limited: bool,
) -> dict[str, str]:
    """Translate observation+profile values into a dict of network evidence."""
    veh = vehicle_clearance.lower()
    if veh in _VEHICLE_CLEARANCE_TO_STATE:
        veh_state = _VEHICLE_CLEARANCE_TO_STATE[veh]
    elif vehicle_clearance in VEHICLE_CLEARANCE_STATES:
        veh_state = vehicle_clearance
    else:
        raise ValueError(f"Unknown vehicle clearance: {vehicle_clearance!r}")
    return {
        "HazardSeverity": severity,
        "Urgency": urgency,
        "WindCategory": wind,
        "PrecipCategory": precip,
        "EmergingThreat": threat,
        "HomeFloor": home_floor,
        "VehicleClearance": veh_state,
        "MobilityLimited": "True" if mobility_limited else "False",
    }


class RiskInference:
    """Cached inference engine over a fixed network.

    Build it once, query it many times. Constructing pgmpy's
    VariableElimination is non-trivial; reusing the engine across queries
    saves significant time during the agent's polling loop.
    """

    def __init__(self, network: DiscreteBayesianNetwork | None = None) -> None:
        self.network = network or build_network()
        self._infer = VariableElimination(self.network)

    def assess(self, evidence: dict[str, str]) -> RiskAssessment:
        """Compute P(RiskLevel | evidence) and return a typed assessment."""
        # pgmpy's query takes a list of variables and an evidence dict.
        result = self._infer.query(
            variables=["RiskLevel"],
            evidence=evidence,
            show_progress=False,
        )
        # pgmpy returns a DiscreteFactor; convert to a {state_name: prob} dict.
        states = result.state_names["RiskLevel"]
        probs = result.values  # shape (4,)
        posterior = {s: float(probs[i]) for i, s in enumerate(states)}
        argmax_state = max(posterior, key=posterior.get)
        return RiskAssessment(
            posterior=posterior,
            argmax=argmax_state,
            argmax_prob=posterior[argmax_state],
            evidence=dict(evidence),
        )


__all__ = [
    "HAZARD_SEVERITY_STATES",
    "URGENCY_STATES",
    "WIND_STATES",
    "PRECIP_STATES",
    "EMERGING_THREAT_STATES",
    "HOME_FLOOR_STATES",
    "VEHICLE_CLEARANCE_STATES",
    "MOBILITY_STATES",
    "HAZARD_IMPACT_STATES",
    "RISK_LEVEL_STATES",
    "build_network",
    "encode_evidence",
    "RiskAssessment",
    "RiskInference",
]
