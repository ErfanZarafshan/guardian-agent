"""Risk Engine: the integrator.

`RiskEngine.assess(observation, profile)` is the single function the agent
calls to convert "what's happening + who's affected" into "what should the
risk score be." It composes:

  1. The Bayesian network (bayesian.py) for probabilistic fusion.
  2. The ML threat classifier (classifier.py) for emerging-threat evidence.
  3. The user profile (profile.py) for vulnerability factors.
  4. The unified weather observation (weather/observation.py) for hazard
     evidence.

This module owns no probability tables of its own — it only does plumbing.
That separation keeps the math testable independently from the integration.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..logging_setup import get_logger
from ..profile import UserProfile
from ..weather.observation import WeatherObservation
from .bayesian import (
    RiskAssessment,
    RiskInference,
    encode_evidence,
)
from .classifier import ThreatBucket, ThreatClassifier, ThreatScore


log = get_logger(__name__)


@dataclass(frozen=True)
class EngineResult:
    """Result of one full perceive-reason cycle.

    Combines the Bayesian assessment with the inputs that produced it, so the
    planner (Phase 6) and any explainability layer have full provenance.
    """
    assessment: RiskAssessment
    threat_score: ThreatScore | None
    observation: WeatherObservation
    profile_user_id: str

    def summary_lines(self) -> list[str]:
        lines = [
            f"User: {self.profile_user_id}",
            f"Observed at: {self.observation.observed_at}",
            f"Sources: {', '.join(self.observation.sources) or '(none)'}",
            f"Active alerts: {len(self.observation.alerts)}",
            "",
            "Evidence used:",
        ]
        for k, v in self.assessment.evidence.items():
            lines.append(f"  {k:>18s} = {v}")
        if self.threat_score is not None:
            lines.append(
                f"  (Classifier P(severe in 24h) = {self.threat_score.probability:.3f})"
            )
        lines += [
            "",
            f"Posterior P(RiskLevel | evidence):",
            self.assessment.as_table(),
            "",
            f"==> Argmax: {self.assessment.argmax}  ({self.assessment.argmax_prob:.1%})",
        ]
        return lines


class RiskEngine:
    """Stateful wrapper that holds the network, inference engine, and (optional)
    threat classifier. Construct it once at agent startup, then call
    `.assess(obs, profile)` per cycle.
    """

    def __init__(
        self,
        classifier: ThreatClassifier | None = None,
        inference: RiskInference | None = None,
    ) -> None:
        self.classifier = classifier
        self.inference = inference or RiskInference()

    # ------------------------------------------------------------------

    def _threat_bucket(
        self, observation: WeatherObservation, profile: UserProfile
    ) -> tuple[ThreatBucket, ThreatScore | None]:
        """Run the ML classifier if we have one and the profile has a county.

        Falls back to ThreatBucket.LOW if the classifier is missing, the
        profile lacks a county FIPS, or the prediction errors out — none of
        these should crash the agent.
        """
        if self.classifier is None or self.classifier.pipeline is None:
            return ThreatBucket.LOW, None
        if not profile.location.county_fips:
            log.debug(
                "No county_fips in profile; skipping classifier and using LOW threat."
            )
            return ThreatBucket.LOW, None

        # Map state-name string from somewhere; the profile doesn't currently
        # have a structured state field, so we extract from the address tail.
        # If we can't, default to LOUISIANA (the project's primary geography).
        state = _extract_state_name(profile.location.address) or "LOUISIANA"
        try:
            score = self.classifier.score_blank(
                state=state,
                county_fips=profile.location.county_fips,
                observed_at=observation.observed_at,
            )
        except Exception as e:  # pragma: no cover — defensive
            log.warning("Classifier scoring failed: %s", e)
            return ThreatBucket.LOW, None
        return score.bucket, score

    # ------------------------------------------------------------------

    def assess(
        self, observation: WeatherObservation, profile: UserProfile
    ) -> EngineResult:
        """Perceive-and-reason: produce a RiskAssessment for this user."""
        threat_bucket, threat_score = self._threat_bucket(observation, profile)

        evidence = encode_evidence(
            severity=observation.max_severity.value,
            urgency=observation.max_urgency.value,
            wind=observation.wind_category.value,
            precip=observation.precip_category.value,
            threat=threat_bucket.value,
            home_floor=profile.home_floor_state.value,
            vehicle_clearance=profile.vehicle.clearance.value,
            mobility_limited=profile.medical.mobility_limited,
        )
        assessment = self.inference.assess(evidence)

        return EngineResult(
            assessment=assessment,
            threat_score=threat_score,
            observation=observation,
            profile_user_id=profile.user_id,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Map of state names (uppercase) we recognize. Used to extract a state from
# the profile's free-form address field.
_STATE_NAMES = {
    "ALABAMA", "ALASKA", "ARIZONA", "ARKANSAS", "CALIFORNIA", "COLORADO",
    "CONNECTICUT", "DELAWARE", "FLORIDA", "GEORGIA", "HAWAII", "IDAHO",
    "ILLINOIS", "INDIANA", "IOWA", "KANSAS", "KENTUCKY", "LOUISIANA",
    "MAINE", "MARYLAND", "MASSACHUSETTS", "MICHIGAN", "MINNESOTA",
    "MISSISSIPPI", "MISSOURI", "MONTANA", "NEBRASKA", "NEVADA",
    "NEW HAMPSHIRE", "NEW JERSEY", "NEW MEXICO", "NEW YORK",
    "NORTH CAROLINA", "NORTH DAKOTA", "OHIO", "OKLAHOMA", "OREGON",
    "PENNSYLVANIA", "RHODE ISLAND", "SOUTH CAROLINA", "SOUTH DAKOTA",
    "TENNESSEE", "TEXAS", "UTAH", "VERMONT", "VIRGINIA", "WASHINGTON",
    "WEST VIRGINIA", "WISCONSIN", "WYOMING",
}
# Two-letter to full name (Gulf Coast set is enough for our purposes).
_STATE_ABBREV = {
    "AL": "ALABAMA", "FL": "FLORIDA", "LA": "LOUISIANA",
    "MS": "MISSISSIPPI", "TX": "TEXAS",
}


def _extract_state_name(address: str) -> str | None:
    """Best-effort: pull a state name out of a free-form address string."""
    if not address:
        return None
    upper = address.upper()
    for name in _STATE_NAMES:
        if name in upper:
            return name
    # Try abbreviations as standalone words ("Baton Rouge, LA 70803")
    for token in upper.replace(",", " ").split():
        token = token.strip(" ,.")
        if token in _STATE_ABBREV:
            return _STATE_ABBREV[token]
    return None


__all__ = ["RiskEngine", "EngineResult"]
