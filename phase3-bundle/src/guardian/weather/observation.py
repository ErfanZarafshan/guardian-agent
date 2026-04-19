"""Unified weather observation: the normalized input to the Bayesian engine.

This module is the boundary between "real world is messy" and "Bayesian network
wants clean discrete evidence." It defines:

  - Categorical enums (SeverityLevel, UrgencyLevel, WindCategory, PrecipCategory)
    that match the nodes in the Bayesian network. Adding a new state here is a
    breaking change for risk/bayesian.py.

  - Continuous thresholds (WIND_THRESHOLDS_MPH, PRECIP_THRESHOLDS_IN_HR) with
    citations. These are the lookup tables used to bucket continuous numbers
    from OWM into the discrete categories the Bayesian network consumes.

  - WeatherAlert: a normalized alert, regardless of which API it came from.

  - WeatherObservation: the top-level aggregate. One observation = one
    perceive-cycle's worth of data. The Bayesian engine ingests this whole
    object as evidence.

Why enums?  pgmpy CPTs are keyed by string state names. Using enums instead of
bare strings gives us IDE autocomplete and catches typos at import time.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Categorical enums (Bayesian network state spaces)
# ---------------------------------------------------------------------------

class SeverityLevel(str, Enum):
    """Maps to NWS severity, and to OWM alert severity when present.

    Ordering matters: NONE < MINOR < MODERATE < SEVERE < EXTREME.
    """
    NONE = "None"
    MINOR = "Minor"
    MODERATE = "Moderate"
    SEVERE = "Severe"
    EXTREME = "Extreme"


class UrgencyLevel(str, Enum):
    """Direct passthrough of the NWS urgency field."""
    UNKNOWN = "Unknown"
    PAST = "Past"
    FUTURE = "Future"
    EXPECTED = "Expected"
    IMMEDIATE = "Immediate"


class CertaintyLevel(str, Enum):
    UNKNOWN = "Unknown"
    UNLIKELY = "Unlikely"
    POSSIBLE = "Possible"
    LIKELY = "Likely"
    OBSERVED = "Observed"


class WindCategory(str, Enum):
    CALM = "Calm"
    BREEZY = "Breezy"
    STRONG = "Strong"
    DAMAGING = "Damaging"


class PrecipCategory(str, Enum):
    NONE = "None"
    LIGHT = "Light"
    MODERATE = "Moderate"
    HEAVY = "Heavy"
    EXTREME = "Extreme"


# ---------------------------------------------------------------------------
# Thresholds
#
# These buckets are what turn continuous sensor readings into the discrete
# states our Bayesian network was designed around.
# ---------------------------------------------------------------------------

# Beaufort scale references (NOAA):
#   <13 mph   = gentle/moderate breeze
#   13-31 mph = fresh/strong breeze
#   32-54 mph = near gale to gale
#   >54 mph   = strong gale and above (damaging)
# See: https://www.weather.gov/mfl/beaufort
WIND_THRESHOLDS_MPH: dict[WindCategory, tuple[float, float]] = {
    WindCategory.CALM:     (0.0, 13.0),
    WindCategory.BREEZY:   (13.0, 32.0),
    WindCategory.STRONG:   (32.0, 55.0),
    WindCategory.DAMAGING: (55.0, float("inf")),
}

# NWS flash-flood guidance uses rainfall rates in inches/hour:
#   <0.1  in/hr = light
#   0.1-0.3 in/hr = moderate
#   0.3-1.0 in/hr = heavy
#   >1.0  in/hr = extreme (flash-flood-producing)
# See: https://www.weather.gov/safety/flood
PRECIP_THRESHOLDS_IN_HR: dict[PrecipCategory, tuple[float, float]] = {
    PrecipCategory.NONE:     (0.0, 0.01),
    PrecipCategory.LIGHT:    (0.01, 0.10),
    PrecipCategory.MODERATE: (0.10, 0.30),
    PrecipCategory.HEAVY:    (0.30, 1.00),
    PrecipCategory.EXTREME:  (1.00, float("inf")),
}


def bucket_wind_mph(speed_mph: float) -> WindCategory:
    """Bucket a wind speed into its Bayesian-network category."""
    for cat, (lo, hi) in WIND_THRESHOLDS_MPH.items():
        if lo <= speed_mph < hi:
            return cat
    return WindCategory.DAMAGING  # speeds >= max bucket


def bucket_precip_in_hr(rate_in_hr: float) -> PrecipCategory:
    """Bucket a precipitation rate (inches per hour) into a category."""
    if rate_in_hr < 0:
        rate_in_hr = 0.0
    for cat, (lo, hi) in PRECIP_THRESHOLDS_IN_HR.items():
        if lo <= rate_in_hr < hi:
            return cat
    return PrecipCategory.EXTREME


# ---------------------------------------------------------------------------
# Normalized alert
# ---------------------------------------------------------------------------

AlertSource = Literal["nws", "owm"]


class WeatherAlert(BaseModel):
    """A single normalized alert. Source-agnostic.

    Whatever API produced this, it has been translated into these fields.
    Raw source data is kept for debugging and audit.
    """

    source: AlertSource
    event: str  # e.g. "Flash Flood Warning", "Tornado Warning"
    headline: str | None = None
    description: str | None = None
    severity: SeverityLevel = SeverityLevel.NONE
    urgency: UrgencyLevel = UrgencyLevel.UNKNOWN
    certainty: CertaintyLevel = CertaintyLevel.UNKNOWN
    onset: datetime | None = None
    expires: datetime | None = None
    sender: str | None = None
    raw: dict = Field(default_factory=dict, repr=False)

    def is_active(self, at: datetime | None = None) -> bool:
        """Is this alert currently in effect at `at` (default: now, UTC)?"""
        now = at or datetime.now(timezone.utc)
        if self.onset and now < self.onset:
            return False
        if self.expires and now > self.expires:
            return False
        return True


# ---------------------------------------------------------------------------
# Top-level observation
# ---------------------------------------------------------------------------

class WeatherObservation(BaseModel):
    """One perceive-cycle's worth of weather evidence.

    Built by `guardian.weather.aggregator.observe()`, consumed by the Bayesian
    engine in Phase 5. The `sources` list records which APIs contributed so
    downstream logic can tell partial observations from full ones.
    """

    observed_at: datetime
    latitude: float
    longitude: float

    # --- Raw numerics (None = no data for that field from any source) ---
    temperature_f: float | None = None
    wind_speed_mph: float | None = None
    wind_gust_mph: float | None = None
    precip_rate_in_hr: float | None = None
    precip_prob_pct: float | None = None  # short-term forecast probability
    humidity_pct: float | None = None
    pressure_mb: float | None = None
    visibility_mi: float | None = None

    # --- Active alerts ---
    alerts: list[WeatherAlert] = Field(default_factory=list)

    # --- Bookkeeping ---
    sources: list[AlertSource] = Field(default_factory=list)
    nws_zone_id: str | None = None

    # ---- Derived categorical states (what the Bayesian network reads) ----

    @property
    def wind_category(self) -> WindCategory:
        if self.wind_speed_mph is None:
            return WindCategory.CALM
        # Use gust if it's materially higher and present
        observed = self.wind_speed_mph
        if self.wind_gust_mph is not None and self.wind_gust_mph > observed:
            observed = self.wind_gust_mph
        return bucket_wind_mph(observed)

    @property
    def precip_category(self) -> PrecipCategory:
        if self.precip_rate_in_hr is None:
            return PrecipCategory.NONE
        return bucket_precip_in_hr(self.precip_rate_in_hr)

    @property
    def max_severity(self) -> SeverityLevel:
        """Highest alert severity currently active. NONE if no active alerts."""
        active = [a for a in self.alerts if a.is_active(self.observed_at)]
        if not active:
            return SeverityLevel.NONE
        ordering = {s: i for i, s in enumerate(SeverityLevel)}
        return max(active, key=lambda a: ordering[a.severity]).severity

    @property
    def max_urgency(self) -> UrgencyLevel:
        """Most pressing urgency among active alerts."""
        active = [a for a in self.alerts if a.is_active(self.observed_at)]
        if not active:
            return UrgencyLevel.UNKNOWN
        # Ordering: Immediate > Expected > Future > Past > Unknown
        rank = {
            UrgencyLevel.UNKNOWN: 0,
            UrgencyLevel.PAST: 1,
            UrgencyLevel.FUTURE: 2,
            UrgencyLevel.EXPECTED: 3,
            UrgencyLevel.IMMEDIATE: 4,
        }
        return max(active, key=lambda a: rank[a.urgency]).urgency

    @property
    def has_active_alerts(self) -> bool:
        return any(a.is_active(self.observed_at) for a in self.alerts)


__all__ = [
    "SeverityLevel",
    "UrgencyLevel",
    "CertaintyLevel",
    "WindCategory",
    "PrecipCategory",
    "WIND_THRESHOLDS_MPH",
    "PRECIP_THRESHOLDS_IN_HR",
    "bucket_wind_mph",
    "bucket_precip_in_hr",
    "WeatherAlert",
    "WeatherObservation",
    "AlertSource",
]
