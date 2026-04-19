"""User profile: schema, validation, storage, and derived Bayesian states.

The profile captures user-specific vulnerability factors that Guardian Agent's
Bayesian Risk Engine needs as evidence:

  - Home type and floor level (drives flood-damage CPTs)
  - Vehicle clearance (drives evacuation-feasibility CPTs)
  - Medical vulnerabilities (drives urgency and action selection)
  - Emergency contacts (drives notification fan-out)

The schema is declared with pydantic v2 so load-time validation catches
malformed profiles loudly and immediately. Derived Bayesian states (see
`BayesianEvidence`) are computed on the fly rather than stored, so the raw
JSON stays the single source of truth and can be hand-edited.
"""

from __future__ import annotations

import json
import re
from datetime import time
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums for the discrete states the Bayesian network ultimately consumes.
# Defining them here (not in bayesian.py) because the profile is the source of
# truth for user-facing values; bayesian.py will import these.
# ---------------------------------------------------------------------------

class HomeType(str, Enum):
    APARTMENT = "apartment"
    HOUSE = "house"
    MOBILE_HOME = "mobile_home"
    CONDO = "condo"
    OTHER = "other"


class VehicleClearance(str, Enum):
    NONE = "none"
    LOW = "low"       # sedan, coupe
    MEDIUM = "medium" # crossover, minivan
    HIGH = "high"     # SUV, truck, lifted vehicle


class HomeFloorState(str, Enum):
    """Derived state used as Bayesian evidence."""
    GROUND = "Ground"
    UPPER = "Upper"
    ELEVATED = "Elevated"


class RiskNotifyLevel(str, Enum):
    """Risk threshold at which a contact should be notified."""
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Nested schemas
# ---------------------------------------------------------------------------

class Location(BaseModel):
    address: str = Field(min_length=1, max_length=300)
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    # NWS zone ID (e.g. "LAZ036"). Optional because we can resolve it from lat/lon.
    nws_zone_id: str | None = Field(default=None, pattern=r"^[A-Z]{2}[A-Z0-9]{3,5}$")
    # 5-digit FIPS county code (e.g. "22033"). Optional.
    county_fips: str | None = Field(default=None, pattern=r"^\d{5}$")


class Home(BaseModel):
    type: HomeType
    # Floor level the user lives on. 1 = ground. Max 200 covers skyscrapers.
    floor_level: int = Field(ge=1, le=200)
    # True for pier-and-beam / stilt houses where floor 1 is still elevated.
    elevated: bool = False
    # FEMA flood zone code (AE, X, VE, etc.) or None.
    flood_zone: str | None = Field(default=None, max_length=10)
    has_generator: bool = False
    has_storm_shutters: bool = False


class Vehicle(BaseModel):
    owns_vehicle: bool = True
    clearance: VehicleClearance = VehicleClearance.LOW
    four_wheel_drive: bool = False

    @model_validator(mode="after")
    def _clearance_none_iff_no_vehicle(self) -> "Vehicle":
        if not self.owns_vehicle and self.clearance != VehicleClearance.NONE:
            # Silently normalize rather than raise: a user who toggles
            # owns_vehicle off shouldn't have to also zero out clearance.
            object.__setattr__(self, "clearance", VehicleClearance.NONE)
            object.__setattr__(self, "four_wheel_drive", False)
        return self


class Medical(BaseModel):
    mobility_limited: bool = False
    oxygen_dependent: bool = False
    refrigerated_medication: bool = False
    chronic_conditions: list[str] = Field(default_factory=list)


class EmergencyContact(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    relationship: str = Field(min_length=1, max_length=50)
    phone: str
    notify_on: list[RiskNotifyLevel] = Field(
        default_factory=lambda: [RiskNotifyLevel.HIGH, RiskNotifyLevel.CRITICAL]
    )

    @field_validator("phone")
    @classmethod
    def _validate_phone(cls, v: str) -> str:
        # E.164: + followed by 8-15 digits. Loose enough to accept most intl.
        if not re.fullmatch(r"\+\d{8,15}", v):
            raise ValueError(
                f"Phone number must be in E.164 format (e.g. +15551234567), got {v!r}"
            )
        return v


class QuietHours(BaseModel):
    start: time
    end: time


class Preferences(BaseModel):
    language: Literal["en", "es"] = "en"
    # IANA timezone string; not exhaustively validated here.
    timezone: str = "America/Chicago"
    quiet_hours: QuietHours | None = None
    allow_smart_home_actions: bool = True


# ---------------------------------------------------------------------------
# Top-level profile
# ---------------------------------------------------------------------------

class UserProfile(BaseModel):
    """Root profile object. Validated on construction."""

    user_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    name: str = Field(min_length=1, max_length=100)
    location: Location
    home: Home
    vehicle: Vehicle
    medical: Medical
    emergency_contacts: list[EmergencyContact] = Field(default_factory=list)
    preferences: Preferences = Field(default_factory=Preferences)

    # ----- Derived helpers used by the Bayesian engine (Phase 5) --------

    @property
    def home_floor_state(self) -> HomeFloorState:
        """Translate raw floor info into the Bayesian 'HomeFloor' state."""
        if self.home.elevated:
            return HomeFloorState.ELEVATED
        if self.home.floor_level >= 2:
            return HomeFloorState.UPPER
        return HomeFloorState.GROUND

    @property
    def is_medically_vulnerable(self) -> bool:
        """True if any medical factor raises baseline urgency."""
        m = self.medical
        return bool(
            m.mobility_limited
            or m.oxygen_dependent
            or m.refrigerated_medication
            or m.chronic_conditions
        )

    def contacts_to_notify(self, level: RiskNotifyLevel) -> list[EmergencyContact]:
        """Contacts whose notify_on threshold has been reached at this level.

        Note: CRITICAL should also notify contacts set to HIGH, and HIGH should
        also notify contacts set to MODERATE. We implement that as a ranking.
        """
        rank = {
            RiskNotifyLevel.MODERATE: 1,
            RiskNotifyLevel.HIGH: 2,
            RiskNotifyLevel.CRITICAL: 3,
        }
        current = rank[level]
        return [
            c
            for c in self.emergency_contacts
            if any(rank[n] <= current for n in c.notify_on)
        ]


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------

def load_profile(path: str | Path) -> UserProfile:
    """Load and validate a profile from a JSON file.

    Raises pydantic.ValidationError with a readable message on malformed data.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Profile not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    return UserProfile.model_validate(data)


def save_profile(profile: UserProfile, path: str | Path) -> None:
    """Write a profile to disk as pretty-printed JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        profile.model_dump_json(indent=2, exclude_none=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "HomeType",
    "VehicleClearance",
    "HomeFloorState",
    "RiskNotifyLevel",
    "Location",
    "Home",
    "Vehicle",
    "Medical",
    "EmergencyContact",
    "QuietHours",
    "Preferences",
    "UserProfile",
    "load_profile",
    "save_profile",
]
