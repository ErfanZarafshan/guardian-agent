"""NOAA National Weather Service API client (free, no API key required).

Endpoints we use:
  - GET /points/{lat},{lon}              -> office, grid, forecastZone
  - GET /alerts/active                    -> filtered by zone or point
  - GET /gridpoints/{office}/{x},{y}/forecast/hourly

Docs: https://www.weather.gov/documentation/services-web-api

Note: NWS insists on a descriptive User-Agent with contact info. We pass the
one configured in .env (NWS_USER_AGENT). A bad UA may get rate-limited.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..config import get_config
from ..logging_setup import get_logger
from .http import (
    WeatherAPIParseError,
    build_session,
    get_json,
)
from .observation import (
    CertaintyLevel,
    SeverityLevel,
    UrgencyLevel,
    WeatherAlert,
)


log = get_logger(__name__)

BASE_URL = "https://api.weather.gov"


# ---------------------------------------------------------------------------
# Point metadata (resolves lat/lon -> office/grid/zone)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PointMetadata:
    office: str  # WFO id, e.g. "LIX" for New Orleans/Baton Rouge
    grid_x: int
    grid_y: int
    forecast_zone_id: str  # e.g. "LAZ036"
    county_zone_id: str | None = None
    timezone: str | None = None


# ---------------------------------------------------------------------------
# Severity / urgency normalization
# ---------------------------------------------------------------------------

_SEVERITY_MAP = {
    "Unknown": SeverityLevel.NONE,
    "Minor": SeverityLevel.MINOR,
    "Moderate": SeverityLevel.MODERATE,
    "Severe": SeverityLevel.SEVERE,
    "Extreme": SeverityLevel.EXTREME,
}

_URGENCY_MAP = {
    "Unknown": UrgencyLevel.UNKNOWN,
    "Past": UrgencyLevel.PAST,
    "Future": UrgencyLevel.FUTURE,
    "Expected": UrgencyLevel.EXPECTED,
    "Immediate": UrgencyLevel.IMMEDIATE,
}

_CERTAINTY_MAP = {
    "Unknown": CertaintyLevel.UNKNOWN,
    "Unlikely": CertaintyLevel.UNLIKELY,
    "Possible": CertaintyLevel.POSSIBLE,
    "Likely": CertaintyLevel.LIKELY,
    "Observed": CertaintyLevel.OBSERVED,
}


def _parse_iso(dt_str: str | None) -> datetime | None:
    if not dt_str:
        return None
    try:
        # NWS uses ISO 8601 with offset, e.g. "2025-06-15T14:00:00-05:00"
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _feature_to_alert(feature: dict) -> WeatherAlert:
    """Translate a single NWS alert GeoJSON feature into WeatherAlert."""
    props = feature.get("properties", {}) or {}
    return WeatherAlert(
        source="nws",
        event=props.get("event", "Unknown Event"),
        headline=props.get("headline"),
        description=props.get("description"),
        severity=_SEVERITY_MAP.get(props.get("severity", "Unknown"), SeverityLevel.NONE),
        urgency=_URGENCY_MAP.get(props.get("urgency", "Unknown"), UrgencyLevel.UNKNOWN),
        certainty=_CERTAINTY_MAP.get(
            props.get("certainty", "Unknown"), CertaintyLevel.UNKNOWN
        ),
        onset=_parse_iso(props.get("onset") or props.get("effective")),
        expires=_parse_iso(props.get("expires") or props.get("ends")),
        sender=props.get("senderName"),
        raw=props,
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class NWSClient:
    """Thin typed wrapper around the NWS REST API.

    Constructed without arguments uses config from .env. Pass `session=` to
    inject a mock in tests.
    """

    def __init__(self, session=None, user_agent: str | None = None) -> None:
        ua = user_agent or get_config().nws_user_agent
        self.session = session or build_session(ua)

    # ----- Point metadata --------------------------------------------------

    def get_point_metadata(self, latitude: float, longitude: float) -> PointMetadata:
        """Resolve forecast office, grid, and zone ID for lat/lon."""
        url = f"{BASE_URL}/points/{latitude:.4f},{longitude:.4f}"
        data = get_json(self.session, url)
        try:
            props = data["properties"]
            zone_url = props["forecastZone"]
            forecast_zone_id = zone_url.rstrip("/").rsplit("/", 1)[-1]
            county_url = props.get("county")
            county_zone_id = (
                county_url.rstrip("/").rsplit("/", 1)[-1] if county_url else None
            )
            return PointMetadata(
                office=props["gridId"],
                grid_x=int(props["gridX"]),
                grid_y=int(props["gridY"]),
                forecast_zone_id=forecast_zone_id,
                county_zone_id=county_zone_id,
                timezone=props.get("timeZone"),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise WeatherAPIParseError(
                f"Unexpected /points response shape: {e}"
            ) from e

    # ----- Active alerts ---------------------------------------------------

    def get_active_alerts_by_zone(self, zone_id: str) -> list[WeatherAlert]:
        """Fetch all active alerts for an NWS forecast zone."""
        url = f"{BASE_URL}/alerts/active/zone/{zone_id}"
        data = get_json(self.session, url)
        features = data.get("features", [])
        return [_feature_to_alert(f) for f in features]

    def get_active_alerts_by_point(
        self, latitude: float, longitude: float
    ) -> list[WeatherAlert]:
        """Fetch active alerts for a lat/lon point (alternative to zone)."""
        url = f"{BASE_URL}/alerts/active"
        params = {"point": f"{latitude:.4f},{longitude:.4f}"}
        data = get_json(self.session, url, params=params)
        features = data.get("features", [])
        return [_feature_to_alert(f) for f in features]

    # ----- Forecast --------------------------------------------------------

    def get_hourly_forecast(self, meta: PointMetadata) -> list[dict]:
        """Return the hourly forecast period list.

        Each entry has keys like: startTime, endTime, temperature,
        temperatureUnit, windSpeed, windDirection, shortForecast,
        probabilityOfPrecipitation.
        """
        url = (
            f"{BASE_URL}/gridpoints/{meta.office}/{meta.grid_x},{meta.grid_y}/"
            "forecast/hourly"
        )
        data = get_json(self.session, url)
        try:
            return list(data["properties"]["periods"])
        except (KeyError, TypeError) as e:
            raise WeatherAPIParseError(
                f"Unexpected /forecast/hourly response shape: {e}"
            ) from e


__all__ = ["NWSClient", "PointMetadata", "BASE_URL"]
