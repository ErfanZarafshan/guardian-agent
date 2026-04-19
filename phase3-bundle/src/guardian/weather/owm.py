"""OpenWeatherMap free-tier API client.

Endpoints used (free tier, no card required):
  - GET /data/2.5/weather?lat={}&lon={}&appid={key}&units=imperial
  - GET /data/2.5/forecast?lat={}&lon={}&appid={key}&units=imperial

Docs: https://openweathermap.org/current and /forecast5

Free-tier limitations vs. One Call 3.0:
  - No minute-by-minute precipitation (we use current rain.1h instead).
  - No built-in alerts endpoint (NWS is the sole source of alerts for us).
  - 3-hour forecast granularity, 5 days ahead.

That's fine for our purposes: NWS covers alerts, and the Bayesian network only
needs current conditions + a short-horizon forecast to compute evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from ..config import get_config
from ..logging_setup import get_logger
from .http import (
    WeatherAPIConfigError,
    WeatherAPIParseError,
    build_session,
    get_json,
)


log = get_logger(__name__)

BASE_URL = "https://api.openweathermap.org/data/2.5"
USER_AGENT = "GuardianAgent/0.1 (academic project)"


@dataclass(frozen=True)
class CurrentConditions:
    """Current-conditions snapshot from OWM /weather."""
    observed_at: datetime
    temperature_f: float | None
    humidity_pct: float | None
    pressure_mb: float | None
    wind_speed_mph: float | None
    wind_gust_mph: float | None
    visibility_mi: float | None  # OWM gives meters; we convert
    precip_rate_in_hr: float | None  # from rain.1h / snow.1h fields
    weather_main: str | None
    weather_description: str | None
    raw: dict


@dataclass(frozen=True)
class ForecastEntry:
    """One 3-hour block from OWM /forecast."""
    valid_at: datetime
    temperature_f: float | None
    wind_speed_mph: float | None
    wind_gust_mph: float | None
    precip_rate_in_hr: float | None
    precip_prob_pct: float | None
    weather_main: str | None


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------

def _meters_to_miles(m: float | None) -> float | None:
    return None if m is None else m / 1609.344


def _mm_to_inches(mm: float | None) -> float | None:
    return None if mm is None else mm / 25.4


def _ts_to_utc(epoch_s: float | None) -> datetime | None:
    if epoch_s is None:
        return None
    return datetime.fromtimestamp(epoch_s, tz=timezone.utc)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_current(payload: dict) -> CurrentConditions:
    try:
        main = payload.get("main", {}) or {}
        wind = payload.get("wind", {}) or {}
        rain = payload.get("rain", {}) or {}
        snow = payload.get("snow", {}) or {}
        weather_list = payload.get("weather") or []
        weather0 = weather_list[0] if weather_list else {}

        # Precipitation: OWM reports mm over the last 1 hour (rain.1h, snow.1h).
        # We combine rain + snow, then convert to inches/hour.
        rain_mm = rain.get("1h")
        snow_mm = snow.get("1h")
        if rain_mm is None and snow_mm is None:
            precip_in_hr: float | None = None
        else:
            precip_in_hr = _mm_to_inches((rain_mm or 0) + (snow_mm or 0))

        return CurrentConditions(
            observed_at=_ts_to_utc(payload.get("dt")) or datetime.now(timezone.utc),
            temperature_f=main.get("temp"),
            humidity_pct=main.get("humidity"),
            pressure_mb=main.get("pressure"),
            wind_speed_mph=wind.get("speed"),
            wind_gust_mph=wind.get("gust"),
            visibility_mi=_meters_to_miles(payload.get("visibility")),
            precip_rate_in_hr=precip_in_hr,
            weather_main=weather0.get("main"),
            weather_description=weather0.get("description"),
            raw=payload,
        )
    except (KeyError, TypeError, IndexError) as e:
        raise WeatherAPIParseError(f"Unexpected /weather payload: {e}") from e


def _parse_forecast_entry(item: dict) -> ForecastEntry:
    main = item.get("main", {}) or {}
    wind = item.get("wind", {}) or {}
    rain = item.get("rain", {}) or {}
    snow = item.get("snow", {}) or {}
    weather_list = item.get("weather") or []
    weather0 = weather_list[0] if weather_list else {}

    # Forecast blocks give rain over last 3h as rain.3h; divide to get in/hr
    rain_mm_3h = rain.get("3h")
    snow_mm_3h = snow.get("3h")
    if rain_mm_3h is None and snow_mm_3h is None:
        precip_in_hr: float | None = None
    else:
        precip_in_hr = _mm_to_inches(((rain_mm_3h or 0) + (snow_mm_3h or 0)) / 3.0)

    pop = item.get("pop")  # probability 0.0-1.0
    precip_prob_pct = None if pop is None else pop * 100.0

    return ForecastEntry(
        valid_at=_ts_to_utc(item.get("dt")) or datetime.now(timezone.utc),
        temperature_f=main.get("temp"),
        wind_speed_mph=wind.get("speed"),
        wind_gust_mph=wind.get("gust"),
        precip_rate_in_hr=precip_in_hr,
        precip_prob_pct=precip_prob_pct,
        weather_main=weather0.get("main"),
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class OWMClient:
    """Thin wrapper around OpenWeatherMap's free /weather and /forecast."""

    def __init__(self, api_key: str | None = None, session=None) -> None:
        cfg = get_config()
        self.api_key = api_key or cfg.owm_api_key
        if not self.api_key:
            raise WeatherAPIConfigError(
                "OWM_API_KEY is not set. Add it to your .env file."
            )
        self.session = session or build_session(USER_AGENT)

    def get_current(self, latitude: float, longitude: float) -> CurrentConditions:
        """Fetch current conditions at lat/lon in imperial units."""
        params = {
            "lat": f"{latitude:.4f}",
            "lon": f"{longitude:.4f}",
            "appid": self.api_key,
            "units": "imperial",
        }
        payload = get_json(self.session, f"{BASE_URL}/weather", params=params)
        return _parse_current(payload)

    def get_forecast(self, latitude: float, longitude: float) -> list[ForecastEntry]:
        """Fetch the 5-day / 3-hour forecast."""
        params = {
            "lat": f"{latitude:.4f}",
            "lon": f"{longitude:.4f}",
            "appid": self.api_key,
            "units": "imperial",
        }
        payload = get_json(self.session, f"{BASE_URL}/forecast", params=params)
        entries = payload.get("list") or []
        return [_parse_forecast_entry(e) for e in entries]


__all__ = ["OWMClient", "CurrentConditions", "ForecastEntry", "BASE_URL"]
