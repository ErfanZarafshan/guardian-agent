"""Tests for the weather data layer.

Uses pytest-mock to patch session.get so no real HTTP calls happen. Canned
JSON fixtures mirror the shape of real NWS and OWM responses.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
import requests

from guardian.weather.aggregator import observe
from guardian.weather.http import (
    WeatherAPIConfigError,
    WeatherAPIParseError,
    WeatherAPIRequestError,
    build_session,
    get_json,
)
from guardian.weather.nws import NWSClient, PointMetadata
from guardian.weather.observation import (
    PRECIP_THRESHOLDS_IN_HR,
    WIND_THRESHOLDS_MPH,
    CertaintyLevel,
    PrecipCategory,
    SeverityLevel,
    UrgencyLevel,
    WeatherAlert,
    WeatherObservation,
    WindCategory,
    bucket_precip_in_hr,
    bucket_wind_mph,
)
from guardian.weather.owm import OWMClient


# ---------------------------------------------------------------------------
# Helpers for building fake responses
# ---------------------------------------------------------------------------

def _fake_response(json_payload: dict, status: int = 200) -> MagicMock:
    r = MagicMock(spec=requests.Response)
    r.status_code = status
    r.json.return_value = json_payload
    r.text = str(json_payload)
    return r


# ---------------------------------------------------------------------------
# observation.py: bucketing logic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mph, expected",
    [
        (0.0, WindCategory.CALM),
        (12.9, WindCategory.CALM),
        (13.0, WindCategory.BREEZY),
        (25.0, WindCategory.BREEZY),
        (32.0, WindCategory.STRONG),
        (54.9, WindCategory.STRONG),
        (55.0, WindCategory.DAMAGING),
        (120.0, WindCategory.DAMAGING),
    ],
)
def test_bucket_wind(mph: float, expected: WindCategory) -> None:
    assert bucket_wind_mph(mph) is expected


@pytest.mark.parametrize(
    "rate, expected",
    [
        (-1.0, PrecipCategory.NONE),
        (0.0, PrecipCategory.NONE),
        (0.005, PrecipCategory.NONE),
        (0.05, PrecipCategory.LIGHT),
        (0.15, PrecipCategory.MODERATE),
        (0.5, PrecipCategory.HEAVY),
        (1.5, PrecipCategory.EXTREME),
    ],
)
def test_bucket_precip(rate: float, expected: PrecipCategory) -> None:
    assert bucket_precip_in_hr(rate) is expected


def test_wind_thresholds_cover_range() -> None:
    """The bucket table should tile 0..inf with no gaps or overlaps."""
    edges = sorted(lo for lo, hi in WIND_THRESHOLDS_MPH.values())
    assert edges == [0.0, 13.0, 32.0, 55.0]


def test_precip_thresholds_cover_range() -> None:
    edges = sorted(lo for lo, hi in PRECIP_THRESHOLDS_IN_HR.values())
    assert edges == [0.0, 0.01, 0.10, 0.30, 1.00]


# ---------------------------------------------------------------------------
# observation.py: WeatherObservation derived properties
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def test_observation_uses_gust_over_sustained_wind() -> None:
    obs = WeatherObservation(
        observed_at=_now(),
        latitude=30.0,
        longitude=-91.0,
        wind_speed_mph=20.0,  # Breezy
        wind_gust_mph=60.0,   # Damaging
    )
    assert obs.wind_category is WindCategory.DAMAGING


def test_observation_no_alerts_gives_none_severity() -> None:
    obs = WeatherObservation(observed_at=_now(), latitude=30.0, longitude=-91.0)
    assert obs.max_severity is SeverityLevel.NONE
    assert obs.max_urgency is UrgencyLevel.UNKNOWN
    assert obs.has_active_alerts is False


def test_observation_max_severity_across_alerts() -> None:
    now = _now()
    obs = WeatherObservation(
        observed_at=now,
        latitude=30.0,
        longitude=-91.0,
        alerts=[
            WeatherAlert(
                source="nws",
                event="Heat Advisory",
                severity=SeverityLevel.MODERATE,
                urgency=UrgencyLevel.EXPECTED,
                onset=now - timedelta(hours=1),
                expires=now + timedelta(hours=1),
            ),
            WeatherAlert(
                source="nws",
                event="Tornado Warning",
                severity=SeverityLevel.EXTREME,
                urgency=UrgencyLevel.IMMEDIATE,
                onset=now - timedelta(minutes=5),
                expires=now + timedelta(minutes=25),
            ),
        ],
    )
    assert obs.max_severity is SeverityLevel.EXTREME
    assert obs.max_urgency is UrgencyLevel.IMMEDIATE
    assert obs.has_active_alerts is True


def test_observation_ignores_expired_alerts() -> None:
    now = _now()
    obs = WeatherObservation(
        observed_at=now,
        latitude=30.0,
        longitude=-91.0,
        alerts=[
            WeatherAlert(
                source="nws",
                event="Old Tornado Warning",
                severity=SeverityLevel.EXTREME,
                urgency=UrgencyLevel.IMMEDIATE,
                onset=now - timedelta(hours=3),
                expires=now - timedelta(hours=1),  # expired
            ),
        ],
    )
    assert obs.max_severity is SeverityLevel.NONE
    assert obs.has_active_alerts is False


def test_alert_is_active_open_ended() -> None:
    """No onset/expires -> treat as always active."""
    alert = WeatherAlert(source="nws", event="X")
    assert alert.is_active() is True


# ---------------------------------------------------------------------------
# http.py
# ---------------------------------------------------------------------------

def test_build_session_sets_user_agent() -> None:
    s = build_session("TestAgent/1.0 (test@example.com)")
    assert s.headers["User-Agent"] == "TestAgent/1.0 (test@example.com)"


def test_get_json_raises_on_http_error(mocker) -> None:
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _fake_response({}, status=500)
    with pytest.raises(WeatherAPIRequestError):
        get_json(session, "https://example.com")


def test_get_json_raises_on_non_json(mocker) -> None:
    session = MagicMock(spec=requests.Session)
    r = MagicMock(spec=requests.Response)
    r.status_code = 200
    r.json.side_effect = ValueError("not json")
    r.text = "<html>"
    session.get.return_value = r
    with pytest.raises(WeatherAPIParseError):
        get_json(session, "https://example.com")


def test_get_json_raises_on_network_error() -> None:
    session = MagicMock(spec=requests.Session)
    session.get.side_effect = requests.ConnectionError("boom")
    with pytest.raises(WeatherAPIRequestError):
        get_json(session, "https://example.com")


# ---------------------------------------------------------------------------
# NWS client
# ---------------------------------------------------------------------------

def _nws_points_payload() -> dict:
    return {
        "properties": {
            "gridId": "LIX",
            "gridX": 65,
            "gridY": 77,
            "forecastZone": "https://api.weather.gov/zones/forecast/LAZ036",
            "county": "https://api.weather.gov/zones/county/LAC033",
            "timeZone": "America/Chicago",
        }
    }


def _nws_alerts_payload_empty() -> dict:
    return {"features": []}


def _nws_alerts_payload_tornado() -> dict:
    return {
        "features": [
            {
                "properties": {
                    "event": "Tornado Warning",
                    "headline": "Tornado Warning for East Baton Rouge Parish",
                    "description": "At 3:15 PM, a tornado was reported...",
                    "severity": "Extreme",
                    "urgency": "Immediate",
                    "certainty": "Observed",
                    "onset": "2026-04-19T15:00:00-05:00",
                    "expires": "2026-04-19T15:45:00-05:00",
                    "senderName": "NWS New Orleans LA",
                }
            }
        ]
    }


def test_nws_get_point_metadata() -> None:
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _fake_response(_nws_points_payload())
    client = NWSClient(session=session)
    meta = client.get_point_metadata(30.4133, -91.18)
    assert meta == PointMetadata(
        office="LIX",
        grid_x=65,
        grid_y=77,
        forecast_zone_id="LAZ036",
        county_zone_id="LAC033",
        timezone="America/Chicago",
    )


def test_nws_point_metadata_raises_on_malformed() -> None:
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _fake_response({"properties": {}})
    client = NWSClient(session=session)
    with pytest.raises(WeatherAPIParseError):
        client.get_point_metadata(30.0, -91.0)


def test_nws_alerts_empty() -> None:
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _fake_response(_nws_alerts_payload_empty())
    client = NWSClient(session=session)
    alerts = client.get_active_alerts_by_zone("LAZ036")
    assert alerts == []


def test_nws_alerts_normalized_correctly() -> None:
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _fake_response(_nws_alerts_payload_tornado())
    client = NWSClient(session=session)
    alerts = client.get_active_alerts_by_zone("LAZ036")
    assert len(alerts) == 1
    a = alerts[0]
    assert a.source == "nws"
    assert a.event == "Tornado Warning"
    assert a.severity is SeverityLevel.EXTREME
    assert a.urgency is UrgencyLevel.IMMEDIATE
    assert a.certainty is CertaintyLevel.OBSERVED
    assert a.sender == "NWS New Orleans LA"
    assert a.onset is not None
    assert a.expires is not None


def test_nws_alerts_with_missing_fields_still_parse() -> None:
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _fake_response(
        {"features": [{"properties": {"event": "Mystery Event"}}]}
    )
    client = NWSClient(session=session)
    alerts = client.get_active_alerts_by_zone("LAZ036")
    assert len(alerts) == 1
    assert alerts[0].event == "Mystery Event"
    assert alerts[0].severity is SeverityLevel.NONE
    assert alerts[0].urgency is UrgencyLevel.UNKNOWN


# ---------------------------------------------------------------------------
# OWM client
# ---------------------------------------------------------------------------

def _owm_current_payload() -> dict:
    return {
        "dt": 1_760_000_000,
        "main": {
            "temp": 88.7,
            "humidity": 72,
            "pressure": 1012,
        },
        "wind": {"speed": 14.3, "gust": 22.0},
        "visibility": 16093,  # ~10 miles in meters
        "rain": {"1h": 2.54},  # 1 inch in mm
        "weather": [{"main": "Rain", "description": "heavy rain"}],
    }


def test_owm_raises_without_api_key(monkeypatch) -> None:
    from guardian import config as config_module

    monkeypatch.setattr(config_module, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("OWM_API_KEY", raising=False)
    config_module.get_config.cache_clear()
    try:
        with pytest.raises(WeatherAPIConfigError):
            OWMClient()
    finally:
        config_module.get_config.cache_clear()


def test_owm_parses_current_conditions() -> None:
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _fake_response(_owm_current_payload())
    client = OWMClient(api_key="fake-key", session=session)
    c = client.get_current(30.0, -91.0)
    assert c.temperature_f == 88.7
    assert c.humidity_pct == 72
    assert c.wind_speed_mph == 14.3
    assert c.wind_gust_mph == 22.0
    # Visibility: 16093m -> ~10 miles
    assert c.visibility_mi is not None and 9.9 < c.visibility_mi < 10.1
    # Precip: 2.54mm in 1h -> 0.1 in/hr
    assert c.precip_rate_in_hr is not None
    assert abs(c.precip_rate_in_hr - 0.1) < 1e-6
    assert c.weather_main == "Rain"


def test_owm_handles_missing_precip() -> None:
    payload = _owm_current_payload()
    payload.pop("rain")
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _fake_response(payload)
    client = OWMClient(api_key="fake-key", session=session)
    c = client.get_current(30.0, -91.0)
    assert c.precip_rate_in_hr is None


def test_owm_forecast_parses_3h_blocks() -> None:
    payload = {
        "list": [
            {
                "dt": 1_760_000_000,
                "main": {"temp": 75.0},
                "wind": {"speed": 10.0},
                "rain": {"3h": 7.62},  # 3mm/hr average
                "pop": 0.6,
                "weather": [{"main": "Rain"}],
            }
        ]
    }
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _fake_response(payload)
    client = OWMClient(api_key="fake-key", session=session)
    entries = client.get_forecast(30.0, -91.0)
    assert len(entries) == 1
    e = entries[0]
    assert e.temperature_f == 75.0
    assert e.precip_prob_pct == pytest.approx(60.0)
    # 7.62mm over 3h -> 2.54mm/hr -> 0.1 in/hr
    assert e.precip_rate_in_hr == pytest.approx(0.1, abs=1e-6)


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def _stub_nws(alerts: list[WeatherAlert]) -> MagicMock:
    nws = MagicMock(spec=NWSClient)
    nws.get_active_alerts_by_zone.return_value = alerts
    nws.get_active_alerts_by_point.return_value = alerts
    nws.get_point_metadata.return_value = PointMetadata(
        office="LIX",
        grid_x=65,
        grid_y=77,
        forecast_zone_id="LAZ036",
    )
    return nws


def _stub_owm(temperature_f: float = 85.0, wind_mph: float = 15.0) -> MagicMock:
    owm = MagicMock(spec=OWMClient)
    from guardian.weather.owm import CurrentConditions

    owm.get_current.return_value = CurrentConditions(
        observed_at=_now(),
        temperature_f=temperature_f,
        humidity_pct=60.0,
        pressure_mb=1013.0,
        wind_speed_mph=wind_mph,
        wind_gust_mph=None,
        visibility_mi=10.0,
        precip_rate_in_hr=None,
        weather_main="Clear",
        weather_description="clear sky",
        raw={},
    )
    return owm


def test_observe_combines_sources() -> None:
    obs = observe(
        latitude=30.4,
        longitude=-91.2,
        nws_zone_id="LAZ036",
        nws=_stub_nws([]),
        owm=_stub_owm(),
    )
    assert "nws" in obs.sources
    assert "owm" in obs.sources
    assert obs.temperature_f == 85.0
    assert obs.nws_zone_id == "LAZ036"
    assert not obs.has_active_alerts


def test_observe_still_works_when_owm_fails() -> None:
    from guardian.weather.http import WeatherAPIRequestError

    owm = MagicMock(spec=OWMClient)
    owm.get_current.side_effect = WeatherAPIRequestError("down")
    obs = observe(
        latitude=30.4,
        longitude=-91.2,
        nws_zone_id="LAZ036",
        nws=_stub_nws([]),
        owm=owm,
    )
    assert obs.sources == ["nws"]
    assert obs.temperature_f is None


def test_observe_still_works_when_nws_fails() -> None:
    from guardian.weather.http import WeatherAPIRequestError

    nws = MagicMock(spec=NWSClient)
    nws.get_active_alerts_by_zone.side_effect = WeatherAPIRequestError("down")
    obs = observe(
        latitude=30.4,
        longitude=-91.2,
        nws_zone_id="LAZ036",
        nws=nws,
        owm=_stub_owm(),
    )
    assert obs.sources == ["owm"]
    assert obs.alerts == []


def test_observe_raises_when_both_sources_fail() -> None:
    from guardian.weather.http import WeatherAPIRequestError

    nws = MagicMock(spec=NWSClient)
    nws.get_active_alerts_by_zone.side_effect = WeatherAPIRequestError("down")
    owm = MagicMock(spec=OWMClient)
    owm.get_current.side_effect = WeatherAPIRequestError("down")
    with pytest.raises(ValueError):
        observe(
            latitude=30.4,
            longitude=-91.2,
            nws_zone_id="LAZ036",
            nws=nws,
            owm=owm,
        )


def test_observe_resolves_zone_when_not_provided() -> None:
    nws = _stub_nws([])
    observe(
        latitude=30.4,
        longitude=-91.2,
        nws_zone_id=None,
        nws=nws,
        owm=_stub_owm(),
    )
    nws.get_active_alerts_by_point.assert_called_once()
    nws.get_point_metadata.assert_called_once()


def test_observe_skip_owm_flag() -> None:
    obs = observe(
        latitude=30.4,
        longitude=-91.2,
        nws_zone_id="LAZ036",
        nws=_stub_nws([]),
        skip_owm=True,
    )
    assert obs.sources == ["nws"]
