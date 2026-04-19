"""Aggregator: calls NWS + OWM and produces one WeatherObservation.

This is the single function the rest of the agent will call:

    obs = observe(latitude=30.4, longitude=-91.2)

Failure handling: if one source fails, we still return an observation built
from whatever succeeded, with `sources` listing only the ones that worked. A
`ValueError` is raised only if *both* sources fail.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..logging_setup import get_logger
from .http import WeatherAPIError
from .nws import NWSClient
from .observation import AlertSource, WeatherObservation
from .owm import OWMClient


log = get_logger(__name__)


def observe(
    latitude: float,
    longitude: float,
    nws_zone_id: str | None = None,
    nws: NWSClient | None = None,
    owm: OWMClient | None = None,
    skip_owm: bool = False,
) -> WeatherObservation:
    """Produce a unified WeatherObservation for the given location.

    Parameters
    ----------
    latitude, longitude
        Location to observe, in decimal degrees.
    nws_zone_id
        If provided, skip NWS point-metadata lookup and query alerts by zone
        directly. If None, we query alerts by point.
    nws, owm
        Optional injected clients (for tests). If omitted, we build them from
        .env config.
    skip_owm
        If True, don't attempt to query OpenWeatherMap. Useful when the OWM
        key is not yet activated.
    """
    nws = nws or NWSClient()
    if skip_owm:
        owm = None
    elif owm is None:
        try:
            owm = OWMClient()
        except WeatherAPIError as e:
            log.warning("OWM client unavailable: %s", e)
            owm = None

    sources: list[AlertSource] = []

    # --- NWS: alerts (and zone resolution if needed) ---
    alerts = []
    resolved_zone = nws_zone_id
    try:
        if resolved_zone:
            alerts = nws.get_active_alerts_by_zone(resolved_zone)
        else:
            alerts = nws.get_active_alerts_by_point(latitude, longitude)
            # Best-effort zone lookup so later calls don't re-resolve.
            try:
                meta = nws.get_point_metadata(latitude, longitude)
                resolved_zone = meta.forecast_zone_id
            except WeatherAPIError as e:
                log.debug("NWS zone resolution failed: %s", e)
        sources.append("nws")
    except WeatherAPIError as e:
        log.warning("NWS fetch failed: %s", e)

    # --- OWM: current conditions ---
    current = None
    if owm is not None:
        try:
            current = owm.get_current(latitude, longitude)
            sources.append("owm")
        except WeatherAPIError as e:
            log.warning("OWM fetch failed: %s", e)

    if not sources:
        raise ValueError(
            "Both NWS and OWM failed to return data; cannot build an observation."
        )

    obs = WeatherObservation(
        observed_at=datetime.now(timezone.utc),
        latitude=latitude,
        longitude=longitude,
        temperature_f=current.temperature_f if current else None,
        wind_speed_mph=current.wind_speed_mph if current else None,
        wind_gust_mph=current.wind_gust_mph if current else None,
        precip_rate_in_hr=current.precip_rate_in_hr if current else None,
        humidity_pct=current.humidity_pct if current else None,
        pressure_mb=current.pressure_mb if current else None,
        visibility_mi=current.visibility_mi if current else None,
        alerts=alerts,
        sources=sources,
        nws_zone_id=resolved_zone,
    )
    return obs


__all__ = ["observe"]
