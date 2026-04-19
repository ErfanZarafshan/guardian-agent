"""OpenWeatherMap API client.

Implemented in **Phase 3**. Supports two modes:

  - "free":    /data/2.5/weather + /data/2.5/forecast (no card required)
  - "onecall": One Call API 3.0 (requires account with card, first 1000/day free)

Will expose:

  - get_current(lat, lon) -> CurrentConditions
  - get_forecast(lat, lon) -> HourlyForecast
  - get_minute_precip(lat, lon) -> MinutePrecip  # onecall mode only
"""
