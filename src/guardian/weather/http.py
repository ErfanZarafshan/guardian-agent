"""Shared HTTP infrastructure for weather clients.

Centralizes:
  - a `requests.Session` with automatic retry on transient failures
  - a sensible timeout
  - typed custom exceptions so callers can handle API failure vs. parse
    failure vs. configuration failure separately.
"""

from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DEFAULT_TIMEOUT_S = 10.0


class WeatherAPIError(RuntimeError):
    """Base class for weather-API failures."""


class WeatherAPIRequestError(WeatherAPIError):
    """Network, HTTP-status, or timeout failure."""


class WeatherAPIParseError(WeatherAPIError):
    """The API responded, but the payload was unexpectedly shaped."""


class WeatherAPIConfigError(WeatherAPIError):
    """Misconfiguration — e.g. missing API key, malformed URL."""


def build_session(user_agent: str) -> requests.Session:
    """Return a requests.Session with retry + User-Agent pre-set.

    Retries: 3 attempts with exponential backoff on 429, 500, 502, 503, 504.
    """
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "application/json",
        }
    )
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def get_json(session: requests.Session, url: str, params: dict | None = None) -> dict:
    """GET a URL, raise WeatherAPIRequestError on failure, return parsed JSON."""
    try:
        resp = session.get(url, params=params, timeout=DEFAULT_TIMEOUT_S)
    except requests.RequestException as e:
        raise WeatherAPIRequestError(f"Request failed: {url} ({e})") from e

    if resp.status_code >= 400:
        raise WeatherAPIRequestError(
            f"HTTP {resp.status_code} from {url}: {resp.text[:300]}"
        )

    try:
        return resp.json()
    except ValueError as e:
        raise WeatherAPIParseError(f"Non-JSON response from {url}: {e}") from e
