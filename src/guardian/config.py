"""Configuration loader.

Reads environment variables from a `.env` file (if present) and exposes them as
typed attributes on a single `Config` dataclass. All modules in the package
should import `get_config()` rather than reading `os.environ` directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    # --- OpenWeatherMap ---
    owm_api_key: str
    owm_mode: str  # "free" or "onecall"

    # --- NWS ---
    nws_user_agent: str

    # --- Twilio ---
    twilio_account_sid: str
    twilio_auth_token: str
    twilio_from_number: str

    # --- SMS safety rails ---
    sms_dry_run: bool           # if True, SMS is logged instead of sent
    sms_max_per_run: int        # hard cap on real SMS per dispatch call

    # --- Runtime ---
    poll_interval_seconds: int
    log_level: str

    @property
    def twilio_configured(self) -> bool:
        return bool(
            self.twilio_account_sid
            and self.twilio_auth_token
            and self.twilio_from_number
            and self.twilio_account_sid != "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        )

    @property
    def owm_configured(self) -> bool:
        return bool(self.owm_api_key and self.owm_api_key != "your_openweathermap_api_key_here")


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Load configuration once per process and cache it."""
    load_dotenv(PROJECT_ROOT / ".env")

    return Config(
        owm_api_key=os.getenv("OWM_API_KEY", ""),
        owm_mode=os.getenv("OWM_MODE", "free").lower(),
        nws_user_agent=os.getenv(
            "NWS_USER_AGENT", "GuardianAgent/0.1 (unspecified@example.com)"
        ),
        twilio_account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
        twilio_auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
        twilio_from_number=os.getenv("TWILIO_FROM_NUMBER", ""),
        sms_dry_run=os.getenv("SMS_DRY_RUN", "true").strip().lower() in ("1", "true", "yes"),
        sms_max_per_run=int(os.getenv("SMS_MAX_PER_RUN", "10")),
        poll_interval_seconds=int(os.getenv("POLL_INTERVAL_SECONDS", "300")),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )
