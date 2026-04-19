"""Smoke tests: verify the project scaffolding is intact.

These should pass immediately after `pip install -e .` on a fresh clone.
"""

from __future__ import annotations

import importlib

import pytest


PACKAGES = [
    "guardian",
    "guardian.config",
    "guardian.logging_setup",
    "guardian.profile",
    "guardian.agent",
    "guardian.weather",
    "guardian.weather.nws",
    "guardian.weather.owm",
    "guardian.weather.observation",
    "guardian.risk",
    "guardian.risk.bayesian",
    "guardian.risk.classifier",
    "guardian.planning",
    "guardian.planning.planner",
    "guardian.output",
    "guardian.output.sms",
    "guardian.output.smart_home",
]


@pytest.mark.parametrize("module_name", PACKAGES)
def test_module_importable(module_name: str) -> None:
    """Every module in the package should import cleanly."""
    importlib.import_module(module_name)


def test_version_exposed() -> None:
    import guardian

    assert isinstance(guardian.__version__, str)
    assert guardian.__version__.count(".") == 2


def test_config_loads_with_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_config() should work even without a .env file, using defaults."""
    # Clear any cached config and environment so we test true defaults.
    from guardian import config as config_module

    config_module.get_config.cache_clear()
    for var in [
        "OWM_API_KEY",
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN",
        "TWILIO_FROM_NUMBER",
    ]:
        monkeypatch.delenv(var, raising=False)

    cfg = config_module.get_config()
    assert cfg.owm_configured is False
    assert cfg.twilio_configured is False
    assert cfg.poll_interval_seconds > 0
