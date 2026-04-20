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
    "guardian.profile_cli",
    "guardian.agent",
    "guardian.weather",
    "guardian.weather.http",
    "guardian.weather.nws",
    "guardian.weather.owm",
    "guardian.weather.observation",
    "guardian.weather.aggregator",
    "guardian.weather.demo",
    "guardian.risk",
    "guardian.risk.bayesian",
    "guardian.risk.classifier",
    "guardian.risk.features",
    "guardian.risk.risk_engine",
    "guardian.risk.data",
    "guardian.risk.data.storm_events",
    "guardian.risk.data.synthetic",
    "guardian.planning",
    "guardian.planning.planner",
    "guardian.planning.actions",
    "guardian.output",
    "guardian.output.sms",
    "guardian.output.smart_home",
    "guardian.output.console",
    "guardian.output.dispatch",
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
    """get_config() should work with no env vars set, producing unconfigured flags.

    Note: get_config() normally calls load_dotenv() on a real .env file. For
    this unit test we mock that out so the test is independent of whether
    the developer has a populated .env in the project root.
    """
    from guardian import config as config_module

    # Prevent the real .env file from being loaded during this test.
    monkeypatch.setattr(config_module, "load_dotenv", lambda *a, **k: None)

    for var in [
        "OWM_API_KEY",
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN",
        "TWILIO_FROM_NUMBER",
    ]:
        monkeypatch.delenv(var, raising=False)

    config_module.get_config.cache_clear()
    try:
        cfg = config_module.get_config()
        assert cfg.owm_configured is False
        assert cfg.twilio_configured is False
        assert cfg.poll_interval_seconds > 0
    finally:
        # Clear cache so later code sees the real config again.
        config_module.get_config.cache_clear()
