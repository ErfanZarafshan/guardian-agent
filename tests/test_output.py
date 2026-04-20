"""Tests for the output layer: SMS, smart-home, console, and coordinator."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from guardian.output.console import ConsoleDispatcher
from guardian.output.dispatch import Dispatcher, dispatch_actions
from guardian.output.smart_home import SmartHomeDispatcher
from guardian.output.sms import SMSDispatcher
from guardian.planning.actions import Action, ActionChannel, ActionKind


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sms_action(
    recipients: tuple[str, ...] = ("+15555550100",),
    message: str = "test message",
) -> Action:
    return Action(
        kind=ActionKind.NOTIFY_CONTACTS,
        channel=ActionChannel.SMS,
        message=message,
        rationale="test",
        recipients=recipients,
    )


def _console_action(kind: ActionKind = ActionKind.NOTIFY_USER) -> Action:
    return Action(
        kind=kind, channel=ActionChannel.CONSOLE,
        message="console msg", rationale="test",
    )


def _smart_home_action(
    kind: ActionKind = ActionKind.ACTIVATE_FLOOD_LIGHTS,
    metadata: dict | None = None,
) -> Action:
    return Action(
        kind=kind, channel=ActionChannel.SMART_HOME,
        message="smart home msg", rationale="test",
        metadata=metadata or {},
    )


def _force_config(monkeypatch, **overrides) -> None:
    """Force a fresh config load with test-supplied environment variables."""
    from guardian import config as config_module

    monkeypatch.setattr(config_module, "load_dotenv", lambda *a, **k: None)
    defaults = {
        "OWM_API_KEY": "",
        "TWILIO_ACCOUNT_SID": "",
        "TWILIO_AUTH_TOKEN": "",
        "TWILIO_FROM_NUMBER": "",
        "SMS_DRY_RUN": "true",
        "SMS_MAX_PER_RUN": "10",
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        monkeypatch.setenv(k, str(v))
    config_module.get_config.cache_clear()


@pytest.fixture(autouse=True)
def _reset_config_cache():
    """Ensure the config cache is reset before AND after each test."""
    from guardian import config as config_module
    config_module.get_config.cache_clear()
    yield
    config_module.get_config.cache_clear()


# ---------------------------------------------------------------------------
# SMS dispatcher
# ---------------------------------------------------------------------------

def test_sms_no_actions_returns_empty_report() -> None:
    d = SMSDispatcher(client=MagicMock())  # client irrelevant since no sms actions
    report = d.dispatch([_console_action(), _smart_home_action()])
    assert report.attempted == 0
    assert report.sent == 0


def test_sms_dry_run_when_flag_set(monkeypatch) -> None:
    _force_config(
        monkeypatch,
        TWILIO_ACCOUNT_SID="ACrealsid",
        TWILIO_AUTH_TOKEN="realtoken",
        TWILIO_FROM_NUMBER="+15551111111",
        SMS_DRY_RUN="true",
    )
    client = MagicMock()
    d = SMSDispatcher(client=client)
    report = d.dispatch([_sms_action()])
    assert report.attempted == 1
    assert report.dry_run == 1
    assert report.sent == 0
    # Client must NOT be called in dry-run mode.
    client.messages.create.assert_not_called()


def test_sms_dry_run_when_unconfigured(monkeypatch) -> None:
    _force_config(monkeypatch, SMS_DRY_RUN="false")  # no Twilio creds
    client = MagicMock()
    d = SMSDispatcher(client=client)
    report = d.dispatch([_sms_action()])
    # Should fall back to dry-run because credentials aren't configured.
    assert report.dry_run == 1
    assert report.sent == 0
    client.messages.create.assert_not_called()


def test_sms_live_send_when_configured_and_not_dry_run(monkeypatch) -> None:
    _force_config(
        monkeypatch,
        TWILIO_ACCOUNT_SID="ACrealsid",
        TWILIO_AUTH_TOKEN="realtoken",
        TWILIO_FROM_NUMBER="+15551111111",
        SMS_DRY_RUN="false",
    )
    client = MagicMock()
    client.messages.create.return_value = MagicMock(sid="SM_fake_sid")
    d = SMSDispatcher(client=client)
    report = d.dispatch([_sms_action(recipients=("+15552223333",),
                                     message="hello")])
    assert report.sent == 1
    assert report.dry_run == 0
    client.messages.create.assert_called_once()
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["to"] == "+15552223333"
    assert kwargs["from_"] == "+15551111111"
    assert kwargs["body"] == "hello"


def test_sms_failure_is_recorded_not_raised(monkeypatch) -> None:
    _force_config(
        monkeypatch,
        TWILIO_ACCOUNT_SID="ACrealsid",
        TWILIO_AUTH_TOKEN="realtoken",
        TWILIO_FROM_NUMBER="+15551111111",
        SMS_DRY_RUN="false",
    )
    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("twilio 401")
    d = SMSDispatcher(client=client)
    report = d.dispatch([_sms_action()])
    assert report.failed == 1
    assert report.ok is False
    assert "twilio 401" in report.results[0].error


def test_sms_dedupes_identical_pairs(monkeypatch) -> None:
    _force_config(
        monkeypatch,
        TWILIO_ACCOUNT_SID="ACrealsid",
        TWILIO_AUTH_TOKEN="realtoken",
        TWILIO_FROM_NUMBER="+15551111111",
        SMS_DRY_RUN="false",
    )
    client = MagicMock()
    client.messages.create.return_value = MagicMock(sid="SM_fake")
    d = SMSDispatcher(client=client)
    # Two identical actions should result in one send.
    report = d.dispatch([_sms_action(), _sms_action()])
    assert report.sent == 1


def test_sms_respects_max_per_run_cap(monkeypatch) -> None:
    _force_config(
        monkeypatch,
        TWILIO_ACCOUNT_SID="ACrealsid",
        TWILIO_AUTH_TOKEN="realtoken",
        TWILIO_FROM_NUMBER="+15551111111",
        SMS_DRY_RUN="false",
        SMS_MAX_PER_RUN="2",
    )
    client = MagicMock()
    client.messages.create.return_value = MagicMock(sid="SM_fake")
    d = SMSDispatcher(client=client)
    # 4 distinct recipients, cap 2 -> 2 sent, 2 skipped.
    actions = [
        _sms_action(recipients=("+15550000001",)),
        _sms_action(recipients=("+15550000002",)),
        _sms_action(recipients=("+15550000003",)),
        _sms_action(recipients=("+15550000004",)),
    ]
    report = d.dispatch(actions)
    assert report.sent == 2
    assert report.skipped == 2


def test_sms_fans_out_to_multiple_recipients(monkeypatch) -> None:
    _force_config(monkeypatch)  # dry-run default
    d = SMSDispatcher(client=MagicMock())
    action = _sms_action(recipients=("+15550001111", "+15550002222"))
    report = d.dispatch([action])
    assert report.attempted == 2


def test_sms_mode_summary_strings(monkeypatch) -> None:
    _force_config(monkeypatch, SMS_DRY_RUN="true")
    assert "DRY-RUN" in SMSDispatcher(client=MagicMock()).mode_summary

    _force_config(monkeypatch, SMS_DRY_RUN="false")
    assert "UNCONFIGURED" in SMSDispatcher(client=MagicMock()).mode_summary

    _force_config(
        monkeypatch, SMS_DRY_RUN="false",
        TWILIO_ACCOUNT_SID="ACreal", TWILIO_AUTH_TOKEN="tok",
        TWILIO_FROM_NUMBER="+15551111111",
    )
    assert "LIVE" in SMSDispatcher(client=MagicMock()).mode_summary


# ---------------------------------------------------------------------------
# Smart-home dispatcher
# ---------------------------------------------------------------------------

def test_smart_home_mocks_flood_lights() -> None:
    d = SmartHomeDispatcher()
    report = d.dispatch([_smart_home_action(ActionKind.ACTIVATE_FLOOD_LIGHTS)])
    assert report.attempted == 1
    assert report.mocked == 1
    assert "flood lights" in report.results[0].detail


def test_smart_home_mocks_alarm() -> None:
    d = SmartHomeDispatcher()
    report = d.dispatch([_smart_home_action(ActionKind.SOUND_ALARM)])
    assert report.mocked == 1
    assert "alarm" in report.results[0].detail.lower()


def test_smart_home_thermostat_includes_setpoint() -> None:
    d = SmartHomeDispatcher()
    action = _smart_home_action(
        ActionKind.ADJUST_THERMOSTAT, metadata={"setpoint_f": 72},
    )
    report = d.dispatch([action])
    assert "72" in report.results[0].detail


def test_smart_home_ignores_non_smart_home_actions() -> None:
    d = SmartHomeDispatcher()
    report = d.dispatch([_sms_action(), _console_action()])
    assert report.attempted == 0


def test_smart_home_unsupported_kind_is_recorded_not_raised() -> None:
    d = SmartHomeDispatcher()
    weird = Action(
        kind=ActionKind.NOTIFY_USER,  # wrong kind for this channel
        channel=ActionChannel.SMART_HOME,
        message="x", rationale="test",
    )
    report = d.dispatch([weird])
    assert report.unsupported == 1
    assert report.mocked == 0


# ---------------------------------------------------------------------------
# Console dispatcher
# ---------------------------------------------------------------------------

def test_console_dispatches_only_console_channel_actions() -> None:
    d = ConsoleDispatcher()
    report = d.dispatch([
        _console_action(ActionKind.LOG_ONLY),
        _console_action(ActionKind.NOTIFY_USER),
        _sms_action(),
        _smart_home_action(),
    ])
    assert report.attempted == 2


def test_console_results_preserve_message() -> None:
    d = ConsoleDispatcher()
    report = d.dispatch([Action(
        kind=ActionKind.NOTIFY_USER, channel=ActionChannel.CONSOLE,
        message="hello world", rationale="x",
    )])
    assert report.results[0].message == "hello world"


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

def test_coordinator_routes_each_channel(monkeypatch) -> None:
    _force_config(monkeypatch)  # dry-run default
    d = Dispatcher(
        console=ConsoleDispatcher(),
        sms=SMSDispatcher(client=MagicMock()),
        smart_home=SmartHomeDispatcher(),
    )
    actions = [
        _console_action(ActionKind.NOTIFY_USER),
        _sms_action(),
        _smart_home_action(),
    ]
    report = d.dispatch(actions)
    assert report.console.attempted == 1
    assert report.sms.attempted == 1
    assert report.smart_home.attempted == 1


def test_coordinator_one_channel_failure_does_not_abort_others(monkeypatch) -> None:
    """An SMS failure should not prevent smart-home or console dispatch."""
    _force_config(
        monkeypatch,
        TWILIO_ACCOUNT_SID="ACreal", TWILIO_AUTH_TOKEN="tok",
        TWILIO_FROM_NUMBER="+15551111111", SMS_DRY_RUN="false",
    )
    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("twilio down")
    d = Dispatcher(
        console=ConsoleDispatcher(),
        sms=SMSDispatcher(client=client),
        smart_home=SmartHomeDispatcher(),
    )
    actions = [
        _console_action(ActionKind.NOTIFY_USER),
        _sms_action(),
        _smart_home_action(ActionKind.SOUND_ALARM),
    ]
    report = d.dispatch(actions)
    assert report.sms.failed == 1
    assert report.console.attempted == 1
    assert report.smart_home.mocked == 1
    assert report.ok is False


def test_coordinator_summary_string(monkeypatch) -> None:
    _force_config(monkeypatch)
    report = dispatch_actions([
        _console_action(),
        _sms_action(),
        _smart_home_action(),
    ])
    summary = report.summary()
    assert "console=1" in summary
    assert "sms=1" in summary
    assert "smart_home=1" in summary
