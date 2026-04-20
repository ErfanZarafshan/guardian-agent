"""Dispatch coordinator: route Actions to the right dispatcher.

`dispatch_actions(actions)` is the single entry point the agent loop uses.
It routes each Action to the dispatcher matching its channel, aggregates
results, and returns a DispatchReport with per-channel reports. Failure of
one dispatcher does NOT abort the others.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..logging_setup import get_logger
from ..planning.actions import Action
from .console import ConsoleDispatcher, ConsoleDispatchReport
from .sms import SMSDispatcher, SMSDispatchReport
from .smart_home import SmartHomeDispatcher, SmartHomeDispatchReport


log = get_logger(__name__)


@dataclass(frozen=True)
class DispatchReport:
    """Aggregated report across all channels."""
    console: ConsoleDispatchReport
    sms: SMSDispatchReport
    smart_home: SmartHomeDispatchReport

    @property
    def ok(self) -> bool:
        return self.sms.ok  # only SMS can fail in the current implementation

    def summary(self) -> str:
        return (
            f"Dispatched: console={self.console.attempted}, "
            f"sms={self.sms.attempted} "
            f"(sent={self.sms.sent}, dry_run={self.sms.dry_run}, "
            f"failed={self.sms.failed}), "
            f"smart_home={self.smart_home.attempted} "
            f"(mocked={self.smart_home.mocked})"
        )


@dataclass
class Dispatcher:
    """Container for the three per-channel dispatchers.

    In tests, you can swap any dispatcher for a mock by constructing this
    directly. The default constructor wires up real dispatchers that read
    config from .env.
    """
    console: ConsoleDispatcher
    sms: SMSDispatcher
    smart_home: SmartHomeDispatcher

    @classmethod
    def default(cls) -> "Dispatcher":
        return cls(
            console=ConsoleDispatcher(),
            sms=SMSDispatcher(),
            smart_home=SmartHomeDispatcher(),
        )

    def dispatch(self, actions: list[Action]) -> DispatchReport:
        return DispatchReport(
            console=self.console.dispatch(actions),
            sms=self.sms.dispatch(actions),
            smart_home=self.smart_home.dispatch(actions),
        )


def dispatch_actions(actions: list[Action]) -> DispatchReport:
    """Convenience: build a default Dispatcher and dispatch `actions`."""
    return Dispatcher.default().dispatch(actions)


__all__ = [
    "Dispatcher",
    "DispatchReport",
    "dispatch_actions",
]
