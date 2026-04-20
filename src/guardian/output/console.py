"""Console output dispatcher — the user-facing alert surface in the prototype.

In a deployed version, this would be the "notify user" channel — push
notification on mobile, widget on a smart display, speaker announcement.
For the prototype, we render tailored alerts to the terminal using rich.

This dispatcher also handles LOG_ONLY actions, which exist so the planner can
express "there's nothing urgent; just record this decision" without needing
a no-op special case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from rich.console import Console
from rich.panel import Panel

from ..logging_setup import get_logger
from ..planning.actions import Action, ActionChannel, ActionKind


log = get_logger(__name__)
_console = Console()


# Severity styling per action kind. Higher-urgency actions get louder styling.
_STYLES: dict[ActionKind, str] = {
    ActionKind.RECOMMEND_EVACUATE: "bold red on black",
    ActionKind.RECOMMEND_SHELTER:  "bold yellow",
    ActionKind.NOTIFY_USER:        "bold cyan",
    ActionKind.LOG_ONLY:           "dim",
}


@dataclass(frozen=True)
class ConsoleResult:
    kind: ActionKind
    dispatched_at: datetime
    message: str


@dataclass(frozen=True)
class ConsoleDispatchReport:
    attempted: int
    results: tuple[ConsoleResult, ...] = field(default_factory=tuple)


class ConsoleDispatcher:
    """Prints user-facing alerts. Stateless."""

    def _dispatch_one(self, action: Action) -> ConsoleResult:
        style = _STYLES.get(action.kind, "")
        title = action.kind.value.replace("_", " ").upper()
        panel = Panel(action.message, title=title, style=style, border_style=style)
        _console.print(panel)
        return ConsoleResult(
            kind=action.kind,
            dispatched_at=datetime.now(timezone.utc),
            message=action.message,
        )

    def dispatch(self, actions: list[Action]) -> ConsoleDispatchReport:
        cons_actions = [a for a in actions if a.channel is ActionChannel.CONSOLE]
        results = [self._dispatch_one(a) for a in cons_actions]
        return ConsoleDispatchReport(
            attempted=len(results),
            results=tuple(results),
        )


__all__ = ["ConsoleDispatcher", "ConsoleDispatchReport", "ConsoleResult"]
