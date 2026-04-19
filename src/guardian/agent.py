"""Main Guardian Agent — perceive, reason, act.

Implemented in **Phase 8**. For now this module exposes a Click-based CLI
entry point that verifies the environment is wired up correctly. Run:

    python -m guardian.agent --check

Later phases wire in the perceive-reason-act loop:

    1. PERCEIVE: poll NWS + OWM for current conditions and alerts.
    2. REASON:   run the ML classifier, set Bayesian network evidence,
                 query RiskLevel posterior.
    3. ACT:      planner selects actions; dispatchers execute them.
    4. SLEEP:    wait POLL_INTERVAL_SECONDS, repeat.
"""

from __future__ import annotations

from pathlib import Path

import click

from . import __version__
from .config import get_config
from .logging_setup import get_logger


log = get_logger(__name__)


@click.command()
@click.option(
    "--profile",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to user profile JSON.",
)
@click.option("--once", is_flag=True, help="Run one perceive-reason-act cycle and exit.")
@click.option("--check", is_flag=True, help="Verify environment + config and exit.")
def main(profile: Path | None, once: bool, check: bool) -> None:
    """Guardian Agent command-line entry point."""
    log.info("Guardian Agent v%s starting.", __version__)
    cfg = get_config()

    if check:
        log.info("OpenWeatherMap configured: %s", cfg.owm_configured)
        log.info("OpenWeatherMap mode:       %s", cfg.owm_mode)
        log.info("Twilio configured:         %s", cfg.twilio_configured)
        log.info("NWS User-Agent:            %s", cfg.nws_user_agent)
        log.info("Poll interval (s):         %s", cfg.poll_interval_seconds)
        log.info("Log level:                 %s", cfg.log_level)
        if profile:
            log.info("Profile path:              %s", profile)
        return

    if not profile:
        raise click.UsageError("--profile is required unless --check is set.")

    log.warning("Main agent loop not yet implemented (lands in Phase 8).")


if __name__ == "__main__":
    main()
