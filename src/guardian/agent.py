"""Guardian Agent — CLI entry point for the main loop.

Commands:

    python -m guardian.agent check --profile config/my_profile.json
    python -m guardian.agent run   --profile config/my_profile.json --once
    python -m guardian.agent run   --profile config/my_profile.json
    python -m guardian.agent run   --profile config/my_profile.json \\
                                   --model models/threat_classifier.joblib \\
                                   --interval 60 --max-cycles 10

The `run` command wires up every part of the system — profile loader, weather
aggregator (NWS + OWM), optional ML classifier, Bayesian risk engine, rule
planner, three-channel dispatcher — and drives the perceive-reason-act loop
until interrupted or `--max-cycles` is reached.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click
from rich.console import Console

from . import __version__
from .config import get_config
from .logging_setup import get_logger
from .loop import AgentState, run_cycle
from .output.dispatch import Dispatcher
from .profile import load_profile
from .risk.classifier import ThreatClassifier
from .risk.risk_engine import RiskEngine


log = get_logger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Shared option set
# ---------------------------------------------------------------------------

_profile_option = click.option(
    "--profile",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to user profile JSON.",
)
_model_option = click.option(
    "--model",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional path to a trained ThreatClassifier .joblib.",
)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(version=__version__, prog_name="Guardian Agent")
def cli() -> None:
    """Guardian Agent — personalized weather hazard safety assistant."""


@cli.command()
@click.option(
    "--profile",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional profile path to also validate and summarize.",
)
def check(profile: Path | None) -> None:
    """Verify environment and config; optionally load and show a profile."""
    cfg = get_config()
    console.rule("[bold cyan]Guardian Agent — environment check")
    console.print(f"Version:                [bold]{__version__}[/bold]")
    console.print(f"OpenWeatherMap:         configured={cfg.owm_configured}, "
                  f"mode={cfg.owm_mode}")
    console.print(f"NWS User-Agent:         {cfg.nws_user_agent}")
    console.print(f"Twilio:                 configured={cfg.twilio_configured}")
    console.print(f"SMS_DRY_RUN:            {cfg.sms_dry_run}")
    console.print(f"SMS_MAX_PER_RUN:        {cfg.sms_max_per_run}")
    console.print(f"Poll interval (s):      {cfg.poll_interval_seconds}")
    console.print(f"Log level:              {cfg.log_level}")

    if profile:
        try:
            p = load_profile(profile)
            console.rule("[bold cyan]Profile")
            console.print(f"User: [bold]{p.name}[/bold] ({p.user_id})")
            console.print(f"Location: {p.location.address}")
            console.print(f"  lat/lon: ({p.location.latitude}, {p.location.longitude})")
            console.print(f"  NWS zone: {p.location.nws_zone_id or '(unresolved)'}")
            console.print(f"  county FIPS: {p.location.county_fips or '(unset)'}")
            console.print(f"Home floor state: {p.home_floor_state.value}")
            console.print(f"Medically vulnerable: {p.is_medically_vulnerable}")
            console.print(f"Emergency contacts: {len(p.emergency_contacts)}")
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Failed to load profile:[/red] {e}")
            sys.exit(2)


@cli.command()
@_profile_option
@_model_option
@click.option("--once", is_flag=True, help="Run one cycle and exit.")
@click.option(
    "--interval",
    type=int,
    default=None,
    help="Seconds between cycles (default: POLL_INTERVAL_SECONDS from .env).",
)
@click.option(
    "--max-cycles",
    type=int,
    default=None,
    help="Stop after this many cycles (default: run until interrupted).",
)
@click.option(
    "--log-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/cycles.jsonl"),
    show_default=True,
    help="Where to append per-cycle JSONL records.",
)
@click.option(
    "--no-owm",
    is_flag=True,
    help="Skip OpenWeatherMap (useful if key isn't activated yet).",
)
def run(
    profile: Path,
    model: Path | None,
    once: bool,
    interval: int | None,
    max_cycles: int | None,
    log_file: Path,
    no_owm: bool,
) -> None:
    """Run the agent loop: perceive → reason → act, cycle after cycle."""
    cfg = get_config()

    # Load everything up front so failures surface before we start looping.
    user_profile = load_profile(profile)

    classifier = None
    if model is not None:
        try:
            classifier = ThreatClassifier.load(model)
            console.print(f"[green]Loaded classifier:[/green] {model}")
            if classifier.metrics:
                console.print(
                    f"  (test ROC-AUC = {classifier.metrics.roc_auc:.3f}, "
                    f"trained on {classifier.metrics.n_train:,} cells)"
                )
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Failed to load classifier:[/red] {e}")
            sys.exit(2)
    else:
        console.print("[dim]No classifier; EmergingThreat evidence stays Low.[/dim]")

    engine = RiskEngine(classifier=classifier)
    dispatcher = Dispatcher.default()
    state = AgentState()

    poll_s = interval if interval is not None else cfg.poll_interval_seconds

    console.rule(
        f"[bold green]Guardian Agent v{__version__} starting for "
        f"{user_profile.name}"
    )
    console.print(f"Poll interval: {poll_s}s   SMS mode: {dispatcher.sms.mode_summary}")
    console.print(f"Logging cycles to: {log_file}")
    console.print()

    cycle_n = 0
    try:
        while True:
            cycle_n += 1
            console.rule(f"[bold]Cycle #{cycle_n}")
            report = run_cycle(
                profile=user_profile,
                engine=engine,
                dispatcher=dispatcher,
                state=state,
                cycle_log_path=log_file,
                skip_owm=no_owm,
            )
            _print_cycle_summary(report)

            if once or (max_cycles is not None and cycle_n >= max_cycles):
                break

            console.print(f"[dim]Sleeping {poll_s}s until next cycle...[/dim]\n")
            time.sleep(poll_s)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user. Exiting cleanly.[/yellow]")


def _print_cycle_summary(report) -> None:  # type: ignore[no-untyped-def]
    if report.error:
        console.print(f"[red]Error:[/red] {report.error}")
        return
    dedup = "yes" if report.fingerprint_matched_previous else "no"
    console.print(
        f"Sources: {', '.join(report.observation_sources) or '(none)'}   "
        f"Active alerts: {report.has_active_alerts}"
    )
    for state, p in report.risk_posterior.items():
        bar = "█" * int(round(p * 20))
        console.print(f"  {state:>10s}  {p:6.1%}  {bar}")
    console.print(
        f"[bold]Argmax:[/bold] {report.risk_argmax}   "
        f"Planned: {report.actions_planned}   "
        f"Dispatched: {report.actions_dispatched}   "
        f"Suppressed by dedup: {report.actions_suppressed_by_dedup}   "
        f"Fingerprint matched previous: {dedup}"
    )
    console.print(f"[dim]{report.dispatch_summary}[/dim]")


if __name__ == "__main__":
    cli()
