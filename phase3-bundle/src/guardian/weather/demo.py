"""Live demo: fetch weather for a profile and pretty-print the observation.

Usage:
    python -m guardian.weather.demo --profile config/my_profile.json

This script hits the real NWS and OWM APIs, so it:
  - requires a working internet connection
  - requires OWM_API_KEY in .env (unless you pass --no-owm)
  - requires the OWM key to have been activated (can take ~2 hours after signup)

Output shows raw numerics, the derived Bayesian categories, and any active
alerts. Use this after Phase 3 to sanity-check that perception is working
before moving on to Phase 4.
"""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from ..profile import load_profile
from .aggregator import observe


console = Console()


@click.command()
@click.option(
    "--profile",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to a user profile JSON (used for lat/lon).",
)
@click.option(
    "--no-owm",
    is_flag=True,
    help="Skip OWM (useful if your key isn't activated yet).",
)
def main(profile: Path, no_owm: bool) -> None:
    """Fetch a live WeatherObservation and print it."""
    p = load_profile(profile)
    console.rule(f"[bold cyan]Observing for {p.name} at {p.location.address}")
    if no_owm:
        console.print("[yellow]--no-owm set; using NWS only.[/yellow]")

    obs = observe(
        latitude=p.location.latitude,
        longitude=p.location.longitude,
        nws_zone_id=p.location.nws_zone_id,
        skip_owm=no_owm,
    )

    # --- Scalars table ---
    t = Table(title="Current conditions", show_header=False)
    t.add_column("Field", style="cyan", no_wrap=True)
    t.add_column("Value")
    t.add_row("Observed at (UTC)", str(obs.observed_at))
    t.add_row("Sources", ", ".join(obs.sources) or "(none)")
    t.add_row("NWS zone", obs.nws_zone_id or "(unresolved)")
    t.add_row("Temperature (F)", _fmt(obs.temperature_f))
    t.add_row("Wind speed (mph)", _fmt(obs.wind_speed_mph))
    t.add_row("Wind gust (mph)", _fmt(obs.wind_gust_mph))
    t.add_row("Precip rate (in/hr)", _fmt(obs.precip_rate_in_hr))
    t.add_row("Humidity (%)", _fmt(obs.humidity_pct))
    t.add_row("Pressure (mb)", _fmt(obs.pressure_mb))
    t.add_row("Visibility (mi)", _fmt(obs.visibility_mi))
    console.print(t)

    # --- Derived categories ---
    d = Table(title="Derived Bayesian categories", show_header=False)
    d.add_column("Field", style="cyan")
    d.add_column("Category")
    d.add_row("Wind category", obs.wind_category.value)
    d.add_row("Precip category", obs.precip_category.value)
    d.add_row("Max alert severity", obs.max_severity.value)
    d.add_row("Max alert urgency", obs.max_urgency.value)
    console.print(d)

    # --- Alerts ---
    if obs.alerts:
        a = Table(title=f"Active alerts ({len(obs.alerts)})")
        a.add_column("Event")
        a.add_column("Severity")
        a.add_column("Urgency")
        a.add_column("Expires")
        for alert in obs.alerts:
            a.add_row(
                alert.event,
                alert.severity.value,
                alert.urgency.value,
                str(alert.expires) if alert.expires else "-",
            )
        console.print(a)
    else:
        console.print("[green]No active weather alerts for this location.[/green]")


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


if __name__ == "__main__":
    main()
