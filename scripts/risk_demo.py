"""Risk engine scenario walkthrough.

A canned set of four scenarios that demonstrate Guardian Agent's Bayesian
reasoning. For each, we synthesize a WeatherObservation, run it through the
RiskEngine for two contrasting user profiles, and print the resulting
posterior side-by-side.

The scenarios are:

  1. Clear day, no alerts — sanity check that no-evidence stays Low.
  2. Heavy rain + flash flood watch — moderate severity, urgency Expected.
  3. Tornado warning — Extreme severity + Immediate urgency.
  4. Tropical storm — severe wind + heavy precip.

For each, "User A" is the baseline (upper-floor apartment, SUV, no medical
issues) and "User B" is the highly vulnerable case (ground-floor apartment,
no vehicle, mobility-limited). Watch the posterior diverge.

Usage:
    python scripts/risk_demo.py
    python scripts/risk_demo.py --scenario tornado
    python scripts/risk_demo.py --model models/threat_classifier.joblib
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

# Allow running directly without `pip install -e .`
HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from guardian.profile import (  # noqa: E402
    EmergencyContact,
    Home,
    HomeType,
    Location,
    Medical,
    UserProfile,
    Vehicle,
    VehicleClearance,
)
from guardian.risk.classifier import ThreatClassifier  # noqa: E402
from guardian.risk.risk_engine import RiskEngine  # noqa: E402
from guardian.weather.observation import (  # noqa: E402
    CertaintyLevel,
    SeverityLevel,
    UrgencyLevel,
    WeatherAlert,
    WeatherObservation,
)


console = Console()


# ---------------------------------------------------------------------------
# Scenario synthesis
# ---------------------------------------------------------------------------

def _now_utc() -> datetime:
    return datetime(2026, 7, 15, 18, 0, 0, tzinfo=timezone.utc)


def scenario_clear() -> WeatherObservation:
    return WeatherObservation(
        observed_at=_now_utc(),
        latitude=30.4133, longitude=-91.18,
        temperature_f=82.0, wind_speed_mph=6.0,
        humidity_pct=55.0, precip_rate_in_hr=0.0,
        sources=["nws", "owm"],
        nws_zone_id="LAZ036",
        alerts=[],
    )


def scenario_heavy_rain() -> WeatherObservation:
    return WeatherObservation(
        observed_at=_now_utc(),
        latitude=30.4133, longitude=-91.18,
        temperature_f=74.0, wind_speed_mph=15.0,
        humidity_pct=92.0, precip_rate_in_hr=0.45,
        sources=["nws", "owm"],
        nws_zone_id="LAZ036",
        alerts=[
            WeatherAlert(
                source="nws",
                event="Flash Flood Watch",
                headline="Flash Flood Watch in effect through 11 PM CDT",
                severity=SeverityLevel.MODERATE,
                urgency=UrgencyLevel.EXPECTED,
                certainty=CertaintyLevel.LIKELY,
            ),
        ],
    )


def scenario_tornado() -> WeatherObservation:
    return WeatherObservation(
        observed_at=_now_utc(),
        latitude=30.4133, longitude=-91.18,
        temperature_f=78.0, wind_speed_mph=42.0, wind_gust_mph=68.0,
        humidity_pct=85.0, precip_rate_in_hr=0.6,
        sources=["nws", "owm"],
        nws_zone_id="LAZ036",
        alerts=[
            WeatherAlert(
                source="nws",
                event="Tornado Warning",
                headline="Tornado Warning for East Baton Rouge Parish until 6:45 PM",
                severity=SeverityLevel.EXTREME,
                urgency=UrgencyLevel.IMMEDIATE,
                certainty=CertaintyLevel.OBSERVED,
            ),
        ],
    )


def scenario_tropical() -> WeatherObservation:
    return WeatherObservation(
        observed_at=_now_utc(),
        latitude=30.4133, longitude=-91.18,
        temperature_f=80.0, wind_speed_mph=58.0, wind_gust_mph=85.0,
        humidity_pct=95.0, precip_rate_in_hr=1.2,
        sources=["nws", "owm"],
        nws_zone_id="LAZ036",
        alerts=[
            WeatherAlert(
                source="nws",
                event="Tropical Storm Warning",
                severity=SeverityLevel.SEVERE,
                urgency=UrgencyLevel.EXPECTED,
                certainty=CertaintyLevel.LIKELY,
            ),
        ],
    )


SCENARIOS = {
    "clear":        ("Clear day, no advisories",        scenario_clear),
    "heavy_rain":   ("Heavy rain + flash flood watch",  scenario_heavy_rain),
    "tornado":      ("Tornado warning",                  scenario_tornado),
    "tropical":     ("Tropical storm warning",           scenario_tropical),
}


# ---------------------------------------------------------------------------
# Synthetic users
# ---------------------------------------------------------------------------

def _make_location() -> Location:
    return Location(
        address="100 Demo St, Baton Rouge, LA 70803",
        latitude=30.4133, longitude=-91.18,
        nws_zone_id="LAZ036", county_fips="22033",
    )


def user_baseline() -> UserProfile:
    """Baseline user: 3rd-floor apartment, SUV, no medical issues."""
    return UserProfile(
        user_id="baseline",
        name="Alex (baseline)",
        location=_make_location(),
        home=Home(type=HomeType.APARTMENT, floor_level=3, elevated=False),
        vehicle=Vehicle(owns_vehicle=True, clearance=VehicleClearance.HIGH,
                        four_wheel_drive=True),
        medical=Medical(),
        emergency_contacts=[
            EmergencyContact(name="Family", relationship="parent",
                             phone="+15555550100"),
        ],
    )


def user_vulnerable() -> UserProfile:
    """Vulnerable user: ground-floor apartment, no vehicle, mobility-limited."""
    return UserProfile(
        user_id="vulnerable",
        name="Sam (vulnerable)",
        location=_make_location(),
        home=Home(type=HomeType.APARTMENT, floor_level=1, elevated=False,
                  flood_zone="AE"),
        vehicle=Vehicle(owns_vehicle=False, clearance=VehicleClearance.NONE),
        medical=Medical(mobility_limited=True),
        emergency_contacts=[
            EmergencyContact(name="Family", relationship="sibling",
                             phone="+15555550101"),
        ],
    )


# ---------------------------------------------------------------------------
# Pretty-print one scenario
# ---------------------------------------------------------------------------

def _render_one(engine: RiskEngine, label: str, observation: WeatherObservation) -> None:
    console.rule(f"[bold cyan]{label}")
    if observation.alerts:
        for a in observation.alerts:
            console.print(
                f"  [yellow]Active:[/yellow] {a.event}  "
                f"({a.severity.value} / {a.urgency.value})"
            )
    else:
        console.print("  [green]No active alerts.[/green]")
    console.print(
        f"  Wind: {observation.wind_category.value}, "
        f"Precip: {observation.precip_category.value}, "
        f"MaxSeverity: {observation.max_severity.value}, "
        f"Urgency: {observation.max_urgency.value}"
    )
    console.print()

    baseline = user_baseline()
    vulnerable = user_vulnerable()

    r_a = engine.assess(observation, baseline)
    r_b = engine.assess(observation, vulnerable)

    table = Table(show_header=True, header_style="bold")
    table.add_column("RiskLevel")
    table.add_column(f"User A: {baseline.name}", justify="right")
    table.add_column(f"User B: {vulnerable.name}", justify="right")

    for state in r_a.assessment.posterior:
        pa = r_a.assessment.posterior[state]
        pb = r_b.assessment.posterior[state]
        bar_a = "█" * int(round(pa * 20))
        bar_b = "█" * int(round(pb * 20))
        table.add_row(
            state,
            f"{pa:6.1%}  [cyan]{bar_a}[/cyan]",
            f"{pb:6.1%}  [magenta]{bar_b}[/magenta]",
        )
    console.print(table)
    console.print(
        f"  Argmax — User A: [bold]{r_a.assessment.argmax}[/bold] "
        f"({r_a.assessment.argmax_prob:.1%})    "
        f"User B: [bold]{r_b.assessment.argmax}[/bold] "
        f"({r_b.assessment.argmax_prob:.1%})"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--scenario",
    type=click.Choice(list(SCENARIOS) + ["all"]),
    default="all",
    help="Which scenario to run.",
)
@click.option(
    "--model",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional path to a trained ThreatClassifier .joblib.",
)
def main(scenario: str, model: Path | None) -> None:
    """Walk through Bayesian risk scenarios with two contrasting users."""
    console.print("[bold]Guardian Agent — Bayesian Risk Engine demo[/bold]\n")

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
        except Exception as e:
            console.print(f"[red]Failed to load classifier:[/red] {e}")
    else:
        console.print(
            "[dim]No classifier provided; EmergingThreat evidence stays Low. "
            "Pass --model to load a trained joblib.[/dim]"
        )

    engine = RiskEngine(classifier=classifier)
    console.print()

    selected = list(SCENARIOS.items()) if scenario == "all" else [
        (scenario, SCENARIOS[scenario])
    ]
    for _key, (label, factory) in selected:
        _render_one(engine, label, factory())
        console.print()


if __name__ == "__main__":
    main()
