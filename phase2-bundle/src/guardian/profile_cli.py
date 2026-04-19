"""Interactive CLI for creating, inspecting, and validating user profiles.

Run:

    python -m guardian.profile_cli create --output config/my_profile.json
    python -m guardian.profile_cli show config/my_profile.json
    python -m guardian.profile_cli validate config/my_profile.json

The `create` command walks the user through every field with sensible defaults
so they never have to hand-write JSON. Output is the same format that
`load_profile` consumes.
"""

from __future__ import annotations

import sys
from datetime import time
from pathlib import Path

import click
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from .profile import (
    EmergencyContact,
    Home,
    HomeType,
    Location,
    Medical,
    Preferences,
    QuietHours,
    RiskNotifyLevel,
    UserProfile,
    Vehicle,
    VehicleClearance,
    load_profile,
    save_profile,
)


console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prompt_enum(label: str, enum_cls: type, default):  # type: ignore[no-untyped-def]
    """Prompt for an enum value, showing available choices."""
    choices = [e.value for e in enum_cls]
    return click.prompt(
        label,
        type=click.Choice(choices, case_sensitive=False),
        default=default.value if hasattr(default, "value") else default,
        show_choices=True,
    )


def _prompt_multi_enum(label: str, enum_cls: type, default_list: list):  # type: ignore[no-untyped-def]
    """Prompt for a comma-separated set of enum values."""
    choices_str = ", ".join(e.value for e in enum_cls)
    default_str = ",".join(v.value for v in default_list)
    raw = click.prompt(
        f"{label} (comma-separated from: {choices_str})",
        default=default_str,
    )
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    return [enum_cls(t) for t in tokens]


def _print_profile_table(profile: UserProfile) -> None:
    table = Table(title=f"Profile: {profile.name} ({profile.user_id})", show_lines=False)
    table.add_column("Section", style="cyan", no_wrap=True)
    table.add_column("Value")

    loc = profile.location
    table.add_row("Location", f"{loc.address}\n({loc.latitude:.4f}, {loc.longitude:.4f})")

    h = profile.home
    table.add_row(
        "Home",
        f"type={h.type.value}, floor={h.floor_level}, elevated={h.elevated}, "
        f"flood_zone={h.flood_zone or '-'}",
    )
    table.add_row("Home (derived)", f"BayesianEvidence = {profile.home_floor_state.value}")

    v = profile.vehicle
    table.add_row(
        "Vehicle",
        f"owns={v.owns_vehicle}, clearance={v.clearance.value}, 4WD={v.four_wheel_drive}",
    )

    m = profile.medical
    table.add_row(
        "Medical",
        f"mobility_limited={m.mobility_limited}, oxygen={m.oxygen_dependent}, "
        f"refrig_meds={m.refrigerated_medication}, conditions={m.chronic_conditions or 'none'}",
    )
    table.add_row("Medical (derived)", f"vulnerable = {profile.is_medically_vulnerable}")

    if profile.emergency_contacts:
        contacts = "\n".join(
            f"- {c.name} ({c.relationship}) {c.phone} notify={[n.value for n in c.notify_on]}"
            for c in profile.emergency_contacts
        )
    else:
        contacts = "(none)"
    table.add_row("Contacts", contacts)

    p = profile.preferences
    quiet = f"{p.quiet_hours.start}–{p.quiet_hours.end}" if p.quiet_hours else "off"
    table.add_row(
        "Preferences",
        f"lang={p.language}, tz={p.timezone}, quiet={quiet}, "
        f"smart_home={p.allow_smart_home_actions}",
    )

    console.print(table)


# ---------------------------------------------------------------------------
# Interactive prompts per section
# ---------------------------------------------------------------------------

def _prompt_location() -> Location:
    console.rule("[bold cyan]Location")
    address = click.prompt("Home address", type=str)
    latitude = click.prompt("Latitude", type=float)
    longitude = click.prompt("Longitude", type=float)
    nws_zone = click.prompt("NWS zone ID (optional, e.g. LAZ036)", type=str, default="", show_default=False)
    county = click.prompt("5-digit county FIPS (optional)", type=str, default="", show_default=False)
    return Location(
        address=address,
        latitude=latitude,
        longitude=longitude,
        nws_zone_id=nws_zone or None,
        county_fips=county or None,
    )


def _prompt_home() -> Home:
    console.rule("[bold cyan]Home")
    home_type = _prompt_enum("Home type", HomeType, HomeType.APARTMENT)
    floor = click.prompt("Floor level (1 = ground)", type=int, default=1)
    elevated = click.confirm("Is the home elevated (pier-and-beam, stilts)?", default=False)
    flood_zone = click.prompt("FEMA flood zone (optional, e.g. AE, X, VE)", default="", show_default=False)
    has_generator = click.confirm("Do you have a backup generator?", default=False)
    has_shutters = click.confirm("Do you have storm shutters?", default=False)
    return Home(
        type=HomeType(home_type),
        floor_level=floor,
        elevated=elevated,
        flood_zone=flood_zone or None,
        has_generator=has_generator,
        has_storm_shutters=has_shutters,
    )


def _prompt_vehicle() -> Vehicle:
    console.rule("[bold cyan]Vehicle")
    owns = click.confirm("Do you own a vehicle?", default=True)
    if not owns:
        return Vehicle(owns_vehicle=False, clearance=VehicleClearance.NONE, four_wheel_drive=False)
    clearance = _prompt_enum(
        "Vehicle clearance (low=sedan, medium=crossover, high=SUV/truck)",
        VehicleClearance,
        VehicleClearance.LOW,
    )
    four_wheel = click.confirm("Four-wheel drive?", default=False)
    return Vehicle(
        owns_vehicle=True,
        clearance=VehicleClearance(clearance),
        four_wheel_drive=four_wheel,
    )


def _prompt_medical() -> Medical:
    console.rule("[bold cyan]Medical")
    console.print("[dim]These flags affect evacuation urgency and action selection.[/dim]")
    mobility = click.confirm("Any mobility limitations?", default=False)
    oxygen = click.confirm("Oxygen-dependent?", default=False)
    refrig = click.confirm("Refrigerated medications (e.g. insulin)?", default=False)
    conditions_raw = click.prompt(
        "Chronic conditions (comma-separated, or blank)", default="", show_default=False
    )
    conditions = [c.strip() for c in conditions_raw.split(",") if c.strip()]
    return Medical(
        mobility_limited=mobility,
        oxygen_dependent=oxygen,
        refrigerated_medication=refrig,
        chronic_conditions=conditions,
    )


def _prompt_contacts() -> list[EmergencyContact]:
    console.rule("[bold cyan]Emergency contacts")
    contacts: list[EmergencyContact] = []
    while True:
        if contacts and not click.confirm("Add another contact?", default=False):
            break
        if not contacts and not click.confirm("Add an emergency contact?", default=True):
            break
        name = click.prompt("  Name", type=str)
        relationship = click.prompt("  Relationship", type=str, default="friend")
        phone = click.prompt("  Phone (E.164, e.g. +15551234567)", type=str)
        notify_raw = _prompt_multi_enum(
            "  Notify on which risk levels?",
            RiskNotifyLevel,
            [RiskNotifyLevel.HIGH, RiskNotifyLevel.CRITICAL],
        )
        try:
            contact = EmergencyContact(
                name=name, relationship=relationship, phone=phone, notify_on=notify_raw
            )
        except ValidationError as e:
            console.print(f"[red]Invalid contact:[/red] {e}")
            continue
        contacts.append(contact)
    return contacts


def _prompt_preferences() -> Preferences:
    console.rule("[bold cyan]Preferences")
    lang = click.prompt("Language", type=click.Choice(["en", "es"]), default="en")
    tz = click.prompt("Timezone (IANA)", default="America/Chicago")
    want_quiet = click.confirm("Set quiet hours?", default=True)
    quiet: QuietHours | None = None
    if want_quiet:
        start_raw = click.prompt("  Quiet hours start (HH:MM, 24h)", default="22:00")
        end_raw = click.prompt("  Quiet hours end (HH:MM, 24h)", default="07:00")
        quiet = QuietHours(
            start=time.fromisoformat(start_raw),
            end=time.fromisoformat(end_raw),
        )
    smart_home = click.confirm("Allow Guardian Agent to trigger smart-home actions?", default=True)
    return Preferences(
        language=lang,  # type: ignore[arg-type]
        timezone=tz,
        quiet_hours=quiet,
        allow_smart_home_actions=smart_home,
    )


# ---------------------------------------------------------------------------
# Click commands
# ---------------------------------------------------------------------------

@click.group()
def cli() -> None:
    """Guardian Agent — user profile tooling."""


@cli.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Where to write the new profile JSON.",
)
@click.option("--force", is_flag=True, help="Overwrite the file if it exists.")
def create(output: Path, force: bool) -> None:
    """Interactively build a new user profile."""
    if output.exists() and not force:
        console.print(
            f"[yellow]{output} already exists.[/yellow] "
            "Re-run with --force to overwrite, or pick a different path."
        )
        sys.exit(1)

    console.print("[bold green]Guardian Agent — profile setup[/bold green]")
    console.print("Answer each prompt. Press Enter to accept the [dim]default[/dim].\n")

    user_id = click.prompt("User ID (letters, digits, _, - only)", default="me")
    name = click.prompt("Display name", type=str)

    try:
        profile = UserProfile(
            user_id=user_id,
            name=name,
            location=_prompt_location(),
            home=_prompt_home(),
            vehicle=_prompt_vehicle(),
            medical=_prompt_medical(),
            emergency_contacts=_prompt_contacts(),
            preferences=_prompt_preferences(),
        )
    except ValidationError as e:
        console.print(f"[red]Profile failed validation:[/red]\n{e}")
        sys.exit(2)

    save_profile(profile, output)
    console.print(f"\n[bold green]Saved[/bold green] → {output}")
    _print_profile_table(profile)


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def show(path: Path) -> None:
    """Pretty-print a profile file."""
    try:
        profile = load_profile(path)
    except (ValidationError, ValueError) as e:
        console.print(f"[red]Failed to load {path}:[/red]\n{e}")
        sys.exit(2)
    _print_profile_table(profile)


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def validate(path: Path) -> None:
    """Validate a profile file and report any schema errors."""
    try:
        load_profile(path)
    except (ValidationError, ValueError) as e:
        console.print(f"[red]Invalid:[/red]\n{e}")
        sys.exit(1)
    console.print(f"[green]OK[/green] — {path} is a valid profile.")


if __name__ == "__main__":
    cli()
