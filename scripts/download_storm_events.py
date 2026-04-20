"""Download NOAA Storm Events Database CSVs.

NOAA hosts the bulk Storm Events CSVs at:
  https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/

For each year, three .csv.gz files exist (details, fatalities, locations).
This script grabs the *details* file for the year range you specify and
saves them to data/raw/storm_events/.

Usage:
    python scripts/download_storm_events.py --start 2018 --end 2024
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import requests
from rich.console import Console


BASE = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
# Filename pattern: StormEvents_details-ftp_v1.0_dYYYY_cYYYYMMDD.csv.gz
# The cYYYYMMDD suffix is a generation date that NOAA updates.
# We fetch the directory listing to find the current filename for each year.

INDEX_URL = BASE  # plain index page

console = Console()


def _list_year_files(year: int, session: requests.Session) -> list[str]:
    """Return all *details* filenames published for a given year."""
    resp = session.get(INDEX_URL, timeout=30)
    resp.raise_for_status()
    html = resp.text
    needle = f"StormEvents_details-ftp_v1.0_d{year}_"
    out = []
    for token in html.split('"'):
        if token.startswith(needle) and token.endswith(".csv.gz"):
            out.append(token)
    return out


@click.command()
@click.option("--start", type=int, default=2018, show_default=True)
@click.option("--end", type=int, default=2024, show_default=True)
@click.option(
    "--out",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/raw/storm_events"),
    show_default=True,
)
def main(start: int, end: int, out: Path) -> None:
    if start > end:
        click.echo("--start must be <= --end", err=True)
        sys.exit(1)
    out.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "GuardianAgent/0.1 (academic project)"

    for year in range(start, end + 1):
        console.rule(f"[bold cyan]{year}")
        try:
            candidates = _list_year_files(year, session)
        except requests.RequestException as e:
            console.print(f"[red]Index fetch failed:[/red] {e}")
            continue
        if not candidates:
            console.print(f"[yellow]No details file found for {year}.[/yellow]")
            continue
        # Prefer the most recently generated file
        fname = sorted(candidates)[-1]
        url = BASE + fname
        target = out / fname
        if target.exists():
            console.print(f"[green]Already have[/green] {fname}")
            continue
        console.print(f"Downloading {fname} ...")
        try:
            with session.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with target.open("wb") as fh:
                    for chunk in r.iter_content(chunk_size=64 * 1024):
                        fh.write(chunk)
            mb = target.stat().st_size / (1024 * 1024)
            console.print(f"  [green]ok[/green] ({mb:.1f} MB)")
        except requests.RequestException as e:
            console.print(f"  [red]failed:[/red] {e}")
            if target.exists():
                target.unlink()

    console.rule("[bold green]Done")
    console.print(f"Files in {out}:")
    for p in sorted(out.glob("*.csv.gz")):
        mb = p.stat().st_size / (1024 * 1024)
        console.print(f"  {p.name} ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
