"""Loader for the NOAA Storm Events Database CSV files.

Schema reference:
  https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/

Each year has three CSV families:
  - StormEvents_details-ftp_v1.0_dYYYY_cYYYYMMDD.csv.gz   (one row per event)
  - StormEvents_locations-ftp_v1.0_dYYYY_cYYYYMMDD.csv.gz (one row per location)
  - StormEvents_fatalities-ftp_v1.0_dYYYY_cYYYYMMDD.csv.gz

We only use the *details* files. They contain event_type, state, county, date,
magnitude, damage, etc.

This module:
  - Defines the canonical column subset we care about.
  - Loads a directory of CSV(.gz) files into a single normalized DataFrame.
  - Filters to Gulf Coast states by default.
  - Parses damage strings like "10.00K" / "1.50M" into floats.
  - Tags each event with whether it counts as a "severe escalation" outcome
    (tornado, flash flood, hurricane, etc.) using documented thresholds.
"""

from __future__ import annotations

import gzip
import io
import re
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GULF_COAST_STATES: tuple[str, ...] = (
    "LOUISIANA",
    "TEXAS",
    "MISSISSIPPI",
    "ALABAMA",
    "FLORIDA",
)

# The columns we actually use. Loading only these keeps memory low.
USED_COLUMNS: tuple[str, ...] = (
    "EVENT_ID",
    "EPISODE_ID",
    "STATE",
    "STATE_FIPS",
    "CZ_TYPE",  # "C" county, "Z" zone, "M" marine
    "CZ_FIPS",
    "CZ_NAME",
    "BEGIN_YEARMONTH",
    "BEGIN_DAY",
    "BEGIN_TIME",
    "END_YEARMONTH",
    "END_DAY",
    "END_TIME",
    "BEGIN_DATE_TIME",
    "END_DATE_TIME",
    "EVENT_TYPE",
    "MAGNITUDE",
    "MAGNITUDE_TYPE",
    "DAMAGE_PROPERTY",
    "DAMAGE_CROPS",
    "INJURIES_DIRECT",
    "DEATHS_DIRECT",
    "BEGIN_LAT",
    "BEGIN_LON",
)


# Event types that count as "severe escalation outcomes" — these are the
# events whose occurrence we want our classifier to predict ahead of time.
# Selection is based on NWS impact classifications and proposal scope.
SEVERE_EVENT_TYPES: frozenset[str] = frozenset({
    "Tornado",
    "Flash Flood",
    "Hurricane",
    "Hurricane (Typhoon)",
    "Tropical Storm",
    "Storm Surge/Tide",
    "Excessive Heat",
    # Magnitude-conditional ones handled in is_severe(): hail, t-storm wind
})

# For magnitude-conditional severity (NWS Storm Data Preparation criteria):
SEVERE_HAIL_INCHES = 1.0          # 1" hail = severe
SEVERE_TSTM_WIND_MPH = 58.0       # 58 mph = severe thunderstorm criterion


# ---------------------------------------------------------------------------
# Damage string parsing
# ---------------------------------------------------------------------------

_DAMAGE_RE = re.compile(r"^\s*([0-9.]+)\s*([KMB]?)\s*$", re.IGNORECASE)
_DAMAGE_MULTIPLIERS = {"": 1.0, "K": 1e3, "M": 1e6, "B": 1e9}


def parse_damage(value) -> float:
    """Parse a Storm Events damage string like '10.00K' / '1.5M' / '0' to float.

    Returns 0.0 for missing/unparseable values rather than raising.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.0
    s = str(value).strip()
    if not s or s == "0":
        return 0.0
    m = _DAMAGE_RE.match(s)
    if not m:
        return 0.0
    num, suffix = m.group(1), m.group(2).upper()
    try:
        return float(num) * _DAMAGE_MULTIPLIERS.get(suffix, 1.0)
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# Severity classification
# ---------------------------------------------------------------------------

def is_severe_event(event_type: str, magnitude: float | None) -> bool:
    """Decide whether an event row qualifies as a 'severe escalation'."""
    if event_type in SEVERE_EVENT_TYPES:
        return True
    mag = magnitude if magnitude and not pd.isna(magnitude) else 0.0
    if event_type == "Hail" and mag >= SEVERE_HAIL_INCHES:
        return True
    if event_type == "Thunderstorm Wind" and mag >= SEVERE_TSTM_WIND_MPH:
        return True
    return False


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _read_one(path: Path) -> pd.DataFrame:
    """Read one Storm Events details CSV (gzipped or plain)."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            df = pd.read_csv(io.StringIO(fh.read()), low_memory=False)
    else:
        df = pd.read_csv(path, low_memory=False)
    # Keep only columns we actually use; some older years lack certain columns.
    keep = [c for c in USED_COLUMNS if c in df.columns]
    return df[keep].copy()


def load_storm_events(
    source: Path,
    states: tuple[str, ...] = GULF_COAST_STATES,
) -> pd.DataFrame:
    """Load all Storm Events detail CSVs from a directory or a single file.

    Returns a normalized DataFrame with parsed dates, damages, and a
    `is_severe` boolean column.
    """
    source = Path(source)
    if source.is_file():
        files = [source]
    elif source.is_dir():
        files = sorted(
            list(source.glob("*details*.csv.gz"))
            + list(source.glob("*details*.csv"))
        )
        if not files:
            raise FileNotFoundError(
                f"No Storm Events details CSVs found in {source}. "
                "Expected files matching *details*.csv[.gz]."
            )
    else:
        raise FileNotFoundError(f"{source} does not exist.")

    frames = [_read_one(p) for p in files]
    df = pd.concat(frames, ignore_index=True)

    # State filter (Storm Events stores state names in uppercase, but be safe).
    if states:
        wanted = {s.upper() for s in states}
        df = df[df["STATE"].astype(str).str.upper().isin(wanted)].copy()

    # Parse begin/end datetimes. NOAA's BEGIN_DATE_TIME format is like
    # "15-JUN-23 14:30:00".
    for col in ("BEGIN_DATE_TIME", "END_DATE_TIME"):
        if col in df.columns:
            # Try the canonical NOAA format first, fall back to inference.
            parsed = pd.to_datetime(df[col], format="%d-%b-%y %H:%M:%S", errors="coerce")
            if parsed.isna().all():
                parsed = pd.to_datetime(df[col], errors="coerce")
            df[col] = parsed

    # Parse damages.
    if "DAMAGE_PROPERTY" in df.columns:
        df["damage_property_usd"] = df["DAMAGE_PROPERTY"].apply(parse_damage)
    else:
        df["damage_property_usd"] = 0.0
    if "DAMAGE_CROPS" in df.columns:
        df["damage_crops_usd"] = df["DAMAGE_CROPS"].apply(parse_damage)
    else:
        df["damage_crops_usd"] = 0.0

    # Severity tag.
    if "MAGNITUDE" in df.columns:
        mag = pd.to_numeric(df["MAGNITUDE"], errors="coerce")
    else:
        mag = pd.Series([None] * len(df))
    df["is_severe"] = [
        is_severe_event(et, m) for et, m in zip(df["EVENT_TYPE"].astype(str), mag)
    ]

    # Convenience columns.
    df["year"] = df["BEGIN_DATE_TIME"].dt.year if "BEGIN_DATE_TIME" in df.columns else pd.NA
    df["month"] = df["BEGIN_DATE_TIME"].dt.month if "BEGIN_DATE_TIME" in df.columns else pd.NA

    # Build a 5-digit county FIPS where possible (state_fips * 1000 + cz_fips,
    # but only for rows where CZ_TYPE == "C" — county vs zone).
    def _county_fips(row) -> str | None:
        if str(row.get("CZ_TYPE", "")).upper() != "C":
            return None
        try:
            sf = int(row["STATE_FIPS"])
            cf = int(row["CZ_FIPS"])
            return f"{sf:02d}{cf:03d}"
        except (TypeError, ValueError, KeyError):
            return None

    df["county_fips"] = df.apply(_county_fips, axis=1)

    return df.reset_index(drop=True)


__all__ = [
    "GULF_COAST_STATES",
    "USED_COLUMNS",
    "SEVERE_EVENT_TYPES",
    "SEVERE_HAIL_INCHES",
    "SEVERE_TSTM_WIND_MPH",
    "parse_damage",
    "is_severe_event",
    "load_storm_events",
]
