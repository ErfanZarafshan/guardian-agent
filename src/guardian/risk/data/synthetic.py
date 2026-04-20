"""Synthetic NOAA Storm Events generator.

Produces a CSV that is *schema-compatible* with the real Storm Events DB and
contains a deliberately learnable signal. We use it for two purposes:

  1. Testing the training pipeline end-to-end without a 500MB download.
  2. Letting CI verify that the model actually learns (synthetic AUC > 0.7).

Embedded signal — designed to mirror real Gulf Coast climatology:

  - Event frequency itself varies seasonally: summer (Jun–Sep) has ~4x more
    events than winter, peaking in August. Hurricane season is real.
  - Within each month, the per-event probability of being "severe" (tornado,
    flash flood, tropical system, hail >= 1", t-storm wind >= 58 mph) also
    rises in summer.
  - Louisiana has a stronger seasonal effect than the rest of the Gulf.
  - East Baton Rouge parish (county_fips=22033) gets an extra severity bump
    so that location features carry signal too.

A classifier with month + state + event-history features should easily beat
random on this dataset (target AUC > 0.7 in tests).
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


# State FIPS for our Gulf Coast set (matching real NOAA codes).
_STATE_FIPS = {
    "LOUISIANA": 22,
    "TEXAS": 48,
    "MISSISSIPPI": 28,
    "ALABAMA": 1,
    "FLORIDA": 12,
}

# A handful of representative counties per state (real FIPS, 3-digit county part).
_COUNTIES_BY_STATE = {
    "LOUISIANA":   [33, 71, 51, 17, 19],   # E. Baton Rouge, Orleans, Jefferson, Calcasieu, Caddo
    "TEXAS":       [201, 113, 29, 439, 157],
    "MISSISSIPPI": [49, 47, 59, 81, 75],
    "ALABAMA":     [73, 97, 117, 89, 125],
    "FLORIDA":     [86, 95, 11, 99, 31],
}

# Distribution of event types. Severe types appear with a baseline rate
# that's modulated by state and season.
_NONSEVERE_TYPES = [
    "Heavy Rain", "Dense Fog", "Heat Advisory",
    "Frost/Freeze", "Lightning", "Thunderstorm Wind",
]
_SEVERE_TYPES = [
    "Tornado", "Flash Flood", "Hail",
    "Tropical Storm", "Excessive Heat",
]


def _seasonal_event_weight(month: int) -> float:
    """Relative event frequency by month. Peaks in Aug, troughs in Jan/Feb."""
    # Smooth annual cycle peaking at month 8 (August)
    # Range: ~0.3 in winter to ~1.7 in late summer.
    return 1.0 + 0.7 * math.cos(2 * math.pi * (month - 8) / 12)


def _severe_probability(state: str, county_id: int, month: int) -> float:
    """Per-event probability that this event is severe."""
    base = 0.10
    if state == "LOUISIANA":
        seasonal = 0.55 if month in (6, 7, 8, 9) else 0.05
    else:
        seasonal = 0.30 if month in (6, 7, 8, 9) else 0.10
    # East Baton Rouge bump
    if state == "LOUISIANA" and county_id == 33:
        seasonal += 0.15
    return min(0.95, base + seasonal * 0.5)


def _draw_seasonal_datetime(rng: random.Random, start: datetime, end: datetime) -> datetime:
    """Draw a datetime weighted by seasonal event frequency.

    Uses rejection sampling against the maximum seasonal weight (~1.7).
    """
    span_seconds = (end - start).total_seconds()
    max_weight = 1.7
    while True:
        offset = rng.uniform(0, span_seconds)
        candidate = start + timedelta(seconds=offset)
        weight = _seasonal_event_weight(candidate.month)
        if rng.random() * max_weight < weight:
            return candidate


def generate_synthetic_events(
    out_path: Path,
    *,
    n_events: int = 5000,
    start_year: int = 2018,
    end_year: int = 2024,
    seed: int = 42,
) -> Path:
    """Write a Storm-Events-shaped CSV to `out_path`. Returns the path."""
    rng = random.Random(seed)
    rows = []

    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31, 23, 59)

    state_names = list(_STATE_FIPS)

    for event_id in range(n_events):
        state = rng.choice(state_names)
        state_fips = _STATE_FIPS[state]
        county_id = rng.choice(_COUNTIES_BY_STATE[state])

        begin_dt = _draw_seasonal_datetime(rng, start_date, end_date)
        end_dt = begin_dt + timedelta(minutes=rng.randint(15, 240))

        p_severe = _severe_probability(state, county_id, begin_dt.month)
        if rng.random() < p_severe:
            event_type = rng.choice(_SEVERE_TYPES)
        else:
            event_type = rng.choice(_NONSEVERE_TYPES)

        # Magnitude: for Hail and Tstm Wind, choose a value that respects
        # our severity definition (severe Hail >= 1", severe TSTM Wind >= 58 mph).
        magnitude = ""
        magnitude_type = ""
        if event_type == "Hail":
            magnitude = round(rng.uniform(0.25, 2.5), 2) if rng.random() < 0.7 else ""
        elif event_type == "Thunderstorm Wind":
            magnitude = rng.randint(35, 90) if rng.random() < 0.7 else ""
            magnitude_type = "EG" if magnitude else ""

        damage_choices = ["", "0", "1.00K", "5.00K", "25.00K", "100.00K", "500.00K", "1.00M"]
        rows.append({
            "EVENT_ID": 100000 + event_id,
            "EPISODE_ID": 50000 + (event_id // 3),
            "STATE": state,
            "STATE_FIPS": state_fips,
            "CZ_TYPE": "C",
            "CZ_FIPS": county_id,
            "CZ_NAME": f"County{county_id}",
            "BEGIN_YEARMONTH": int(begin_dt.strftime("%Y%m")),
            "BEGIN_DAY": begin_dt.day,
            "BEGIN_TIME": int(begin_dt.strftime("%H%M")),
            "END_YEARMONTH": int(end_dt.strftime("%Y%m")),
            "END_DAY": end_dt.day,
            "END_TIME": int(end_dt.strftime("%H%M")),
            "BEGIN_DATE_TIME": begin_dt.strftime("%d-%b-%y %H:%M:%S").upper(),
            "END_DATE_TIME": end_dt.strftime("%d-%b-%y %H:%M:%S").upper(),
            "EVENT_TYPE": event_type,
            "MAGNITUDE": magnitude,
            "MAGNITUDE_TYPE": magnitude_type,
            "DAMAGE_PROPERTY": rng.choice(damage_choices),
            "DAMAGE_CROPS": rng.choice(damage_choices),
            "INJURIES_DIRECT": rng.choices([0, 0, 0, 1, 3], k=1)[0],
            "DEATHS_DIRECT": rng.choices([0, 0, 0, 0, 1], k=1)[0],
            "BEGIN_LAT": round(rng.uniform(28.0, 32.0), 4),
            "BEGIN_LON": round(rng.uniform(-95.0, -85.0), 4),
        })

    df = pd.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


__all__ = ["generate_synthetic_events"]
