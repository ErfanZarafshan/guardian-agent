"""Feature engineering for the threat classifier.

The Storm Events DB is an *event log*, not a time series. To turn it into a
supervised learning problem we reframe it as a per-(county, day) prediction:

    For each (county, day) cell:
      Features  = season, location, recent event history in that county
      Label     = 1 iff a severe event begins in that county within
                  the LOOKAHEAD_HOURS that follow.

This gives us a binary classification task with strong base-rate and
seasonality signal that the model can pick up.

Inference uses the same feature vector at runtime: the agent looks up its
county_fips from the user profile, derives time-of-year features, and uses
recent NWS alerts as a lightweight stand-in for "recent events."
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOOKAHEAD_HOURS = 24       # Predict severe event in the next 24h
LOOKBACK_DAYS_SHORT = 7    # Short window for event-history features
LOOKBACK_DAYS_LONG = 30    # Long window for severe-event-history features


# Numeric feature columns, in the order the model sees them.
NUMERIC_FEATURES: tuple[str, ...] = (
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    "is_summer",
    "events_past_7d",
    "severe_events_past_30d",
    "hours_since_last_event",
    "hours_since_last_severe",
    "max_damage_past_7d_log",
    "had_advisory_past_24h",
)

# Categorical feature columns. We one-hot encode `state` (small cardinality);
# county_fips is too high-cardinality for one-hot — we use it as a hash bucket.
CATEGORICAL_FEATURES: tuple[str, ...] = ("state",)

COUNTY_HASH_BUCKETS = 64  # number of buckets for county_fips hashing


# ---------------------------------------------------------------------------
# Per-cell feature computation
# ---------------------------------------------------------------------------

def _cyclical(value: float, period: float) -> tuple[float, float]:
    """Encode a periodic value as (sin, cos)."""
    angle = 2.0 * math.pi * value / period
    return math.sin(angle), math.cos(angle)


def _hours_between(t1: datetime, t0: datetime) -> float:
    return max(0.0, (t1 - t0).total_seconds() / 3600.0)


@dataclass
class CellFeatures:
    """Features computed for one (county, observed_at) cell.

    Stored as a plain dict-like object so we can convert to a DataFrame row.
    """
    state: str
    county_fips: str
    observed_at: datetime
    month_sin: float
    month_cos: float
    doy_sin: float
    doy_cos: float
    is_summer: int
    events_past_7d: int
    severe_events_past_30d: int
    hours_since_last_event: float
    hours_since_last_severe: float
    max_damage_past_7d_log: float
    had_advisory_past_24h: int

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "county_fips": self.county_fips,
            "observed_at": self.observed_at,
            **{f: getattr(self, f) for f in NUMERIC_FEATURES},
        }


def compute_cell_features(
    events: pd.DataFrame,
    state: str,
    county_fips: str,
    observed_at: datetime,
) -> CellFeatures:
    """Compute features for one (county, time) cell using its event history."""
    # Normalize observed_at to a naive (timezone-unaware) datetime so it can be
    # compared against pandas' naive datetime64 Series. Historical Storm Events
    # data is naive; weather observations flowing in from the agent are
    # tz-aware (UTC). We convert to UTC then drop the tzinfo.
    if observed_at.tzinfo is not None:
        observed_at = observed_at.astimezone(timezone.utc).replace(tzinfo=None)

    short_window_start = observed_at - timedelta(days=LOOKBACK_DAYS_SHORT)
    long_window_start = observed_at - timedelta(days=LOOKBACK_DAYS_LONG)
    advisory_window_start = observed_at - timedelta(hours=24)

    # Restrict to this county + strictly past events
    if "county_fips" in events.columns:
        county_mask = events["county_fips"] == county_fips
    else:
        county_mask = pd.Series([False] * len(events))
    past_mask = events["BEGIN_DATE_TIME"] < observed_at
    rel = events[county_mask & past_mask]

    short = rel[rel["BEGIN_DATE_TIME"] >= short_window_start]
    long = rel[rel["BEGIN_DATE_TIME"] >= long_window_start]
    advisory = rel[rel["BEGIN_DATE_TIME"] >= advisory_window_start]

    if len(rel) > 0:
        last_any = rel["BEGIN_DATE_TIME"].max()
        hrs_since_last = _hours_between(observed_at, last_any.to_pydatetime())
    else:
        hrs_since_last = float(LOOKBACK_DAYS_LONG * 24)  # capped large value

    severe_rel = rel[rel["is_severe"]]
    if len(severe_rel) > 0:
        last_sev = severe_rel["BEGIN_DATE_TIME"].max()
        hrs_since_sev = _hours_between(observed_at, last_sev.to_pydatetime())
    else:
        hrs_since_sev = float(LOOKBACK_DAYS_LONG * 24)

    max_damage_short = float(short["damage_property_usd"].max()) if len(short) else 0.0
    log_damage = math.log1p(max_damage_short)

    month_sin, month_cos = _cyclical(observed_at.month, 12.0)
    doy_sin, doy_cos = _cyclical(observed_at.timetuple().tm_yday, 365.0)

    return CellFeatures(
        state=state,
        county_fips=county_fips,
        observed_at=observed_at,
        month_sin=month_sin,
        month_cos=month_cos,
        doy_sin=doy_sin,
        doy_cos=doy_cos,
        is_summer=1 if observed_at.month in (6, 7, 8, 9) else 0,
        events_past_7d=len(short),
        severe_events_past_30d=int(long["is_severe"].sum()),
        hours_since_last_event=hrs_since_last,
        hours_since_last_severe=hrs_since_sev,
        max_damage_past_7d_log=log_damage,
        had_advisory_past_24h=1 if len(advisory) > 0 else 0,
    )


def label_cell(
    events: pd.DataFrame,
    county_fips: str,
    observed_at: datetime,
    lookahead_hours: int = LOOKAHEAD_HOURS,
) -> int:
    """Label = 1 iff a severe event begins in `county_fips` within lookahead."""
    if observed_at.tzinfo is not None:
        observed_at = observed_at.astimezone(timezone.utc).replace(tzinfo=None)
    horizon = observed_at + timedelta(hours=lookahead_hours)
    if "county_fips" not in events.columns:
        return 0
    mask = (
        (events["county_fips"] == county_fips)
        & (events["BEGIN_DATE_TIME"] >= observed_at)
        & (events["BEGIN_DATE_TIME"] < horizon)
        & (events["is_severe"])
    )
    return int(mask.any())


# ---------------------------------------------------------------------------
# Training-set construction
# ---------------------------------------------------------------------------

def build_training_set(
    events: pd.DataFrame,
    *,
    grid_hours: int = 24,
    seed: int = 0,
    max_cells_per_county: int | None = None,
) -> pd.DataFrame:
    """Construct a (county, time) cell DataFrame with features + label.

    `grid_hours` controls how often we sample cells per county (24 = once a
    day). `max_cells_per_county` caps the result for memory; None = no cap.
    """
    if "BEGIN_DATE_TIME" not in events.columns:
        raise ValueError("events DataFrame missing BEGIN_DATE_TIME column")

    events = events.dropna(subset=["BEGIN_DATE_TIME", "county_fips"]).copy()
    if len(events) == 0:
        raise ValueError("No events with valid datetime+county_fips")

    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for (state, county_fips), group in events.groupby(["STATE", "county_fips"]):
        county_min = group["BEGIN_DATE_TIME"].min()
        county_max = group["BEGIN_DATE_TIME"].max()
        if pd.isna(county_min) or pd.isna(county_max):
            continue

        # Generate a grid of timestamps for this county.
        delta = timedelta(hours=grid_hours)
        timestamps = []
        t = county_min.to_pydatetime()
        end = county_max.to_pydatetime()
        while t <= end:
            timestamps.append(t)
            t += delta

        if max_cells_per_county and len(timestamps) > max_cells_per_county:
            idx = rng.choice(len(timestamps), max_cells_per_county, replace=False)
            timestamps = [timestamps[i] for i in sorted(idx)]

        for ts in timestamps:
            features = compute_cell_features(events, state, county_fips, ts)
            row = features.to_dict()
            row["label"] = label_cell(events, county_fips, ts)
            rows.append(row)

    if not rows:
        raise ValueError("No training cells produced; check input data.")
    return pd.DataFrame(rows)


__all__ = [
    "LOOKAHEAD_HOURS",
    "LOOKBACK_DAYS_SHORT",
    "LOOKBACK_DAYS_LONG",
    "NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
    "COUNTY_HASH_BUCKETS",
    "CellFeatures",
    "compute_cell_features",
    "label_cell",
    "build_training_set",
]
