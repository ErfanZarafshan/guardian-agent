"""Tests for the threat-classifier pipeline.

Uses synthetic data exclusively so tests are fast, deterministic, and don't
require downloading 500MB of NOAA CSVs.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from guardian.risk.classifier import (
    DEFAULT_BUCKET_THRESHOLDS,
    ThreatBucket,
    ThreatClassifier,
    bucket_probability,
)
from guardian.risk.data.storm_events import (
    SEVERE_HAIL_INCHES,
    SEVERE_TSTM_WIND_MPH,
    is_severe_event,
    load_storm_events,
    parse_damage,
)
from guardian.risk.data.synthetic import generate_synthetic_events
from guardian.risk.features import (
    LOOKAHEAD_HOURS,
    NUMERIC_FEATURES,
    build_training_set,
    compute_cell_features,
    label_cell,
)


# ---------------------------------------------------------------------------
# Damage parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("0", 0.0),
        ("", 0.0),
        ("10.00K", 10_000.0),
        ("1.5K", 1_500.0),
        ("2M", 2_000_000.0),
        ("1.5M", 1_500_000.0),
        ("3.0B", 3e9),
        ("garbage", 0.0),
        (None, 0.0),
    ],
)
def test_parse_damage(raw, expected) -> None:
    assert parse_damage(raw) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Severity classification
# ---------------------------------------------------------------------------

def test_is_severe_named_types() -> None:
    assert is_severe_event("Tornado", None) is True
    assert is_severe_event("Flash Flood", 0.0) is True
    assert is_severe_event("Hurricane", None) is True
    assert is_severe_event("Excessive Heat", None) is True


def test_is_severe_hail_threshold() -> None:
    assert is_severe_event("Hail", SEVERE_HAIL_INCHES) is True
    assert is_severe_event("Hail", SEVERE_HAIL_INCHES + 0.5) is True
    assert is_severe_event("Hail", SEVERE_HAIL_INCHES - 0.5) is False
    assert is_severe_event("Hail", None) is False


def test_is_severe_tstm_wind_threshold() -> None:
    assert is_severe_event("Thunderstorm Wind", SEVERE_TSTM_WIND_MPH) is True
    assert is_severe_event("Thunderstorm Wind", SEVERE_TSTM_WIND_MPH - 1) is False


def test_is_severe_random_events_not_severe() -> None:
    assert is_severe_event("Dense Fog", None) is False
    assert is_severe_event("Heat Advisory", None) is False
    assert is_severe_event("Lightning", None) is False


# ---------------------------------------------------------------------------
# Synthetic generator + loader
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_csv(tmp_path_factory) -> Path:
    """Module-scoped: generate the CSV once and reuse across tests."""
    out = tmp_path_factory.mktemp("synth") / "events.csv"
    generate_synthetic_events(out, n_events=6000, seed=7)
    return out


@pytest.fixture(scope="module")
def synthetic_df(synthetic_csv: Path) -> pd.DataFrame:
    return load_storm_events(synthetic_csv)


def test_loader_normalizes_columns(synthetic_df: pd.DataFrame) -> None:
    for col in (
        "BEGIN_DATE_TIME",
        "EVENT_TYPE",
        "STATE",
        "county_fips",
        "is_severe",
        "damage_property_usd",
        "year",
        "month",
    ):
        assert col in synthetic_df.columns, f"missing {col}"


def test_loader_filters_to_gulf_coast(synthetic_df: pd.DataFrame) -> None:
    states = set(synthetic_df["STATE"].dropna().str.upper())
    assert states.issubset(
        {"LOUISIANA", "TEXAS", "MISSISSIPPI", "ALABAMA", "FLORIDA"}
    )


def test_loader_parses_dates(synthetic_df: pd.DataFrame) -> None:
    assert synthetic_df["BEGIN_DATE_TIME"].notna().any()
    assert pd.api.types.is_datetime64_any_dtype(synthetic_df["BEGIN_DATE_TIME"])


def test_loader_county_fips_format(synthetic_df: pd.DataFrame) -> None:
    fips = synthetic_df["county_fips"].dropna()
    assert (fips.str.len() == 5).all()


def test_loader_severity_signal_present(synthetic_df: pd.DataFrame) -> None:
    """Synthetic data has more severe events in summer LA than winter LA."""
    la = synthetic_df[synthetic_df["STATE"] == "LOUISIANA"]
    summer_rate = la[la["month"].isin([6, 7, 8, 9])]["is_severe"].mean()
    winter_rate = la[la["month"].isin([12, 1, 2])]["is_severe"].mean()
    assert summer_rate > winter_rate, (
        f"Expected summer LA severe-rate > winter LA severe-rate; "
        f"got {summer_rate:.3f} vs {winter_rate:.3f}"
    )


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def test_compute_cell_features_returns_all_numeric(synthetic_df: pd.DataFrame) -> None:
    state = "LOUISIANA"
    fips = synthetic_df[synthetic_df["STATE"] == state]["county_fips"].dropna().iloc[0]
    obs_at = datetime(2022, 7, 15, 12, 0, 0)
    f = compute_cell_features(synthetic_df, state, fips, obs_at)
    d = f.to_dict()
    for col in NUMERIC_FEATURES:
        assert col in d
        assert isinstance(d[col], (int, float))


def test_compute_cell_features_cyclical_bounds(synthetic_df: pd.DataFrame) -> None:
    f = compute_cell_features(
        synthetic_df, "LOUISIANA", "22033", datetime(2022, 1, 1)
    )
    assert -1.0 <= f.month_sin <= 1.0
    assert -1.0 <= f.month_cos <= 1.0


def test_label_cell_positive_when_severe_event_imminent() -> None:
    df = pd.DataFrame({
        "BEGIN_DATE_TIME": pd.to_datetime([
            "2022-07-15 13:00",
            "2022-07-15 16:00",
        ]),
        "county_fips": ["22033", "22033"],
        "is_severe": [True, False],
    })
    obs = datetime(2022, 7, 15, 12, 0, 0)
    assert label_cell(df, "22033", obs, lookahead_hours=LOOKAHEAD_HOURS) == 1


def test_label_cell_zero_when_no_imminent_severe() -> None:
    df = pd.DataFrame({
        "BEGIN_DATE_TIME": pd.to_datetime(["2022-07-20 12:00"]),
        "county_fips": ["22033"],
        "is_severe": [True],
    })
    obs = datetime(2022, 7, 15, 12, 0, 0)
    # Severe event is 5 days away, outside 24h horizon.
    assert label_cell(df, "22033", obs) == 0


def test_label_cell_zero_for_different_county() -> None:
    df = pd.DataFrame({
        "BEGIN_DATE_TIME": pd.to_datetime(["2022-07-15 13:00"]),
        "county_fips": ["48201"],
        "is_severe": [True],
    })
    obs = datetime(2022, 7, 15, 12, 0, 0)
    assert label_cell(df, "22033", obs) == 0


def test_build_training_set_shape(synthetic_df: pd.DataFrame) -> None:
    cells = build_training_set(
        synthetic_df, grid_hours=72, max_cells_per_county=20, seed=0
    )
    assert "label" in cells.columns
    assert len(cells) > 0
    assert set(NUMERIC_FEATURES).issubset(cells.columns)
    assert "state" in cells.columns
    # Labels should be 0/1
    assert set(cells["label"].unique()).issubset({0, 1})


def test_build_training_set_has_both_classes(synthetic_df: pd.DataFrame) -> None:
    cells = build_training_set(
        synthetic_df, grid_hours=72, max_cells_per_county=30, seed=0
    )
    assert (cells["label"] == 1).any(), "no positive labels"
    assert (cells["label"] == 0).any(), "no negative labels"


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------

def test_bucket_probability_thresholds() -> None:
    low_max, med_max = DEFAULT_BUCKET_THRESHOLDS
    assert bucket_probability(low_max - 0.01) is ThreatBucket.LOW
    assert bucket_probability(low_max) is ThreatBucket.MEDIUM
    assert bucket_probability(med_max - 0.01) is ThreatBucket.MEDIUM
    assert bucket_probability(med_max) is ThreatBucket.HIGH
    assert bucket_probability(0.99) is ThreatBucket.HIGH


# ---------------------------------------------------------------------------
# Classifier end-to-end (uses synthetic data; fast)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_classifier(synthetic_df: pd.DataFrame) -> ThreatClassifier:
    """Train once per test module."""
    return ThreatClassifier.fit(
        synthetic_df,
        grid_hours=72,            # coarser grid -> fewer cells -> faster
        max_cells_per_county=40,
        param_grid={
            "clf__n_estimators": [80],
            "clf__max_depth": [2],
            "clf__learning_rate": [0.1],
        },
        cv_folds=2,
        random_state=0,
    )


def test_classifier_trains_and_has_metrics(trained_classifier: ThreatClassifier) -> None:
    assert trained_classifier.pipeline is not None
    assert trained_classifier.metrics is not None
    m = trained_classifier.metrics
    assert m.n_train > 0 and m.n_test > 0
    assert 0.0 <= m.accuracy <= 1.0


def test_classifier_beats_random_on_synthetic(trained_classifier: ThreatClassifier) -> None:
    """Synthetic data has a learnable signal; AUC should clearly exceed 0.5."""
    assert trained_classifier.metrics is not None
    assert trained_classifier.metrics.roc_auc > 0.6, (
        f"AUC too low: {trained_classifier.metrics.roc_auc:.3f}"
    )


def test_classifier_score_blank(trained_classifier: ThreatClassifier) -> None:
    score = trained_classifier.score_blank(
        state="LOUISIANA",
        county_fips="22033",
        observed_at=datetime(2022, 7, 15, 12, 0, 0),
    )
    assert 0.0 <= score.probability <= 1.0
    assert score.bucket in (ThreatBucket.LOW, ThreatBucket.MEDIUM, ThreatBucket.HIGH)
    assert "month_sin" in score.feature_summary


def test_classifier_seasonal_signal(trained_classifier: ThreatClassifier) -> None:
    """Summer LA score should be greater than winter LA score (signal in synth)."""
    summer = trained_classifier.score_blank("LOUISIANA", "22033", datetime(2022, 7, 15))
    winter = trained_classifier.score_blank("LOUISIANA", "22033", datetime(2022, 1, 15))
    assert summer.probability > winter.probability, (
        f"Expected summer prob > winter; got summer={summer.probability:.3f}, "
        f"winter={winter.probability:.3f}"
    )


def test_classifier_save_and_load(
    trained_classifier: ThreatClassifier, tmp_path: Path
) -> None:
    out = tmp_path / "model.joblib"
    trained_classifier.save(out)
    assert out.exists()

    loaded = ThreatClassifier.load(out)
    assert loaded.metrics is not None
    assert loaded.metrics.roc_auc == pytest.approx(
        trained_classifier.metrics.roc_auc
    )

    # Inference should still work and match.
    obs_at = datetime(2022, 8, 1, 12, 0, 0)
    s1 = trained_classifier.score_blank("LOUISIANA", "22033", obs_at)
    s2 = loaded.score_blank("LOUISIANA", "22033", obs_at)
    assert s1.probability == pytest.approx(s2.probability)


def test_classifier_save_unfitted_fails(tmp_path: Path) -> None:
    clf = ThreatClassifier()
    with pytest.raises(RuntimeError):
        clf.save(tmp_path / "x.joblib")


def test_score_blank_works_on_newer_pandas(trained_classifier: ThreatClassifier) -> None:
    """Regression test for two distinct datetime issues seen on pandas 2.x +
    Python 3.13:

      1. Empty pd.to_datetime() calls infer dtype=datetime64[s], which
         can't be compared against stdlib datetime. Fixed by constructing
         the empty DataFrame with explicit datetime64[ns].
      2. tz-aware stdlib datetime can't be compared against naive
         datetime64[ns] Series. Fixed by stripping tzinfo on entry to
         compute_cell_features / label_cell.

    This test exercises both paths by passing a timezone-aware datetime —
    which is what the agent does in production, since WeatherObservation
    uses datetime.now(timezone.utc).
    """
    from datetime import timezone as tz_mod

    # Should not raise — tz-aware datetime is the realistic case.
    score = trained_classifier.score_blank(
        state="LOUISIANA",
        county_fips="22033",
        observed_at=datetime(2024, 7, 15, 12, 0, 0, tzinfo=tz_mod.utc),
    )
    assert 0.0 <= score.probability <= 1.0

    # And the naive case should still work too.
    score_naive = trained_classifier.score_blank(
        state="LOUISIANA",
        county_fips="22033",
        observed_at=datetime(2024, 7, 15, 12, 0, 0),
    )
    assert 0.0 <= score_naive.probability <= 1.0
