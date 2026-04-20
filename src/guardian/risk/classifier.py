"""Supervised threat-pattern classifier.

Wraps a sklearn pipeline (preprocessing + GradientBoostingClassifier) with:

  - .fit(events_df) — feature engineering + GridSearchCV training
  - .save() / .load() via joblib
  - .score_observation(...) — runtime inference returning ThreatScore

The saved model bundles the sklearn pipeline plus the metadata needed to
reproduce inference (feature order, training metrics, sklearn version).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from .features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_training_set,
    compute_cell_features,
)


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

class ThreatBucket(str, Enum):
    """Discrete bucket consumed by the Bayesian network as evidence."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


# Probability cutoffs separating Low/Medium/High. Tunable; defaults reflect a
# conservative-but-actionable split (so most observations are Low, some
# Medium, few High).
DEFAULT_BUCKET_THRESHOLDS = (0.20, 0.50)


@dataclass
class ThreatScore:
    probability: float          # P(severe event in next LOOKAHEAD_HOURS)
    bucket: ThreatBucket
    feature_summary: dict       # what went into the prediction (for logging)


@dataclass
class TrainingMetrics:
    """Held-out test metrics. Saved alongside the model."""
    n_train: int
    n_test: int
    positive_rate_train: float
    positive_rate_test: float
    accuracy: float
    f1: float
    roc_auc: float
    avg_precision: float
    confusion_matrix: list[list[int]]
    best_params: dict[str, Any] = field(default_factory=dict)
    sklearn_version: str = sklearn.__version__

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------

def _build_pipeline() -> Pipeline:
    """Numeric scaling + categorical one-hot + GradientBoostingClassifier.

    We don't tune n_estimators/max_depth/learning_rate to extreme values to
    keep training tractable on a laptop with synthetic data in CI.
    """
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(NUMERIC_FEATURES)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
    )
    model = GradientBoostingClassifier(random_state=0)
    return Pipeline([("pre", pre), ("clf", model)])


_DEFAULT_PARAM_GRID = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [2, 3],
    "clf__learning_rate": [0.05, 0.1],
}


def bucket_probability(
    p: float,
    thresholds: tuple[float, float] = DEFAULT_BUCKET_THRESHOLDS,
) -> ThreatBucket:
    low_max, med_max = thresholds
    if p < low_max:
        return ThreatBucket.LOW
    if p < med_max:
        return ThreatBucket.MEDIUM
    return ThreatBucket.HIGH


# ---------------------------------------------------------------------------
# Classifier wrapper
# ---------------------------------------------------------------------------

class ThreatClassifier:
    """Train, save, load, and run inference for the threat classifier."""

    def __init__(
        self,
        pipeline: Pipeline | None = None,
        metrics: TrainingMetrics | None = None,
        bucket_thresholds: tuple[float, float] = DEFAULT_BUCKET_THRESHOLDS,
    ) -> None:
        self.pipeline = pipeline
        self.metrics = metrics
        self.bucket_thresholds = bucket_thresholds

    # ---- Training ---------------------------------------------------------

    @classmethod
    def fit(
        cls,
        events: pd.DataFrame,
        *,
        grid_hours: int = 24,
        max_cells_per_county: int | None = 200,
        param_grid: dict | None = None,
        cv_folds: int = 3,
        test_size: float = 0.2,
        random_state: int = 0,
    ) -> "ThreatClassifier":
        """Build training cells, run GridSearchCV, return a fitted classifier."""
        cells = build_training_set(
            events,
            grid_hours=grid_hours,
            max_cells_per_county=max_cells_per_county,
            seed=random_state,
        )

        feature_cols = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
        X = cells[feature_cols]
        y = cells["label"].astype(int).values

        if y.sum() == 0:
            raise ValueError(
                "Training set has zero positive examples; "
                "cannot train classifier."
            )
        if y.sum() == len(y):
            raise ValueError(
                "Training set has zero negative examples; "
                "cannot train classifier."
            )

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state,
        )

        pipeline = _build_pipeline()
        grid = GridSearchCV(
            pipeline,
            param_grid=param_grid or _DEFAULT_PARAM_GRID,
            cv=cv_folds,
            scoring="roc_auc",
            n_jobs=1,  # 1 for reproducibility / lower memory in CI
        )
        # Class imbalance: severe-event base rate is low (~5-10%). Without
        # sample weighting, the model collapses to "always predict negative."
        # We use sklearn's balanced inverse-frequency weighting computed
        # from the training set.
        sample_weight = compute_sample_weight("balanced", y_tr)
        grid.fit(X_tr, y_tr, **{"clf__sample_weight": sample_weight})
        best = grid.best_estimator_

        y_pred = best.predict(X_te)
        y_proba = best.predict_proba(X_te)[:, 1]

        metrics = TrainingMetrics(
            n_train=len(X_tr),
            n_test=len(X_te),
            positive_rate_train=float(y_tr.mean()),
            positive_rate_test=float(y_te.mean()),
            accuracy=float(accuracy_score(y_te, y_pred)),
            f1=float(f1_score(y_te, y_pred, zero_division=0)),
            roc_auc=float(roc_auc_score(y_te, y_proba)),
            avg_precision=float(average_precision_score(y_te, y_proba)),
            confusion_matrix=confusion_matrix(y_te, y_pred).tolist(),
            best_params=grid.best_params_,
        )
        return cls(pipeline=best, metrics=metrics)

    # ---- Persistence ------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save pipeline + metrics + thresholds to a single .joblib file."""
        if self.pipeline is None:
            raise RuntimeError("Cannot save an unfitted classifier.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "pipeline": self.pipeline,
            "metrics": self.metrics,
            "bucket_thresholds": self.bucket_thresholds,
            "feature_order": list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES),
            "sklearn_version": sklearn.__version__,
        }
        joblib.dump(bundle, path)

    @classmethod
    def load(cls, path: str | Path) -> "ThreatClassifier":
        bundle = joblib.load(Path(path))
        return cls(
            pipeline=bundle["pipeline"],
            metrics=bundle.get("metrics"),
            bucket_thresholds=tuple(bundle.get(
                "bucket_thresholds", DEFAULT_BUCKET_THRESHOLDS
            )),
        )

    # ---- Inference --------------------------------------------------------

    def score_cell(
        self,
        events: pd.DataFrame,
        state: str,
        county_fips: str,
        observed_at: datetime,
    ) -> ThreatScore:
        """Predict P(severe in next 24h) for a single (county, time) cell."""
        if self.pipeline is None:
            raise RuntimeError("Classifier not loaded/fitted.")
        features = compute_cell_features(events, state, county_fips, observed_at)
        row = pd.DataFrame([{
            "state": features.state,
            **{f: getattr(features, f) for f in NUMERIC_FEATURES},
        }])
        proba = float(self.pipeline.predict_proba(row)[0, 1])
        return ThreatScore(
            probability=proba,
            bucket=bucket_probability(proba, self.bucket_thresholds),
            feature_summary=features.to_dict(),
        )

    def score_blank(
        self,
        state: str,
        county_fips: str,
        observed_at: datetime,
    ) -> ThreatScore:
        """Predict using only seasonal/location features, no event history.

        Used at runtime when we don't have a recent local event log loaded —
        the agent falls back to base-rate-by-season-and-place.
        """
        # Construct the empty events DataFrame with explicit datetime64[ns]
        # dtype. Newer pandas versions infer datetime64[s] for empty
        # to_datetime calls, which then fails comparison against stdlib
        # datetime in compute_cell_features. Specifying ns explicitly avoids
        # that mismatch.
        empty = pd.DataFrame({
            "BEGIN_DATE_TIME": pd.Series(dtype="datetime64[ns]"),
            "county_fips": pd.Series(dtype="object"),
            "is_severe": pd.Series(dtype="bool"),
            "damage_property_usd": pd.Series(dtype="float64"),
        })
        return self.score_cell(empty, state, county_fips, observed_at)


__all__ = [
    "ThreatBucket",
    "ThreatScore",
    "TrainingMetrics",
    "ThreatClassifier",
    "bucket_probability",
    "DEFAULT_BUCKET_THRESHOLDS",
]
