"""Cached construction of the heavyweight Guardian objects.

Building the Bayesian network and loading the joblib classifier is slow
(~1-2s). We cache them with @st.cache_resource so they're built once per
Streamlit server process, not once per user interaction.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from guardian.risk.classifier import ThreatClassifier
from guardian.risk.risk_engine import RiskEngine


# Repository root — needed to find models/ at runtime regardless of CWD.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "threat_classifier.joblib"


@st.cache_resource(show_spinner="Loading Bayesian network and classifier…")
def get_engine() -> RiskEngine:
    """Build the RiskEngine once per server process.

    If a trained classifier exists at models/threat_classifier.joblib, it's
    loaded; otherwise the engine runs without an emerging-threat signal
    (EmergingThreat=Low).
    """
    classifier = None
    model_path = Path(os.environ.get("GUARDIAN_MODEL_PATH", DEFAULT_MODEL_PATH))
    if model_path.exists():
        try:
            classifier = ThreatClassifier.load(model_path)
        except Exception as e:
            st.warning(f"Could not load classifier at {model_path}: {e}")
    return RiskEngine(classifier=classifier)


def get_classifier_metrics_summary() -> str | None:
    """If a classifier is loaded, return a one-line metrics summary."""
    engine = get_engine()
    if engine.classifier is None or engine.classifier.metrics is None:
        return None
    m = engine.classifier.metrics
    return (
        f"ROC-AUC = {m.roc_auc:.3f} on {m.n_test:,} held-out cells "
        f"(trained on {m.n_train:,})"
    )
