"""Streamlit session state helpers.

Streamlit reruns the script on every interaction. To preserve the user's
profile, last assessment, and preferences across page navigations, we use
st.session_state. This module centralizes all the keys so we don't sprinkle
magic strings throughout the UI files.
"""

from __future__ import annotations

import streamlit as st


# Keys
PROFILE = "profile"                  # UserProfile or None
LAST_OBSERVATION = "last_observation"  # WeatherObservation or None
LAST_ENGINE_RESULT = "last_engine_result"  # EngineResult or None
SIMULATED_OBSERVATION = "simulated_observation"  # WeatherObservation or None
LLM_PROVIDER = "llm_provider"        # "anthropic" | "openai" | "off"
LLM_API_KEY = "llm_api_key"          # str
SHARED_ENGINE = "_shared_engine"     # cached RiskEngine (singleton-ish)


def get(key: str, default=None):
    return st.session_state.get(key, default)


def set_(key: str, value) -> None:
    st.session_state[key] = value


def clear(key: str) -> None:
    if key in st.session_state:
        del st.session_state[key]


def ensure_defaults() -> None:
    """Initialize session state on first page load."""
    defaults = {
        PROFILE: None,
        LAST_OBSERVATION: None,
        LAST_ENGINE_RESULT: None,
        SIMULATED_OBSERVATION: None,
        LLM_PROVIDER: "off",
        LLM_API_KEY: "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def has_profile() -> bool:
    return get(PROFILE) is not None
