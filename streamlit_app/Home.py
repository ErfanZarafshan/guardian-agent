"""Guardian Agent — Streamlit demo entry page.

Sets up the page config, lets the user pick or build a profile, and stores
it in session state. Other pages read st.session_state[PROFILE].

Run locally:
    streamlit run streamlit_app/Home.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `guardian` importable when running `streamlit run streamlit_app/Home.py`
# from the repo root. Without this, Streamlit's process can't find src/.
_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_THIS.parent) not in sys.path:
    sys.path.insert(0, str(_THIS.parent))

import streamlit as st

import state
from cached_engine import get_classifier_metrics_summary, get_engine
from components.profile_form import EXAMPLES, render_profile_form


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Guardian Agent — Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

state.ensure_defaults()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🛡️ Guardian Agent")
st.caption(
    "A context-aware AI agent that turns real-time weather data into "
    "personalized safety guidance. CSC 4444G, Spring 2026, LSU."
)

with st.sidebar:
    st.header("How this works")
    st.markdown("""
    1. **Set up a profile** on this page (your home, vehicle, medical needs).
    2. **Live Weather**: see current conditions at your location.
    3. **Risk Assessment**: the Bayesian engine + ML classifier compute your
       personal risk and the planner picks actions.
    4. **Scenario Simulator**: try hypothetical hazards without waiting for
       real weather.
    5. **About**: methodology + GitHub link.
    """)
    st.divider()
    st.subheader("LLM explainer (optional)")
    state.set_(state.LLM_PROVIDER, st.selectbox(
        "Provider",
        options=["off", "anthropic", "openai"],
        index=["off", "anthropic", "openai"].index(state.get(state.LLM_PROVIDER, "off")),
        help="If set, the Risk Assessment page can rephrase results in plain English.",
    ))
    if state.get(state.LLM_PROVIDER) != "off":
        state.set_(state.LLM_API_KEY, st.text_input(
            "API key", type="password",
            value=state.get(state.LLM_API_KEY, ""),
            help="Stored in session memory only; not persisted.",
        ))
    st.divider()
    metrics = get_classifier_metrics_summary()
    if metrics:
        st.success(f"ML classifier loaded:\n\n{metrics}")
    else:
        st.warning(
            "No trained classifier loaded. Bayesian engine still works, "
            "but the EmergingThreat node defaults to Low."
        )


# ---------------------------------------------------------------------------
# Profile setup
# ---------------------------------------------------------------------------

st.header("1. Build your profile")

st.markdown(
    "Pick an example to start with, or fill in the form below from scratch. "
    "Your profile drives the personalization in the rest of the demo — a "
    "ground-floor apartment with a mobility-limited resident gets very "
    "different recommendations than an elevated home with a 4WD pickup."
)

example_label = st.selectbox(
    "Pre-built examples",
    options=["(start from scratch)"] + list(EXAMPLES.keys()),
    index=1,  # default to baseline example
    help="You can edit any field after picking an example.",
)
prefill = EXAMPLES.get(example_label)

st.markdown("---")
profile = render_profile_form(prefilled=prefill)

if profile is not None:
    cols = st.columns([1, 1, 4])
    with cols[0]:
        if st.button("💾 Save profile", type="primary", use_container_width=True):
            state.set_(state.PROFILE, profile)
            # Clear any stale assessment from a previous profile.
            state.set_(state.LAST_OBSERVATION, None)
            state.set_(state.LAST_ENGINE_RESULT, None)
            state.set_(state.SIMULATED_OBSERVATION, None)
            st.success(
                f"Profile saved for **{profile.name}**. "
                f"Now visit the **Live Weather** or **Scenario Simulator** "
                f"page in the left sidebar."
            )
    with cols[1]:
        if state.has_profile() and st.button("🗑️ Clear", use_container_width=True):
            state.set_(state.PROFILE, None)
            state.set_(state.LAST_OBSERVATION, None)
            state.set_(state.LAST_ENGINE_RESULT, None)
            state.set_(state.SIMULATED_OBSERVATION, None)
            st.rerun()


# ---------------------------------------------------------------------------
# Footer / status
# ---------------------------------------------------------------------------

st.markdown("---")
if state.has_profile():
    p = state.get(state.PROFILE)
    st.success(
        f"✅ Active profile: **{p.name}** "
        f"(home: {p.home_floor_state.value}, "
        f"vehicle: {p.vehicle.clearance.value}, "
        f"medically vulnerable: {p.is_medically_vulnerable})"
    )
else:
    st.info("👆 Save a profile above to enable the other pages.")

# Pre-warm the engine cache on first load so subsequent page navigations are
# fast. Safe to do unconditionally — get_engine() is cached.
_ = get_engine()
