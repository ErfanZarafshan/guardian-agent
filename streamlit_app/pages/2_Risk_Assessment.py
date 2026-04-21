"""Risk Assessment page — runs the engine on the live observation.

Shows the Bayesian posterior, the planned actions, and optionally an LLM
plain-English explanation. This is the centerpiece of the demo.
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[2]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_THIS.parent.parent) not in sys.path:
    sys.path.insert(0, str(_THIS.parent.parent))

import streamlit as st

import state
from cached_engine import get_engine
from components.action_card import render_actions
from components.llm import explain
from components.posterior_chart import render_posterior
from guardian.planning import plan_actions


st.set_page_config(page_title="Risk Assessment — Guardian Agent",
                   page_icon="🧠", layout="wide")
state.ensure_defaults()

st.title("🧠 Risk Assessment")
st.caption(
    "The Bayesian Risk Engine fuses the live weather evidence, the ML "
    "classifier's emerging-threat signal, and your profile's vulnerability "
    "factors into a posterior over four risk levels. The rule planner then "
    "picks an ordered list of actions."
)


# ---------------------------------------------------------------------------
# Pre-conditions
# ---------------------------------------------------------------------------

profile = state.get(state.PROFILE)
if profile is None:
    st.warning("⚠️ No profile loaded. Visit the **Home** page first.")
    st.stop()

obs = state.get(state.LAST_OBSERVATION)
if obs is None:
    st.warning(
        "⚠️ No weather observation yet. Visit **Live Weather** to fetch one, "
        "or use the **Scenario Simulator** for hypothetical weather."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Run engine
# ---------------------------------------------------------------------------

engine = get_engine()

with st.spinner("Running Bayesian inference and planning…"):
    result = engine.assess(obs, profile)
    actions = plan_actions(result.assessment.argmax, profile, obs)
state.set_(state.LAST_ENGINE_RESULT, result)


# ---------------------------------------------------------------------------
# Top: posterior chart + key facts
# ---------------------------------------------------------------------------

cols = st.columns([2, 1])
with cols[0]:
    render_posterior(result.assessment.posterior, title=f"Risk for {profile.name}")

with cols[1]:
    st.markdown("### Evidence used")
    ev = result.assessment.evidence
    st.markdown(
        f"- **Hazard severity:** `{ev['HazardSeverity']}`\n"
        f"- **Urgency:** `{ev['Urgency']}`\n"
        f"- **Wind:** `{ev['WindCategory']}`\n"
        f"- **Precipitation:** `{ev['PrecipCategory']}`\n"
        f"- **ML emerging threat:** `{ev['EmergingThreat']}`\n"
        f"- **Home floor:** `{ev['HomeFloor']}`\n"
        f"- **Vehicle clearance:** `{ev['VehicleClearance']}`\n"
        f"- **Mobility limited:** `{ev['MobilityLimited']}`"
    )
    if result.threat_score is not None:
        st.caption(
            f"_Classifier P(severe in 24h) = "
            f"{result.threat_score.probability:.1%} → "
            f"bucket {result.threat_score.bucket.value}_"
        )


# ---------------------------------------------------------------------------
# Planned actions
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader(f"📋 Planned actions ({len(actions)})")
st.caption(
    "Each action carries a rationale explaining why the planner selected it. "
    "Actions are sorted by priority — most urgent first. SMS recipients are "
    "shown but **never actually messaged from this demo site**."
)
render_actions(actions)


# ---------------------------------------------------------------------------
# Optional LLM explanation
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("💬 Plain-English explanation (optional)")

provider = state.get(state.LLM_PROVIDER, "off")
api_key = state.get(state.LLM_API_KEY, "")

if provider == "off":
    st.info(
        "Turn on an LLM provider in the sidebar of the Home page to enable "
        "a conversational explanation of this assessment. The recommendation "
        "logic itself is fully decided by the Bayesian network and rule "
        "planner — the LLM only re-phrases."
    )
elif not api_key:
    st.warning(
        f"Provider set to **{provider}** but no API key entered. "
        "Add your key in the Home page sidebar."
    )
else:
    if st.button("✨ Explain in plain English", type="primary"):
        with st.spinner(f"Asking {provider}…"):
            llm_result = explain(
                provider=provider,
                api_key=api_key,
                profile=profile,
                observation=obs,
                risk_argmax=result.assessment.argmax,
                posterior=dict(result.assessment.posterior),
                actions=actions,
            )
        if llm_result.ok:
            with st.container(border=True):
                st.markdown(llm_result.text)
                st.caption(f"_Generated by {llm_result.provider}/{llm_result.model}_")
        else:
            st.error(f"LLM call failed: {llm_result.error}")
