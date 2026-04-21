"""Scenario Simulator — override the real weather with hypothetical hazards.

This is the killer demo feature. In a presentation you can show how the
agent's posterior shifts under different scenarios without waiting for real
weather. Pick a preset (Tornado Warning, Flash Flood, etc.) or build a
custom evidence set with sliders.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
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
from guardian.weather.observation import (
    CertaintyLevel,
    SeverityLevel,
    UrgencyLevel,
    WeatherAlert,
    WeatherObservation,
)


st.set_page_config(page_title="Scenario Simulator — Guardian Agent",
                   page_icon="🎮", layout="wide")
state.ensure_defaults()

st.title("🎮 Scenario Simulator")
st.caption(
    "Don't want to wait for real weather to test the agent? Pick a preset "
    "scenario or dial up a custom one. The Bayesian engine and planner run "
    "the same way as on real data."
)


profile = state.get(state.PROFILE)
if profile is None:
    st.warning("⚠️ No profile loaded. Visit the **Home** page first.")
    st.stop()


# ---------------------------------------------------------------------------
# Scenario presets
# ---------------------------------------------------------------------------

PRESETS = {
    "Clear day": dict(
        wind_mph=5.0, precip_in_hr=0.0,
        alert_event=None, severity="None", urgency="Unknown",
    ),
    "Heat advisory (summer afternoon)": dict(
        wind_mph=8.0, precip_in_hr=0.0,
        alert_event="Excessive Heat Warning", severity="Moderate", urgency="Expected",
    ),
    "Flash Flood Watch": dict(
        wind_mph=15.0, precip_in_hr=0.45,
        alert_event="Flash Flood Watch", severity="Moderate", urgency="Expected",
    ),
    "Flash Flood WARNING (severe)": dict(
        wind_mph=20.0, precip_in_hr=1.2,
        alert_event="Flash Flood Warning", severity="Severe", urgency="Immediate",
    ),
    "Severe Thunderstorm": dict(
        wind_mph=45.0, precip_in_hr=0.7,
        alert_event="Severe Thunderstorm Warning", severity="Severe", urgency="Expected",
    ),
    "Tornado Warning": dict(
        wind_mph=42.0, precip_in_hr=0.6,
        alert_event="Tornado Warning", severity="Extreme", urgency="Immediate",
    ),
    "Tropical Storm Warning": dict(
        wind_mph=58.0, precip_in_hr=1.2,
        alert_event="Tropical Storm Warning", severity="Severe", urgency="Expected",
    ),
    "Hurricane Warning": dict(
        wind_mph=85.0, precip_in_hr=2.0,
        alert_event="Hurricane Warning", severity="Extreme", urgency="Immediate",
    ),
}


with st.sidebar:
    st.markdown(f"**Profile:** {profile.name}")
    st.divider()
    preset_label = st.selectbox(
        "Scenario preset",
        options=list(PRESETS.keys()),
        index=0,
        help="Pick a preset, or override fields below to customize.",
    )
preset = PRESETS[preset_label]


# ---------------------------------------------------------------------------
# Custom overrides
# ---------------------------------------------------------------------------

st.markdown("### Hazard parameters")
cols = st.columns(2)
with cols[0]:
    wind_mph = st.slider(
        "Sustained wind (mph)",
        min_value=0.0, max_value=120.0,
        value=float(preset["wind_mph"]), step=1.0,
    )
    wind_gust_mph = st.slider(
        "Wind gust (mph)",
        min_value=0.0, max_value=160.0,
        value=float(preset["wind_mph"]) + 10.0, step=1.0,
        help="Used in addition to sustained wind. Set equal to sustained for "
             "no gusts.",
    )
with cols[1]:
    precip_in_hr = st.slider(
        "Precipitation rate (inches/hour)",
        min_value=0.0, max_value=4.0,
        value=float(preset["precip_in_hr"]), step=0.05,
    )
    temp_f = st.slider(
        "Temperature (°F)",
        min_value=-20.0, max_value=120.0, value=82.0, step=1.0,
    )

st.markdown("### Active alert")
cols = st.columns([2, 1, 1])
with cols[0]:
    alert_event = st.text_input(
        "Alert event name (blank = no alert)",
        value=preset["alert_event"] or "",
        help="e.g. 'Tornado Warning', 'Flash Flood Watch', 'Excessive Heat Warning'",
    )
with cols[1]:
    severity_options = [s.value for s in SeverityLevel]
    severity = st.selectbox(
        "Severity",
        options=severity_options,
        index=severity_options.index(preset["severity"]),
    )
with cols[2]:
    urgency_options = [u.value for u in UrgencyLevel]
    urgency = st.selectbox(
        "Urgency",
        options=urgency_options,
        index=urgency_options.index(preset["urgency"]),
    )


# ---------------------------------------------------------------------------
# Build observation + run engine
# ---------------------------------------------------------------------------

now = datetime.now(timezone.utc)
alerts: list[WeatherAlert] = []
if alert_event.strip():
    alerts.append(WeatherAlert(
        source="nws",
        event=alert_event.strip(),
        severity=SeverityLevel(severity),
        urgency=UrgencyLevel(urgency),
        certainty=CertaintyLevel.LIKELY,
    ))

simulated_obs = WeatherObservation(
    observed_at=now,
    latitude=profile.location.latitude,
    longitude=profile.location.longitude,
    temperature_f=temp_f,
    wind_speed_mph=wind_mph,
    wind_gust_mph=wind_gust_mph,
    precip_rate_in_hr=precip_in_hr,
    sources=["nws", "owm"],   # pretend both fed in
    nws_zone_id=profile.location.nws_zone_id,
    alerts=alerts,
)
state.set_(state.SIMULATED_OBSERVATION, simulated_obs)

engine = get_engine()
result = engine.assess(simulated_obs, profile)
actions = plan_actions(result.assessment.argmax, profile, simulated_obs)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

st.markdown("---")
cols = st.columns([2, 1])
with cols[0]:
    render_posterior(
        result.assessment.posterior,
        title=f"Risk for {profile.name} under '{preset_label}' (custom)",
    )
with cols[1]:
    st.markdown("### Evidence used")
    ev = result.assessment.evidence
    st.markdown(
        f"- Hazard severity: `{ev['HazardSeverity']}`\n"
        f"- Urgency: `{ev['Urgency']}`\n"
        f"- Wind: `{ev['WindCategory']}`\n"
        f"- Precipitation: `{ev['PrecipCategory']}`\n"
        f"- ML emerging threat: `{ev['EmergingThreat']}`\n"
        f"- Home floor: `{ev['HomeFloor']}`\n"
        f"- Vehicle clearance: `{ev['VehicleClearance']}`\n"
        f"- Mobility limited: `{ev['MobilityLimited']}`"
    )
    if result.threat_score is not None:
        st.caption(
            f"_Classifier P(severe in 24h) = "
            f"{result.threat_score.probability:.1%}_"
        )


st.markdown("---")
st.subheader(f"📋 Planned actions ({len(actions)})")
render_actions(actions)


# ---------------------------------------------------------------------------
# Optional LLM
# ---------------------------------------------------------------------------

st.markdown("---")
provider = state.get(state.LLM_PROVIDER, "off")
api_key = state.get(state.LLM_API_KEY, "")

if provider != "off" and api_key:
    if st.button("✨ Explain this scenario in plain English"):
        with st.spinner(f"Asking {provider}…"):
            llm_result = explain(
                provider=provider, api_key=api_key,
                profile=profile, observation=simulated_obs,
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
