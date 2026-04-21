"""Live Weather page — fetches real NWS + OWM data for the user's location.

This is the "perceive" half of the agent, made visible. The user sees what
the agent sees before reasoning happens.
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
from guardian.weather.aggregator import observe
from guardian.weather.http import WeatherAPIError


st.set_page_config(page_title="Live Weather — Guardian Agent",
                   page_icon="🌦️", layout="wide")
state.ensure_defaults()

st.title("🌦️ Live Weather")
st.caption(
    "Real-time observations for your profile's location, pulled from "
    "NOAA's National Weather Service (alerts) and OpenWeatherMap (current "
    "conditions)."
)

profile = state.get(state.PROFILE)
if profile is None:
    st.warning("⚠️ No profile loaded yet. Go to the **Home** page first.")
    st.stop()


with st.sidebar:
    st.markdown(f"**Profile:** {profile.name}")
    st.markdown(f"**Location:** {profile.location.address}")
    st.divider()
    skip_owm = st.checkbox(
        "Skip OpenWeatherMap",
        value=False,
        help="Use this if your OWM key isn't activated yet (NWS alone gives "
             "alerts but no current conditions).",
    )


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

if st.button("🔄 Fetch live weather now", type="primary"):
    with st.spinner("Calling NWS and OpenWeatherMap..."):
        try:
            obs = observe(
                latitude=profile.location.latitude,
                longitude=profile.location.longitude,
                nws_zone_id=profile.location.nws_zone_id,
                skip_owm=skip_owm,
            )
            state.set_(state.LAST_OBSERVATION, obs)
            # Invalidate any prior assessment since the world changed.
            state.set_(state.LAST_ENGINE_RESULT, None)
            st.success(f"Fetched data from: {', '.join(obs.sources) or '(none)'}")
        except WeatherAPIError as e:
            st.error(f"Weather fetch failed: {e}")
        except ValueError as e:
            st.error(f"Both NWS and OWM failed: {e}")


obs = state.get(state.LAST_OBSERVATION)
if obs is None:
    st.info("👆 Click the button above to fetch the latest weather.")
    st.stop()


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Current conditions")

cols = st.columns(4)
with cols[0]:
    st.metric(
        "Temperature",
        f"{obs.temperature_f:.0f} °F" if obs.temperature_f is not None else "—",
    )
with cols[1]:
    st.metric(
        "Wind",
        f"{obs.wind_speed_mph:.0f} mph" if obs.wind_speed_mph is not None else "—",
        delta=f"category: {obs.wind_category.value}",
        delta_color="off",
    )
with cols[2]:
    st.metric(
        "Precipitation",
        f"{obs.precip_rate_in_hr:.2f} in/hr" if obs.precip_rate_in_hr is not None else "0",
        delta=f"category: {obs.precip_category.value}",
        delta_color="off",
    )
with cols[3]:
    st.metric(
        "Humidity",
        f"{obs.humidity_pct:.0f}%" if obs.humidity_pct is not None else "—",
    )

st.markdown(
    f"**Observed at (UTC):** {obs.observed_at}  \n"
    f"**Sources:** {', '.join(obs.sources) or '(none)'}  \n"
    f"**NWS zone:** {obs.nws_zone_id or '(unresolved)'}"
)


# ---------------------------------------------------------------------------
# Active alerts
# ---------------------------------------------------------------------------

st.subheader("Active alerts")

active_alerts = [a for a in obs.alerts if a.is_active(obs.observed_at)]
if not active_alerts:
    st.success("✅ No active weather alerts for this location.")
else:
    for alert in active_alerts:
        with st.container(border=True):
            st.markdown(f"### 🚨 {alert.event}")
            st.markdown(
                f"**Severity:** {alert.severity.value}  \n"
                f"**Urgency:** {alert.urgency.value}  \n"
                f"**Certainty:** {alert.certainty.value}  \n"
                f"**Source:** {alert.sender or 'NWS'}"
            )
            if alert.headline:
                st.markdown(f"**Headline:** {alert.headline}")
            if alert.description:
                with st.expander("Full description"):
                    st.text(alert.description)


# ---------------------------------------------------------------------------
# Next step
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown(
    "👉 Now jump to **Risk Assessment** in the sidebar to run the Bayesian "
    "engine and planner against this observation."
)
