"""About page — methodology, references, and links."""

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
from cached_engine import get_classifier_metrics_summary


st.set_page_config(page_title="About — Guardian Agent",
                   page_icon="📖", layout="wide")
state.ensure_defaults()


st.title("📖 About Guardian Agent")
st.caption("CSC 4444G Artificial Intelligence — Spring 2026 — Louisiana State University")


st.markdown("""
## What this is

Guardian Agent is an intelligent, context-aware agent that continuously
monitors real-time weather and environmental hazard data to provide
*personalized*, actionable safety guidance. Existing systems like the
Wireless Emergency Alerts that buzz your phone are reactive and generic —
the same Tornado Warning is broadcast to everyone in a polygon, regardless
of whether the recipient lives on the ground floor of a flood-prone
apartment or in a third-story unit with storm shutters.

Guardian Agent addresses this gap by combining **probabilistic reasoning**
with **user-specific knowledge**, shifting emergency preparedness from
broadcast alarms to individualized guidance before, during, and after a
hazardous event.
""")


st.markdown("## Architecture")
st.markdown("""
The agent is a goal-based agent in the sense of Russell & Norvig (4e), Ch. 2.
Its perceive → reason → act cycle threads through six components:

```
Weather APIs (NWS + OWM) ──┐
                           ├─► Bayesian Risk Engine ─► Planner ─► Dispatchers
NOAA Storm Events ──► ML ──┘                                       │
                                                                   ├─ SMS (Twilio)
User Profile  ─────────────────────────────────────────────────────┤
                                                                   └─ Smart home (mock)
```
""")

st.markdown("""
**Reasoning core.** A Bayesian network with 10 discrete nodes connected as
a polytree, with conditional probability tables specified by domain-knowledge
scoring functions. Inference is exact, via Variable Elimination (Russell &
Norvig, Ch. 13.4.1).

**ML threat classifier.** A scikit-learn Gradient Boosting Classifier
trained on the NOAA Storm Events Database (2018–2024, Gulf Coast states,
~106k events). Selected by GridSearchCV with stratified cross-validation;
trained with class-balanced sample weighting to address the ~1% positive
base rate.

**Action planner.** Deterministic rules. The argmax of the Bayesian
posterior plus user profile factors map to a fixed catalog of 8 action
kinds across 3 dispatch channels (console, SMS, smart-home). Every action
carries a rationale field so its selection is post-hoc auditable.

**LLM enhancement (optional).** When configured, an LLM (Anthropic Claude
or OpenAI GPT) re-phrases the system's output into plain-English
guidance. The LLM does *not* alter the recommendation logic; the
safety-critical reasoning remains fully grounded in the deterministic
Bayesian + planner pipeline.
""")

metrics = get_classifier_metrics_summary()
if metrics:
    st.success(f"Currently loaded ML classifier: {metrics}")


st.markdown("## Limitations")
st.markdown("""
- **Smart-home integration is mocked.** A production deployment would
  swap in a real Alexa Skills Kit or Google Home Actions implementation
  via the same `SmartHomeDispatcher` interface.
- **SMS dispatch is dry-run only on this demo site.** This is an
  intentional safety choice — random web visitors should not be able to
  trigger real SMS sends on the developer's Twilio account.
- **The classifier uses calendar + location features only**, not radar or
  atmospheric soundings. Real meteorological nowcasting models with those
  inputs hit ROC-AUC 0.85–0.95; ours sits at 0.76. Honest tradeoff for a
  semester project.
- **Profiles are not persisted** between visits to this site. Each session
  is fresh.
""")


st.markdown("## References")
st.markdown("""
1. NOAA Storm Events Database — https://www.ncdc.noaa.gov/stormevents/
2. NWS API — https://www.weather.gov/documentation/services-web-api
3. OpenWeatherMap API — https://openweathermap.org/api
4. Russell & Norvig, *Artificial Intelligence: A Modern Approach* (4th ed.)
5. Ankan & Panda, "pgmpy: Probabilistic Graphical Models Using Python,"
   *SciPy 2015*.
6. Pedregosa et al., "Scikit-learn: Machine Learning in Python,"
   *JMLR* 12, 2011.
""")


st.markdown("## Source code")
st.markdown(
    "📦 [github.com/ErfanZarafshan/guardian-agent]"
    "(https://github.com/ErfanZarafshan/guardian-agent)"
)
