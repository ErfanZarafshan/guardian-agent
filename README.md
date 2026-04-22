# 🛡️ Guardian Agent

> **A personalized AI safety agent that turns real-time weather data into
> tailored guidance, grounded in Bayesian probabilistic reasoning.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-streamlit.app-FF4B4B?logo=streamlit)](https://guardianagent.streamlit.app/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-239%20passing-brightgreen)](#testing)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

**Course:** CSC 4444G Artificial Intelligence — Spring 2026
**Institution:** Louisiana State University, Division of Computer Science and Engineering
**Instructor:** Prof. Keith G. Mills
**Authors:** Erfan Zarafshan, Maby Gavilan Abanto

📄 **[Read the full report (PDF)](docs/report.pdf)** &nbsp;•&nbsp;
🎬 **[Try the live demo](https://guardianagent.streamlit.app/)** &nbsp;•&nbsp;
📊 **[Demo video script](docs/DEMO_VIDEO_SCRIPT.md)**

---

## What is Guardian Agent?

The U.S. Wireless Emergency Alert system buzzes every phone in a polygon
with the same text. A 78-year-old on the ground floor of a flood-prone
apartment with no car gets the same message as a 25-year-old on the third
floor with a 4WD pickup. The appropriate response is materially different.

Guardian Agent fuses three signals — **live weather data** (NWS + OWM), an
**ML threat classifier** trained on NOAA Storm Events, and a **structured
user profile** — through a **10-node Bayesian network** to produce a
*personal* posterior over four risk levels (Low/Moderate/High/Critical).
A deterministic rule planner then maps the posterior to a prioritized
list of actions: notify, shelter, evacuate, sound alarm, message
contacts, etc., each with a written rationale.

An optional LLM (Claude or GPT) rephrases the result into plain English.
The LLM cannot alter recommendations — safety-critical reasoning stays
inside the deterministic Bayesian + planner pipeline.

## Architecture

```
                         ┌──────────────────────────┐
  Weather APIs           │                          │
  (NWS + OWM) ──────────►│   Bayesian Risk Engine   │
                         │   (10-node DAG, pgmpy)   │
  ML Threat Classifier ─►│                          │
  (sklearn, NOAA)        │   exact inference via    │
                         │   Variable Elimination   │
  User Profile ─────────►│                          │
  (pydantic)             └────────────┬─────────────┘
                                      │
                                      ▼
                            ┌────────────────────┐
                            │   Rule Planner     │
                            │   8 actions × 3    │
                            │   channels with    │
                            │   audit rationale  │
                            └─────────┬──────────┘
                                      │
                       ┌──────────────┼──────────────┐
                       ▼              ▼              ▼
                  ┌─────────┐   ┌─────────┐   ┌────────────┐
                  │ Console │   │   SMS   │   │ Smart Home │
                  │  rich   │   │ Twilio  │   │   (mock)   │
                  └─────────┘   └─────────┘   └────────────┘
```

## Repository layout

```
guardian-agent/
├── src/guardian/
│   ├── agent.py              # CLI entry point (check / run)
│   ├── loop.py               # Core perceive-reason-act cycle
│   ├── profile.py            # User profile schema (pydantic)
│   ├── config.py             # Environment + API key management
│   ├── weather/              # NWS + OWM clients & aggregator
│   ├── risk/                 # Bayesian network + ML classifier
│   ├── planning/             # Action catalog + rule planner
│   └── output/               # SMS / smart-home / console dispatch
├── streamlit_app/            # Interactive web demo (5 pages)
├── scripts/                  # Training, demos, smoke tests
├── tests/                    # 239 tests across 9 files
├── models/                   # Trained classifier (.joblib)
├── data/                     # Storm Events training data + cycle logs
├── docs/                     # Architecture notes + this report
├── config/                   # Example .env + example profiles
├── requirements.txt          # Core dependencies
└── requirements-streamlit.txt
```

---

## Quick start (5 minutes)

### 1. Clone and set up the environment

```bash
git clone https://github.com/ErfanZarafshan/guardian-agent.git
cd guardian-agent
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

> **Python 3.11 or newer required.** Tested on 3.11, 3.12, and 3.13.

### 2. Configure API keys (free tiers)

Copy the example environment file and fill in your keys:

```bash
cp config/example.env .env
```

Edit `.env`:

```ini
# OpenWeatherMap free tier — https://openweathermap.org/api
OWM_API_KEY=your_owm_key_here

# NWS asks for a contact email in the User-Agent header
NWS_USER_AGENT=GuardianAgent/0.1 (your_email@example.com)

# Twilio is OPTIONAL — leave SMS_DRY_RUN=true to skip it entirely
SMS_DRY_RUN=true
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_FROM_NUMBER=
```

> ⚠️ **`SMS_DRY_RUN=true` is the default and recommended setting.** Real
> SMS is only sent when you explicitly set this to `false` AND provide
> Twilio credentials.

### 3. Build a user profile

```bash
python -m guardian.profile_cli interactive --output config/my_profile.json
```

Or copy the example:

```bash
cp config/example_profile.json config/my_profile.json
```

### 4. Verify the environment

```bash
python -m guardian.agent check --profile config/my_profile.json
```

Expected output: a config summary plus profile details. You should see
`OWM configured: True` and `SMS_DRY_RUN: True`.

---

## How to prove it works

The grader has **two avenues** to verify the project. Either works.

### Avenue A — Run it yourself (recommended)

#### Run one end-to-end agent cycle against live weather

```bash
python -m guardian.agent run \
    --profile config/my_profile.json \
    --model models/threat_classifier.joblib \
    --once
```

Expected output: a startup banner, one Cycle #1 with a posterior bar
chart, dispatched action(s), and a JSON record appended to
`data/cycles.jsonl`.

#### Run the four-scenario reasoning demo

```bash
python scripts/risk_demo.py --model models/threat_classifier.joblib
```

Shows the Bayesian engine + planner output for two contrasting user
profiles across four hazard scenarios (clear day, flash flood watch,
tornado warning, tropical storm warning). Two of the four scenarios
exhibit *argmax flips* between users — the core personalization result.

#### Launch the interactive web app locally

```bash
streamlit run streamlit_app/Home.py
```

Browser opens at `http://localhost:8501`. Five pages: Profile setup,
Live Weather, Risk Assessment, Scenario Simulator, About.

### Avenue B — Visit the deployed demo

🌐 **<https://guardianagent.streamlit.app/>**

No installation required. Click through the five pages on the left
sidebar; the **Scenario Simulator** is the most informative — pick
"Tornado Warning" or "Flash Flood Warning" and watch the posterior
react in real time.

---

## Testing

```bash
pytest -q
```

Expected: **239 tests pass**.

| Test file | Tests | Component |
|---|---|---|
| `test_smoke.py` | 32 | Module import smoke test |
| `test_profile.py` | 27 | User-profile schema |
| `test_weather.py` | 41 | NWS + OWM clients |
| `test_classifier.py` | 33 | ML threat classifier |
| `test_bayesian.py` | 21 | Bayesian network construction |
| `test_risk_engine.py` | 14 | End-to-end engine integration |
| `test_planner.py` | 37 | Rule planner |
| `test_output.py` | 19 | SMS / smart-home / console dispatchers |
| `test_loop.py` | 15 | Main agent perceive-reason-act loop |

---

## Key results

- **Trained classifier:** ROC-AUC = **0.759** on 21,281 held-out
  county-day cells (training set: 85,120 cells from 106k NOAA Storm
  Events, 5 Gulf Coast states, 2018–2024).
- **Personalization:** Same hazard, different users → different argmax
  in 2 of 4 demo scenarios (Flash Flood Watch and Tropical Storm
  Warning).
- **Dedup behavior verified:** 3-cycle continuous run with 60-second
  polling correctly suppresses duplicate non-console actions while
  maintaining a console heartbeat.
- **All 239 tests pass** on Python 3.13 / macOS.

---

## Documentation

- 📄 [`docs/report.pdf`](docs/report.pdf) — Full ICML-formatted report
- 🎬 [`docs/DEMO_VIDEO_SCRIPT.md`](docs/DEMO_VIDEO_SCRIPT.md) — Demo screencast script
- 🏗️ [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — Architecture deep-dive

---

## Limitations

The honest limits, also called out in the report:

- **The classifier uses calendar+location features only**, not radar
  reflectivity. Mature meteorological nowcasting systems hit
  ROC-AUC 0.85–0.95 with radar; ours sits at 0.76. The Bayesian fusion
  layer compensates partially by leaning on NWS alert evidence.
- **Smart-home dispatch is mocked.** A production deployment would swap
  the mock for a real Alexa Skills Kit integration via the same
  `SmartHomeDispatcher` interface.
- **The deployed web demo always operates in SMS dry-run mode** — random
  visitors should not be able to trigger Twilio sends on the developer's
  account.
- **Profiles are not persisted between visits.** Each browser session is
  fresh.

---

## Acknowledgements

We thank Prof. Keith G. Mills for guidance throughout the semester and
for reviewing the project proposal. We thank the National Weather
Service and OpenWeatherMap for providing free public weather APIs, and
NOAA for maintaining the Storm Events Database used to train the
classifier.

## License

MIT — see [`LICENSE`](LICENSE) for details.
