# Guardian Agent

Personalized weather-hazard safety agent for CSC 4444G (Spring 2026, LSU).

Live demo: https://guardianagent.streamlit.app/

## Demo Video
A walkthrough of the system is available at: https://www.youtube.com/watch?v=x_1lZ8YvUxU

## Authors

- Erfan Zarafshan
- Maby Gavilan Abanto

Course: CSC 4444G Artificial Intelligence, Spring 2026, LSU.
Instructor: Prof. Keith G. Mills.

## What it does

Most weather emergency alerts (like the WEA messages that come to your phone) send the same text to everyone in an area. This project tries to do better by combining the live weather data with a profile of the user (where they live, what floor, whether they have a vehicle, medical needs) and producing a risk level and a list of recommended actions that are specific to that person.

The reasoning is done by a Bayesian network with 10 nodes (built with pgmpy). Two of the inputs to the network are real-time weather (NWS alerts and OpenWeatherMap conditions) and the output of a gradient-boosted classifier we trained on NOAA Storm Events data. The other inputs come from the user profile.

After the network produces a posterior over four risk levels (Low, Moderate, High, Critical), a rule-based planner picks actions from a fixed set (notify the user, recommend shelter, recommend evacuation, contact emergency contacts by SMS, sound a smart-home alarm, etc.). Each action is sent to the right output channel: console, Twilio SMS, or a mock smart-home dispatcher.

There is also an optional plain-English explainer that uses an LLM (Claude or ChatGPT) to rephrase the result. The LLM only rephrases; it does not change the recommendations.

## Repo layout

```
src/guardian/
  agent.py        CLI entry point
  loop.py         Main perceive-reason-act loop
  profile.py      User profile schema (pydantic)
  config.py       Loads .env
  weather/        NWS + OpenWeatherMap clients
  risk/           Bayesian network + ML classifier
  planning/       Action catalog + rule planner
  output/         SMS / smart-home / console dispatchers

streamlit_app/    Web demo (5 pages)
scripts/          Training, demo scripts, smoke tests
tests/            239 tests, 9 files
models/           Trained classifier (.joblib)
data/             Storm Events training data, cycle logs
docs/             Report (.tex + .pdf), architecture notes
config/           Example .env, example profiles
```

## Setup

Requires Python 3.11 or newer (we developed on 3.13).

```
git clone https://github.com/ErfanZarafshan/guardian-agent.git
cd guardian-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the example env file and add your API keys:

```
cp config/example.env .env
```

Then open `.env` and set:

- `OWM_API_KEY` from https://openweathermap.org/api (free tier)
- `NWS_USER_AGENT` to something like `GuardianAgent/0.1 (your_email@example.com)` (NWS asks for a contact email)
- Leave `SMS_DRY_RUN=true` unless you set up Twilio (which is optional)

Build a profile (interactive prompts) or copy the example:

```
python -m guardian.profile_cli interactive --output config/my_profile.json
```

or

```
cp config/example_profile.json config/my_profile.json
```

Check that everything is wired up:

```
python -m guardian.agent check --profile config/my_profile.json
```

## How to run / verify

The simplest way to see the project working is the deployed web app: https://guardianagent.streamlit.app/. The Scenario Simulator page is the most informative: pick a hazard like Tornado Warning or Flash Flood Warning and the posterior updates immediately. Switch profiles and the same hazard produces different recommendations.

To run locally:

**One agent cycle against live weather** (this hits the real NWS and OpenWeatherMap APIs):

```
python -m guardian.agent run \
    --profile config/my_profile.json \
    --model models/threat_classifier.joblib \
    --once
```

**Reasoning demo** with two contrasting profiles across four hazard scenarios:

```
python scripts/risk_demo.py --model models/threat_classifier.joblib
```

In two of the four scenarios (Flash Flood Watch and Tropical Storm Warning), the two profiles end up at different risk levels even though the weather is identical. This was the main thing we wanted to show.

**Web app locally:**

```
streamlit run streamlit_app/Home.py
```

Opens at http://localhost:8501.

## Tests

```
pytest -q
```

Should report 239 passed.

| Test file | Tests | What it covers |
|---|---|---|
| test_smoke.py | 32 | Imports |
| test_profile.py | 27 | Profile schema |
| test_weather.py | 41 | NWS + OWM clients |
| test_classifier.py | 33 | ML classifier |
| test_bayesian.py | 21 | Bayesian network construction |
| test_risk_engine.py | 14 | Engine integration |
| test_planner.py | 37 | Planner |
| test_output.py | 19 | SMS / smart-home / console dispatch |
| test_loop.py | 15 | Main loop |

## Results

- Classifier trained on 85,120 county-day cells from 5 Gulf Coast states (LA, TX, MS, AL, FL), 2018-2024. Test set ROC-AUC = 0.759 on 21,281 held-out cells.
- Personalization works: same hazard, different profiles, different argmax in 2 of the 4 demo scenarios.
- Continuous-mode dedup verified: a 3-cycle run at 60-second intervals correctly suppresses duplicate non-console actions when the world hasn't changed.
- All 239 tests pass on Python 3.13.

## Limitations

A few things we want to be upfront about:

- The classifier only uses calendar features (month, day-of-year, county FIPS) and a 30-day rolling event count. It does not use radar. Real meteorological nowcasting systems with radar reach AUC 0.85-0.95; we get 0.76.
- The smart-home dispatcher is a mock. The interface is set up the way an Alexa Skills Kit integration would need it, but we did not actually deploy a skill.
- The deployed web app always operates in SMS dry-run mode so that random visitors cannot trigger Twilio sends on our account.
- User profiles are not persisted between visits to the deployed site.

## Documentation

- `docs/report.pdf`: full project report (ICML format)
- `docs/report.tex`, `docs/refs.bib`: LaTeX source
- `docs/ARCHITECTURE.md`: more detail on each module
- `docs/REPORT_BUILD.md`: how to rebuild the report PDF
- `docs/DEMO_VIDEO_SCRIPT.md`: script for the demo video

## License

MIT.
