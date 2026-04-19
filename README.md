# Guardian Agent

A context-aware intelligent agent that monitors real-time weather and environmental
hazard data and provides personalized, actionable safety guidance based on user
vulnerability factors (home type, vehicle capabilities, medical needs, contacts).

**Course:** CSC 4444G Artificial Intelligence — Spring 2026 — Louisiana State University

## Architecture

```
Weather & Hazard APIs (NWS, OWM) ──┐
                                   ├─► Bayesian Risk Engine ─► Planner ─► Outputs
Historical data ─► ML Classifier ──┘                                       │
                                                                           ├─ SMS (Twilio)
User Profile Store ────────────────────────────────────────────────────────┤
                                                                           └─ Smart home (mock)
```

Core model: **goal-based intelligent agent** (AIMA Ch. 2), reasoning under
uncertainty with a **Bayesian network** (AIMA Ch. 13, pgmpy), augmented by a
supervised **Gradient Boosting classifier** trained on the NOAA Storm Events
Database for emerging-threat detection.

## Repository layout

```
guardian-agent/
├── src/guardian/           # Main package
│   ├── profile.py          # User profile schema + storage
│   ├── weather/            # NWS + OpenWeatherMap clients
│   ├── risk/               # Bayesian network + ML classifier
│   ├── planning/           # Rule-augmented action planner
│   ├── output/             # SMS + smart-home dispatchers
│   └── agent.py            # Main perceive-reason-act loop
├── scripts/                # Standalone utilities (data download, training)
├── notebooks/              # Exploratory data analysis
├── data/                   # Raw + processed datasets (gitignored)
├── models/                 # Serialized classifier + Bayesian network
├── config/                 # Example .env and profile templates
├── tests/                  # pytest suite
└── docs/                   # Architecture + design notes
```

## Setup

### 1. Clone and create virtual environment

```bash
git clone https://github.com/YOUR_USERNAME/guardian-agent.git
cd guardian-agent
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                 # install guardian package in editable mode
```

### 2. Configure API keys

```bash
cp config/example.env .env
# Then edit .env and fill in OWM_API_KEY, TWILIO_ACCOUNT_SID, etc.
```

### 3. Create your user profile

```bash
cp config/example_profile.json config/my_profile.json
# Edit my_profile.json to reflect your home, vehicle, medical, contacts
```

### 4. Run the smoke test

```bash
pytest -v
```

## Usage (will be implemented in later phases)

```bash
python -m guardian.agent --profile config/my_profile.json --once
```

## References

1. NOAA Storm Events Database — https://www.ncdc.noaa.gov/stormevents/
2. NWS API — https://www.weather.gov/documentation/services-web-api
3. OpenWeatherMap API — https://openweathermap.org/api
4. Russell & Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed.
5. Ankan & Panda, *pgmpy*, SciPy 2015.

## License

Academic project. Not licensed for production use.
