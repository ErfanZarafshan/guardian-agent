# Guardian Agent — Architecture

This document captures the design rationale, data flow, and the mapping from
AIMA concepts to our implementation. It will grow as each phase lands.

## 1. Agent model

Guardian Agent is a **goal-based agent** in the sense of Russell & Norvig (4e),
Chapter 2. Its goals are: (a) keep the user safe, and (b) keep the user's
designated contacts informed, subject to (c) not spamming during quiet hours
and (d) respecting smart-home opt-out. The environment is:

| Property          | Value                                        |
| ----------------- | -------------------------------------------- |
| Observable        | Partially — APIs lag behind reality.         |
| Deterministic     | No — weather evolution is stochastic.        |
| Episodic          | No — decisions depend on history.            |
| Static            | No — conditions change during deliberation.  |
| Discrete          | Discretized — observations bucketed for CPTs. |
| Agents            | Single agent, single user.                   |

The partial-observability + stochasticity is exactly why a **Bayesian
network** is the right reasoning substrate (AIMA Ch. 13).

## 2. Data flow (one perceive-reason-act cycle)

```
┌──────────────┐   ┌──────────────┐
│ NWS /alerts  │   │ OWM current  │
│ NWS forecast │   │ OWM forecast │
└──────┬───────┘   └──────┬───────┘
       │                  │
       ▼                  ▼
┌────────────────────────────────────┐
│  Observation normalizer            │  (weather/observation.py)
│  → unified WeatherObservation      │
└──────────────────┬─────────────────┘
                   │
                   ▼
┌────────────────────────────────────┐     ┌────────────────────────┐
│  ML threat classifier              │◄────┤ NOAA Storm Events      │
│  (risk/classifier.py, offline)     │     │ (training, Phase 4)    │
│  emits P(emerging_threat)          │     └────────────────────────┘
└──────────────────┬─────────────────┘
                   │
                   ▼
┌────────────────────────────────────┐     ┌────────────────────────┐
│  Bayesian Risk Engine              │◄────┤ User Profile           │
│  (risk/bayesian.py)                │     │ (profile.py)           │
│  evidence = obs + classifier +     │     │ sets HomeFloor,        │
│             profile; query         │     │ VehicleClearance,      │
│  RiskLevel ∈ {Low..Critical}       │     │ MobilityLimited        │
└──────────────────┬─────────────────┘     └────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────┐
│  Planner                           │  (planning/planner.py)
│  rules + posterior → action set    │
└──────┬─────────────────────┬───────┘
       │                     │
       ▼                     ▼
┌──────────────┐      ┌──────────────┐
│ Twilio SMS   │      │ Smart home   │
│ (output/sms) │      │ (output/     │
│              │      │  smart_home) │
└──────────────┘      └──────────────┘
```

## 3. Bayesian network sketch

```
          EmergingThreat (from ML classifier)
                  │
                  ▼
HazardSeverity ──► RiskLevel ◄── HomeFloor
                   ▲   ▲
       PrecipRate ─┘   └─ VehicleClearance
       WindSpeed ──────► RiskLevel
                   ▲
                   └─ MobilityLimited
```

(Final structure will be refined in Phase 5 with d-separation checks.)

## 4. Phase status

| Phase | Component                       | Status        |
| ----- | ------------------------------- | ------------- |
| 1     | Scaffolding                     | **Done**      |
| 2     | User profile                    | Not started   |
| 3     | Weather data layer              | Not started   |
| 4     | ML threat classifier            | Not started   |
| 5     | Bayesian risk engine            | Not started   |
| 6     | Planner                         | Not started   |
| 7     | Output layer                    | Not started   |
| 8     | Main loop                       | Not started   |
| 9     | Demo + write-up                 | Not started   |
