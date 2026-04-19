"""Bayesian Risk Assessment Engine (Phase 5).

Core of Guardian Agent's reasoning. Builds a Bayesian network (pgmpy) over:

  Observable nodes:
    - HazardSeverity        (None / Minor / Moderate / Severe / Extreme)
    - Urgency               (Future / Expected / Immediate / Past)
    - PrecipRate            (None / Light / Moderate / Heavy / Extreme)
    - WindSpeed             (Calm / Breezy / Strong / Damaging)
    - EmergingThreat        (Low / Med / High)  ← from ML classifier

  User profile nodes (set as evidence):
    - HomeFloor             (Ground / Upper / Elevated)
    - VehicleClearance      (Low / Medium / High)
    - MobilityLimited       (True / False)

  Target (query) node:
    - RiskLevel             (Low / Moderate / High / Critical)

CPTs will be partly learned from the Storm Events dataset with
pgmpy.estimators.MaximumLikelihoodEstimator and partly hand-specified from
domain knowledge (e.g., P(Flood damage | Ground floor, Flash flood warning)).

Inference at runtime via pgmpy.inference.VariableElimination.
"""
