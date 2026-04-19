"""Supervised threat-pattern classifier (Phase 4).

Trained on the NOAA Storm Events Database (2005–2025, Gulf Coast states).
Produces a probability distribution over event-escalation categories from
current weather features. Output feeds the Bayesian network as an
"emerging_threat" evidence node.

Model: sklearn.ensemble.GradientBoostingClassifier (selected by CV).
Serialized via joblib to models/threat_classifier.joblib.
"""
