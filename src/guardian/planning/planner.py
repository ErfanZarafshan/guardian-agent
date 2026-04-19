"""Rule-augmented action planner (Phase 6).

Takes the posterior risk distribution from the Bayesian network plus the user
profile, and selects a set of actions from a fixed catalog:

  - NOTIFY_USER              (push/console alert with tailored guidance)
  - NOTIFY_CONTACTS          (SMS via Twilio to contacts whose notify_on matches)
  - ACTIVATE_SMART_HOME      (flood lights, alarm, thermostat adjust)
  - RECOMMEND_EVACUATE       (with route/shelter info if available)
  - RECOMMEND_SHELTER        (shelter-in-place with location specifics)
  - LOG_ONLY                 (quiet hours or low risk)

Selection rules combine: (a) argmax(RiskLevel posterior), (b) user vulnerability
flags, (c) user preferences (quiet hours, smart-home opt-in).
"""
