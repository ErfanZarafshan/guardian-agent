"""Smart-home action dispatcher (Phase 7).

For the prototype, this is a **mock** that logs intended device actions
(flood lights on, audible alarm, thermostat adjust) with timestamps rather
than calling the real Alexa Smart Home Skill API or Google Home Actions SDK.

Interface is designed so a real Alexa Skill implementation can replace the
mock later without changing any planner code.
"""
