"""Twilio SMS dispatcher (Phase 7).

Sends pre-drafted, scenario-specific SMS alerts to the user's emergency
contacts. Templates live alongside this module and are parameterized on
{user_name, address, hazard, recommended_action, onset_time}.

During development / without Twilio credentials, the dispatcher falls back to
a DryRunSMS mode that logs messages to stdout instead of sending them.
"""
