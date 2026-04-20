"""Action catalog for Guardian Agent.

An Action is a concrete thing the agent wants to happen — send an SMS, turn
on flood lights, display a console alert, recommend evacuation. The Phase 6
planner produces a list of Actions from the current risk assessment; the
Phase 7 dispatchers consume Actions and execute them against real services
(Twilio, mock smart-home, console).

Design choices:
  - Actions are immutable dataclasses. Side-effect-free construction.
  - Every Action carries a `rationale` so the user/log can explain WHY the
    action was chosen. This is core to Guardian Agent's trust story.
  - Actions have a `kind` (what to do) and a `channel` (which dispatcher
    should handle it). The two are orthogonal — NOTIFY_USER can go to
    console or SMS depending on severity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ActionKind(str, Enum):
    """What the action wants to make happen."""
    LOG_ONLY = "log_only"
    NOTIFY_USER = "notify_user"
    NOTIFY_CONTACTS = "notify_contacts"
    RECOMMEND_SHELTER = "recommend_shelter"
    RECOMMEND_EVACUATE = "recommend_evacuate"
    ACTIVATE_FLOOD_LIGHTS = "activate_flood_lights"
    SOUND_ALARM = "sound_alarm"
    ADJUST_THERMOSTAT = "adjust_thermostat"


class ActionChannel(str, Enum):
    """Where the action is dispatched to."""
    CONSOLE = "console"     # printed to the running terminal/log
    SMS = "sms"             # Twilio SMS to user or contacts
    SMART_HOME = "smart_home"  # smart-home API (mocked in prototype)


# Priority used for sorting the action list in the planner output.
# Higher priority actions come first so a human reading the log sees the
# most urgent guidance at the top.
_PRIORITY: dict[ActionKind, int] = {
    ActionKind.RECOMMEND_EVACUATE: 100,
    ActionKind.RECOMMEND_SHELTER:   90,
    ActionKind.SOUND_ALARM:         85,
    ActionKind.NOTIFY_CONTACTS:     70,
    ActionKind.NOTIFY_USER:         60,
    ActionKind.ACTIVATE_FLOOD_LIGHTS: 40,
    ActionKind.ADJUST_THERMOSTAT:     30,
    ActionKind.LOG_ONLY:             10,
}


@dataclass(frozen=True)
class Action:
    """One planned action.

    Fields:
      kind:      what the action is
      channel:   who dispatches it
      message:   the tailored human-readable text
      rationale: why the planner chose this action (for audit/explainability)
      recipients: list of phone numbers (only populated for SMS actions)
      metadata:  dispatcher-specific extra info (e.g. thermostat setpoint)
    """
    kind: ActionKind
    channel: ActionChannel
    message: str
    rationale: str
    recipients: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict = field(default_factory=dict)

    @property
    def priority(self) -> int:
        return _PRIORITY.get(self.kind, 0)


# ---------------------------------------------------------------------------
# Hazard-specific message templates
#
# Each hazard has distinct "shelter" and "evacuate" guidance. Templates
# accept a dict of variables for substitution. Missing variables are left
# as empty strings rather than raising — safer during a live event.
# ---------------------------------------------------------------------------

HAZARD_TEMPLATES: dict[str, dict[str, str]] = {
    "Tornado Warning": {
        "shelter": (
            "TORNADO WARNING {address}. Shelter NOW in the lowest interior "
            "room, away from windows. Cover your head. Stay put until NWS "
            "cancels or {expires} passes."
        ),
        "evacuate": (
            "TORNADO WARNING {address}. A safer location is preferred. If "
            "you cannot shelter here, move to a sturdy building nearby. "
            "Do NOT try to outrun a tornado in a vehicle."
        ),
        "contacts_sms": (
            "URGENT: Tornado Warning for {name} at {address}. Expires {expires}. "
            "They are {sheltering_in_place}. Sent by Guardian Agent."
        ),
    },
    "Flash Flood Warning": {
        "shelter": (
            "FLASH FLOOD WARNING {address}. Move to higher ground inside your "
            "building (upper floor, attic). Do NOT walk or drive through flood "
            "water. Turn around, don't drown."
        ),
        "evacuate": (
            "FLASH FLOOD {address}. If water is rising in your home, "
            "evacuate to higher ground immediately. Avoid any road covered "
            "by water — six inches can knock you down, two feet carries a car."
        ),
        "contacts_sms": (
            "URGENT: Flash flood {address} affects {name}. They may need "
            "help reaching higher ground. Sent by Guardian Agent."
        ),
    },
    "Flash Flood Watch": {
        "shelter": (
            "Flash flood conditions possible near {address}. Move valuables "
            "off the floor; be ready to relocate to an upper level if water "
            "begins entering the building."
        ),
        "evacuate": (
            "Flash flood watch {address}. Consider moving to higher ground "
            "before water rises, especially if you are on the ground floor."
        ),
        "contacts_sms": (
            "Flash flood watch {address}. {name} is in a flood-prone area "
            "and may need assistance. Sent by Guardian Agent."
        ),
    },
    "Tropical Storm Warning": {
        "shelter": (
            "TROPICAL STORM {address}. Stay indoors, away from windows. "
            "Charge phones now, fill bathtubs with water, and expect power "
            "loss. Winds may exceed 70 mph."
        ),
        "evacuate": (
            "TROPICAL STORM {address}. Complete any evacuation BEFORE winds "
            "reach 40 mph — bridges and elevated roads will close. Follow "
            "evacuation routes posted by local emergency management."
        ),
        "contacts_sms": (
            "Tropical storm warning at {address} for {name}. Storm expected "
            "{expires}. Sent by Guardian Agent."
        ),
    },
    "Hurricane Warning": {
        "shelter": (
            "HURRICANE WARNING {address}. If you did not evacuate, shelter "
            "in the strongest interior room. Stay away from windows. Do not "
            "venture outside during the eye; the back wall is coming."
        ),
        "evacuate": (
            "HURRICANE WARNING {address}. If an evacuation order is in "
            "effect for your zone, LEAVE NOW. Bring medications, IDs, pets. "
            "Follow designated evacuation routes."
        ),
        "contacts_sms": (
            "Hurricane warning at {address} for {name}. Please confirm they "
            "have evacuated or have shelter supplies. Sent by Guardian Agent."
        ),
    },
    "Excessive Heat Warning": {
        "shelter": (
            "EXCESSIVE HEAT {address}. Stay in air-conditioned spaces. "
            "Hydrate. Avoid strenuous activity. Check on elderly neighbors."
        ),
        "evacuate": (
            "EXCESSIVE HEAT {address}. If you have no AC, move to a cooling "
            "center. Heat illness develops quickly — do not delay."
        ),
        "contacts_sms": (
            "Excessive heat warning for {name} at {address}. They may need "
            "to relocate to a cool space. Sent by Guardian Agent."
        ),
    },
    # Generic fallback when we don't have a specific template for an alert.
    "__default__": {
        "shelter": (
            "Weather advisory in effect at {address}: {event}. Stay indoors, "
            "monitor local news, and follow official guidance."
        ),
        "evacuate": (
            "Severe weather at {address}: {event}. Consider moving to a "
            "safer location if local authorities advise evacuation."
        ),
        "contacts_sms": (
            "Weather alert affecting {name} at {address}: {event}. "
            "Sent by Guardian Agent."
        ),
    },
}


def get_template(event: str, kind: str) -> str:
    """Look up a message template by event name and action kind.

    `kind` is one of 'shelter', 'evacuate', 'contacts_sms'. Falls back to
    the default templates if the event isn't in HAZARD_TEMPLATES.
    """
    bucket = HAZARD_TEMPLATES.get(event) or HAZARD_TEMPLATES["__default__"]
    return bucket.get(kind, HAZARD_TEMPLATES["__default__"][kind])


def render(template: str, **vars) -> str:  # type: ignore[no-untyped-def]
    """Format a template, leaving missing variables as empty strings."""
    class _SafeDict(dict):  # type: ignore[type-arg]
        def __missing__(self, key: str) -> str:  # type: ignore[override]
            return ""
    return template.format_map(_SafeDict(**vars))


__all__ = [
    "ActionKind",
    "ActionChannel",
    "Action",
    "HAZARD_TEMPLATES",
    "get_template",
    "render",
]
