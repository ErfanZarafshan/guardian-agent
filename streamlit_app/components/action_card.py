"""Streamlit component for rendering one Action as a card."""

from __future__ import annotations

import streamlit as st

from guardian.planning.actions import Action, ActionChannel, ActionKind


# Emoji per kind — gives the cards instant scan-ability.
_EMOJI: dict[ActionKind, str] = {
    ActionKind.RECOMMEND_EVACUATE:    "🚨",
    ActionKind.RECOMMEND_SHELTER:     "🏠",
    ActionKind.SOUND_ALARM:           "🔔",
    ActionKind.NOTIFY_CONTACTS:       "📱",
    ActionKind.NOTIFY_USER:           "💬",
    ActionKind.ACTIVATE_FLOOD_LIGHTS: "💡",
    ActionKind.ADJUST_THERMOSTAT:     "🌡️",
    ActionKind.LOG_ONLY:              "📝",
}

_CHANNEL_BADGE: dict[ActionChannel, str] = {
    ActionChannel.CONSOLE:    "🖥️ Console",
    ActionChannel.SMS:        "📡 SMS",
    ActionChannel.SMART_HOME: "🏡 Smart Home",
}


def render_actions(actions: list[Action]) -> None:
    """Render an ordered list of Actions as cards."""
    if not actions:
        st.info("No actions planned for this assessment.")
        return

    for i, action in enumerate(actions, 1):
        _render_one(i, action)


def _render_one(index: int, action: Action) -> None:
    emoji = _EMOJI.get(action.kind, "•")
    title = action.kind.value.replace("_", " ").title()
    channel = _CHANNEL_BADGE.get(action.channel, action.channel.value)

    with st.container(border=True):
        cols = st.columns([0.07, 0.93])
        with cols[0]:
            st.markdown(f"<div style='font-size:2em;text-align:center'>{emoji}</div>",
                        unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f"**{index}. {title}**  &nbsp; *{channel}*")
            st.caption(f"_Why:_ {action.rationale}")
            st.markdown(f"> {action.message}")
            if action.recipients:
                st.markdown("**Recipients (dry-run, not actually sent):**")
                for r in action.recipients:
                    st.code(r, language=None)
