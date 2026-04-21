"""LLM-powered plain-English explanation of a risk assessment.

This is an OPTIONAL enhancement: the recommendation logic is fully decided
by the Bayesian network and rule planner. The LLM only re-phrases the
result into conversational guidance.

Supports two providers, selectable at call time:
  - Anthropic Claude (claude-3-5-haiku-latest by default — fast and cheap)
  - OpenAI ChatGPT  (gpt-4o-mini by default — fast and cheap)

Errors are caught and returned as user-readable messages rather than raising.
"""

from __future__ import annotations

from dataclasses import dataclass

from guardian.planning.actions import Action
from guardian.profile import UserProfile
from guardian.weather.observation import WeatherObservation


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    ok: bool
    error: str | None = None


SYSTEM_PROMPT = """You are Guardian Agent's safety explainer.

The user has run a probabilistic risk-assessment system that combines real-time
weather data, an ML threat classifier, and a Bayesian network. The system has
already produced its recommended actions. Your job is NOT to second-guess
those recommendations or invent new ones. Your job is to take the technical
output and explain it to the user in plain, calm, supportive English.

Guidelines:
- Speak in the second person ("you").
- 2-3 short paragraphs, max ~150 words total.
- Do NOT add new safety advice the system did not produce.
- Do NOT contradict the system's recommendations.
- Do NOT use alarmist language for low-risk situations.
- Do mention specific personal factors (their floor level, mobility, vehicle)
  where they explain *why* the recommendation is what it is.
- End with one sentence reminding the user this is an academic prototype, not
  a substitute for official emergency services."""


def build_user_prompt(
    profile: UserProfile,
    observation: WeatherObservation,
    risk_argmax: str,
    posterior: dict[str, float],
    actions: list[Action],
) -> str:
    """Assemble the structured input the LLM rephrases."""
    posterior_lines = "\n".join(
        f"- {state}: {prob:.0%}" for state, prob in posterior.items()
    )
    alerts_lines = (
        "\n".join(f"- {a.event} ({a.severity.value}, {a.urgency.value})"
                  for a in observation.alerts if a.is_active(observation.observed_at))
        or "- (none)"
    )
    actions_lines = "\n".join(
        f"- {a.kind.value.replace('_', ' ').upper()}: {a.message}"
        for a in actions
    ) or "- (none)"

    return f"""USER PROFILE:
- Name: {profile.name}
- Address: {profile.location.address}
- Home: {profile.home.type.value}, floor {profile.home.floor_level} (state: {profile.home_floor_state.value})
- Vehicle: owns={profile.vehicle.owns_vehicle}, clearance={profile.vehicle.clearance.value}
- Medically vulnerable: {profile.is_medically_vulnerable}
- Mobility limited: {profile.medical.mobility_limited}

CURRENT WEATHER:
- Temperature: {observation.temperature_f}°F
- Wind: {observation.wind_speed_mph} mph (category: {observation.wind_category.value})
- Precip: {observation.precip_rate_in_hr} in/hr (category: {observation.precip_category.value})
- Active alerts:
{alerts_lines}

RISK ASSESSMENT (Bayesian posterior):
{posterior_lines}
- Most likely: {risk_argmax}

ACTIONS THE SYSTEM RECOMMENDS:
{actions_lines}

Now write a plain-English explanation following the guidelines."""


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

def _explain_with_anthropic(api_key: str, user_prompt: str) -> LLMResult:
    try:
        from anthropic import Anthropic
    except ImportError:
        return LLMResult(
            text="", provider="anthropic", model="?", ok=False,
            error="anthropic package not installed. Run: pip install anthropic"
        )
    model = "claude-haiku-4-5"
    try:
        client = Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model,
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        # Extract text from content blocks
        text_parts = [b.text for b in resp.content if hasattr(b, "text")]
        return LLMResult(
            text="\n\n".join(text_parts).strip(),
            provider="anthropic", model=model, ok=True,
        )
    except Exception as e:  # noqa: BLE001
        return LLMResult(
            text="", provider="anthropic", model=model, ok=False,
            error=f"Anthropic API call failed: {e}"
        )


def _explain_with_openai(api_key: str, user_prompt: str) -> LLMResult:
    try:
        from openai import OpenAI
    except ImportError:
        return LLMResult(
            text="", provider="openai", model="?", ok=False,
            error="openai package not installed. Run: pip install openai"
        )
    model = "gpt-4o-mini"
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            max_tokens=400,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return LLMResult(
            text=resp.choices[0].message.content.strip(),
            provider="openai", model=model, ok=True,
        )
    except Exception as e:  # noqa: BLE001
        return LLMResult(
            text="", provider="openai", model=model, ok=False,
            error=f"OpenAI API call failed: {e}"
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def explain(
    provider: str,
    api_key: str,
    profile: UserProfile,
    observation: WeatherObservation,
    risk_argmax: str,
    posterior: dict[str, float],
    actions: list[Action],
) -> LLMResult:
    """Generate a plain-English explanation of the assessment."""
    if not api_key:
        return LLMResult(
            text="", provider=provider, model="?", ok=False,
            error="No API key provided.",
        )
    user_prompt = build_user_prompt(
        profile, observation, risk_argmax, posterior, actions,
    )
    if provider == "anthropic":
        return _explain_with_anthropic(api_key, user_prompt)
    if provider == "openai":
        return _explain_with_openai(api_key, user_prompt)
    return LLMResult(
        text="", provider=provider, model="?", ok=False,
        error=f"Unknown provider: {provider}",
    )
