"""User profile: schema, validation, and JSON-backed storage.

Implemented in **Phase 2**. The profile captures vulnerability factors that the
Bayesian network uses as evidence when computing posterior risk:

  - Home type and floor level (drives flood-damage conditional probabilities)
  - Vehicle clearance (drives evacuation-feasibility conditional probabilities)
  - Medical vulnerabilities (drives urgency and action selection)
  - Emergency contacts (drives the notification fan-out)

Schema is declared with pydantic v2 so we get validation on load.
"""

from __future__ import annotations

# Placeholder — full implementation lands in Phase 2.
__all__ = ["UserProfile", "load_profile", "save_profile"]


class UserProfile:  # noqa: D401 — placeholder
    """Placeholder — replaced by a pydantic model in Phase 2."""


def load_profile(path):  # type: ignore[no-untyped-def]
    raise NotImplementedError("Phase 2")


def save_profile(profile, path):  # type: ignore[no-untyped-def]
    raise NotImplementedError("Phase 2")
