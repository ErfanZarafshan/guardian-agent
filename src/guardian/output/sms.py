"""SMS output dispatcher (Twilio + dry-run fallback).

Single entry point: `SMSDispatcher.dispatch(actions)`. Routes each SMS-channel
Action to the Twilio API (if credentials are set AND dry-run is off), or to
a dry-run sink that logs messages instead of sending them.

Safety rails built in:

  - DRY_RUN is the DEFAULT. You must explicitly set SMS_DRY_RUN=false in .env
    to send real messages. This is a deliberate guardrail during development.
  - Per-dispatch cap (SMS_MAX_PER_RUN) limits blast radius of bugs.
  - Deduplication: (phone, message) tuples are sent once per dispatch.
  - Each attempt produces a SMSResult record even on failure.

The dispatcher is constructed lazily. It doesn't import `twilio` at module
load time, so the rest of the package keeps working without the twilio
library installed — useful if a CI environment only tests the core logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..config import get_config
from ..logging_setup import get_logger
from ..planning.actions import Action, ActionChannel


if TYPE_CHECKING:
    from twilio.rest import Client as TwilioClient


log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SMSResult:
    """One send attempt. Immutable audit record."""
    recipient: str
    message: str
    mode: str  # "sent" | "dry_run" | "skipped" | "failed"
    sid: str | None = None      # Twilio message SID if sent
    error: str | None = None    # error message if failed


@dataclass(frozen=True)
class SMSDispatchReport:
    """Result of calling dispatch() on a list of Actions."""
    attempted: int
    sent: int
    dry_run: int
    skipped: int
    failed: int
    results: tuple[SMSResult, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.failed == 0


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

class SMSDispatcher:
    """Twilio-backed SMS dispatcher with dry-run fallback.

    Construct once, call `dispatch(actions)` per agent cycle. The underlying
    Twilio Client is built lazily on first real send, so dry-run mode never
    requires the twilio package.
    """

    def __init__(self, client: "TwilioClient | None" = None) -> None:
        self.cfg = get_config()
        self._client = client  # optional pre-built/mock client for tests
        self._client_built = client is not None

    # ------------------------------------------------------------------

    @property
    def mode_summary(self) -> str:
        """Human-readable description of the dispatcher's current mode."""
        if self.cfg.sms_dry_run:
            return "DRY-RUN (SMS_DRY_RUN=true; messages will be logged, not sent)"
        if not self.cfg.twilio_configured:
            return "UNCONFIGURED (no Twilio credentials; falling back to dry-run)"
        return f"LIVE (sending as {self.cfg.twilio_from_number})"

    # ------------------------------------------------------------------

    def _get_client(self):  # type: ignore[no-untyped-def]
        """Build the Twilio client on first use."""
        if self._client_built:
            return self._client
        try:
            from twilio.rest import Client  # local import keeps import cost low
        except ImportError as e:
            log.warning("twilio package not installed: %s", e)
            self._client = None
            self._client_built = True
            return None
        self._client = Client(
            self.cfg.twilio_account_sid, self.cfg.twilio_auth_token
        )
        self._client_built = True
        return self._client

    # ------------------------------------------------------------------

    def _send_one(self, recipient: str, message: str) -> SMSResult:
        """Send a single SMS; choose dry-run vs live based on config."""
        if self.cfg.sms_dry_run or not self.cfg.twilio_configured:
            mode_reason = "dry_run" if self.cfg.sms_dry_run else "unconfigured"
            log.info("[SMS %s] to=%s | %s", mode_reason, recipient, message[:120])
            return SMSResult(recipient=recipient, message=message, mode="dry_run")

        client = self._get_client()
        if client is None:
            return SMSResult(
                recipient=recipient,
                message=message,
                mode="failed",
                error="twilio package not installed",
            )

        try:
            resp = client.messages.create(
                to=recipient,
                from_=self.cfg.twilio_from_number,
                body=message,
            )
        except Exception as e:  # noqa: BLE001 — Twilio raises a variety of errors
            log.warning("SMS send failed to %s: %s", recipient, e)
            return SMSResult(
                recipient=recipient,
                message=message,
                mode="failed",
                error=str(e),
            )

        sid = getattr(resp, "sid", None)
        log.info("[SMS sent] to=%s sid=%s", recipient, sid)
        return SMSResult(recipient=recipient, message=message, mode="sent", sid=sid)

    # ------------------------------------------------------------------

    def dispatch(self, actions: list[Action]) -> SMSDispatchReport:
        """Dispatch all SMS-channel actions in `actions`."""
        sms_actions = [a for a in actions if a.channel is ActionChannel.SMS]
        if not sms_actions:
            return SMSDispatchReport(attempted=0, sent=0, dry_run=0, skipped=0, failed=0)

        # Flatten (action, recipient) pairs. One action can have multiple
        # recipients (e.g., notify two contacts).
        pairs: list[tuple[str, str]] = []
        for a in sms_actions:
            if not a.recipients:
                log.debug("SMS action has no recipients, skipping: %s", a.kind.value)
                continue
            for phone in a.recipients:
                pairs.append((phone, a.message))

        # Dedupe within this dispatch.
        seen: set[tuple[str, str]] = set()
        unique_pairs = []
        for p in pairs:
            if p in seen:
                continue
            seen.add(p)
            unique_pairs.append(p)

        # Cap.
        cap = self.cfg.sms_max_per_run
        truncated = unique_pairs[:cap]
        skipped_count = len(unique_pairs) - len(truncated)
        if skipped_count:
            log.warning(
                "SMS dispatch truncated from %d to %d by SMS_MAX_PER_RUN cap.",
                len(unique_pairs), cap,
            )

        results = [self._send_one(phone, msg) for phone, msg in truncated]

        # Tack on 'skipped' results for the dropped-by-cap ones.
        for phone, msg in unique_pairs[cap:]:
            results.append(SMSResult(
                recipient=phone, message=msg, mode="skipped",
                error=f"hit SMS_MAX_PER_RUN cap of {cap}",
            ))

        sent = sum(1 for r in results if r.mode == "sent")
        dry = sum(1 for r in results if r.mode == "dry_run")
        skipped = sum(1 for r in results if r.mode == "skipped")
        failed = sum(1 for r in results if r.mode == "failed")

        return SMSDispatchReport(
            attempted=len(results),
            sent=sent,
            dry_run=dry,
            skipped=skipped,
            failed=failed,
            results=tuple(results),
        )


__all__ = ["SMSDispatcher", "SMSDispatchReport", "SMSResult"]
