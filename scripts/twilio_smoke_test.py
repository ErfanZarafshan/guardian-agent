"""Smoke-test Twilio credentials by sending one message.

Usage:
    python scripts/twilio_smoke_test.py --to +15551234567

Checks:
  - That TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN / TWILIO_FROM_NUMBER are set.
  - That the twilio package is installed.
  - That the credentials actually work by sending a real SMS to --to.

IMPORTANT: this script intentionally IGNORES SMS_DRY_RUN. It sends a real SMS
against the supplied recipient. Use it once after you configure Twilio to
confirm the trial account is wired up correctly; then set SMS_DRY_RUN=true
in .env again for normal development.

The recipient must be a verified Twilio trial number, otherwise Twilio will
refuse to deliver.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console

# Allow running without `pip install -e .`
HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from guardian.config import get_config  # noqa: E402


console = Console()


@click.command()
@click.option(
    "--to", required=True,
    help="Recipient phone in E.164 (e.g. +15551234567). Must be Twilio-verified.",
)
@click.option(
    "--body", default="Guardian Agent Twilio smoke test. If you got this, SMS works.",
    help="Message body.",
)
def main(to: str, body: str) -> None:
    cfg = get_config()

    console.rule("[bold cyan]Twilio smoke test")
    console.print(f"Account SID: {cfg.twilio_account_sid[:8]}... "
                  f"(set: {bool(cfg.twilio_account_sid)})")
    console.print(f"Auth Token:  {'set' if cfg.twilio_auth_token else 'MISSING'}")
    console.print(f"From number: {cfg.twilio_from_number or 'MISSING'}")
    console.print(f"twilio_configured flag: {cfg.twilio_configured}")

    if not cfg.twilio_configured:
        console.print("[red]Twilio is not configured in .env.[/red] "
                      "Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER.")
        sys.exit(1)

    try:
        from twilio.rest import Client
    except ImportError:
        console.print("[red]twilio package not installed.[/red] "
                      "Run: pip install twilio")
        sys.exit(1)

    client = Client(cfg.twilio_account_sid, cfg.twilio_auth_token)
    console.print(f"\nSending real SMS to {to} ...")
    try:
        msg = client.messages.create(to=to, from_=cfg.twilio_from_number, body=body)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Twilio send failed:[/red] {e}")
        sys.exit(2)

    console.print(f"[green]OK[/green] — Twilio accepted the send.")
    console.print(f"  SID:    {msg.sid}")
    console.print(f"  Status: {getattr(msg, 'status', '?')}")
    console.print(
        "\n[dim]Check the recipient phone. If it didn't arrive, look at "
        "https://console.twilio.com/us1/monitor/logs/sms for delivery status.[/dim]"
    )


if __name__ == "__main__":
    main()
