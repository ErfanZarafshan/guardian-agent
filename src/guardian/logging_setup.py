"""Structured logging with rich formatting."""

from __future__ import annotations

import logging

from rich.logging import RichHandler

from .config import get_config


_configured = False


def get_logger(name: str) -> logging.Logger:
    """Return a package-scoped logger, configuring root once."""
    global _configured
    if not _configured:
        cfg = get_config()
        logging.basicConfig(
            level=cfg.log_level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        )
        _configured = True
    return logging.getLogger(name)
