"""Content-safe JSON logging with daily three-day rotation."""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import UTC, datetime
from typing import Any

from .config import Settings

_STANDARD_RECORD_KEYS = set(logging.makeLogRecord({}).__dict__) | {
    "message",
    "asctime",
}

# Keep the formatter content-safe even if a future call site accidentally passes
# an inbound body, token, prompt, or another user-controlled value via ``extra``.
# Every value in this allowlist is a bounded technical identifier, counter, flag,
# timestamp, or stable error code used by the service's current log events.
_SAFE_EXTRA_KEYS = frozenset(
    {
        "attempt",
        "callbacks_deleted",
        "card_id",
        "cards_seen",
        "code",
        "comment_id",
        "duplicate",
        "duration_ms",
        "event_id",
        "event_type",
        "events_deleted",
        "log_files_deleted",
        "next_attempt_at",
        "result_id",
        "temp_files_deleted",
    }
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_KEYS and key in _SAFE_EXTRA_KEYS:
                payload[key] = value
        if record.exc_info:
            payload["exception_type"] = record.exc_info[0].__name__
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


def configure_logging(settings: Settings) -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    formatter = JsonFormatter()
    handlers: list[logging.Handler] = []

    stdout = logging.StreamHandler(sys.stdout)
    stdout.setFormatter(formatter)
    handlers.append(stdout)

    settings.log_dir.mkdir(parents=True, exist_ok=True)
    rotating = logging.handlers.TimedRotatingFileHandler(
        settings.log_dir / "bridge.log",
        when="midnight",
        interval=1,
        backupCount=settings.log_retention_days,
        encoding="utf-8",
        utc=True,
    )
    rotating.setFormatter(formatter)
    handlers.append(rotating)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    for handler in handlers:
        root.addHandler(handler)

    # Access logs may contain query strings; the service has explicit safe request logs.
    logging.getLogger("uvicorn.access").disabled = True
    logging.getLogger("httpx").setLevel(logging.WARNING)
