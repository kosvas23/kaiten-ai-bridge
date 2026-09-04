"""Extract only technical trigger identifiers from untrusted Kaiten webhooks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from .errors import InvalidWebhookError
from .schemas import Trigger

SUPPORTED_EVENTS = {"card:add", "comment:add"}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _event_name(payload: Mapping[str, Any]) -> str | None:
    raw = _first(
        payload.get("event_type"),
        payload.get("event"),
        payload.get("type"),
        payload.get("action"),
        _mapping(payload.get("data")).get("event_type"),
    )
    if isinstance(raw, Mapping):
        raw = _first(raw.get("type"), raw.get("name"), raw.get("action"))
    if not isinstance(raw, str):
        return None
    normalized = raw.strip().casefold().replace("_", ":")
    aliases = {
        "card:created": "card:add",
        "card:create": "card:add",
        "comment:created": "comment:add",
        "comment:create": "comment:add",
    }
    return aliases.get(normalized, normalized)


def parse_webhook(payload: Mapping[str, Any]) -> Trigger | None:
    """Parse a trigger without returning or retaining any user content.

    Unsupported, well-formed event types return ``None`` and are acknowledged
    without queueing.  Supported events with missing identifiers are rejected.
    """

    event_type = _event_name(payload)
    if not event_type:
        raise InvalidWebhookError("event type is missing")
    if event_type not in SUPPORTED_EVENTS:
        return None

    data = _mapping(payload.get("data"))
    card = _mapping(_first(payload.get("card"), data.get("card")))
    comment = _mapping(_first(payload.get("comment"), data.get("comment")))
    entity = _mapping(_first(payload.get("entity"), data.get("entity")))

    comment_id = _positive_int(
        _first(
            payload.get("comment_id"),
            data.get("comment_id"),
            data.get("id") if event_type == "comment:add" else None,
            comment.get("id"),
            entity.get("id") if event_type == "comment:add" else None,
        )
    )
    card_id = _positive_int(
        _first(
            payload.get("card_id"),
            data.get("card_id"),
            data.get("id") if event_type == "card:add" else None,
            card.get("id"),
            comment.get("card_id"),
            _mapping(comment.get("card")).get("id"),
            entity.get("id") if event_type == "card:add" else None,
        )
    )
    timestamp = _first(
        payload.get("created_at"),
        payload.get("timestamp"),
        data.get("created_at"),
        data.get("created"),
        comment.get("created_at"),
        card.get("created_at"),
    )

    try:
        return Trigger.model_validate(
            {
                "event_type": event_type,
                "card_id": card_id,
                "comment_id": comment_id if event_type == "comment:add" else None,
                "source_created_at": timestamp,
            }
        )
    except ValidationError as exc:
        raise InvalidWebhookError("supported event has invalid identifiers") from exc
