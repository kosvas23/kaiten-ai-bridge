from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from kaiten_ai_bridge.errors import InvalidWebhookError
from kaiten_ai_bridge.schemas import Trigger
from kaiten_ai_bridge.webhook import parse_webhook


def test_card_add_uses_official_data_id_and_created_shape() -> None:
    """Kaiten's documented card:add body puts the card itself in ``data``."""

    trigger = parse_webhook(
        {
            "event": "card:add",
            "data": {
                "id": 69_416_673,
                "created": "2026-09-01T12:00:00.123Z",
                "title": "This webhook title is not trusted",
                "description": "This body must never become source content",
                "author": {"full_name": "Webhook Author", "email": "secret@example.test"},
                "files": [{"id": 8, "name": "not-persisted.pdf"}],
            },
        }
    )

    assert trigger == Trigger(
        event_type="card:add",
        card_id=69_416_673,
        source_created_at=datetime(2026, 9, 1, 12, 0, 0, 123_000, tzinfo=UTC),
    )
    assert trigger.event_key == "card:add:69416673"


def test_comment_add_uses_official_data_id_card_id_and_created_shape() -> None:
    """Kaiten's documented comment:add data.id is the comment, not the card."""

    trigger = parse_webhook(
        {
            "event": "comment:add",
            "data": {
                "id": 987,
                "card_id": 69_416_673,
                "created": "2026-09-01T15:02:03+03:00",
                "text": "Never trust or persist this text",
                "internal": False,
                "author": {"id": 10, "email": "private@example.test"},
            },
        }
    )

    assert trigger == Trigger(
        event_type="comment:add",
        card_id=69_416_673,
        comment_id=987,
        source_created_at=datetime(2026, 9, 1, 15, 2, 3, tzinfo=timezone(timedelta(hours=3))),
    )
    # Pydantic preserves the supplied offset, while equality compares the same instant.
    assert trigger.source_created_at is not None
    assert trigger.source_created_at.astimezone(UTC) == datetime(2026, 9, 1, 12, 2, 3, tzinfo=UTC)
    assert trigger.event_key == "comment:add:987"


@pytest.mark.parametrize("event", ["file:add", "file:update", "card:update", "card:move"])
def test_unsupported_events_are_acknowledgeable_without_identifiers(event: str) -> None:
    assert parse_webhook({"event": event, "data": {"text": "ignored"}}) is None


def test_trigger_has_no_surface_for_webhook_content_persistence() -> None:
    payload = {
        "event": "comment:add",
        "data": {
            "id": 12,
            "card_id": 34,
            "created": "2026-09-01T12:00:00Z",
            "text": "highly sensitive body",
            "description": "also sensitive",
            "files": [{"name": "customer-list.xlsx", "content": "base64"}],
            "author": {"full_name": "Person", "email": "person@example.test"},
        },
    }

    trigger = parse_webhook(payload)

    assert trigger is not None
    assert set(Trigger.model_fields) == {
        "event_type",
        "card_id",
        "comment_id",
        "source_created_at",
    }
    serialized = trigger.model_dump_json()
    for forbidden in (
        "highly sensitive body",
        "also sensitive",
        "customer-list.xlsx",
        "base64",
        "Person",
        "person@example.test",
    ):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"data": {"id": 1}},
        {"event": "card:add", "data": {"created": "2026-09-01T12:00:00Z"}},
        {"event": "card:add", "data": {"id": True}},
        {"event": "comment:add", "data": {"id": 2}},
        {"event": "comment:add", "data": {"id": 2, "card_id": 0}},
        {"event": "comment:add", "data": {"id": "not-an-id", "card_id": 1}},
    ],
)
def test_missing_or_invalid_supported_event_identifiers_are_rejected(payload: dict) -> None:
    with pytest.raises(InvalidWebhookError):
        parse_webhook(payload)


def test_legacy_nested_shapes_are_supported_without_using_content() -> None:
    trigger = parse_webhook(
        {
            "type": "comment_created",
            "data": {
                "comment": {
                    "id": "81",
                    "card": {"id": "91", "title": "ignored"},
                    "text": "ignored",
                    "created_at": "2026-09-01T12:00:00Z",
                }
            },
        }
    )

    assert trigger is not None
    assert trigger.event_type == "comment:add"
    assert trigger.card_id == 91
    assert trigger.comment_id == 81
