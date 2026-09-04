from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kaiten_ai_bridge.app import create_app
from kaiten_ai_bridge.config import Settings
from kaiten_ai_bridge.errors import KaitenUnavailableError
from kaiten_ai_bridge.logging_config import JsonFormatter
from kaiten_ai_bridge.storage import (
    CallbackOutcome,
    CallbackReservation,
    EnqueueResult,
    EventRecord,
    EventStatus,
)

NOW = "2026-09-02T09:00:00.000000Z"
CALLBACK_TOKEN = "callback-token-that-must-not-leak"
PRIVATE_BODY = "private-body-that-must-not-leak"


class FakeStorage:
    def __init__(self, timeline: list[str]) -> None:
        self.timeline = timeline
        self.initialized = False
        self.closed = False
        self.enqueue_error: Exception | None = None
        self.reserve_error: Exception | None = None
        self.events: dict[str, EventRecord] = {}
        self.service_status: dict[str, dict[str, str]] = {
            "last_webhook": {"value": "accepted", "updated_at": NOW},
            "last_reconciliation_result": {"value": "ok", "updated_at": NOW},
            "last_cleanup": {"value": "ok", "updated_at": NOW},
        }
        self.jobs: dict[str, int] = {"job-known": 701}
        self.callback_states: dict[str, tuple[str, str]] = {}
        self.released: list[str] = []

    async def initialize(self) -> None:
        self.initialized = True

    async def close(self) -> None:
        self.closed = True

    async def set_service_status(self, key: str, value: str) -> None:
        self.service_status[key] = {"value": value, "updated_at": NOW}

    async def enqueue_event(
        self,
        *,
        event_type: str,
        card_id: int,
        event_key: str,
        comment_id: int | None = None,
        source_created_at: datetime | str | None = None,
    ) -> EnqueueResult:
        if self.enqueue_error is not None:
            raise self.enqueue_error
        existing = self.events.get(event_key)
        if existing is not None:
            return EnqueueResult(existing, False)
        if isinstance(source_created_at, datetime):
            source_timestamp = source_created_at.isoformat()
        else:
            source_timestamp = source_created_at
        event = EventRecord(
            id=len(self.events) + 1,
            event_type=event_type,
            card_id=card_id,
            comment_id=comment_id,
            event_key=event_key,
            status=EventStatus.PENDING,
            error_code=None,
            attempts=0,
            next_attempt_at=NOW,
            notification_sent=False,
            source_created_at=source_timestamp,
            created_at=NOW,
            updated_at=NOW,
            claimed_at=None,
            success_at=None,
        )
        self.events[event_key] = event
        self.timeline.append("durably-enqueued")
        return EnqueueResult(event, True)

    async def reserve_callback(
        self,
        *,
        result_id: str,
        job_id: str,
        stale_before: datetime,
    ) -> CallbackReservation:
        del stale_before
        if self.reserve_error is not None:
            raise self.reserve_error
        card_id = self.jobs.get(job_id)
        if card_id is None:
            return CallbackReservation(CallbackOutcome.UNKNOWN_JOB, result_id, job_id, None)
        existing = self.callback_states.get(result_id)
        if existing is not None:
            existing_job, status = existing
            if existing_job != job_id:
                return CallbackReservation(CallbackOutcome.CONFLICT, result_id, job_id, card_id)
            outcome = CallbackOutcome.DUPLICATE if status == "success" else CallbackOutcome.BUSY
            return CallbackReservation(outcome, result_id, job_id, card_id)
        self.callback_states[result_id] = (job_id, "reserved")
        return CallbackReservation(CallbackOutcome.RESERVED, result_id, job_id, card_id)

    async def release_callback(self, result_id: str) -> bool:
        state = self.callback_states.get(result_id)
        if state is None or state[1] != "reserved":
            return False
        del self.callback_states[result_id]
        self.released.append(result_id)
        return True

    async def mark_callback_success(self, result_id: str) -> bool:
        job_id, status = self.callback_states[result_id]
        if status == "success":
            return True
        assert self.timeline[-1] == "published"
        self.callback_states[result_id] = (job_id, "success")
        self.timeline.append("callback-committed")
        return True

    async def health_snapshot(self) -> dict[str, Any]:
        return {
            "service": self.service_status,
            "queue": {"waiting": 2, "errors": 1, "queue_size": 3},
        }


class FakeKaiten:
    def __init__(self, timeline: list[str]) -> None:
        self.timeline = timeline
        self.closed = False
        self.healthy = True
        self.comment_error: Exception | None = None
        self.comments: list[tuple[int, str]] = []

    async def close(self) -> None:
        self.closed = True

    async def healthcheck(self) -> bool:
        return self.healthy

    async def add_public_comment(self, card_id: int, text: str) -> None:
        self.timeline.append("publish-attempt")
        if self.comment_error is not None:
            raise self.comment_error
        self.comments.append((card_id, text))
        self.timeline.append("published")


class FakeReceiver:
    def __init__(self) -> None:
        self.closed = False
        self.healthy = True

    async def close(self) -> None:
        self.closed = True

    async def healthcheck(self) -> bool:
        return self.healthy


@dataclass(slots=True)
class AppHarness:
    app: FastAPI
    settings: Settings
    storage: FakeStorage
    kaiten: FakeKaiten
    receiver: FakeReceiver
    timeline: list[str]


@pytest.fixture
def app_factory(tmp_path: Path) -> Callable[..., AppHarness]:
    sequence = count()

    def factory(**overrides: object) -> AppHarness:
        instance = next(sequence)
        values: dict[str, object] = {
            "environment": "test",
            "log_dir": tmp_path / f"logs-{instance}",
            "database_path": tmp_path / f"bridge-{instance}.sqlite3",
            "temp_dir": tmp_path / f"tmp-{instance}",
            "kaiten_base_url": "https://tagat.kaiten.test",
            "kaiten_token": "kaiten-token-that-must-not-leak",
            "kaiten_space_id": 1,
            "kaiten_board_id": 2,
            "kaiten_queue_column_id": 3,
            "kaiten_work_column_id": 4,
            "kaiten_done_column_id": 5,
            "ai_bearer_token": "receiver-token-that-must-not-leak",
            "callback_bearer_token": CALLBACK_TOKEN,
            "max_incoming_body_bytes": 8_192,
        }
        values.update(overrides)
        settings = Settings(**values)  # type: ignore[arg-type]
        timeline: list[str] = []
        storage = FakeStorage(timeline)
        kaiten = FakeKaiten(timeline)
        receiver = FakeReceiver()
        app = create_app(
            settings,
            storage=storage,  # type: ignore[arg-type]
            kaiten=kaiten,  # type: ignore[arg-type]
            receiver=receiver,  # type: ignore[arg-type]
            start_background=False,
        )
        return AppHarness(app, settings, storage, kaiten, receiver, timeline)

    return factory


def webhook_payload(*, card_id: int = 101) -> dict[str, Any]:
    return {
        "event": "card:add",
        "data": {
            "id": card_id,
            "created": "2026-09-02T09:00:00Z",
            "description": PRIVATE_BODY,
        },
    }


def callback_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "result_id": "result-1",
        "job_id": "job-known",
        "status": "completed",
        "result": {"text": "published result", "files": []},
        "created_at": "2026-09-02T09:05:00Z",
    }
    payload.update(overrides)
    return payload


def callback_headers(token: str = CALLBACK_TOKEN) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_factory_uses_injected_dependencies_without_background_tasks(
    app_factory: Callable[..., AppHarness],
) -> None:
    harness = app_factory()

    with TestClient(harness.app) as client:
        assert harness.storage.initialized
        assert harness.app.state.bridge.storage is harness.storage
        assert harness.app.state.bridge.kaiten is harness.kaiten
        assert harness.app.state.bridge.receiver is harness.receiver
        assert harness.app.state.bridge._tasks == []
        assert client.get("/live").json() == {"status": "ok"}

    assert harness.storage.closed
    assert harness.kaiten.closed
    assert harness.receiver.closed


def test_webhook_returns_202_only_after_durable_enqueue_and_deduplicates(
    app_factory: Callable[..., AppHarness],
) -> None:
    harness = app_factory()

    with TestClient(harness.app) as client:
        first = client.post(harness.settings.webhook_path, json=webhook_payload())
        second = client.post(harness.settings.webhook_path, json=webhook_payload())

    assert first.status_code == 202
    assert first.json() == {
        "accepted": True,
        "duplicate": False,
        "event_key": "card:add:101",
    }
    assert second.status_code == 202
    assert second.json() == {
        "accepted": True,
        "duplicate": True,
        "event_key": "card:add:101",
    }
    assert list(harness.storage.events) == ["card:add:101"]
    assert harness.timeline == ["durably-enqueued"]
    assert harness.kaiten.comments == []


def test_unsupported_webhook_is_acknowledged_with_empty_204(
    app_factory: Callable[..., AppHarness],
) -> None:
    harness = app_factory()

    with TestClient(harness.app) as client:
        response = client.post(
            harness.settings.webhook_path,
            json={"event": "card:move", "data": {"text": PRIVATE_BODY}},
        )

    assert response.status_code == 204
    assert response.content == b""
    assert harness.storage.events == {}
    assert harness.storage.service_status["last_webhook"]["value"] == "ignored"


@pytest.mark.parametrize(
    ("body", "expected_status", "detail"),
    [
        (b"{not-json", 400, "invalid_json"),
        (b"[]", 400, "json_object_required"),
        (b'{"event":"card:add","data":{}}', 422, "invalid_webhook"),
    ],
)
def test_malformed_webhook_returns_stable_4xx(
    app_factory: Callable[..., AppHarness],
    body: bytes,
    expected_status: int,
    detail: str,
) -> None:
    harness = app_factory()

    with TestClient(harness.app) as client:
        response = client.post(
            harness.settings.webhook_path,
            content=body,
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == expected_status
    assert response.json() == {"detail": detail}
    assert harness.storage.events == {}


def test_webhook_storage_failure_is_non_2xx_and_does_not_echo_exception(
    app_factory: Callable[..., AppHarness],
) -> None:
    harness = app_factory()
    harness.storage.enqueue_error = OSError(f"database failed: {PRIVATE_BODY} {CALLBACK_TOKEN}")

    with TestClient(harness.app) as client:
        response = client.post(harness.settings.webhook_path, json=webhook_payload())

    assert response.status_code == 503
    assert response.json() == {"detail": "queue_unavailable"}
    assert PRIVATE_BODY not in response.text
    assert CALLBACK_TOKEN not in response.text


@pytest.mark.parametrize("endpoint", ["webhook", "callback"])
def test_incoming_body_limit_applies_to_both_json_endpoints(
    app_factory: Callable[..., AppHarness], endpoint: str
) -> None:
    harness = app_factory(max_incoming_body_bytes=64)
    path = (
        harness.settings.webhook_path if endpoint == "webhook" else harness.settings.callback_path
    )
    headers = callback_headers() if endpoint == "callback" else {}

    with TestClient(harness.app) as client:
        response = client.post(path, json={"padding": "x" * 128}, headers=headers)

    assert response.status_code == 413
    assert response.json() == {"detail": "request_too_large"}


@pytest.mark.asyncio
async def test_chunked_body_is_rejected_as_soon_as_stream_crosses_limit(
    app_factory: Callable[..., AppHarness],
) -> None:
    harness = app_factory(max_incoming_body_bytes=64)
    consumed: list[int] = []

    async def streamed_body():
        chunks = (b'{"padding":"', b"x" * 64, b'this-chunk-must-not-be-read"}')
        for index, chunk in enumerate(chunks):
            consumed.append(index)
            yield chunk

    transport = httpx.ASGITransport(app=harness.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            harness.settings.webhook_path,
            content=streamed_body(),
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 413
    assert response.json() == {"detail": "request_too_large"}
    assert consumed == [0, 1]
    assert harness.storage.events == {}


def test_negative_content_length_is_rejected_before_body_read(
    app_factory: Callable[..., AppHarness],
) -> None:
    harness = app_factory()

    with TestClient(harness.app) as client:
        response = client.post(
            harness.settings.webhook_path,
            content=b"{}",
            headers={"Content-Type": "application/json", "Content-Length": "-1"},
        )

    assert response.status_code == 400
    assert response.json() == {"detail": "invalid_content_length"}
    assert harness.storage.events == {}


@pytest.mark.parametrize("endpoint", ["webhook", "callback"])
def test_json_endpoints_reject_wrong_content_type(
    app_factory: Callable[..., AppHarness], endpoint: str
) -> None:
    harness = app_factory()
    path = (
        harness.settings.webhook_path if endpoint == "webhook" else harness.settings.callback_path
    )
    headers = {"Content-Type": "text/plain"}
    if endpoint == "callback":
        headers.update(callback_headers())

    with TestClient(harness.app) as client:
        response = client.post(path, content=b"{}", headers=headers)

    assert response.status_code == 415
    assert response.json() == {"detail": "unsupported_media_type"}


def test_json_content_type_with_charset_is_accepted(
    app_factory: Callable[..., AppHarness],
) -> None:
    harness = app_factory()
    body = json.dumps(webhook_payload()).encode()

    with TestClient(harness.app) as client:
        response = client.post(
            harness.settings.webhook_path,
            content=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
        )

    assert response.status_code == 202


def test_callback_requires_exact_bearer_token_before_processing_body(
    app_factory: Callable[..., AppHarness],
) -> None:
    harness = app_factory()
    payload = callback_payload(result={"text": PRIVATE_BODY, "files": []})

    with TestClient(harness.app) as client:
        unauthorized = [
            client.post(harness.settings.callback_path, json=payload),
            client.post(
                harness.settings.callback_path,
                json=payload,
                headers=callback_headers("wrong-token"),
            ),
            client.post(
                harness.settings.callback_path,
                json=payload,
                headers={"Authorization": f"Basic {CALLBACK_TOKEN}"},
            ),
        ]
        assert harness.storage.callback_states == {}
        assert harness.kaiten.comments == []
        authorized = client.post(
            harness.settings.callback_path,
            json=payload,
            headers=callback_headers(),
        )

    assert all(response.status_code == 401 for response in unauthorized)
    assert all(response.json() == {"detail": "unauthorized"} for response in unauthorized)
    assert authorized.status_code == 200
    assert authorized.json() == {"status": "published", "card_id": 701}


@pytest.mark.parametrize(
    "payload",
    [
        {
            "schema_version": 1,
            "result_id": "result-missing-fields",
            "job_id": "job-known",
        },
        callback_payload(result_id=""),
        callback_payload(unexpected=PRIVATE_BODY),
    ],
)
def test_callback_schema_validation_is_422_without_echoing_payload(
    app_factory: Callable[..., AppHarness], payload: dict[str, Any]
) -> None:
    harness = app_factory()

    with TestClient(harness.app) as client:
        response = client.post(
            harness.settings.callback_path,
            json=payload,
            headers=callback_headers(),
        )

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid_callback"}
    assert PRIVATE_BODY not in response.text


@pytest.mark.parametrize(
    ("overrides", "detail"),
    [
        ({"schema_version": 2}, "unsupported_schema_version"),
        ({"status": "failed"}, "unsupported_callback_status"),
        ({"result": {}}, "empty_callback_result"),
        (
            {"result": {"text": "text", "files": [{"name": "answer.pdf"}]}},
            "result_files_not_configured",
        ),
        ({"result": {"text": "x" * 4_097}}, "callback_text_exceeds_kaiten_limit"),
    ],
)
def test_callback_semantic_validation_is_422(
    app_factory: Callable[..., AppHarness], overrides: dict[str, Any], detail: str
) -> None:
    harness = app_factory()
    payload = callback_payload(result_id=f"result-{detail}", **overrides)

    with TestClient(harness.app) as client:
        response = client.post(
            harness.settings.callback_path,
            json=payload,
            headers=callback_headers(),
        )

    assert response.status_code == 422
    assert response.json() == {"detail": detail}
    assert harness.kaiten.comments == []


def test_callback_duplicate_is_2xx_without_second_publication(
    app_factory: Callable[..., AppHarness],
) -> None:
    harness = app_factory()
    payload = callback_payload(result_id="result-duplicate")

    with TestClient(harness.app) as client:
        first = client.post(
            harness.settings.callback_path,
            json=payload,
            headers=callback_headers(),
        )
        second = client.post(
            harness.settings.callback_path,
            json=payload,
            headers=callback_headers(),
        )

    assert first.status_code == 200
    assert first.json() == {"status": "published", "card_id": 701}
    assert second.status_code == 200
    assert second.json() == {"status": "duplicate", "card_id": 701}
    assert harness.kaiten.comments == [(701, "published result")]
    assert harness.timeline == ["publish-attempt", "published", "callback-committed"]


def test_callback_unknown_busy_conflict_and_storage_errors_are_non_2xx(
    app_factory: Callable[..., AppHarness],
) -> None:
    harness = app_factory()
    harness.storage.callback_states["result-busy"] = ("job-known", "reserved")
    harness.storage.callback_states["result-conflict"] = ("other-job", "success")

    with TestClient(harness.app) as client:
        unknown = client.post(
            harness.settings.callback_path,
            json=callback_payload(result_id="result-unknown", job_id="missing-job"),
            headers=callback_headers(),
        )
        busy = client.post(
            harness.settings.callback_path,
            json=callback_payload(result_id="result-busy"),
            headers=callback_headers(),
        )
        conflict = client.post(
            harness.settings.callback_path,
            json=callback_payload(result_id="result-conflict"),
            headers=callback_headers(),
        )
        harness.storage.reserve_error = RuntimeError(f"storage: {PRIVATE_BODY}")
        failed = client.post(
            harness.settings.callback_path,
            json=callback_payload(result_id="result-storage-error"),
            headers=callback_headers(),
        )

    assert unknown.status_code == 503
    assert unknown.json() == {"detail": "callback_retry_required"}
    assert busy.status_code == 503
    assert busy.json() == {"detail": "callback_retry_required"}
    assert conflict.status_code == 409
    assert conflict.json() == {"detail": "result_id_conflict"}
    assert failed.status_code == 503
    assert failed.json() == {"detail": "callback_processing_failed"}
    assert PRIVATE_BODY not in failed.text
    assert harness.kaiten.comments == []


def test_callback_returns_2xx_only_after_publication_and_releases_failed_reservation(
    app_factory: Callable[..., AppHarness],
) -> None:
    harness = app_factory()
    payload = callback_payload(result_id="result-retry")
    harness.kaiten.comment_error = KaitenUnavailableError(f"kaiten: {PRIVATE_BODY}")

    with TestClient(harness.app) as client:
        failed = client.post(
            harness.settings.callback_path,
            json=payload,
            headers=callback_headers(),
        )
        assert harness.kaiten.comments == []
        assert "result-retry" not in harness.storage.callback_states
        harness.kaiten.comment_error = None
        succeeded = client.post(
            harness.settings.callback_path,
            json=payload,
            headers=callback_headers(),
        )

    assert failed.status_code == 503
    assert failed.json() == {"detail": "kaiten_temporarily_unavailable"}
    assert succeeded.status_code == 200
    assert succeeded.json() == {"status": "published", "card_id": 701}
    assert harness.storage.released == ["result-retry"]
    assert harness.kaiten.comments == [(701, "published result")]
    assert harness.timeline[-2:] == ["published", "callback-committed"]


def test_unexpected_publication_error_is_safe_non_2xx(
    app_factory: Callable[..., AppHarness],
) -> None:
    harness = app_factory()
    harness.kaiten.comment_error = RuntimeError(f"unexpected: {PRIVATE_BODY} {CALLBACK_TOKEN}")

    with TestClient(harness.app) as client:
        response = client.post(
            harness.settings.callback_path,
            json=callback_payload(result_id="result-unexpected"),
            headers=callback_headers(),
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "callback_processing_failed"}
    assert PRIVATE_BODY not in response.text
    assert CALLBACK_TOKEN not in response.text
    assert harness.kaiten.comments == []


def test_health_reports_dependency_state_while_live_stays_ok(
    app_factory: Callable[..., AppHarness],
) -> None:
    harness = app_factory()

    with TestClient(harness.app) as client:
        healthy = client.get(harness.settings.health_path)
        harness.kaiten.healthy = False
        harness.receiver.healthy = False
        degraded = client.get(harness.settings.health_path)
        live = client.get("/live")

    assert healthy.status_code == 200
    assert healthy.json() == {
        "status": "ok",
        "process": "running",
        "kaiten_api": "available",
        "receiver": "available",
        "last_webhook_at": NOW,
        "last_reconciliation_at": NOW,
        "last_reconciliation_result": "ok",
        "pending_events": 2,
        "error_events": 1,
        "queue_size": 3,
        "last_cleanup_at": NOW,
    }
    assert degraded.status_code == 200
    assert degraded.json()["status"] == "degraded"
    assert degraded.json()["kaiten_api"] == "unavailable"
    assert degraded.json()["receiver"] == "unavailable"
    assert live.status_code == 200
    assert live.json() == {"status": "ok"}


def test_endpoint_responses_and_logs_do_not_contain_body_or_bearer_token(
    app_factory: Callable[..., AppHarness],
) -> None:
    harness = app_factory()
    callback = callback_payload(
        result_id="result-content-safety",
        result={"text": PRIVATE_BODY, "files": []},
    )

    with TestClient(harness.app) as client:
        responses = [
            client.post(harness.settings.webhook_path, json=webhook_payload(card_id=999)),
            client.post(
                harness.settings.callback_path,
                json=callback,
                headers=callback_headers(),
            ),
        ]

    for response in responses:
        assert PRIVATE_BODY not in response.text
        assert CALLBACK_TOKEN not in response.text
    log_text = (harness.settings.log_dir / "bridge.log").read_text(encoding="utf-8")
    assert "webhook_enqueued" in log_text
    assert "callback_published" in log_text
    assert PRIVATE_BODY not in log_text
    assert CALLBACK_TOKEN not in log_text


def test_json_formatter_ignores_arbitrary_content_bearing_extras() -> None:
    record = logging.LogRecord(
        name="kaiten_ai_bridge.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="safe_event",
        args=(),
        exc_info=None,
    )
    record.card_id = 123
    record.body = PRIVATE_BODY
    record.payload = {"text": PRIVATE_BODY}
    record.token = CALLBACK_TOKEN
    record.authorization = f"Bearer {CALLBACK_TOKEN}"

    formatted = JsonFormatter().format(record)
    parsed = json.loads(formatted)

    assert parsed["event"] == "safe_event"
    assert parsed["card_id"] == 123
    assert "body" not in parsed
    assert "payload" not in parsed
    assert "token" not in parsed
    assert "authorization" not in parsed
    assert PRIVATE_BODY not in formatted
    assert CALLBACK_TOKEN not in formatted
