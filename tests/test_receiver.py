from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from kaiten_ai_bridge.config import Settings
from kaiten_ai_bridge.errors import (
    DataRejectedError,
    ReceiverRejectedError,
    ReceiverUnavailableError,
)
from kaiten_ai_bridge.receiver import ReceiverClient


def settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "environment": "test",
        "temp_dir": tmp_path,
        "kaiten_base_url": "https://tagat.kaiten.ru",
        "kaiten_token": "kaiten-secret",
        "kaiten_space_id": 834_556,
        "kaiten_board_id": 1_868_751,
        "kaiten_queue_column_id": 6_467_479,
        "kaiten_work_column_id": 6_467_480,
        "kaiten_done_column_id": 6_467_481,
        "ai_scheme": "http",
        "ai_host": "127.0.0.1",
        "ai_port": 8090,
        "ai_path": "/api/v1/jobs",
        "ai_bearer_token": "receiver-secret",
    }
    values.update(overrides)
    return Settings(**values)


@pytest.mark.asyncio
async def test_submit_sends_json_bearer_and_idempotency_key_and_accepts_only_job_id(
    tmp_path: Path,
) -> None:
    requests: list[httpx.Request] = []
    payload = {
        "idempotency_key": "comment:add:456",
        "prompt": "Сделайте расчёт",
        "files": [],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.method == "POST"
        assert request.url == httpx.URL("http://127.0.0.1:8090/api/v1/jobs")
        assert request.headers["authorization"] == "Bearer receiver-secret"
        assert request.headers["content-type"] == "application/json"
        assert request.headers["idempotency-key"] == "comment:add:456"
        assert json.loads(request.content) == payload
        assert "Сделайте расчёт" in request.content.decode("utf-8")
        return httpx.Response(202, json={"status": "accepted", "job_id": " job-17 "})

    async with ReceiverClient(settings(tmp_path), httpx.MockTransport(handler)) as client:
        assert await client.submit(payload) == "job-17"

    assert len(requests) == 1


@pytest.mark.parametrize(
    ("status", "content", "json_body"),
    [
        (200, None, {"job_id": "job-17"}),
        (201, None, {"job_id": "job-17"}),
        (202, None, {}),
        (202, None, {"job_id": "   "}),
        (202, b"not-json", None),
    ],
)
@pytest.mark.asyncio
async def test_submit_requires_exactly_202_and_nonempty_job_id(
    tmp_path: Path,
    status: int,
    content: bytes | None,
    json_body: dict[str, object] | None,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if content is not None:
            return httpx.Response(status, content=content)
        return httpx.Response(status, json=json_body)

    payload = {"idempotency_key": "card:add:17", "prompt": "text"}
    async with ReceiverClient(settings(tmp_path), httpx.MockTransport(handler)) as client:
        with pytest.raises(ReceiverRejectedError):
            await client.submit(payload)


@pytest.mark.asyncio
async def test_submit_rejects_missing_idempotency_key_before_network(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(202, json={"job_id": "job-17"})

    async with ReceiverClient(settings(tmp_path), httpx.MockTransport(handler)) as client:
        with pytest.raises(ReceiverRejectedError, match="outgoing_idempotency_key_invalid"):
            await client.submit({"prompt": "text"})

    assert calls == 0


@pytest.mark.asyncio
async def test_outgoing_body_limit_is_measured_on_exact_utf8_json(tmp_path: Path) -> None:
    payload = {"idempotency_key": "card:add:17", "prompt": "Расчёт"}
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.content == encoded
        return httpx.Response(202, json={"job_id": "job-17"})

    exact = settings(tmp_path, max_outgoing_body_bytes=len(encoded))
    async with ReceiverClient(exact, httpx.MockTransport(handler)) as client:
        assert await client.submit(payload) == "job-17"

    too_small = settings(tmp_path, max_outgoing_body_bytes=len(encoded) - 1)
    async with ReceiverClient(too_small, httpx.MockTransport(handler)) as client:
        with pytest.raises(DataRejectedError) as raised:
            await client.submit(payload)
        assert raised.value.code == "outgoing_body_too_large"
        assert "общий размер" in raised.value.user_message

    assert calls == 1


@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        (400, ReceiverRejectedError),
        (422, ReceiverRejectedError),
        (429, ReceiverUnavailableError),
        (500, ReceiverUnavailableError),
        (503, ReceiverUnavailableError),
    ],
)
@pytest.mark.asyncio
async def test_receiver_http_failures_are_classified_for_retry(
    tmp_path: Path,
    status: int,
    error_type: type[Exception],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status)

    payload = {"idempotency_key": "card:add:17", "prompt": "text"}
    async with ReceiverClient(settings(tmp_path), httpx.MockTransport(handler)) as client:
        with pytest.raises(error_type):
            await client.submit(payload)


@pytest.mark.asyncio
async def test_receiver_timeout_is_retryable_unavailability(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("receiver was too slow", request=request)

    payload = {"idempotency_key": "card:add:17", "prompt": "text"}
    async with ReceiverClient(settings(tmp_path), httpx.MockTransport(handler)) as client:
        with pytest.raises(ReceiverUnavailableError) as raised:
            await client.submit(payload)

    assert raised.value.retryable is True
