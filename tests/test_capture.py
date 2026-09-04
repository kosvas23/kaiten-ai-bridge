from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from kaiten_ai_bridge.capture import create_capture_app


def test_capture_receiver_persists_exact_json_and_is_idempotent(tmp_path: Path) -> None:
    app = create_capture_app(output_dir=tmp_path, bearer_token="local-secret")
    payload = {
        "schema_version": 1,
        "event_type": "initial",
        "idempotency_key": "card:add:17",
        "card": {"id": 17, "url": "https://example.test/card/17"},
        "comment_id": None,
        "ai_job_id": None,
        "request_type": "Тест",
        "prompt": "Проверка",
        "files": [],
        "created_at": "2026-09-03T00:00:00Z",
    }
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    headers = {
        "Authorization": "Bearer local-secret",
        "Content-Type": "application/json",
        "Idempotency-Key": "card:add:17",
    }

    with TestClient(app) as client:
        first = client.post("/api/v1/jobs", content=body, headers=headers)
        second = client.post("/api/v1/jobs", content=body, headers=headers)
        health = client.head("/api/v1/jobs", headers={"Authorization": "Bearer local-secret"})

    assert first.status_code == 202
    assert second.json() == first.json()
    assert health.status_code == 204
    captures = list(tmp_path.glob("*.json"))
    assert len(captures) == 1
    assert captures[0].read_bytes() == body


def test_capture_receiver_rejects_unauthorized_and_invalid_payload(tmp_path: Path) -> None:
    app = create_capture_app(output_dir=tmp_path, bearer_token="local-secret")
    with TestClient(app) as client:
        unauthorized = client.post("/api/v1/jobs", json={})
        invalid = client.post(
            "/api/v1/jobs",
            json={"idempotency_key": "wrong"},
            headers={
                "Authorization": "Bearer local-secret",
                "Idempotency-Key": "key",
            },
        )

    assert unauthorized.status_code == 401
    assert invalid.status_code == 422
    assert not list(tmp_path.glob("*.json"))
