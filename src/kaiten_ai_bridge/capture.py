"""Loopback-only diagnostic receiver for full local integration tests."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response


def _authorized(request: Request, expected: str) -> bool:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    return scheme.casefold() == "bearer" and hmac.compare_digest(token, expected)


def _read_limit(value: str | None) -> int:
    try:
        parsed = int(value or 20 * 1024 * 1024)
    except ValueError as exc:
        raise ValueError("BRIDGE_CAPTURE_MAX_BODY_BYTES must be an integer") from exc
    if parsed <= 0:
        raise ValueError("BRIDGE_CAPTURE_MAX_BODY_BYTES must be positive")
    return parsed


async def _limited_body(request: Request, maximum: int) -> bytes:
    declared = request.headers.get("content-length")
    if declared:
        try:
            if int(declared) > maximum:
                raise OverflowError
        except ValueError as exc:
            raise ValueError("invalid_content_length") from exc
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > maximum:
            raise OverflowError
        body.extend(chunk)
    return bytes(body)


def _validate_payload(payload: Any, idempotency_key: str | None) -> tuple[int, str]:
    if not isinstance(payload, dict):
        raise ValueError("json_object_required")
    if not idempotency_key or payload.get("idempotency_key") != idempotency_key:
        raise ValueError("idempotency_key_mismatch")
    event_type = payload.get("event_type")
    if event_type not in {"initial", "comment"}:
        raise ValueError("invalid_event_type")
    card = payload.get("card")
    if not isinstance(card, dict) or isinstance(card.get("id"), bool):
        raise ValueError("invalid_card")
    try:
        card_id = int(card.get("id"))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_card") from exc
    if card_id <= 0:
        raise ValueError("invalid_card")
    if not isinstance(payload.get("prompt"), str) or not isinstance(payload.get("files"), list):
        raise ValueError("invalid_content")
    return card_id, event_type


def _write_capture(output_dir: Path, name: str, body: bytes) -> Path:
    directory = output_dir.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=directory
    )
    temporary = Path(temporary_name)
    try:
        os.chmod(temporary, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        os.chmod(target, 0o600)
        return target
    except BaseException:
        if descriptor >= 0:
            with suppress(OSError):
                os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def create_capture_app(
    *, output_dir: Path, bearer_token: str, maximum_body_bytes: int = 20 * 1024 * 1024
) -> FastAPI:
    if not bearer_token:
        raise ValueError("bearer_token must not be empty")
    app = FastAPI(
        title="Local Corporate AI Capture",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    async def accept_job(request: Request) -> Response:
        if not _authorized(request, bearer_token):
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
        content_type = request.headers.get("content-type", "").split(";", 1)[0].casefold()
        if content_type != "application/json":
            return JSONResponse({"detail": "unsupported_media_type"}, status_code=415)
        try:
            body = await _limited_body(request, maximum_body_bytes)
            payload = json.loads(body)
            key = request.headers.get("idempotency-key")
            card_id, event_type = _validate_payload(payload, key)
        except OverflowError:
            return JSONResponse({"detail": "request_too_large"}, status_code=413)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            return JSONResponse({"detail": str(exc) or "invalid_json"}, status_code=422)

        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        job_id = f"local-{digest[:24]}"
        filename = f"card-{card_id}-{event_type}-{digest[:12]}.json"
        _write_capture(output_dir, filename, body)
        return JSONResponse({"job_id": job_id}, status_code=202)

    async def receiver_health(request: Request) -> Response:
        if not _authorized(request, bearer_token):
            return Response(status_code=401)
        return Response(status_code=204)

    app.add_api_route("/api/v1/jobs", accept_job, methods=["POST"])
    app.add_api_route("/api/v1/jobs", receiver_health, methods=["HEAD"])
    return app


def main() -> None:
    token = os.environ.get("BRIDGE_AI_BEARER_TOKEN", "")
    if not token:
        raise SystemExit("BRIDGE_AI_BEARER_TOKEN is required")
    output_dir = Path(os.environ.get("BRIDGE_CAPTURE_DIR", "data/captures"))
    maximum = _read_limit(os.environ.get("BRIDGE_CAPTURE_MAX_BODY_BYTES"))
    app = create_capture_app(
        output_dir=output_dir,
        bearer_token=token,
        maximum_body_bytes=maximum,
    )
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8090,
        access_log=False,
        log_config=None,
    )


if __name__ == "__main__":
    main()
