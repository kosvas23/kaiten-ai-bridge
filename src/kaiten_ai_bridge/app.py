"""FastAPI application factory."""

from __future__ import annotations

import hmac
import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from .config import Settings, get_settings
from .errors import BridgeError, InvalidWebhookError, UnsupportedResultFilesError
from .kaiten import KaitenClient
from .logging_config import configure_logging
from .receiver import ReceiverClient
from .schemas import CallbackRequest, HealthResponse, WebhookAccepted
from .service import (
    BridgeService,
    CallbackBusyError,
    CallbackConflictError,
    CallbackUnknownJobError,
    InvalidCallbackError,
)
from .storage import SQLiteStorage
from .webhook import parse_webhook


async def _json_body(request: Request, *, maximum: int) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().casefold()
    if content_type != "application/json":
        raise ValueError("unsupported_media_type")
    declared = request.headers.get("content-length")
    if declared:
        try:
            declared_size = int(declared)
            if declared_size < 0:
                raise ValueError("invalid_content_length")
            if declared_size > maximum:
                raise OverflowError("request_too_large")
        except ValueError as exc:
            raise ValueError("invalid_content_length") from exc
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > maximum:
            raise OverflowError("request_too_large")
        body.extend(chunk)
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid_json") from exc
    if not isinstance(payload, dict):
        raise ValueError("json_object_required")
    return payload


def _bearer_authorized(request: Request, expected: str | None) -> bool:
    if expected is None:
        return True
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    return scheme.casefold() == "bearer" and hmac.compare_digest(token, expected)


def create_app(
    settings: Settings | None = None,
    *,
    storage: SQLiteStorage | None = None,
    kaiten: KaitenClient | None = None,
    receiver: ReceiverClient | None = None,
    start_background: bool = True,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings)
    storage = storage or SQLiteStorage(settings.database_path)
    kaiten = kaiten or KaitenClient(settings)
    receiver = receiver or ReceiverClient(settings)
    service = BridgeService(settings, storage, kaiten, receiver)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await storage.initialize()
        if start_background:
            await service.start()
        app.state.bridge = service
        try:
            yield
        finally:
            if start_background:
                await service.stop()
            await receiver.close()
            await kaiten.close()
            await storage.close()

    app = FastAPI(
        title="Kaiten AI Bridge",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    async def webhook_endpoint(request: Request) -> Response:
        try:
            payload = await _json_body(request, maximum=settings.max_incoming_body_bytes)
        except OverflowError:
            return JSONResponse({"detail": "request_too_large"}, status_code=413)
        except ValueError as exc:
            status = 415 if str(exc) == "unsupported_media_type" else 400
            return JSONResponse({"detail": str(exc)}, status_code=status)
        try:
            trigger = parse_webhook(payload)
        except InvalidWebhookError:
            return JSONResponse({"detail": "invalid_webhook"}, status_code=422)
        if trigger is None:
            await storage.set_service_status("last_webhook", "ignored")
            return Response(status_code=204)
        try:
            event, inserted = await service.enqueue_trigger(trigger)
        except Exception:
            # No 2xx: Kaiten must retry because the durable write was not confirmed.
            return JSONResponse({"detail": "queue_unavailable"}, status_code=503)
        response = WebhookAccepted(
            accepted=True,
            duplicate=not inserted,
            event_key=event.event_key,
        )
        return JSONResponse(response.model_dump(), status_code=202)

    async def callback_endpoint(request: Request) -> Response:
        expected = (
            settings.callback_bearer_token.get_secret_value()
            if settings.callback_bearer_token is not None
            else None
        )
        if not _bearer_authorized(request, expected):
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
        try:
            payload = await _json_body(request, maximum=settings.max_incoming_body_bytes)
            callback = CallbackRequest.model_validate(payload)
        except OverflowError:
            return JSONResponse({"detail": "request_too_large"}, status_code=413)
        except ValidationError:
            return JSONResponse({"detail": "invalid_callback"}, status_code=422)
        except ValueError as exc:
            status = 415 if str(exc) == "unsupported_media_type" else 400
            return JSONResponse({"detail": str(exc)}, status_code=status)

        try:
            result = await service.handle_callback(callback)
        except (InvalidCallbackError, UnsupportedResultFilesError) as exc:
            code = exc.code if isinstance(exc, BridgeError) else str(exc)
            return JSONResponse({"detail": code}, status_code=422)
        except CallbackConflictError:
            return JSONResponse({"detail": "result_id_conflict"}, status_code=409)
        except (CallbackBusyError, CallbackUnknownJobError):
            return JSONResponse({"detail": "callback_retry_required"}, status_code=503)
        except BridgeError:
            return JSONResponse({"detail": "kaiten_temporarily_unavailable"}, status_code=503)
        except Exception:
            return JSONResponse({"detail": "callback_processing_failed"}, status_code=503)
        return JSONResponse({"status": result.outcome, "card_id": result.card_id}, status_code=200)

    async def health_endpoint() -> HealthResponse:
        return HealthResponse.model_validate(await service.health())

    async def live_endpoint() -> dict[str, str]:
        return {"status": "ok"}

    app.add_api_route(
        settings.webhook_path,
        webhook_endpoint,
        methods=["POST"],
        name="kaiten-webhook",
    )
    app.add_api_route(
        settings.callback_path,
        callback_endpoint,
        methods=["POST"],
        name="corporate-ai-callback",
    )
    app.add_api_route(
        settings.health_path,
        health_endpoint,
        methods=["GET"],
        response_model=HealthResponse,
        name="health",
    )
    app.add_api_route("/live", live_endpoint, methods=["GET"], name="liveness")
    return app
