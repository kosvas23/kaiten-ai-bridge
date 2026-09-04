"""HTTP adapter for the corporate AI receiver."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import httpx
from pydantic import ValidationError

from .config import Settings
from .errors import DataRejectedError, ReceiverRejectedError, ReceiverUnavailableError
from .schemas import AcceptedJob


class ReceiverClient:
    def __init__(
        self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self.settings = settings
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=settings.ai_connect_timeout_seconds,
                read=settings.ai_read_timeout_seconds,
                write=settings.ai_read_timeout_seconds,
                pool=settings.ai_connect_timeout_seconds,
            ),
            headers={
                "Authorization": f"Bearer {settings.ai_bearer_token.get_secret_value()}",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "kaiten-ai-bridge/0.1",
            },
            transport=transport,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> ReceiverClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def submit(self, payload: Mapping[str, Any]) -> str:
        key = payload.get("idempotency_key")
        if not isinstance(key, str) or not key.strip():
            raise ReceiverRejectedError("outgoing_idempotency_key_invalid")
        try:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ReceiverRejectedError("outgoing_payload_invalid") from exc
        maximum = self.settings.max_outgoing_body_bytes
        if maximum is not None and len(body) > maximum:
            raise DataRejectedError(
                "outgoing_body_too_large",
                "Запрос не был обработан: общий размер текста и файлов превышает "
                "настроенный лимит передачи. Уменьшите количество или размер файлов "
                "и добавьте новый комментарий.",
            )
        try:
            response = await self._client.post(
                self.settings.ai_url,
                content=body,
                headers={"Idempotency-Key": key},
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise ReceiverUnavailableError() from exc
        if response.status_code != 202:
            if response.status_code == 429 or response.status_code >= 500:
                raise ReceiverUnavailableError(f"receiver_http_{response.status_code}")
            raise ReceiverRejectedError(f"receiver_http_{response.status_code}")
        try:
            accepted = AcceptedJob.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise ReceiverRejectedError("receiver_invalid_acceptance") from exc
        job_id = accepted.job_id.strip()
        if not job_id:
            raise ReceiverRejectedError("receiver_invalid_acceptance")
        return job_id

    async def healthcheck(self) -> bool:
        try:
            response = await self._client.request("HEAD", self.settings.ai_url)
        except (httpx.TimeoutException, httpx.NetworkError):
            return False
        return response.status_code < 500
