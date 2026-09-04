"""Validated inbound and outbound HTTP contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Trigger(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_type: Literal["card:add", "comment:add"]
    card_id: int = Field(gt=0)
    comment_id: int | None = Field(default=None, gt=0)
    source_created_at: datetime | None = None

    @model_validator(mode="after")
    def comment_requires_id(self) -> Trigger:
        if self.event_type == "comment:add" and self.comment_id is None:
            raise ValueError("comment:add requires comment_id")
        if self.event_type == "card:add" and self.comment_id is not None:
            raise ValueError("card:add must not contain comment_id")
        return self

    @property
    def event_key(self) -> str:
        suffix = self.comment_id if self.comment_id is not None else self.card_id
        return f"{self.event_type}:{suffix}"


class ResultFile(BaseModel):
    """Reserved forward-compatible shape; publication awaits an agreed contract."""

    model_config = ConfigDict(extra="allow")

    name: str | None = None


class CallbackResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    text: str | None = None
    files: list[ResultFile] = Field(default_factory=list)


class CallbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(ge=1)
    result_id: str = Field(min_length=1, max_length=256)
    job_id: str = Field(min_length=1, max_length=256)
    status: str = Field(min_length=1, max_length=64)
    result: CallbackResult | dict[str, Any] | str | None = None
    created_at: datetime

    def result_text(self) -> str | None:
        if isinstance(self.result, CallbackResult):
            return self.result.text
        if isinstance(self.result, str):
            return self.result
        if isinstance(self.result, dict):
            value = self.result.get("text")
            return value if isinstance(value, str) else None
        return None

    def has_result_files(self) -> bool:
        if isinstance(self.result, CallbackResult):
            return bool(self.result.files)
        if isinstance(self.result, dict):
            value = self.result.get("files")
            return isinstance(value, list) and bool(value)
        return False


class AcceptedJob(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: str | None = None
    job_id: str = Field(min_length=1, max_length=256)


class WebhookAccepted(BaseModel):
    accepted: bool
    duplicate: bool = False
    event_key: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    process: Literal["running"] = "running"
    kaiten_api: Literal["available", "unavailable", "unknown"]
    receiver: Literal["available", "unavailable", "unknown"]
    last_webhook_at: str | None
    last_reconciliation_at: str | None
    last_reconciliation_result: str | None
    pending_events: int
    error_events: int
    queue_size: int
    last_cleanup_at: str | None
