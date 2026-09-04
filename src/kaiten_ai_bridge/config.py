"""Environment-only runtime configuration.

Secrets are represented by ``SecretStr`` and are never included in repr/log output.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urljoin, urlparse

from pydantic import BeforeValidator, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _csv(value: object) -> object:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            return json.loads(stripped)
        return [item.strip() for item in stripped.split(",") if item.strip()]
    return value


CsvStrings = Annotated[list[str], BeforeValidator(_csv)]
CsvInts = Annotated[list[int], BeforeValidator(_csv)]


DEFAULT_EXTENSIONS = [
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".csv",
    ".txt",
    ".md",
    ".sql",
    ".json",
    ".xml",
    ".ppt",
    ".pptx",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
]


class Settings(BaseSettings):
    """Settings loaded exclusively from ``BRIDGE_*`` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="BRIDGE_",
        env_file=None,
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
    )

    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    log_dir: Path = Path("logs")
    database_path: Path = Path("data/bridge.sqlite3")
    temp_dir: Path = Path("tmp")

    bind_host: str = "127.0.0.1"
    bind_port: int = Field(default=8080, ge=1, le=65535)
    webhook_path: str = "/webhooks/kaiten"
    callback_path: str = "/callbacks/corporate-ai"
    health_path: str = "/health"
    callback_bearer_token: SecretStr | None = None
    callback_success_statuses: CsvStrings = Field(default_factory=lambda: ["completed"])
    max_incoming_body_bytes: int = Field(default=2 * 1024 * 1024, gt=0)
    callback_reservation_timeout_seconds: int = Field(default=120, ge=10)
    health_probe_timeout_seconds: float = Field(default=5.0, gt=0, le=30)

    kaiten_base_url: str
    kaiten_token: SecretStr
    kaiten_space_id: int = Field(gt=0)
    kaiten_space_name: str = "ИИ"
    kaiten_board_id: int = Field(gt=0)
    kaiten_board_name: str = "Доска агента"
    kaiten_queue_column_id: int = Field(gt=0)
    kaiten_queue_column_name: str = "Очередь"
    kaiten_work_column_id: int = Field(gt=0)
    kaiten_work_column_name: str = "В работе"
    kaiten_done_column_id: int = Field(gt=0)
    kaiten_done_column_name: str = "Готово"
    kaiten_service_id: int | None = Field(default=None, gt=0)
    kaiten_service_name: str = "ИИ"
    kaiten_request_type_field: str = "Тип заявки"
    kaiten_request_type_field_id: int | None = None
    kaiten_technical_author_ids: CsvInts = Field(default_factory=list)
    kaiten_requests_per_second: int = Field(default=45, ge=1, le=50)

    ai_scheme: Literal["http", "https"] = "http"
    ai_host: str = "127.0.0.1"
    ai_port: int = Field(default=8090, ge=1, le=65535)
    ai_path: str = "/api/v1/jobs"
    ai_bearer_token: SecretStr
    ai_connect_timeout_seconds: float = Field(default=5.0, gt=0)
    ai_read_timeout_seconds: float = Field(default=60.0, gt=0)
    kaiten_timeout_seconds: float = Field(default=30.0, gt=0)

    schema_version: int = Field(default=1, ge=1)
    max_file_bytes: int = Field(default=5 * 1024 * 1024, gt=0)
    allowed_extensions: CsvStrings = Field(default_factory=lambda: list(DEFAULT_EXTENSIONS))
    max_outgoing_body_bytes: int | None = Field(default=None, gt=0)
    retry_schedule_seconds: CsvInts = Field(
        default_factory=lambda: [0, 2, 5, 10, 20, 40, 80, 150, 300]
    )
    reconciliation_interval_seconds: int = Field(default=6 * 60 * 60, ge=60)
    cleanup_interval_seconds: int = Field(default=24 * 60 * 60, ge=60)
    activity_retention_days: int = Field(default=30, ge=1)
    warning_before_close_hours: int = Field(default=24, ge=1)
    dedup_retention_days: int = Field(default=3, ge=1)
    log_retention_days: int = Field(default=3, ge=1)
    temp_retention_days: int = Field(default=3, ge=1)
    card_parallelism: int = Field(default=4, ge=1, le=64)
    worker_poll_seconds: float = Field(default=0.5, gt=0, le=60)
    reconcile_on_startup: bool = True
    cleanup_on_startup: bool = True

    @field_validator("webhook_path", "callback_path", "health_path", "ai_path")
    @classmethod
    def paths_start_with_slash(cls, value: str) -> str:
        return value if value.startswith("/") else f"/{value}"

    @field_validator("allowed_extensions")
    @classmethod
    def normalize_extensions(cls, values: list[str]) -> list[str]:
        normalized = sorted({v.lower() if v.startswith(".") else f".{v.lower()}" for v in values})
        if not normalized:
            raise ValueError("allowed_extensions must not be empty")
        return normalized

    @field_validator("retry_schedule_seconds")
    @classmethod
    def validate_retry_schedule(cls, values: list[int]) -> list[int]:
        if not values or values[0] != 0 or any(v < 0 for v in values):
            raise ValueError("retry schedule must start at 0 and contain non-negative seconds")
        if values != sorted(values):
            raise ValueError("retry schedule must be ascending")
        return values

    @field_validator("callback_success_statuses")
    @classmethod
    def normalize_callback_statuses(cls, values: list[str]) -> list[str]:
        normalized = sorted({value.strip() for value in values if value.strip()})
        if not normalized:
            raise ValueError("callback_success_statuses must not be empty")
        return normalized

    @field_validator("kaiten_technical_author_ids")
    @classmethod
    def validate_technical_author_ids(cls, values: list[int]) -> list[int]:
        if any(value <= 0 for value in values):
            raise ValueError("technical author IDs must be positive")
        return sorted(set(values))

    @model_validator(mode="after")
    def validate_distinct_columns(self) -> Settings:
        for name, secret in (
            ("kaiten token", self.kaiten_token),
            ("AI bearer token", self.ai_bearer_token),
        ):
            if not secret.get_secret_value().strip():
                raise ValueError(f"{name} must not be empty")
        if (
            self.callback_bearer_token is not None
            and not self.callback_bearer_token.get_secret_value().strip()
        ):
            raise ValueError("callback bearer token must not be empty")

        kaiten_url = urlparse(self.kaiten_base_url)
        if kaiten_url.scheme not in {"http", "https"} or not kaiten_url.netloc:
            raise ValueError("kaiten_base_url must be an absolute HTTP(S) URL")
        if kaiten_url.username is not None or kaiten_url.password is not None:
            raise ValueError("kaiten_base_url must not contain credentials")
        if self.environment == "production" and kaiten_url.scheme != "https":
            raise ValueError("Kaiten API must use HTTPS in production")

        columns = {
            self.kaiten_queue_column_id,
            self.kaiten_work_column_id,
            self.kaiten_done_column_id,
        }
        if len(columns) != 3:
            raise ValueError("queue, work and done column IDs must be distinct")
        if self.warning_before_close_hours >= self.activity_retention_days * 24:
            raise ValueError("warning interval must be shorter than activity retention")
        if self.environment == "production":
            if self.callback_bearer_token is None:
                raise ValueError("callback bearer token is required in production")
            if self.max_outgoing_body_bytes is None:
                raise ValueError("maximum outgoing body size is required in production")
        return self

    @property
    def kaiten_api_root(self) -> str:
        base = self.kaiten_base_url.rstrip("/") + "/"
        lowered = base.rstrip("/").lower()
        if lowered.endswith("/api/latest") or re.search(r"/api/v\d+$", lowered):
            return base.rstrip("/")
        return urljoin(base, "api/latest")

    @property
    def ai_url(self) -> str:
        return f"{self.ai_scheme}://{self.ai_host}:{self.ai_port}{self.ai_path}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
