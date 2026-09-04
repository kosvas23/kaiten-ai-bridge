from __future__ import annotations

import pytest
from pydantic import ValidationError

from kaiten_ai_bridge.config import Settings


def settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "kaiten_base_url": "https://tagat.kaiten.ru",
        "kaiten_token": "kaiten-super-secret",
        "kaiten_space_id": 834_556,
        "kaiten_board_id": 1_868_751,
        "kaiten_queue_column_id": 6_467_479,
        "kaiten_work_column_id": 6_467_480,
        "kaiten_done_column_id": 6_467_481,
        "ai_bearer_token": "ai-super-secret",
    }
    values.update(overrides)
    return Settings(**values)


def test_urls_paths_dynamic_lists_and_file_limit_are_externalized() -> None:
    configured = settings(
        kaiten_base_url="https://tagat.kaiten.ru/api/latest/",
        ai_scheme="https",
        ai_host="ai.internal",
        ai_port=9443,
        ai_path="v2/jobs",
        callback_path="callbacks/result",
        allowed_extensions="PDF, .DocX, pdf",
        retry_schedule_seconds="0,2,5,10,20,40,80,150,300",
        kaiten_technical_author_ids="12, 15",
        max_file_bytes=5_242_880,
    )

    assert configured.kaiten_api_root == "https://tagat.kaiten.ru/api/latest"
    assert configured.ai_url == "https://ai.internal:9443/v2/jobs"
    assert configured.callback_path == "/callbacks/result"
    assert configured.allowed_extensions == [".docx", ".pdf"]
    assert configured.retry_schedule_seconds == [0, 2, 5, 10, 20, 40, 80, 150, 300]
    assert configured.kaiten_technical_author_ids == [12, 15]
    assert configured.max_file_bytes == 5 * 1024 * 1024


@pytest.mark.parametrize(
    ("base", "expected"),
    [
        ("https://tagat.kaiten.ru", "https://tagat.kaiten.ru/api/latest"),
        ("https://tagat.kaiten.ru/", "https://tagat.kaiten.ru/api/latest"),
        ("https://tagat.kaiten.ru/api/latest", "https://tagat.kaiten.ru/api/latest"),
        ("https://tagat.kaiten.ru/api/v1/", "https://tagat.kaiten.ru/api/v1"),
    ],
)
def test_kaiten_api_root_is_normalized_without_duplicate_api_path(base: str, expected: str) -> None:
    assert settings(kaiten_base_url=base).kaiten_api_root == expected


def test_secrets_are_masked_in_repr_and_dump() -> None:
    configured = settings()
    representation = repr(configured)
    dumped = repr(configured.model_dump())

    assert "kaiten-super-secret" not in representation
    assert "ai-super-secret" not in representation
    assert "kaiten-super-secret" not in dumped
    assert "ai-super-secret" not in dumped
    assert configured.kaiten_token.get_secret_value() == "kaiten-super-secret"


@pytest.mark.parametrize(
    "overrides",
    [
        {"kaiten_work_column_id": 6_467_479},
        {"allowed_extensions": ""},
        {"retry_schedule_seconds": "2,5,10"},
        {"retry_schedule_seconds": "0,10,5"},
        {"retry_schedule_seconds": "0,-1,2"},
        {"kaiten_space_id": 0},
        {"kaiten_service_id": -1},
        {"kaiten_technical_author_ids": "12,-1"},
        {"kaiten_token": "  "},
        {"ai_bearer_token": ""},
        {"kaiten_base_url": "not-a-url"},
        {"kaiten_base_url": "https://user:password@tagat.kaiten.ru"},
    ],
)
def test_invalid_operational_configuration_is_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        settings(**overrides)


def test_warning_must_precede_activity_expiry() -> None:
    with pytest.raises(ValidationError):
        settings(activity_retention_days=1, warning_before_close_hours=24)


def test_production_requires_callback_auth_and_total_body_limit() -> None:
    with pytest.raises(ValidationError):
        settings(environment="production")
    with pytest.raises(ValidationError):
        settings(
            environment="production",
            callback_bearer_token="callback-secret",
        )
    with pytest.raises(ValidationError):
        settings(
            environment="production",
            callback_bearer_token="callback-secret",
            max_outgoing_body_bytes=20 * 1024 * 1024,
            kaiten_base_url="http://tagat.kaiten.ru",
        )
    with pytest.raises(ValidationError):
        settings(
            environment="production",
            callback_bearer_token=" ",
            max_outgoing_body_bytes=20 * 1024 * 1024,
        )

    configured = settings(
        environment="production",
        callback_bearer_token="callback-secret",
        max_outgoing_body_bytes=20 * 1024 * 1024,
        callback_success_statuses="completed,done,completed",
    )
    assert configured.callback_success_statuses == ["completed", "done"]
