from __future__ import annotations

import base64
import hashlib
from datetime import UTC, date, datetime

import pytest

from kaiten_ai_bridge.domain import (
    AttachmentRef,
    CardSnapshot,
    CommentSnapshot,
    NormalizedField,
    PreparedFile,
    author_display_name,
    build_comment_prompt,
    build_initial_prompt,
    build_outgoing_payload,
    deduplicate_prepared,
    deduplicate_refs,
    encoded_json_size,
    humanize_value,
    safe_json,
    validate_file_refs,
    validate_prepared_files,
)
from kaiten_ai_bridge.errors import DataRejectedError


def attachment(
    name: str,
    *,
    file_id: str | None = None,
    size: int | None = None,
    source: str = "карточка",
) -> AttachmentRef:
    return AttachmentRef(
        file_id=file_id,
        name=name,
        size=size,
        mime_type="application/octet-stream",
        download_url=f"https://files.example/{file_id or name}",
        source=source,
    )


def prepared(
    name: str,
    content: bytes,
    *,
    file_id: str | None = None,
    sha256: str | None = None,
) -> PreparedFile:
    return PreparedFile(
        reference=attachment(name, file_id=file_id, size=len(content)),
        content=content,
        sha256=sha256,
    )


@pytest.mark.parametrize(
    ("kind", "value", "labels", "expected"),
    [
        ("select", 7, {"7": "ТКП"}, "ТКП"),
        ("select", {"id": 7}, {"7": "ТКП"}, "ТКП"),
        ("catalog", {"id": 20, "title": "Проект Север"}, None, "Проект Север"),
        ("directory", "Справочное значение", None, "Справочное значение"),
        (
            "user",
            {"id": 99, "full_name": "Иван Иванов", "email": "private@example.test"},
            None,
            "Иван Иванов",
        ),
        ("date", date(2026, 9, 1), None, "2026-09-01"),
        ("datetime", "2026-09-01T15:30:00+03:00", None, "2026-09-01T12:30:00Z"),
        ("checkbox", True, None, "да"),
        ("checkbox", "false", None, "нет"),
    ],
)
def test_dynamic_known_field_types_are_human_readable(
    kind: str,
    value: object,
    labels: dict[str, str] | None,
    expected: str,
) -> None:
    assert humanize_value(kind, value, labels=labels) == expected


def test_multi_value_fields_preserve_api_order_and_resolve_each_value() -> None:
    assert (
        humanize_value("multi_select", [2, 1], labels={"1": "Первый", "2": "Второй"})
        == "Второй, Первый"
    )


def test_unknown_field_is_deterministic_serializable_and_redacted() -> None:
    value = {
        "z": {"api_token": "top-secret", "password": "hunter2"},
        "email": "private@example.test",
        "internal_user_id": 42,
        "a": [float("inf"), datetime(2026, 9, 1, 12, tzinfo=UTC)],
    }

    rendered = humanize_value("future_kaiten_type", value)

    assert rendered == (
        '{"a":["inf","2026-09-01T12:00:00+00:00"],'
        '"email":"[скрыто]","internal_user_id":"[скрыто]",'
        '"z":{"api_token":"[скрыто]",'
        '"password":"[скрыто]"}} [неизвестный тип: future_kaiten_type]'
    )
    assert "top-secret" not in rendered
    assert "hunter2" not in rendered
    assert "private@example.test" not in rendered
    assert "42" not in rendered
    assert safe_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'


def test_author_name_uses_only_display_value() -> None:
    author = {
        "id": 42,
        "full_name": "Мария Петрова",
        "email": "maria@example.test",
        "token": "secret",
    }
    assert author_display_name(author) == "Мария Петрова"


@pytest.mark.parametrize("value", [99, {"id": 99}, {"user_id": 99, "email": "x@test"}])
def test_unresolved_user_value_never_exposes_internal_id_or_email(value: object) -> None:
    rendered = humanize_value("user", value)

    assert rendered == "[пользователь не расшифрован]"
    assert "99" not in rendered
    assert "@" not in rendered


def test_initial_prompt_is_deterministic_sorted_and_markdown_safe() -> None:
    fields = (
        NormalizedField("Язык", "text", "Python *3.14*", property_id=3),
        NormalizedField("Бюджет", "money", "1000", property_id=2),
        NormalizedField("Пустое", "text", "", property_id=1),
    )
    files = (
        attachment("zeta.pdf", file_id="2", size=1024, source="поле `Документы`"),
        attachment("Alpha.xlsx", file_id="1", size=5_242_880, source="карточка"),
    )
    card = CardSnapshot(
        card_id=77,
        url="https://tagat.kaiten.ru/space/834556/boards/card/77?a=1&b=2",
        title="# Нужен <расчёт>",
        request_type="Запрос | срочный",
        service="ИИ",
        space="ИИ",
        board="Доска агента",
        column="Очередь",
        system_type="Тендерная заявка",
        author="Иван *Иванов*",
        fields=fields,
        initial_comments=("Первая строка\n- не список",),
        attachments=files,
    )
    reordered = CardSnapshot(
        card_id=card.card_id,
        url=card.url,
        title=card.title,
        request_type=card.request_type,
        service=card.service,
        space=card.space,
        board=card.board,
        column=card.column,
        system_type=card.system_type,
        author=card.author,
        fields=tuple(reversed(fields)),
        initial_comments=card.initial_comments,
        attachments=tuple(reversed(files)),
    )

    prompt = build_initial_prompt(card, files)

    assert prompt == build_initial_prompt(reordered, tuple(reversed(files)))
    assert prompt.startswith("# Заявка в корпоративную ИИ\n\n## Карточка\n")
    assert "- Название: \\# Нужен \\<расчёт\\>" in prompt
    assert "- Тип заявки: Запрос \\| срочный" in prompt
    assert "- Автор: Иван \\*Иванов\\*" in prompt
    assert prompt.index("- Бюджет: 1000") < prompt.index("- Язык: Python \\*3.14\\*")
    assert "Пустое" not in prompt
    assert "Первая строка\n  \\- не список" in prompt
    assert prompt.index("Alpha.xlsx") < prompt.index("zeta.pdf")
    assert "Alpha.xlsx — 5.00 МиБ — карточка" in prompt
    assert "zeta.pdf — 1.0 КиБ — поле \\`Документы\\`" in prompt
    assert prompt.endswith("\n")


def test_comment_prompt_supports_text_only_files_only_and_rejects_empty() -> None:
    comment = CommentSnapshot(
        comment_id=8,
        card_id=7,
        card_url="https://tagat.kaiten.ru/card/7",
        text="Новая **задача**",
        author="Анна",
        created_at=datetime(2026, 9, 1, 12, tzinfo=UTC),
        is_public=True,
    )
    text_prompt = build_comment_prompt(comment, [])
    assert "Новая \\*\\*задача\\*\\*" in text_prompt
    assert "## Приложенные файлы" not in text_prompt

    files_only = CommentSnapshot(
        comment_id=9,
        card_id=7,
        card_url=comment.card_url,
        text="  ",
        author=None,
        created_at=comment.created_at,
        is_public=True,
    )
    file_prompt = build_comment_prompt(files_only, [attachment("only.pdf", file_id="9", size=10)])
    assert "## Комментарий пользователя" not in file_prompt
    assert "only.pdf — 10 Б — карточка" in file_prompt

    with pytest.raises(DataRejectedError) as raised:
        build_comment_prompt(files_only, [])
    assert raised.value.code == "empty_request"


def test_file_reference_validation_is_all_or_nothing_and_lists_every_violation() -> None:
    files = [
        attachment("valid.PDF", file_id="1", size=100),
        attachment("archive.zip", file_id="2", size=100),
        attachment("huge.docx", file_id="3", size=5_242_881),
        attachment("huge.exe", file_id="4", size=5_242_882),
    ]

    with pytest.raises(DataRejectedError) as raised:
        validate_file_refs(files, max_bytes=5_242_880, allowed_extensions=[".pdf", ".docx"])

    assert raised.value.code == "invalid_files"
    message = raised.value.user_message
    assert "valid.PDF" not in message
    assert "archive.zip" in message and "не поддерживается" in message
    assert "huge.docx" in message and "превышает" in message
    assert message.count("huge.exe") == 2


def test_prepared_file_validation_uses_downloaded_bytes_not_declared_size() -> None:
    reference = attachment("actual.pdf", file_id="1", size=1)
    file = PreparedFile(reference=reference, content=b"x" * 11)

    with pytest.raises(DataRejectedError) as raised:
        validate_prepared_files([file], max_bytes=10, allowed_extensions=[".pdf"])
    assert "actual.pdf" in raised.value.user_message
    assert "превышает" in raised.value.user_message


def test_file_reference_deduplication_prefers_kaiten_id() -> None:
    first = attachment("first.pdf", file_id="55", size=100)
    same_id_other_surface = attachment("renamed.pdf", file_id="55", size=200)
    no_id = attachment("loose.pdf", size=3)
    same_fallback = AttachmentRef(
        file_id=None,
        name="LOOSE.PDF",
        size=3,
        mime_type=no_id.mime_type,
        download_url=no_id.download_url,
        source=no_id.source,
    )

    assert deduplicate_refs([first, same_id_other_surface, no_id, same_fallback]) == [first, no_id]


def test_prepared_deduplication_uses_id_then_content_hash_globally() -> None:
    first = prepared("first.pdf", b"same", file_id="id-1")
    same_id_different_bytes = prepared("renamed.pdf", b"different", file_id="id-1")
    same_hash_other_id = prepared("copy.pdf", b"same", file_id="id-2")
    # Although id-2 was encountered on a hash duplicate, the ID must remain known.
    same_second_id_new_bytes = prepared("changed.pdf", b"new", file_id="id-2")
    unique_without_id = prepared("unique.pdf", b"unique")
    hash_copy_without_id = prepared("unique-copy.pdf", b"unique")

    assert deduplicate_prepared(
        [
            first,
            same_id_different_bytes,
            same_hash_other_id,
            same_second_id_new_bytes,
            unique_without_id,
            hash_copy_without_id,
        ]
    ) == [first, unique_without_id]


def test_outgoing_payload_matches_schema_and_encodes_opaque_bytes() -> None:
    content = b"\x00binary\xffcontent"
    file = prepared("example.bin", content, file_id="123")
    created = datetime(2026, 9, 1, 15, 0, tzinfo=datetime.now().astimezone().tzinfo)

    payload = build_outgoing_payload(
        schema_version=1,
        event_type="comment",
        idempotency_key="comment:add:456",
        card_id=123,
        card_url="https://tagat.kaiten.ru/space/834556/boards/card/123",
        comment_id=456,
        ai_job_id="job-123",
        request_type="Новый тип без релиза",
        prompt="",
        files=[file],
        created_at=created,
    )

    assert list(payload) == [
        "schema_version",
        "event_type",
        "idempotency_key",
        "card",
        "comment_id",
        "ai_job_id",
        "request_type",
        "prompt",
        "files",
        "created_at",
    ]
    assert payload["card"] == {
        "id": 123,
        "url": "https://tagat.kaiten.ru/space/834556/boards/card/123",
    }
    assert payload["comment_id"] == 456
    assert payload["ai_job_id"] == "job-123"
    assert payload["created_at"].endswith("Z")
    assert payload["files"] == [
        {
            "name": "example.bin",
            "mime_type": "application/octet-stream",
            "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "content_base64": base64.b64encode(content).decode("ascii"),
        }
    ]
    assert encoded_json_size(payload) >= len(content)


def test_outgoing_payload_file_order_is_deterministic() -> None:
    alpha = prepared("Alpha.pdf", b"alpha", file_id="2")
    zeta = prepared("zeta.pdf", b"zeta", file_id="1")
    created = datetime(2026, 9, 1, 12, tzinfo=UTC)

    def payload(files: list[PreparedFile]) -> dict[str, object]:
        return build_outgoing_payload(
            schema_version=1,
            event_type="initial",
            idempotency_key="card:add:1",
            card_id=1,
            card_url="https://tagat.kaiten.ru/card/1",
            request_type="Запрос",
            prompt="text",
            files=files,
            created_at=created,
        )

    assert payload([zeta, alpha]) == payload([alpha, zeta])
    assert [item["name"] for item in payload([zeta, alpha])["files"]] == [
        "Alpha.pdf",
        "zeta.pdf",
    ]


def test_outgoing_payload_allows_prompt_only_and_rejects_both_empty() -> None:
    prompt_only = build_outgoing_payload(
        schema_version=1,
        event_type="initial",
        idempotency_key="card:add:1",
        card_id=1,
        card_url="https://tagat.kaiten.ru/card/1",
        request_type="Запрос",
        prompt="Сделайте расчёт",
        files=[],
        created_at=datetime(2026, 9, 1, 12, tzinfo=UTC),
    )
    assert prompt_only["comment_id"] is None
    assert prompt_only["ai_job_id"] is None
    assert prompt_only["files"] == []

    with pytest.raises(DataRejectedError) as raised:
        build_outgoing_payload(
            schema_version=1,
            event_type="initial",
            idempotency_key="card:add:1",
            card_id=1,
            card_url="https://tagat.kaiten.ru/card/1",
            request_type="Запрос",
            prompt=" \n",
            files=[],
        )
    assert raised.value.code == "empty_request"


@pytest.mark.parametrize(
    "arguments",
    [
        {"event_type": "other"},
        {"event_type": "initial", "comment_id": 1},
        {"event_type": "initial", "ai_job_id": "job"},
        {"event_type": "comment", "comment_id": None, "ai_job_id": "job"},
        {"event_type": "comment", "comment_id": 1, "ai_job_id": None},
    ],
)
def test_outgoing_payload_enforces_initial_and_comment_contract(arguments: dict) -> None:
    defaults = {
        "schema_version": 1,
        "event_type": "initial",
        "idempotency_key": "key",
        "card_id": 1,
        "card_url": "https://tagat.kaiten.ru/card/1",
        "request_type": "Запрос",
        "prompt": "text",
        "files": [],
    }
    defaults.update(arguments)
    with pytest.raises(ValueError):
        build_outgoing_payload(**defaults)
