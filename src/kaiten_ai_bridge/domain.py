"""Pure transformation rules for Kaiten data.

This module deliberately performs no persistence, network calls, OCR, or semantic
interpretation.  It turns already-fetched data into a deterministic transport
document and validates files as opaque byte strings.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import PurePath
from typing import Any

from .errors import DataRejectedError

UTC = UTC
_SENSITIVE_KEYS = {"email", "e-mail", "token", "secret", "password", "authorization"}
_DISPLAY_KEYS = ("display_name", "full_name", "name", "title", "label", "text")


@dataclass(frozen=True, slots=True)
class NormalizedField:
    name: str
    kind: str
    value: str
    property_id: int | None = None


@dataclass(frozen=True, slots=True)
class AttachmentRef:
    file_id: str | None
    name: str
    size: int | None
    mime_type: str | None
    download_url: str | None
    source: str
    comment_id: int | None = None
    api_content_path: str | None = None
    custom_property_id: int | None = None
    comment_uid: str | None = None
    custom_property_uid: str | None = None
    entity_type: str | None = None


@dataclass(frozen=True, slots=True)
class PreparedFile:
    reference: AttachmentRef
    content: bytes = field(repr=False)
    sha256: str | None = None

    @property
    def digest(self) -> str:
        return self.sha256 or hashlib.sha256(self.content).hexdigest()


@dataclass(frozen=True, slots=True)
class CardSnapshot:
    card_id: int
    url: str
    title: str
    request_type: str
    service: str | None = None
    space: str | None = None
    board: str | None = None
    column: str | None = None
    column_id: int | None = None
    system_type: str | None = None
    author: str | None = None
    fields: tuple[NormalizedField, ...] = ()
    initial_comments: tuple[str, ...] = ()
    attachments: tuple[AttachmentRef, ...] = ()


@dataclass(frozen=True, slots=True)
class CommentSnapshot:
    comment_id: int
    card_id: int
    card_url: str
    text: str
    author: str | None
    created_at: datetime
    is_public: bool
    author_id: int | None = None
    attachments: tuple[AttachmentRef, ...] = ()


def utc_now() -> datetime:
    return datetime.now(UTC)


def iso_z(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _redact_unknown(value: Any) -> Any:
    """Make an unknown API value serializable without leaking common secrets/PII."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            lowered = key_text.casefold()
            if (
                lowered == "id"
                or lowered.endswith(("_id", "_ids", "_uid", "_uids"))
                or any(sensitive in lowered for sensitive in _SENSITIVE_KEYS)
            ):
                cleaned[key_text] = "[скрыто]"
            else:
                cleaned[key_text] = _redact_unknown(item)
        return cleaned
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_redact_unknown(item) for item in value]
    return str(value)


def safe_json(value: Any) -> str:
    return json.dumps(
        _redact_unknown(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _display_from_mapping(value: Mapping[str, Any]) -> str | None:
    for key in _DISPLAY_KEYS:
        item = value.get(key)
        if item not in (None, ""):
            return str(item)
    first = value.get("first_name") or value.get("firstName")
    last = value.get("last_name") or value.get("lastName")
    combined = " ".join(str(item).strip() for item in (first, last) if item)
    return combined or None


def author_display_name(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, Mapping):
        return _display_from_mapping(value)
    return str(value)


def _lookup(value: Any, labels: Mapping[str, str] | None) -> str | None:
    if not labels:
        return None
    key = str(value)
    return labels.get(key)


def _sequence_text(kind: str, values: Iterable[Any], labels: Mapping[str, str] | None) -> str:
    rendered = [humanize_value(kind, item, labels=labels) for item in values]
    return ", ".join(item for item in rendered if item)


def humanize_value(kind: str | None, value: Any, *, labels: Mapping[str, str] | None = None) -> str:
    """Return a user-facing representation for a dynamic Kaiten field value."""

    normalized_kind = (kind or "unknown").strip().casefold().replace("-", "_")
    if value is None or value == "":
        return ""
    if isinstance(value, (list, tuple, set, frozenset)):
        return _sequence_text(normalized_kind, value, labels)

    if normalized_kind in {"bool", "boolean", "checkbox"}:
        if isinstance(value, str):
            return "да" if value.strip().casefold() in {"1", "true", "yes", "да"} else "нет"
        return "да" if bool(value) else "нет"

    if normalized_kind in {"date", "datetime", "due_date"}:
        if isinstance(value, datetime):
            return iso_z(value)
        if isinstance(value, date):
            return value.isoformat()
        text = str(value).strip()
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                return date.fromisoformat(text).isoformat()
            except ValueError:
                return f"{text} [дата не распознана]"
        return iso_z(parsed)

    if normalized_kind in {"number", "numeric", "integer", "float", "money"}:
        try:
            number = Decimal(str(value))
            return format(number.normalize(), "f")
        except (InvalidOperation, ValueError):
            return str(value)

    if normalized_kind in {"select", "multi_select", "catalog", "directory", "user", "users"}:
        if isinstance(value, Mapping):
            display = _display_from_mapping(value)
            if display:
                return display
            raw_id = value.get("id") or value.get("value_id") or value.get("user_id")
            resolved = _lookup(raw_id, labels)
            if resolved:
                return resolved
            if normalized_kind in {"user", "users"}:
                # An unresolved Kaiten user value commonly contains only an
                # internal user ID. The specification explicitly forbids putting
                # that identifier (or an email) into the user-facing prompt.
                return "[пользователь не расшифрован]"
            return f"{safe_json(value)} [значение не расшифровано]"
        resolved = _lookup(value, labels)
        if resolved:
            return resolved
        if normalized_kind in {"user", "users"}:
            return "[пользователь не расшифрован]"
        if (
            normalized_kind in {"catalog", "directory"}
            and isinstance(value, str)
            and not value.isdigit()
        ):
            return value
        return f"{value} [значение не расшифровано]"

    if normalized_kind in {"attachment", "file", "files"}:
        if isinstance(value, Mapping):
            name = _display_from_mapping(value) or value.get("filename") or "файл"
            size = value.get("size") or value.get("file_size")
            return (
                f"{name} — {human_size(int(size))}" if isinstance(size, (int, float)) else str(name)
            )
        return str(value)

    if normalized_kind in {
        "string",
        "text",
        "url",
        "phone",
        "email",
        "regular",
        "title",
        "description",
    }:
        return str(value)

    if isinstance(value, Mapping):
        display = _display_from_mapping(value)
        if display:
            return f"{display} [неизвестный тип: {kind or 'unknown'}]"
    return f"{safe_json(value)} [неизвестный тип: {kind or 'unknown'}]"


def escape_markdown(value: str) -> str:
    """Escape user-controlled text while preserving line breaks and wording."""

    escaped = value.replace("\\", "\\\\")
    escaped = re.sub(r"([`*_{}\[\]<>|])", r"\\\1", escaped)
    lines: list[str] = []
    for line in escaped.splitlines() or [""]:
        line = re.sub(r"^(\s*)(#{1,6}|[-+]|\d+[.)])\s", r"\1\\\2 ", line)
        lines.append(line)
    return "\n  ".join(lines)


def human_size(size: int) -> str:
    if size < 1024:
        return f"{size} Б"
    if size < 1024**2:
        return f"{size / 1024:.1f} КиБ"
    return f"{size / 1024**2:.2f} МиБ"


def _bullet(label: str, value: str | int) -> str:
    return f"- {escape_markdown(str(label))}: {escape_markdown(str(value))}"


def _nonempty(value: str | None) -> bool:
    return value is not None and bool(value.strip())


def build_initial_prompt(card: CardSnapshot, files: Sequence[AttachmentRef]) -> str:
    """Build the fixed Markdown template for an initial request."""

    lines = ["# Заявка в корпоративную ИИ", "", "## Карточка"]
    lines.append(_bullet("Название", card.title))
    lines.append(_bullet("Тип заявки", card.request_type))
    card_items = (
        ("Системный тип карточки", card.system_type),
        ("Сервис", card.service),
        ("Пространство", card.space),
        ("Доска", card.board),
        ("Колонка", card.column),
        ("Автор", card.author),
        ("Ссылка", card.url),
    )
    lines.extend(_bullet(label, value) for label, value in card_items if _nonempty(value))

    fields = sorted(
        (item for item in card.fields if _nonempty(item.value)),
        key=lambda item: (item.name.casefold(), item.property_id or -1),
    )
    if fields:
        lines.extend(["", "## Данные заявки"])
        lines.extend(_bullet(item.name, item.value) for item in fields)

    comments = [item for item in card.initial_comments if _nonempty(item)]
    if comments:
        lines.extend(["", "## Комментарий пользователя", ""])
        lines.append("\n\n".join(escape_markdown(item) for item in comments))

    if files:
        lines.extend(["", "## Приложенные файлы"])
        for item in sorted(files, key=lambda f: (f.name.casefold(), f.file_id or "", f.source)):
            size = human_size(item.size) if item.size is not None else "размер неизвестен"
            lines.append(
                f"- {escape_markdown(item.name)} — {escape_markdown(size)} — "
                f"{escape_markdown(item.source)}"
            )

    return "\n".join(lines).rstrip() + "\n"


def build_comment_prompt(comment: CommentSnapshot, files: Sequence[AttachmentRef]) -> str:
    if not _nonempty(comment.text) and not files:
        raise DataRejectedError("empty_request", "Пустой комментарий без файлов не был обработан.")

    lines = [
        "# Заявка в корпоративную ИИ",
        "",
        "## Карточка",
        _bullet("Ссылка", comment.card_url),
    ]
    if _nonempty(comment.author):
        lines.append(_bullet("Автор", comment.author or ""))
    if _nonempty(comment.text):
        lines.extend(["", "## Комментарий пользователя", "", escape_markdown(comment.text)])
    if files:
        lines.extend(["", "## Приложенные файлы"])
        for item in sorted(files, key=lambda f: (f.name.casefold(), f.file_id or "", f.source)):
            size = human_size(item.size) if item.size is not None else "размер неизвестен"
            lines.append(
                f"- {escape_markdown(item.name)} — {escape_markdown(size)} — "
                f"{escape_markdown(item.source)}"
            )
    return "\n".join(lines).rstrip() + "\n"


def deduplicate_refs(files: Sequence[AttachmentRef]) -> list[AttachmentRef]:
    result: list[AttachmentRef] = []
    seen_ids: set[str] = set()
    seen_fallback: set[tuple[str, int | None, str | None]] = set()
    for item in files:
        if item.file_id:
            if item.file_id in seen_ids:
                continue
            seen_ids.add(item.file_id)
        else:
            key = (item.name.casefold(), item.size, item.download_url)
            if key in seen_fallback:
                continue
            seen_fallback.add(key)
        result.append(item)
    return result


def deduplicate_prepared(files: Sequence[PreparedFile]) -> list[PreparedFile]:
    result: list[PreparedFile] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for item in files:
        file_id = item.reference.file_id
        if file_id:
            if file_id in seen_ids:
                continue
            seen_ids.add(file_id)
        digest = item.digest
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        result.append(item)
    return result


def validate_file_refs(
    files: Sequence[AttachmentRef], *, max_bytes: int, allowed_extensions: Sequence[str]
) -> None:
    allowed = {item.casefold() for item in allowed_extensions}
    violations: list[str] = []
    for item in files:
        extension = PurePath(item.name).suffix.casefold()
        if extension not in allowed:
            violations.append(f"Формат файла «{item.name}» не поддерживается.")
        if item.size is not None and item.size > max_bytes:
            violations.append(
                f"Файл «{item.name}» превышает допустимый размер {human_size(max_bytes)}."
            )
    if violations:
        message = "Запрос не был обработан. " + " ".join(violations)
        raise DataRejectedError("invalid_files", message)


def validate_prepared_files(
    files: Sequence[PreparedFile], *, max_bytes: int, allowed_extensions: Sequence[str]
) -> None:
    validate_file_refs(
        [
            AttachmentRef(
                file_id=item.reference.file_id,
                name=item.reference.name,
                size=len(item.content),
                mime_type=item.reference.mime_type,
                download_url=None,
                source=item.reference.source,
                comment_id=item.reference.comment_id,
                api_content_path=None,
                custom_property_id=item.reference.custom_property_id,
                comment_uid=item.reference.comment_uid,
                custom_property_uid=item.reference.custom_property_uid,
                entity_type=item.reference.entity_type,
            )
            for item in files
        ],
        max_bytes=max_bytes,
        allowed_extensions=allowed_extensions,
    )


def build_outgoing_payload(
    *,
    schema_version: int,
    event_type: str,
    idempotency_key: str,
    card_id: int,
    card_url: str,
    request_type: str,
    prompt: str,
    files: Sequence[PreparedFile],
    comment_id: int | None = None,
    ai_job_id: str | None = None,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    if event_type not in {"initial", "comment"}:
        raise ValueError("event_type must be initial or comment")
    if not prompt.strip() and not files:
        raise DataRejectedError("empty_request", "Пустой запрос не был обработан.")
    if event_type == "initial" and (comment_id is not None or ai_job_id is not None):
        raise ValueError("initial payload cannot contain comment_id or ai_job_id")
    if event_type == "comment" and (comment_id is None or not ai_job_id):
        raise ValueError("comment payload requires comment_id and ai_job_id")

    return {
        "schema_version": schema_version,
        "event_type": event_type,
        "idempotency_key": idempotency_key,
        "card": {"id": card_id, "url": card_url},
        "comment_id": comment_id,
        "ai_job_id": ai_job_id,
        "request_type": request_type,
        "prompt": prompt,
        "files": [
            {
                "name": item.reference.name,
                "mime_type": item.reference.mime_type or "application/octet-stream",
                "size": len(item.content),
                "sha256": item.digest,
                "content_base64": base64.b64encode(item.content).decode("ascii"),
            }
            for item in sorted(
                files,
                key=lambda prepared: (
                    prepared.reference.name.casefold(),
                    prepared.reference.file_id or "",
                    prepared.digest,
                ),
            )
        ],
        "created_at": iso_z(created_at or utc_now()),
    }


def encoded_json_size(payload: Mapping[str, Any]) -> int:
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
