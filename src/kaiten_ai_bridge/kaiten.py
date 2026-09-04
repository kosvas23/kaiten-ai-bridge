"""Narrow, defensive adapter for the Kaiten REST API."""

from __future__ import annotations

import asyncio
import ipaddress
import os
import tempfile
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from .config import Settings
from .domain import (
    AttachmentRef,
    CardSnapshot,
    CommentSnapshot,
    NormalizedField,
    PreparedFile,
    author_display_name,
    deduplicate_refs,
    human_size,
    humanize_value,
)
from .errors import (
    DataRejectedError,
    KaitenProtocolError,
    KaitenUnavailableError,
    SourceNotReadyError,
)

_STANDARD_FIELDS: dict[str, tuple[str, str]] = {
    "description": ("Описание", "text"),
    "due_date": ("Срок", "date"),
    "planned_start": ("Плановое начало", "date"),
    "planned_end": ("Плановое завершение", "date"),
    "size": ("Размер", "number"),
    "size_text": ("Размер (текст)", "string"),
    "estimate_workload": ("Оценка трудозатрат", "number"),
    "asap": ("Срочно", "boolean"),
    "blocked": ("Заблокировано", "boolean"),
}

_REDIRECT_STATUSES = {301, 302, 303, 307, 308}

_GENERIC_STANDARD_EXCLUDED = {
    "archived",
    "author",
    "blocker",
    "blocks",
    "board",
    "card_properties",
    "checklists",
    "children",
    "column",
    "comments",
    "condition",
    "created",
    "external",
    "files",
    "id",
    "lane",
    "members",
    "name",
    "owner",
    "parents",
    "permissions",
    "properties",
    "public",
    "schema_version",
    "sd_new_comment",
    "service",
    "share_settings",
    "slas",
    "sort_order",
    "space",
    "tags",
    "title",
    "type",
    "uid",
    "updated",
    "version",
}
_GENERIC_STANDARD_SENSITIVE_PARTS = (
    "authorization",
    "avatar",
    "email",
    "password",
    "secret",
    "token",
)
_GENERIC_STANDARD_TECHNICAL_SUFFIXES = (
    "_at",
    "_count",
    "_id",
    "_ids",
    "_total",
    "_uid",
    "_uids",
    "_url",
)


class _SlidingWindowRateLimiter:
    """Bound authenticated Kaiten calls to a rolling one-second window."""

    def __init__(
        self,
        requests_per_second: int,
        *,
        clock: Callable[[], float],
        sleep: Callable[[float], Awaitable[None]],
    ) -> None:
        self._limit = requests_per_second
        self._clock = clock
        self._sleep = sleep
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = self._clock()
                cutoff = now - 1.0
                while self._timestamps and self._timestamps[0] <= cutoff:
                    self._timestamps.popleft()
                if len(self._timestamps) < self._limit:
                    self._timestamps.append(now)
                    return
                delay = max(0.0, self._timestamps[0] + 1.0 - now)
            await self._sleep(delay)


def _api_v1_root(base_url: str) -> str:
    """Build Kaiten's stable JSON API v1 root from the configured company URL."""

    parsed = urlparse(base_url)
    return parsed._replace(path="/api/v1/", params="", query="", fragment="").geturl()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Mapping):
        for key in ("data", "items", "results"):
            nested = value.get(key)
            if isinstance(nested, list):
                return nested
    return [value]


def _identifier(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _nested_name(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    for key in ("title", "name", "value", "full_name"):
        result = value.get(key)
        if result not in (None, ""):
            return str(result)
    return None


def _author_identifier(value: Mapping[str, Any]) -> int | None:
    """Resolve both documented flattened and nested comment author shapes."""

    author_id = _identifier(value.get("author_id"))
    if author_id is not None:
        return author_id
    author = value.get("author")
    return _identifier(author.get("id")) if isinstance(author, Mapping) else None


def _is_initial_service_comment(comment: Mapping[str, Any]) -> bool:
    """Identify the immutable user request created by Service Desk.

    Kaiten may attribute this comment to the same account whose token is used by
    the bridge.  Its ``sd_description`` flag, rather than its author, is the
    authoritative origin signal.  Bridge-authored comments never have this flag
    and remain excluded by the technical-author checks in ``comment_snapshot``.
    """

    return (
        comment.get("sd_description") is True
        and comment.get("internal") is not True
        and comment.get("deleted") is not True
    )


def _is_card_or_property_attachment(reference: AttachmentRef) -> bool:
    if reference.custom_property_id is not None:
        return True
    return reference.comment_id is None and reference.entity_type != "comment"


def _validated_download_url(value: str) -> str:
    """Accept only absolute HTTPS storage URLs outside obvious local networks."""

    parsed = urlparse(value)
    hostname = (parsed.hostname or "").rstrip(".").casefold()
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or hostname in {"localhost", "localhost.localdomain"}
    ):
        raise KaitenProtocolError("file_url_invalid")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return value
    if (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    ):
        raise KaitenProtocolError("file_url_invalid")
    return value


class KaitenClient:
    """Only the Kaiten operations authorized by the specification."""

    def __init__(
        self,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
        *,
        rate_limit_clock: Callable[[], float] | None = None,
        rate_limit_sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self.settings = settings
        self._api_limiter = _SlidingWindowRateLimiter(
            settings.kaiten_requests_per_second,
            clock=rate_limit_clock or time.monotonic,
            sleep=rate_limit_sleep or asyncio.sleep,
        )
        self._client = httpx.AsyncClient(
            base_url=_api_v1_root(settings.kaiten_base_url),
            headers={
                "Authorization": f"Bearer {settings.kaiten_token.get_secret_value()}",
                "Accept": "application/json",
                "User-Agent": "kaiten-ai-bridge/0.1",
                "X-Kaiten-Client": "kaiten-bridge",
                "X-Kaiten-Client-Version": "0.1.0",
            },
            timeout=httpx.Timeout(settings.kaiten_timeout_seconds),
            follow_redirects=False,
            transport=transport,
        )
        # Never attach the Kaiten bearer token to legacy or signed storage URLs.
        self._download_client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.kaiten_timeout_seconds),
            follow_redirects=False,
            headers={"User-Agent": "kaiten-ai-bridge/0.1"},
            transport=transport,
        )
        self._metadata_lock = asyncio.Lock()
        self._property_cache: tuple[float, list[dict[str, Any]]] | None = None
        self._select_cache: dict[int, tuple[float, dict[str, str]]] = {}
        self._catalog_cache: dict[int, tuple[float, dict[str, str]]] = {}
        self._users_cache: dict[int, str] = {}
        self._current_user_id: int | None = None

    async def close(self) -> None:
        await self._download_client.aclose()
        await self._client.aclose()

    async def __aenter__(self) -> KaitenClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def _request(
        self,
        method: str,
        path_or_url: str,
        *,
        expected: set[int] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        await self._api_limiter.acquire()
        try:
            response = await self._client.request(method, path_or_url, **kwargs)
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise KaitenUnavailableError() from exc
        if expected is not None and response.status_code in expected:
            return response
        if response.status_code == 404:
            raise SourceNotReadyError()
        if response.status_code == 429 or response.status_code >= 500:
            raise KaitenUnavailableError(f"kaiten_http_{response.status_code}")
        if 300 <= response.status_code < 400:
            raise KaitenProtocolError(f"kaiten_http_{response.status_code}")
        if response.is_error:
            raise KaitenProtocolError(f"kaiten_http_{response.status_code}")
        return response

    async def _json(self, method: str, path: str, **kwargs: Any) -> Any:
        response = await self._request(method, path, **kwargs)
        try:
            return response.json()
        except ValueError as exc:
            raise KaitenProtocolError("kaiten_invalid_json") from exc

    async def healthcheck(self) -> bool:
        try:
            await self._request("GET", "users/current")
        except (KaitenUnavailableError, KaitenProtocolError, SourceNotReadyError):
            return False
        return True

    async def get_card(self, card_id: int) -> dict[str, Any]:
        data = await self._json("GET", f"cards/{card_id}", params={"broken_api": "false"})
        if not isinstance(data, dict) or _identifier(data.get("id")) != card_id:
            raise SourceNotReadyError("card response is not ready")
        return data

    async def list_target_cards(self) -> list[dict[str, Any]]:
        """List live cards in queue/work columns, handling Kaiten's 100-row page."""

        result: list[dict[str, Any]] = []
        offset = 0
        while True:
            page = _as_list(
                await self._json(
                    "GET",
                    "cards",
                    params={
                        "space_id": self.settings.kaiten_space_id,
                        "board_id": self.settings.kaiten_board_id,
                        "column_ids": (
                            f"{self.settings.kaiten_queue_column_id},"
                            f"{self.settings.kaiten_work_column_id}"
                        ),
                        "condition": 1,
                        "limit": 100,
                        "offset": offset,
                        "broken_api": "false",
                    },
                )
            )
            result.extend(item for item in page if isinstance(item, dict))
            if len(page) < 100:
                break
            offset += len(page)
        return result

    async def list_comments(self, card_id: int) -> list[dict[str, Any]]:
        # Unlike the card list, Kaiten's comments route is not paginated and does
        # not accept limit/offset. Sending those parameters and looping can repeat
        # the same 100+ row response forever.
        data = _as_list(await self._json("GET", f"cards/{card_id}/comments"))
        return [item for item in data if isinstance(item, dict)]

    async def get_comment(self, card_id: int, comment_id: int) -> dict[str, Any]:
        comments = await self.list_comments(card_id)
        for comment in comments:
            if _identifier(comment.get("id")) == comment_id:
                return comment
        raise SourceNotReadyError("comment is not ready")

    async def custom_properties(self, *, force: bool = False) -> list[dict[str, Any]]:
        now = time.monotonic()
        cached = self._property_cache
        if not force and cached and now - cached[0] < 300:
            return cached[1]
        async with self._metadata_lock:
            cached = self._property_cache
            if not force and cached and now - cached[0] < 300:
                return cached[1]
            properties: list[dict[str, Any]] = []
            offset = 0
            while True:
                page = _as_list(
                    await self._json(
                        "GET",
                        "company/custom-properties",
                        params={"compact": "true", "limit": 100, "offset": offset},
                    )
                )
                properties.extend(item for item in page if isinstance(item, dict))
                if len(page) < 100:
                    break
                offset += len(page)
            self._property_cache = (time.monotonic(), properties)
            return properties

    async def select_labels(self, property_id: int) -> dict[str, str]:
        now = time.monotonic()
        cached = self._select_cache.get(property_id)
        if cached and now - cached[0] < 300:
            return cached[1]
        labels: dict[str, str] = {}
        offset = 0
        while True:
            page = _as_list(
                await self._json(
                    "GET",
                    f"company/custom-properties/{property_id}/select-values",
                    params={
                        "limit": 100,
                        "offset": offset,
                        "v2_select_search": "true",
                    },
                )
            )
            for item in page:
                if not isinstance(item, Mapping):
                    continue
                item_id = item.get("id")
                label = item.get("value") or item.get("name") or item.get("title")
                if item_id is not None and label not in (None, ""):
                    labels[str(item_id)] = str(label)
            if len(page) < 100:
                break
            offset += len(page)
        self._select_cache[property_id] = (time.monotonic(), labels)
        return labels

    async def catalog_labels(self, property_id: int) -> dict[str, str]:
        now = time.monotonic()
        cached = self._catalog_cache.get(property_id)
        if cached and now - cached[0] < 300:
            return cached[1]
        labels: dict[str, str] = {}
        offset = 0
        while True:
            page = _as_list(
                await self._json(
                    "GET",
                    f"company/custom-properties/{property_id}/catalog-values",
                    params={"limit": 100, "offset": offset},
                )
            )
            for item in page:
                if not isinstance(item, Mapping):
                    continue
                item_id = item.get("id")
                label = item.get("name") or item.get("title") or item.get("value")
                if item_id is not None and isinstance(label, (str, int, float)):
                    labels[str(item_id)] = str(label)
            if len(page) < 100:
                break
            offset += len(page)
        self._catalog_cache[property_id] = (time.monotonic(), labels)
        return labels

    async def current_user_id(self) -> int | None:
        if self._current_user_id is not None:
            return self._current_user_id
        data = await self._json("GET", "users/current")
        if isinstance(data, Mapping):
            self._current_user_id = _identifier(data.get("id"))
        return self._current_user_id

    async def user_labels(self, ids: Sequence[int]) -> dict[str, str]:
        missing = sorted({item for item in ids if item not in self._users_cache})
        for start in range(0, len(missing), 100):
            chunk = missing[start : start + 100]
            if not chunk:
                continue
            users = _as_list(
                await self._json(
                    "GET",
                    "users",
                    params={"ids": ",".join(str(item) for item in chunk), "limit": 100},
                )
            )
            for user in users:
                if not isinstance(user, Mapping):
                    continue
                user_id = _identifier(user.get("id"))
                display = author_display_name(user)
                if user_id and display:
                    self._users_cache[user_id] = display
        return {
            str(item): self._users_cache[str_id]
            for item in ids
            if (str_id := item) in self._users_cache
        }

    @staticmethod
    def _property_values(card: Mapping[str, Any]) -> dict[int, Any]:
        values: dict[int, Any] = {}
        properties = card.get("properties")
        if isinstance(properties, Mapping):
            for key, value in properties.items():
                text = str(key)
                if text.startswith("id_"):
                    property_id = _identifier(text[3:])
                    if property_id:
                        values[property_id] = value
        for item in _as_list(card.get("card_properties")):
            if not isinstance(item, Mapping):
                continue
            property_id = _identifier(item.get("property_id") or item.get("custom_property_id"))
            if property_id:
                values[property_id] = item.get("value")
        return values

    @staticmethod
    def _attachment_refs(card: Mapping[str, Any]) -> list[AttachmentRef]:
        result: list[AttachmentRef] = []
        for raw in _as_list(card.get("files")):
            if not isinstance(raw, Mapping) or raw.get("deleted") is True:
                continue
            name = raw.get("name") or raw.get("filename")
            if not isinstance(name, str) or not name.strip():
                continue
            property_id = _identifier(raw.get("custom_property_id"))
            comment_id = _identifier(raw.get("comment_id"))
            source = (
                f"кастомное поле {property_id}"
                if property_id
                else (f"комментарий {comment_id}" if comment_id else "карточка")
            )
            size = _identifier(raw.get("size"))
            file_id = str(raw.get("id")) if raw.get("id") is not None else None
            api_content_path: str | None = None
            restricted = _identifier(raw.get("type")) == 11
            if restricted and file_id:
                card_uid = raw.get("card_uid") or card.get("uid")
                entity_type = raw.get("entity_type")
                if entity_type == "comment" and card_uid and raw.get("comment_uid"):
                    api_content_path = (
                        f"cards/{card_uid}/comments/{raw['comment_uid']}/files/{file_id}/content"
                    )
                elif (
                    entity_type == "custom_property" and card_uid and raw.get("custom_property_uid")
                ):
                    api_content_path = (
                        f"cards/{card_uid}/custom-properties/{raw['custom_property_uid']}"
                        f"/files/{file_id}/content"
                    )
                elif card_uid:
                    api_content_path = f"cards/{card_uid}/files/{file_id}/content"
            result.append(
                AttachmentRef(
                    file_id=file_id,
                    name=name,
                    size=size,
                    mime_type=str(raw.get("mime_type")) if raw.get("mime_type") else None,
                    # A type=11 URL is temporary. Always request a fresh signed URL
                    # through the authenticated UID route immediately before use.
                    download_url=(
                        str(raw.get("url")) if raw.get("url") and not restricted else None
                    ),
                    source=source,
                    comment_id=comment_id,
                    api_content_path=api_content_path,
                    custom_property_id=property_id,
                    comment_uid=(str(raw.get("comment_uid")) if raw.get("comment_uid") else None),
                    custom_property_uid=(
                        str(raw.get("custom_property_uid"))
                        if raw.get("custom_property_uid")
                        else None
                    ),
                    entity_type=str(raw.get("entity_type")) if raw.get("entity_type") else None,
                )
            )
        return deduplicate_refs(result)

    async def _enrich_attachment_refs(
        self,
        card: Mapping[str, Any],
        attachments: Sequence[AttachmentRef],
        comments: Sequence[Mapping[str, Any]],
    ) -> list[AttachmentRef]:
        definitions = await self.custom_properties()
        property_names: dict[int, str] = {}
        property_uids: dict[int, str] = {}
        for definition in definitions:
            property_id = _identifier(definition.get("id"))
            if not property_id:
                continue
            if definition.get("name"):
                property_names[property_id] = str(definition["name"])
            if definition.get("uid"):
                property_uids[property_id] = str(definition["uid"])
        comment_uids = {
            comment_id: str(comment["uid"])
            for comment in comments
            if (comment_id := _identifier(comment.get("id"))) and comment.get("uid")
        }
        card_uid = str(card.get("uid")) if card.get("uid") else None
        enriched: list[AttachmentRef] = []
        for item in attachments:
            source = item.source
            api_path = item.api_content_path
            property_uid = item.custom_property_uid
            comment_uid = item.comment_uid
            if item.custom_property_id:
                property_uid = property_uid or property_uids.get(item.custom_property_id)
                property_name = property_names.get(item.custom_property_id)
                source = (
                    f"кастомное поле «{property_name}»"
                    if property_name
                    else f"кастомное поле {item.custom_property_id}"
                )
                if api_path and card_uid and property_uid and item.file_id:
                    api_path = (
                        f"cards/{card_uid}/custom-properties/{property_uid}"
                        f"/files/{item.file_id}/content"
                    )
            elif item.comment_id:
                comment_uid = comment_uid or comment_uids.get(item.comment_id)
                if api_path and card_uid and comment_uid and item.file_id:
                    api_path = (
                        f"cards/{card_uid}/comments/{comment_uid}/files/{item.file_id}/content"
                    )
            enriched.append(
                replace(
                    item,
                    source=source,
                    api_content_path=api_path,
                    comment_uid=comment_uid,
                    custom_property_uid=property_uid,
                )
            )
        return enriched

    async def _normalized_custom_fields(
        self, card: Mapping[str, Any], attachments: Sequence[AttachmentRef]
    ) -> tuple[list[NormalizedField], int | None]:
        definitions = await self.custom_properties()
        by_id = {
            property_id: item
            for item in definitions
            if (property_id := _identifier(item.get("id"))) is not None
        }
        values = self._property_values(card)
        routing_id = self.settings.kaiten_request_type_field_id
        if routing_id is None:
            target = self.settings.kaiten_request_type_field.strip().casefold()
            routing_id = next(
                (
                    property_id
                    for property_id, definition in by_id.items()
                    if str(definition.get("name", "")).strip().casefold() == target
                ),
                None,
            )

        fields: list[NormalizedField] = []
        for property_id, raw_value in values.items():
            definition = by_id.get(property_id, {})
            name = str(definition.get("name") or f"Поле {property_id}")
            kind = str(definition.get("type") or definition.get("kind") or "unknown")
            if kind.casefold() in {"attachment", "file", "files"}:
                matching = [item for item in attachments if item.custom_property_id == property_id]
                value = ", ".join(
                    f"{item.name} — {human_size(item.size)}" if item.size is not None else item.name
                    for item in matching
                )
            else:
                labels: Mapping[str, str] | None = None
                if kind.casefold() in {"select", "multi_select"}:
                    labels = await self.select_labels(property_id)
                elif kind.casefold() in {"catalog", "directory"}:
                    labels = await self.catalog_labels(property_id)
                elif kind.casefold() in {"user", "users"}:
                    user_ids = [
                        user_id
                        for item in _as_list(raw_value)
                        if (
                            user_id := _identifier(
                                item.get("id") if isinstance(item, Mapping) else item
                            )
                        )
                    ]
                    labels = await self.user_labels(user_ids)
                value = humanize_value(kind, raw_value, labels=labels)
            if value.strip():
                fields.append(
                    NormalizedField(
                        name=name,
                        kind=kind,
                        value=value,
                        property_id=property_id,
                    )
                )
        return fields, routing_id

    @staticmethod
    def _standard_fields(card: Mapping[str, Any]) -> list[NormalizedField]:
        fields: list[NormalizedField] = []
        for key, (label, kind) in _STANDARD_FIELDS.items():
            value = card.get(key)
            if value in (None, "", [], {}):
                continue
            rendered = humanize_value(kind, value)
            if rendered:
                fields.append(NormalizedField(name=label, kind=kind, value=rendered))
        tags = [
            _nested_name(item) for item in _as_list(card.get("tags")) if isinstance(item, Mapping)
        ]
        if any(tags):
            fields.append(
                NormalizedField(
                    name="Метки",
                    kind="tags",
                    value=", ".join(item for item in tags if item),
                )
            )

        # Kaiten can add top-level standard fields before this adapter is
        # released. Preserve safe scalar values, but never walk arbitrary API
        # structures or expose identifiers, contacts, tokens, or technical URLs.
        for key in sorted(
            (item for item in card if isinstance(item, str)),
            key=str.casefold,
        ):
            lowered = key.strip().casefold()
            if (
                not lowered
                or lowered.startswith("_")
                or lowered in _STANDARD_FIELDS
                or lowered in _GENERIC_STANDARD_EXCLUDED
                or lowered.endswith(_GENERIC_STANDARD_TECHNICAL_SUFFIXES)
                or any(part in lowered for part in _GENERIC_STANDARD_SENSITIVE_PARTS)
            ):
                continue
            value = card.get(key)
            if isinstance(value, bool):
                kind = "boolean"
            elif isinstance(value, (int, float)):
                kind = "number"
            elif isinstance(value, str):
                if not value.strip():
                    continue
                kind = "string"
            else:
                continue
            rendered = humanize_value(kind, value)
            if not rendered:
                continue
            label = " ".join(key.replace("-", "_").split("_")).strip()
            label = label[:1].upper() + label[1:]
            fields.append(NormalizedField(name=label, kind=kind, value=rendered))
        return fields

    @staticmethod
    def _card_url(base_url: str, space_id: int, card_id: int) -> str:
        parsed = urlparse(base_url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        return f"{origin}/space/{space_id}/boards/card/{card_id}"

    def _validate_card_scope(self, card: Mapping[str, Any]) -> None:
        if (
            "space_id" in card
            and _identifier(card.get("space_id")) != self.settings.kaiten_space_id
        ):
            raise DataRejectedError(
                "card_outside_space",
                "Карточка не относится к настроенному пространству обработки.",
            )
        if _identifier(card.get("board_id")) != self.settings.kaiten_board_id:
            raise DataRejectedError(
                "card_outside_target",
                "Карточка не относится к настроенной доске обработки.",
            )
        if (
            self.settings.kaiten_service_id is not None
            and _identifier(card.get("service_id")) != self.settings.kaiten_service_id
        ):
            raise DataRejectedError(
                "card_outside_service", "Карточка не относится к настроенному сервису."
            )

    async def card_snapshot(
        self,
        card_id: int,
        *,
        correction_comment_id: int | None = None,
    ) -> CardSnapshot:
        card = await self.get_card(card_id)
        self._validate_card_scope(card)
        column_id = _identifier(card.get("column_id"))
        if column_id not in {
            self.settings.kaiten_queue_column_id,
            self.settings.kaiten_work_column_id,
        }:
            raise DataRejectedError("card_not_active", "Карточка уже не находится в обработке.")

        comments = await self.list_comments(card_id)
        all_attachments = await self._enrich_attachment_refs(
            card, self._attachment_refs(card), comments
        )
        correction: dict[str, Any] | None = None
        if correction_comment_id is not None:
            correction = next(
                (item for item in comments if _identifier(item.get("id")) == correction_comment_id),
                None,
            )
            if correction is None:
                raise SourceNotReadyError("correction comment is not ready")

        technical_authors = set(self.settings.kaiten_technical_author_ids)
        current_user = await self.current_user_id()
        if current_user:
            technical_authors.add(current_user)
        initial_comments: list[str] = []
        author: str | None = None
        if correction is not None:
            if correction.get("deleted") is True:
                raise DataRejectedError(
                    "deleted_comment", "Удалённый комментарий не обрабатывается."
                )
            if correction.get("internal") is True:
                raise DataRejectedError("internal_comment", "Внутренняя заметка не обрабатывается.")
            if correction.get("sd_description") is True:
                raise DataRejectedError(
                    "initial_service_comment",
                    "Исходный комментарий Service Desk уже входит в первоначальную заявку.",
                )
            author_id = _author_identifier(correction)
            if author_id in technical_authors:
                raise DataRejectedError(
                    "technical_comment", "Служебный комментарий не обрабатывается."
                )
            text = correction.get("text")
            if isinstance(text, str) and text.strip():
                initial_comments.append(text)
            author = author_display_name(correction.get("author"))
            attachments = [
                item for item in all_attachments if item.comment_id == correction_comment_id
            ]
        else:
            initial_comment_ids: set[int] = set()
            for comment in comments:
                if not _is_initial_service_comment(comment):
                    continue
                comment_id = _identifier(comment.get("id"))
                if comment_id is not None:
                    initial_comment_ids.add(comment_id)
                text = comment.get("text")
                if isinstance(text, str) and text.strip():
                    initial_comments.append(text)
                author = author or author_display_name(comment.get("author"))
            author = author or author_display_name(card.get("owner"))
            attachments = [
                item
                for item in all_attachments
                if _is_card_or_property_attachment(item) or item.comment_id in initial_comment_ids
            ]

        custom_fields, routing_id = await self._normalized_custom_fields(card, attachments)
        request_field = next(
            (item for item in custom_fields if routing_id and item.property_id == routing_id),
            None,
        )
        if request_field is None or not request_field.value.strip():
            raise DataRejectedError(
                "request_type_missing",
                "Запрос не был обработан: заполните поле «Тип заявки» и добавьте новый комментарий.",
            )

        fields = self._standard_fields(card)
        fields.extend(item for item in custom_fields if item.property_id != routing_id)
        return CardSnapshot(
            card_id=card_id,
            url=self._card_url(
                self.settings.kaiten_base_url, self.settings.kaiten_space_id, card_id
            ),
            title=str(card.get("title") or ""),
            request_type=request_field.value,
            service=_nested_name(card.get("service")) or self.settings.kaiten_service_name,
            space=self.settings.kaiten_space_name,
            board=_nested_name(card.get("board")) or self.settings.kaiten_board_name,
            column=_nested_name(card.get("column")) or self._configured_column_name(column_id),
            column_id=column_id,
            system_type=_nested_name(card.get("type")),
            author=author,
            fields=tuple(fields),
            initial_comments=tuple(initial_comments),
            attachments=tuple(attachments),
        )

    async def comment_snapshot(self, card_id: int, comment_id: int) -> tuple[CommentSnapshot, str]:
        card = await self.get_card(card_id)
        self._validate_card_scope(card)
        if _identifier(card.get("column_id")) == self.settings.kaiten_done_column_id:
            raise DataRejectedError("card_closed", "Завершённая карточка не обрабатывается.")
        comment = await self.get_comment(card_id, comment_id)
        author_id = _author_identifier(comment)
        if comment.get("deleted") is True:
            raise DataRejectedError("deleted_comment", "Удалённый комментарий не обрабатывается.")
        if comment.get("internal") is True:
            raise DataRejectedError("internal_comment", "Внутренняя заметка не обрабатывается.")
        if comment.get("sd_description") is True:
            raise DataRejectedError(
                "initial_service_comment",
                "Исходный комментарий Service Desk уже входит в первоначальную заявку.",
            )
        technical_authors = set(self.settings.kaiten_technical_author_ids)
        current_user = await self.current_user_id()
        if current_user:
            technical_authors.add(current_user)
        if author_id in technical_authors:
            raise DataRejectedError("technical_comment", "Служебный комментарий не обрабатывается.")

        raw_created = comment.get("created") or comment.get("created_at")
        try:
            created_at = datetime.fromisoformat(str(raw_created).replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            raise SourceNotReadyError("comment timestamp is not ready") from exc
        attachments = [
            item
            for item in await self._enrich_attachment_refs(
                card, self._attachment_refs(card), [comment]
            )
            if item.comment_id == comment_id
        ]
        text = comment.get("text") if isinstance(comment.get("text"), str) else ""
        snapshot = CommentSnapshot(
            comment_id=comment_id,
            card_id=card_id,
            card_url=self._card_url(
                self.settings.kaiten_base_url, self.settings.kaiten_space_id, card_id
            ),
            text=text,
            author=author_display_name(comment.get("author")),
            created_at=created_at,
            is_public=comment.get("internal") is not True,
            author_id=author_id,
            attachments=tuple(attachments),
        )
        request_type = await self.request_type(card)
        return snapshot, request_type

    async def request_type(self, card: Mapping[str, Any]) -> str:
        attachments = self._attachment_refs(card)
        fields, routing_id = await self._normalized_custom_fields(card, attachments)
        result = next(
            (item.value for item in fields if routing_id and item.property_id == routing_id),
            "",
        )
        if not result:
            raise DataRejectedError(
                "request_type_missing",
                "Запрос не был обработан: заполните поле «Тип заявки» и добавьте новый комментарий.",
            )
        return result

    def _configured_column_name(self, column_id: int | None) -> str | None:
        return {
            self.settings.kaiten_queue_column_id: self.settings.kaiten_queue_column_name,
            self.settings.kaiten_work_column_id: self.settings.kaiten_work_column_name,
            self.settings.kaiten_done_column_id: self.settings.kaiten_done_column_name,
        }.get(column_id)

    async def move_card(self, card_id: int, column_id: int) -> None:
        current = await self.get_card(card_id)
        if _identifier(current.get("column_id")) == column_id:
            return
        response = await self._request("PATCH", f"cards/{card_id}", json={"column_id": column_id})
        try:
            data = response.json()
        except ValueError:
            data = None
        if isinstance(data, Mapping):
            actual = _identifier(data.get("column_id"))
            if actual is not None and actual != column_id:
                raise KaitenProtocolError("card_move_not_confirmed")

    async def add_public_comment(self, card_id: int, text: str) -> None:
        if len(text) > 4096:
            raise KaitenProtocolError("comment_too_long")
        data = await self._json("POST", f"cards/{card_id}/comments", json={"text": text})
        if not isinstance(data, Mapping) or data.get("internal") is not False:
            raise KaitenProtocolError("comment_was_not_confirmed_public")

    @asynccontextmanager
    async def _storage_response(
        self,
        url: str,
        *,
        redirects_remaining: int = 3,
    ) -> AsyncIterator[httpx.Response]:
        validated = _validated_download_url(url)
        async with self._download_client.stream("GET", validated) as response:
            if response.status_code not in _REDIRECT_STATUSES:
                yield response
                return
            if redirects_remaining <= 0:
                raise KaitenProtocolError("file_redirect_limit_exceeded")
            location = response.headers.get("location")
            if not location:
                raise KaitenProtocolError("file_redirect_without_location")
            redirected = urljoin(str(response.request.url), location)
            try:
                redirected = _validated_download_url(redirected)
            except KaitenProtocolError as exc:
                raise KaitenProtocolError("file_redirect_invalid_url") from exc
            async with self._storage_response(
                redirected,
                redirects_remaining=redirects_remaining - 1,
            ) as final_response:
                yield final_response

    @asynccontextmanager
    async def _file_response(self, reference: AttachmentRef) -> AsyncIterator[httpx.Response]:
        if reference.api_content_path:
            await self._api_limiter.acquire()
            async with self._client.stream(
                "GET",
                reference.api_content_path,
                params={"download": "true"},
            ) as response:
                if response.status_code not in _REDIRECT_STATUSES:
                    yield response
                    return
                location = response.headers.get("location")
                if not location:
                    raise KaitenProtocolError("file_redirect_without_location")
                signed_url = urljoin(str(response.request.url), location)
                try:
                    signed_url = _validated_download_url(signed_url)
                except KaitenProtocolError as exc:
                    raise KaitenProtocolError("file_redirect_invalid_url") from exc
                # Deliberately switch clients: the Kaiten bearer must never be
                # forwarded to object storage, even for a same-origin signed URL.
                async with self._storage_response(signed_url) as signed_response:
                    yield signed_response
            return

        url = reference.download_url
        async with self._storage_response(url or "") as response:
            yield response

    @asynccontextmanager
    async def downloaded_file(
        self, reference: AttachmentRef, *, max_bytes: int
    ) -> AsyncIterator[PreparedFile]:
        """Stream an opaque file to a mode-0600 temp file, then remove it."""

        if not reference.download_url and not reference.api_content_path:
            raise SourceNotReadyError("file URL is not ready")
        if reference.size is not None and reference.size > max_bytes:
            raise DataRejectedError(
                "invalid_files",
                f"Запрос не был обработан. Файл «{reference.name}» превышает "
                f"допустимый размер {human_size(max_bytes)}.",
            )
        self.settings.temp_dir.mkdir(parents=True, exist_ok=True)
        descriptor, raw_path = tempfile.mkstemp(prefix="bridge-", dir=self.settings.temp_dir)
        os.close(descriptor)
        path = Path(raw_path)
        try:
            try:
                async with self._file_response(reference) as response:
                    if response.status_code == 404:
                        raise SourceNotReadyError("file is not ready")
                    if response.status_code == 422:
                        raise DataRejectedError(
                            "invalid_files",
                            f"Запрос не был обработан. Файл «{reference.name}» "
                            "отклонён Kaiten как небезопасный.",
                        )
                    if response.status_code == 429 or response.status_code >= 500:
                        raise KaitenUnavailableError(f"file_http_{response.status_code}")
                    if response.is_error:
                        raise KaitenProtocolError(f"file_http_{response.status_code}")
                    declared = response.headers.get("content-length")
                    try:
                        declared_size = int(declared) if declared is not None else None
                    except ValueError as exc:
                        raise KaitenProtocolError("file_invalid_content_length") from exc
                    if declared_size is not None and declared_size < 0:
                        raise KaitenProtocolError("file_invalid_content_length")
                    if declared_size is not None and declared_size > max_bytes:
                        raise DataRejectedError(
                            "invalid_files",
                            f"Запрос не был обработан. Файл «{reference.name}» превышает "
                            f"допустимый размер {human_size(max_bytes)}.",
                        )
                    total = 0
                    with await asyncio.to_thread(path.open, "wb") as output:
                        os.chmod(path, 0o600)
                        async for chunk in response.aiter_bytes():
                            total += len(chunk)
                            if total > max_bytes:
                                raise DataRejectedError(
                                    "invalid_files",
                                    f"Запрос не был обработан. Файл «{reference.name}» превышает "
                                    f"допустимый размер {human_size(max_bytes)}.",
                                )
                            await asyncio.to_thread(output.write, chunk)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                raise KaitenUnavailableError() from exc
            content = await asyncio.to_thread(path.read_bytes)
            yield PreparedFile(reference=reference, content=content)
        finally:
            await asyncio.to_thread(path.unlink, missing_ok=True)
