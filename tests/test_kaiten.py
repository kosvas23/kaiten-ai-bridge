from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from kaiten_ai_bridge.config import Settings
from kaiten_ai_bridge.domain import AttachmentRef, build_initial_prompt
from kaiten_ai_bridge.errors import (
    DataRejectedError,
    KaitenProtocolError,
    KaitenUnavailableError,
    SourceNotReadyError,
)
from kaiten_ai_bridge.kaiten import KaitenClient


def settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "environment": "test",
        "temp_dir": tmp_path,
        "kaiten_base_url": "https://tagat.kaiten.ru/api/latest?ignored=true",
        "kaiten_token": "kaiten-secret",
        "kaiten_space_id": 834_556,
        "kaiten_board_id": 1_868_751,
        "kaiten_queue_column_id": 6_467_479,
        "kaiten_work_column_id": 6_467_480,
        "kaiten_done_column_id": 6_467_481,
        "kaiten_requests_per_second": 50,
        "ai_bearer_token": "ai-secret",
    }
    values.update(overrides)
    return Settings(**values)


def attachment(
    *,
    url: str | None = "https://legacy-files.example/report.pdf",
    api_content_path: str | None = None,
    size: int | None = None,
) -> AttachmentRef:
    return AttachmentRef(
        file_id="file-uid",
        name="report.pdf",
        size=size,
        mime_type="application/pdf",
        download_url=url,
        source="карточка",
        api_content_path=api_content_path,
    )


def assert_directory_empty(path: Path) -> None:
    assert list(path.iterdir()) == []


@pytest.mark.asyncio
async def test_stable_v1_root_bearer_and_broken_api_false_are_used(
    tmp_path: Path,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/v1/cards/17":
            return httpx.Response(200, json={"id": 17})
        if request.url.path == "/api/v1/cards":
            return httpx.Response(200, json=[])
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    async with KaitenClient(settings(tmp_path), httpx.MockTransport(handler)) as client:
        assert await client.get_card(17) == {"id": 17}
        assert await client.list_target_cards() == []

    assert [request.url.path for request in requests] == [
        "/api/v1/cards/17",
        "/api/v1/cards",
    ]
    assert all(request.url.params["broken_api"] == "false" for request in requests)
    assert all(request.headers["authorization"] == "Bearer kaiten-secret" for request in requests)
    assert all(request.headers["x-kaiten-client"] == "kaiten-bridge" for request in requests)
    list_query = requests[1].url.params
    assert list_query["space_id"] == "834556"
    assert list_query["board_id"] == "1868751"
    assert list_query["column_ids"] == "6467479,6467480"
    assert list_query["condition"] == "1"
    assert list_query["limit"] == "100"
    assert list_query["offset"] == "0"


@pytest.mark.asyncio
async def test_comments_route_is_called_once_without_undocumented_pagination(
    tmp_path: Path,
) -> None:
    calls = 0
    comments = [{"id": item, "text": str(item)} for item in range(1, 151)]

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.method == "GET"
        assert request.url.path == "/api/v1/cards/17/comments"
        assert not request.url.query
        return httpx.Response(200, json=comments)

    async with KaitenClient(settings(tmp_path), httpx.MockTransport(handler)) as client:
        result = await client.list_comments(17)

    assert result == comments
    assert calls == 1


@pytest.mark.asyncio
async def test_dynamic_select_catalog_and_user_labels_follow_documented_routes(
    tmp_path: Path,
) -> None:
    seen: list[tuple[str, dict[str, str]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        query = dict(request.url.params)
        seen.append((request.url.path, query))
        offset = int(query.get("offset", "0"))
        if request.url.path.endswith("/12/select-values"):
            assert query["v2_select_search"] == "true"
            assert query["limit"] == "100"
            if offset == 0:
                return httpx.Response(
                    200,
                    json=[{"id": item, "value": f"Вариант {item}"} for item in range(1, 101)],
                )
            return httpx.Response(200, json=[{"id": 101, "value": "Новый тип"}])
        if request.url.path.endswith("/13/catalog-values"):
            assert query["limit"] == "100"
            if offset == 0:
                return httpx.Response(
                    200,
                    json=[{"id": item, "name": f"Запись {item}"} for item in range(1, 101)],
                )
            return httpx.Response(200, json=[{"id": 101, "name": "Проект Север"}])
        if request.url.path == "/api/v1/users":
            assert query == {"ids": "7,9", "limit": "100"}
            return httpx.Response(
                200,
                json=[
                    {"id": 7, "full_name": "Анна Иванова", "email": "private@example.test"},
                    {"id": 9, "full_name": "Пётр Петров", "email": "private2@example.test"},
                ],
            )
        raise AssertionError(f"unexpected request: {request.url}")

    async with KaitenClient(settings(tmp_path), httpx.MockTransport(handler)) as client:
        select = await client.select_labels(12)
        catalog = await client.catalog_labels(13)
        users = await client.user_labels([9, 7])
        # Metadata caches must avoid repeatedly traversing dynamic dictionaries.
        assert await client.select_labels(12) is select
        assert await client.catalog_labels(13) is catalog

    assert select["1"] == "Вариант 1"
    assert select["101"] == "Новый тип"
    assert catalog["101"] == "Проект Север"
    assert users == {"9": "Пётр Петров", "7": "Анна Иванова"}
    assert [query["offset"] for path, query in seen if path.endswith("select-values")] == [
        "0",
        "100",
    ]
    assert [query["offset"] for path, query in seen if path.endswith("catalog-values")] == [
        "0",
        "100",
    ]


@pytest.mark.asyncio
async def test_add_comment_requires_confirmation_that_it_is_public(tmp_path: Path) -> None:
    responses = iter(
        [
            {"id": 1, "internal": False},
            {"id": 2, "internal": True},
        ]
    )
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/v1/cards/17/comments"
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=next(responses))

    async with KaitenClient(settings(tmp_path), httpx.MockTransport(handler)) as client:
        await client.add_public_comment(17, "Готово")
        with pytest.raises(KaitenProtocolError, match="comment_was_not_confirmed_public"):
            await client.add_public_comment(17, "Не подтверждено")

    # The documented create-comment body has only text; Kaiten confirms publicness
    # through internal=false in the response.
    assert bodies == [{"text": "Готово"}, {"text": "Не подтверждено"}]


@pytest.mark.asyncio
async def test_move_card_is_idempotent_and_patches_only_once(tmp_path: Path) -> None:
    current_column = 6_467_479
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal current_column
        methods.append(request.method)
        assert request.url.path == "/api/v1/cards/17"
        if request.method == "GET":
            assert request.url.params["broken_api"] == "false"
            return httpx.Response(200, json={"id": 17, "column_id": current_column})
        assert request.method == "PATCH"
        assert json.loads(request.content) == {"column_id": 6_467_480}
        current_column = 6_467_480
        return httpx.Response(200, json={"id": 17, "column_id": current_column})

    async with KaitenClient(settings(tmp_path), httpx.MockTransport(handler)) as client:
        await client.move_card(17, 6_467_480)
        await client.move_card(17, 6_467_480)

    assert methods == ["GET", "PATCH", "GET"]


@pytest.mark.asyncio
async def test_legacy_download_never_receives_kaiten_authorization(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url == httpx.URL("https://legacy-files.example/report.pdf")
        assert "authorization" not in request.headers
        return httpx.Response(200, content=b"legacy bytes")

    async with KaitenClient(settings(tmp_path), httpx.MockTransport(handler)) as client:
        async with client.downloaded_file(attachment(), max_bytes=100) as prepared:
            assert prepared.content == b"legacy bytes"

    assert len(requests) == 1
    assert_directory_empty(tmp_path)


@pytest.mark.parametrize(
    "url",
    [
        "http://files.example/report.pdf",
        "https://localhost/report.pdf",
        "https://127.0.0.1/report.pdf",
        "https://169.254.169.254/latest/meta-data",
        "https://user:password@files.example/report.pdf",
    ],
)
@pytest.mark.asyncio
async def test_legacy_download_rejects_unsafe_storage_urls_before_network(
    tmp_path: Path,
    url: str,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=b"must not be reached")

    reference = attachment(url=url)
    async with KaitenClient(settings(tmp_path), httpx.MockTransport(handler)) as client:
        with pytest.raises(KaitenProtocolError, match="file_url_invalid"):
            async with client.downloaded_file(reference, max_bytes=100):
                pass

    assert calls == 0
    assert_directory_empty(tmp_path)


@pytest.mark.asyncio
async def test_storage_redirect_is_revalidated_before_following(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "files.example":
            return httpx.Response(302, headers={"location": "https://127.0.0.1/private"})
        raise AssertionError("unsafe redirect must not be requested")

    reference = attachment(url="https://files.example/report.pdf")
    async with KaitenClient(settings(tmp_path), httpx.MockTransport(handler)) as client:
        with pytest.raises(KaitenProtocolError, match="file_redirect_invalid_url"):
            async with client.downloaded_file(reference, max_bytes=100):
                pass

    assert [request.url.host for request in requests] == ["files.example"]
    assert_directory_empty(tmp_path)


@pytest.mark.asyncio
async def test_restricted_fallback_uid_enrichment_and_redirect_do_not_leak_bearer(
    tmp_path: Path,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/v1/company/custom-properties":
            return httpx.Response(
                200,
                json=[{"id": 44, "uid": "property-uid", "name": "Документация"}],
            )
        if request.url.host == "tagat.kaiten.ru":
            assert request.headers["authorization"] == "Bearer kaiten-secret"
            assert request.url.params["download"] == "true"
            return httpx.Response(
                302,
                headers={"location": "https://signed-storage.example/object?signature=secret"},
            )
        assert request.url.host == "signed-storage.example"
        assert "authorization" not in request.headers
        return httpx.Response(200, content=b"restricted bytes")

    card = {
        "uid": "card-uid",
        "files": [
            {
                "id": "comment-file-uid",
                "type": 11,
                "name": "comment.pdf",
                "comment_id": 55,
                "entity_type": "comment",
                # The files[] record intentionally omits comment_uid.
                "url": "https://expired.example/comment",
            },
            {
                "id": "property-file-uid",
                "type": 11,
                "name": "property.pdf",
                "custom_property_id": 44,
                "entity_type": "custom_property",
                # The files[] record intentionally omits custom_property_uid.
                "url": "https://expired.example/property",
            },
        ],
    }
    comments = [{"id": 55, "uid": "comment-uid"}]

    async with KaitenClient(settings(tmp_path), httpx.MockTransport(handler)) as client:
        raw = client._attachment_refs(card)
        enriched = await client._enrich_attachment_refs(card, raw, comments)
        assert enriched[0].api_content_path == (
            "cards/card-uid/comments/comment-uid/files/comment-file-uid/content"
        )
        assert enriched[1].api_content_path == (
            "cards/card-uid/custom-properties/property-uid/files/property-file-uid/content"
        )
        assert enriched[1].source == "кастомное поле «Документация»"
        assert all(item.download_url is None for item in enriched)

        async with client.downloaded_file(enriched[0], max_bytes=100) as prepared:
            assert prepared.content == b"restricted bytes"

    content_request = next(
        request for request in requests if request.url.path.endswith("/comment-file-uid/content")
    )
    assert content_request.url.path == (
        "/api/v1/cards/card-uid/comments/comment-uid/files/comment-file-uid/content"
    )
    storage_request = next(
        request for request in requests if request.url.host == "signed-storage.example"
    )
    assert "authorization" not in storage_request.headers
    assert_directory_empty(tmp_path)


@pytest.mark.asyncio
async def test_configured_rate_limit_includes_restricted_authenticated_request(
    tmp_path: Path,
) -> None:
    now = 0.0
    sleeps: list[float] = []

    def clock() -> float:
        return now

    async def sleep(delay: float) -> None:
        nonlocal now
        sleeps.append(delay)
        now += delay

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/cards/17":
            return httpx.Response(200, json={"id": 17})
        if request.url.host == "tagat.kaiten.ru":
            return httpx.Response(302, headers={"location": "https://storage.example/file"})
        assert request.url.host == "storage.example"
        return httpx.Response(200, content=b"x")

    configured = settings(tmp_path, kaiten_requests_per_second=1)
    client = KaitenClient(
        configured,
        httpx.MockTransport(handler),
        rate_limit_clock=clock,
        rate_limit_sleep=sleep,
    )
    async with client:
        await client.get_card(17)
        reference = attachment(
            url=None,
            api_content_path="cards/card-uid/files/file-uid/content",
        )
        async with client.downloaded_file(reference, max_bytes=10):
            pass

    # One API request fills the one-request window; restricted /content waits.
    # The unauthenticated storage request is deliberately outside the limiter.
    assert sleeps == [pytest.approx(1.0)]


@pytest.mark.asyncio
async def test_file_size_is_checked_before_and_during_download(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        # Declared length is allowed, but the actual stream crosses the boundary.
        return httpx.Response(200, headers={"content-length": "4"}, content=b"12345")

    async with KaitenClient(settings(tmp_path), httpx.MockTransport(handler)) as client:
        with pytest.raises(DataRejectedError) as declared:
            async with client.downloaded_file(attachment(size=5), max_bytes=4):
                pass
        assert declared.value.code == "invalid_files"
        assert calls == 0

        with pytest.raises(DataRejectedError) as streamed:
            async with client.downloaded_file(attachment(size=None), max_bytes=4):
                pass
        assert streamed.value.code == "invalid_files"

    assert calls == 1
    assert_directory_empty(tmp_path)


@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        (404, SourceNotReadyError),
        (422, DataRejectedError),
        (429, KaitenUnavailableError),
    ],
)
@pytest.mark.asyncio
async def test_restricted_file_statuses_have_stable_error_categories(
    tmp_path: Path,
    status: int,
    error_type: type[Exception],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/cards/card-uid/files/file-uid/content"
        return httpx.Response(status)

    reference = attachment(
        url=None,
        api_content_path="cards/card-uid/files/file-uid/content",
    )
    async with KaitenClient(settings(tmp_path), httpx.MockTransport(handler)) as client:
        with pytest.raises(error_type) as raised:
            async with client.downloaded_file(reference, max_bytes=100):
                pass

    if isinstance(raised.value, DataRejectedError):
        assert raised.value.code == "invalid_files"
    assert_directory_empty(tmp_path)


@pytest.mark.asyncio
async def test_card_snapshot_keeps_initial_service_content_from_current_api_user(
    tmp_path: Path,
) -> None:
    card = {
        "id": 17,
        "space_id": 834_556,
        "board_id": 1_868_751,
        "column_id": 6_467_479,
        "title": "Новая заявка",
        "type": {"id": 900, "name": "Тендерная заявка"},
        "owner": {
            "id": 8,
            "full_name": "Иван Иванов",
            "email": "private@example.test",
        },
        "properties": {
            "id_44": 501,
            "id_45": 502,
            "id_46": {"value": "безопасное значение", "internal_user_id": 777},
        },
        "future_standard": "новое стандартное значение",
        "contact_email": "must-not-appear@example.test",
        "files": [
            {
                "id": "card-file",
                "name": "card.pdf",
                "size": 10,
                "url": "https://files.example/card.pdf",
            },
            {
                "id": "initial-file",
                "name": "initial.pdf",
                "size": 11,
                "comment_id": 55,
                "url": "https://files.example/initial.pdf",
            },
            {
                "id": "late-file",
                "name": "late.pdf",
                "size": 12,
                "comment_id": 56,
                "url": "https://files.example/late.pdf",
            },
        ],
    }
    comments = [
        {
            "id": 55,
            "sd_description": True,
            "internal": False,
            "text": "Первоначальное описание",
            "author": {"id": 8, "full_name": "Иван Иванов"},
        },
        {
            "id": 56,
            "internal": False,
            "text": "Поздний комментарий",
            "author": {"id": 9, "full_name": "Пётр Петров"},
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/v1/cards/17":
            return httpx.Response(200, json=card)
        if path == "/api/v1/cards/17/comments":
            return httpx.Response(200, json=comments)
        if path == "/api/v1/company/custom-properties":
            return httpx.Response(
                200,
                json=[
                    {"id": 44, "name": "Тип заявки", "type": "select"},
                    {"id": 45, "name": "Новое поле", "type": "select"},
                    {"id": 46, "name": "Будущий тип", "type": "future_type"},
                ],
            )
        if path.endswith("/44/select-values"):
            return httpx.Response(200, json=[{"id": 501, "value": "Новый тип заявки"}])
        if path.endswith("/45/select-values"):
            return httpx.Response(200, json=[{"id": 502, "value": "Новое значение"}])
        if path == "/api/v1/users/current":
            return httpx.Response(200, json={"id": 8, "full_name": "Иван Иванов"})
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    async with KaitenClient(settings(tmp_path), httpx.MockTransport(handler)) as client:
        snapshot = await client.card_snapshot(17)

    assert snapshot.request_type == "Новый тип заявки"
    assert snapshot.system_type == "Тендерная заявка"
    assert snapshot.initial_comments == ("Первоначальное описание",)
    assert [item.name for item in snapshot.attachments] == ["card.pdf", "initial.pdf"]
    fields = {item.name: item.value for item in snapshot.fields}
    assert fields["Новое поле"] == "Новое значение"
    assert fields["Будущий тип"] == (
        '{"internal_user_id":"[скрыто]","value":"безопасное значение"} '
        "[неизвестный тип: future_type]"
    )
    assert fields["Future standard"] == "новое стандартное значение"
    prompt = build_initial_prompt(snapshot, snapshot.attachments)
    assert "Поздний комментарий" not in prompt
    assert "late.pdf" not in prompt
    assert "private@example.test" not in prompt
    assert "must-not-appear@example.test" not in prompt
    assert "777" not in prompt


@pytest.mark.asyncio
async def test_missing_request_type_is_rejected_without_a_hardcoded_type_list(
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/company/custom-properties"
        return httpx.Response(
            200,
            json=[{"id": 44, "name": "Тип заявки", "type": "select"}],
        )

    configured = settings(tmp_path, kaiten_request_type_field_id=44)
    async with KaitenClient(configured, httpx.MockTransport(handler)) as client:
        with pytest.raises(DataRejectedError) as raised:
            await client.request_type({"properties": {}})

    assert raised.value.code == "request_type_missing"


@pytest.mark.asyncio
async def test_card_snapshot_rejects_explicit_other_space_before_reading_content(
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/cards/17"
        return httpx.Response(
            200,
            json={
                "id": 17,
                "space_id": 999,
                "board_id": 1_868_751,
                "column_id": 6_467_479,
            },
        )

    async with KaitenClient(settings(tmp_path), httpx.MockTransport(handler)) as client:
        with pytest.raises(DataRejectedError) as raised:
            await client.card_snapshot(17)

    assert raised.value.code == "card_outside_space"


@pytest.mark.parametrize(
    ("comment", "expected_code"),
    [
        (
            {
                "id": 55,
                "created": "2026-09-01T12:00:00Z",
                "author": {"id": 777, "full_name": "Corporate AI"},
            },
            "technical_comment",
        ),
        (
            {
                "id": 55,
                "created": "2026-09-01T12:00:00Z",
                "author": {"id": 999, "full_name": "API user"},
                "sd_description": True,
            },
            "initial_service_comment",
        ),
        (
            {
                "id": 55,
                "created": "2026-09-01T12:00:00Z",
                "author": {"id": 999, "full_name": "API user"},
            },
            "technical_comment",
        ),
    ],
)
@pytest.mark.asyncio
async def test_comment_snapshot_ignores_nested_technical_and_initial_service_comments(
    tmp_path: Path,
    comment: dict[str, object],
    expected_code: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/cards/17":
            return httpx.Response(
                200,
                json={
                    "id": 17,
                    "space_id": 834_556,
                    "board_id": 1_868_751,
                    "column_id": 6_467_480,
                },
            )
        if request.url.path == "/api/v1/cards/17/comments":
            return httpx.Response(200, json=[comment])
        if request.url.path == "/api/v1/users/current":
            return httpx.Response(200, json={"id": 999})
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    configured = settings(tmp_path, kaiten_technical_author_ids=[777])
    async with KaitenClient(configured, httpx.MockTransport(handler)) as client:
        with pytest.raises(DataRejectedError) as raised:
            await client.comment_snapshot(17, 55)

    assert raised.value.code == expected_code
