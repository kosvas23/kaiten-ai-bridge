from __future__ import annotations

import asyncio
import copy
import os
from collections import deque
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from kaiten_ai_bridge.config import Settings
from kaiten_ai_bridge.domain import (
    AttachmentRef,
    CardSnapshot,
    CommentSnapshot,
    PreparedFile,
)
from kaiten_ai_bridge.errors import (
    DataRejectedError,
    KaitenUnavailableError,
    ReceiverUnavailableError,
)
from kaiten_ai_bridge.schemas import CallbackRequest, Trigger
from kaiten_ai_bridge.service import (
    INACTIVITY_WARNING_MESSAGE,
    TECHNICAL_ERROR_MESSAGE,
    BridgeService,
    CallbackBusyError,
)
from kaiten_ai_bridge.storage import EventStatus, SQLiteStorage

QUEUE = 101
WORK = 102
DONE = 103
BOARD = 201
SPACE = 301
TECHNICAL_AUTHOR = 777
CURRENT_USER = 900


def configured(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "environment": "test",
        "database_path": tmp_path / "bridge.sqlite3",
        "log_dir": tmp_path / "logs",
        "temp_dir": tmp_path / "tmp",
        "kaiten_base_url": "https://tagat.example.test",
        "kaiten_token": "kaiten-secret",
        "kaiten_space_id": SPACE,
        "kaiten_board_id": BOARD,
        "kaiten_queue_column_id": QUEUE,
        "kaiten_work_column_id": WORK,
        "kaiten_done_column_id": DONE,
        "kaiten_technical_author_ids": [TECHNICAL_AUTHOR],
        "ai_bearer_token": "receiver-secret",
        "allowed_extensions": [".pdf"],
        "max_file_bytes": 10_000,
        "retry_schedule_seconds": [0, 60],
        "card_parallelism": 8,
        "reconciliation_interval_seconds": 60,
        "cleanup_interval_seconds": 60,
        "reconcile_on_startup": False,
        "cleanup_on_startup": False,
    }
    values.update(overrides)
    return Settings(**values)


def card_snapshot(
    card_id: int,
    *,
    title: str = "Новая заявка",
    comments: tuple[str, ...] = ("Помогите с расчётом",),
    attachments: tuple[AttachmentRef, ...] = (),
) -> CardSnapshot:
    return CardSnapshot(
        card_id=card_id,
        url=f"https://tagat.example.test/space/{SPACE}/boards/card/{card_id}",
        title=title,
        request_type="Запрос",
        space=str(SPACE),
        board="ИИ",
        column="Очередь",
        column_id=QUEUE,
        author="Иван Иванов",
        initial_comments=comments,
        attachments=attachments,
    )


def comment_snapshot(
    card_id: int,
    comment_id: int,
    *,
    text: str,
    created_at: datetime,
    attachments: tuple[AttachmentRef, ...] = (),
) -> CommentSnapshot:
    return CommentSnapshot(
        comment_id=comment_id,
        card_id=card_id,
        card_url=f"https://tagat.example.test/space/{SPACE}/boards/card/{card_id}",
        text=text,
        author="Мария",
        created_at=created_at,
        is_public=True,
        author_id=55,
        attachments=attachments,
    )


def attachment(name: str, *, file_id: str, comment_id: int | None = None) -> AttachmentRef:
    return AttachmentRef(
        file_id=file_id,
        name=name,
        size=4,
        mime_type="application/pdf",
        download_url=f"https://files.example.test/{file_id}",
        source="комментарий" if comment_id is not None else "карточка",
        comment_id=comment_id,
    )


class FakeKaiten:
    def __init__(self) -> None:
        self.cards: dict[int, dict[str, Any]] = {}
        self.target_cards: list[dict[str, Any]] = []
        self.comments: dict[int, list[dict[str, Any]]] = {}
        self.initial_results: dict[tuple[int, int | None], object] = {}
        self.comment_results: dict[tuple[int, int], object] = {}
        self.file_contents: dict[str, bytes] = {}
        self.card_snapshot_calls: list[tuple[int, int | None]] = []
        self.comment_snapshot_calls: list[tuple[int, int]] = []
        self.downloads: list[AttachmentRef] = []
        self.move_attempts: list[tuple[int, int]] = []
        self.move_observed_job_ids: list[str | None] = []
        self.move_failures: deque[BaseException] = deque()
        self.comment_attempts: list[tuple[int, str]] = []
        self.public_comments: list[tuple[int, str]] = []
        self.public_comment_failures: deque[BaseException] = deque()
        self.storage: SQLiteStorage | None = None
        self.current_user = CURRENT_USER

    async def get_card(self, card_id: int) -> dict[str, Any]:
        return copy.deepcopy(self.cards[card_id])

    async def list_target_cards(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self.target_cards)

    async def list_comments(self, card_id: int) -> list[dict[str, Any]]:
        return copy.deepcopy(self.comments.get(card_id, []))

    async def current_user_id(self) -> int | None:
        return self.current_user

    async def card_snapshot(
        self, card_id: int, *, correction_comment_id: int | None = None
    ) -> CardSnapshot:
        self.card_snapshot_calls.append((card_id, correction_comment_id))
        result = self.initial_results[(card_id, correction_comment_id)]
        if isinstance(result, BaseException):
            raise result
        assert isinstance(result, CardSnapshot)
        return result

    async def comment_snapshot(self, card_id: int, comment_id: int) -> tuple[CommentSnapshot, str]:
        self.comment_snapshot_calls.append((card_id, comment_id))
        result = self.comment_results[(card_id, comment_id)]
        if isinstance(result, BaseException):
            raise result
        assert isinstance(result, CommentSnapshot)
        return result, "Запрос"

    async def move_card(self, card_id: int, column_id: int) -> None:
        self.move_attempts.append((card_id, column_id))
        if self.storage is not None:
            state = await self.storage.get_card_job(card_id)
            self.move_observed_job_ids.append(state.job_id if state is not None else None)
        if self.move_failures:
            raise self.move_failures.popleft()
        self.cards[card_id]["column_id"] = column_id

    async def add_public_comment(self, card_id: int, text: str) -> None:
        self.comment_attempts.append((card_id, text))
        if self.public_comment_failures:
            raise self.public_comment_failures.popleft()
        self.public_comments.append((card_id, text))

    @asynccontextmanager
    async def downloaded_file(
        self, reference: AttachmentRef, *, max_bytes: int
    ) -> AsyncIterator[PreparedFile]:
        del max_bytes
        self.downloads.append(reference)
        key = reference.file_id or reference.name
        yield PreparedFile(reference=reference, content=self.file_contents.get(key, b"data"))

    async def healthcheck(self) -> bool:
        return True


class FakeReceiver:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.effects: deque[object] = deque()
        self.wait_before_return = False
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def submit(self, payload: Mapping[str, Any]) -> str:
        saved = copy.deepcopy(dict(payload))
        self.calls.append(saved)
        self.entered.set()
        if self.wait_before_return:
            await self.release.wait()
        if self.effects:
            effect = self.effects.popleft()
            if isinstance(effect, BaseException):
                raise effect
            assert isinstance(effect, str)
            return effect
        existing = saved.get("ai_job_id")
        return str(existing or f"job-{saved['card']['id']}")

    async def healthcheck(self) -> bool:
        return True


@pytest_asyncio.fixture
async def storage(tmp_path: Path) -> AsyncIterator[SQLiteStorage]:
    database = SQLiteStorage(tmp_path / "bridge.sqlite3", recover_on_initialize=False)
    await database.initialize()
    try:
        yield database
    finally:
        await database.close()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return configured(tmp_path)


@pytest.fixture
def kaiten(storage: SQLiteStorage) -> FakeKaiten:
    fake = FakeKaiten()
    fake.storage = storage
    return fake


@pytest.fixture
def receiver() -> FakeReceiver:
    return FakeReceiver()


@pytest.fixture
def service(
    settings: Settings,
    storage: SQLiteStorage,
    kaiten: FakeKaiten,
    receiver: FakeReceiver,
) -> BridgeService:
    return BridgeService(settings, storage, kaiten, receiver)  # type: ignore[arg-type]


async def enqueue(
    service: BridgeService,
    *,
    event_type: str,
    card_id: int,
    comment_id: int | None = None,
    created_at: datetime | None = None,
) -> int:
    trigger = Trigger(
        event_type=event_type,  # type: ignore[arg-type]
        card_id=card_id,
        comment_id=comment_id,
        source_created_at=created_at or datetime.now(UTC),
    )
    event, inserted = await service.enqueue_trigger(trigger)
    assert inserted
    return event.id


async def wait_for_event_status(
    storage: SQLiteStorage, event_id: int, expected: EventStatus
) -> None:
    while True:
        event = await storage.get_event(event_id)
        if event is not None and event.status is expected:
            return
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_initial_binds_job_and_moves_only_after_confirmed_acceptance(
    service: BridgeService,
    storage: SQLiteStorage,
    kaiten: FakeKaiten,
    receiver: FakeReceiver,
) -> None:
    card_id = 101
    kaiten.cards[card_id] = {"id": card_id, "board_id": BOARD, "column_id": QUEUE}
    kaiten.initial_results[(card_id, None)] = card_snapshot(card_id)
    receiver.effects.append("accepted-101")
    receiver.wait_before_return = True
    event_id = await enqueue(service, event_type="card:add", card_id=card_id)

    processing = asyncio.create_task(service.run_due_batch())
    await asyncio.wait_for(receiver.entered.wait(), timeout=1)

    state_while_waiting = await storage.get_card_job(card_id)
    assert state_while_waiting is not None and state_while_waiting.job_id is None
    assert kaiten.move_attempts == []

    receiver.release.set()
    assert await processing == 1

    state = await storage.get_card_job(card_id)
    event = await storage.get_event(event_id)
    assert state is not None and state.job_id == "accepted-101"
    assert event is not None and event.status is EventStatus.SUCCESS
    assert kaiten.move_attempts == [(card_id, WORK)]
    assert kaiten.move_observed_job_ids == ["accepted-101"]


@pytest.mark.asyncio
async def test_card_add_runs_before_initial_service_comment_that_arrived_first(
    service: BridgeService,
    storage: SQLiteStorage,
    kaiten: FakeKaiten,
    receiver: FakeReceiver,
) -> None:
    card_id = 105
    comment_id = 1051
    created = datetime.now(UTC)
    kaiten.cards[card_id] = {"id": card_id, "board_id": BOARD, "column_id": QUEUE}
    kaiten.initial_results[(card_id, None)] = card_snapshot(card_id)
    kaiten.comment_results[(card_id, comment_id)] = DataRejectedError(
        "initial_service_comment",
        "Исходный комментарий Service Desk уже входит в первоначальную заявку.",
    )
    comment_event_id = await enqueue(
        service,
        event_type="comment:add",
        card_id=card_id,
        comment_id=comment_id,
        created_at=created,
    )
    initial_event_id = await enqueue(
        service,
        event_type="card:add",
        card_id=card_id,
        created_at=created,
    )

    assert await service.run_due_batch() == 1
    initial = await storage.get_event(initial_event_id)
    waiting_comment = await storage.get_event(comment_event_id)
    assert initial is not None and initial.status is EventStatus.SUCCESS
    assert waiting_comment is not None and waiting_comment.status is EventStatus.PENDING
    assert len(receiver.calls) == 1
    assert kaiten.move_attempts == [(card_id, WORK)]

    assert await service.run_due_batch() == 1
    ignored_comment = await storage.get_event(comment_event_id)
    assert ignored_comment is not None and ignored_comment.status is EventStatus.IGNORED
    assert ignored_comment.error_code == "initial_service_comment"
    assert len(receiver.calls) == 1


@pytest.mark.asyncio
async def test_retry_after_move_failure_uses_bound_job_without_resubmitting(
    service: BridgeService,
    storage: SQLiteStorage,
    kaiten: FakeKaiten,
    receiver: FakeReceiver,
) -> None:
    card_id = 102
    kaiten.cards[card_id] = {"id": card_id, "board_id": BOARD, "column_id": QUEUE}
    kaiten.initial_results[(card_id, None)] = card_snapshot(card_id)
    kaiten.move_failures.append(KaitenUnavailableError())
    receiver.effects.append("accepted-102")
    event_id = await enqueue(service, event_type="card:add", card_id=card_id)

    assert await service.run_due_batch() == 1
    after_failure = await storage.get_event(event_id)
    state = await storage.get_card_job(card_id)
    assert after_failure is not None and after_failure.status is EventStatus.RETRY
    assert state is not None and state.job_id == "accepted-102"
    assert len(receiver.calls) == 1

    [retry] = await storage.claim_due_events(now=datetime.now(UTC) + timedelta(days=1))
    await service._process_event(retry)

    completed = await storage.get_event(event_id)
    assert completed is not None and completed.status is EventStatus.SUCCESS
    assert len(receiver.calls) == 1
    assert kaiten.card_snapshot_calls == [(card_id, None)]
    assert kaiten.move_attempts == [(card_id, WORK), (card_id, WORK)]


@pytest.mark.asyncio
async def test_data_rejection_is_terminal_and_posts_exactly_one_public_comment(
    service: BridgeService,
    storage: SQLiteStorage,
    kaiten: FakeKaiten,
    receiver: FakeReceiver,
) -> None:
    card_id = 103
    message = "Добавьте новый комментарий с файлом PDF до 5 МБ."
    kaiten.cards[card_id] = {"id": card_id, "board_id": BOARD, "column_id": QUEUE}
    kaiten.target_cards = [kaiten.cards[card_id]]
    kaiten.initial_results[(card_id, None)] = DataRejectedError("invalid_files", message)
    event_id = await enqueue(service, event_type="card:add", card_id=card_id)

    assert await service.run_due_batch() == 1
    duplicate, inserted = await service.enqueue_trigger(
        Trigger(event_type="card:add", card_id=card_id)
    )
    assert duplicate.id == event_id and not inserted
    assert await service.run_due_batch() == 0
    await service.reconcile_once()

    event = await storage.get_event(event_id)
    assert event is not None and event.status is EventStatus.REJECTED
    assert event.error_code == "invalid_files" and event.notification_sent
    assert kaiten.public_comments == [(card_id, message)]
    assert receiver.calls == []
    assert kaiten.move_attempts == []


@pytest.mark.asyncio
async def test_technical_failures_follow_schedule_then_exhaust_with_one_notification(
    service: BridgeService,
    storage: SQLiteStorage,
    kaiten: FakeKaiten,
    receiver: FakeReceiver,
) -> None:
    card_id = 104
    kaiten.cards[card_id] = {"id": card_id, "board_id": BOARD, "column_id": QUEUE}
    kaiten.initial_results[(card_id, None)] = card_snapshot(card_id)
    receiver.effects.extend([ReceiverUnavailableError(), ReceiverUnavailableError()])
    event_id = await enqueue(service, event_type="card:add", card_id=card_id)

    assert await service.run_due_batch() == 1
    scheduled = await storage.get_event(event_id)
    assert scheduled is not None and scheduled.status is EventStatus.RETRY
    assert scheduled.attempts == 1 and scheduled.error_code == "receiver_unavailable"
    assert scheduled.next_attempt_at is not None
    created = datetime.fromisoformat(scheduled.created_at.replace("Z", "+00:00"))
    retry_at = datetime.fromisoformat(scheduled.next_attempt_at.replace("Z", "+00:00"))
    assert retry_at == created + timedelta(seconds=60)

    [retry] = await storage.claim_due_events(now=retry_at + timedelta(seconds=1))
    await service._process_event(retry)
    exhausted = await storage.get_event(event_id)

    assert exhausted is not None and exhausted.status is EventStatus.TECHNICAL_ERROR
    assert exhausted.attempts == 2 and exhausted.notification_sent
    assert len(receiver.calls) == 2
    assert kaiten.public_comments == [(card_id, TECHNICAL_ERROR_MESSAGE)]
    assert await service.run_due_batch() == 0


@pytest.mark.asyncio
async def test_comments_are_strictly_filtered_and_sent_one_at_a_time_with_only_new_content(
    service: BridgeService,
    storage: SQLiteStorage,
    kaiten: FakeKaiten,
    receiver: FakeReceiver,
) -> None:
    card_id = 201
    base = datetime.now(UTC) - timedelta(minutes=10)
    await storage.ensure_card_state(card_id=card_id, last_human_activity_at=base)
    await storage.bind_job(card_id=card_id, job_id="dialog-201")
    kaiten.cards[card_id] = {"id": card_id, "board_id": BOARD, "column_id": WORK}
    kaiten.comment_results[(card_id, 2011)] = DataRejectedError(
        "internal_comment", "Внутренняя заметка не обрабатывается."
    )
    first_file = attachment("only-first.pdf", file_id="first", comment_id=2012)
    kaiten.file_contents["first"] = b"first"
    kaiten.comment_results[(card_id, 2012)] = comment_snapshot(
        card_id,
        2012,
        text="FIRST-COMMENT-ONLY",
        created_at=base + timedelta(minutes=1),
        attachments=(first_file,),
    )
    kaiten.comment_results[(card_id, 2013)] = comment_snapshot(
        card_id,
        2013,
        text="SECOND-COMMENT-ONLY",
        created_at=base + timedelta(minutes=2),
    )
    for offset, comment_id in enumerate((2011, 2012, 2013), start=1):
        await enqueue(
            service,
            event_type="comment:add",
            card_id=card_id,
            comment_id=comment_id,
            created_at=base + timedelta(minutes=offset),
        )

    assert await service.run_due_batch() == 1
    ignored = await storage.get_event_by_key("comment:add:2011")
    unchanged = await storage.get_card_job(card_id)
    assert ignored is not None and ignored.status is EventStatus.IGNORED
    assert unchanged is not None and unchanged.last_human_activity_at.startswith(
        base.isoformat(timespec="seconds")[:19]
    )
    assert receiver.calls == [] and kaiten.public_comments == []

    assert await service.run_due_batch() == 1
    assert len(receiver.calls) == 1
    first_payload = receiver.calls[0]
    assert first_payload["event_type"] == "comment"
    assert first_payload["comment_id"] == 2012
    assert first_payload["ai_job_id"] == "dialog-201"
    assert "FIRST-COMMENT-ONLY" in first_payload["prompt"]
    assert "SECOND-COMMENT-ONLY" not in first_payload["prompt"]
    assert [item["name"] for item in first_payload["files"]] == ["only-first.pdf"]

    assert await service.run_due_batch() == 1
    assert [item["comment_id"] for item in receiver.calls] == [2012, 2013]
    assert "FIRST-COMMENT-ONLY" not in receiver.calls[1]["prompt"]
    assert receiver.calls[1]["files"] == []


@pytest.mark.asyncio
async def test_out_of_order_webhook_restores_earlier_remote_comment_before_sending(
    service: BridgeService,
    storage: SQLiteStorage,
    kaiten: FakeKaiten,
    receiver: FakeReceiver,
) -> None:
    card_id = 204
    first_id = 2041
    second_id = 2042
    base = datetime.now(UTC) - timedelta(minutes=5)
    await storage.ensure_card_state(card_id=card_id, last_human_activity_at=base)
    await storage.bind_job(card_id=card_id, job_id="dialog-204")
    kaiten.cards[card_id] = {
        "id": card_id,
        "space_id": SPACE,
        "board_id": BOARD,
        "column_id": WORK,
    }
    kaiten.comments[card_id] = [
        {"id": second_id, "created": base + timedelta(minutes=2), "author_id": 52},
        {"id": first_id, "created": base + timedelta(minutes=1), "author_id": 51},
    ]
    kaiten.comment_results[(card_id, first_id)] = comment_snapshot(
        card_id,
        first_id,
        text="FIRST-REMOTE-COMMENT",
        created_at=base + timedelta(minutes=1),
    )
    kaiten.comment_results[(card_id, second_id)] = comment_snapshot(
        card_id,
        second_id,
        text="SECOND-WEBHOOK-COMMENT",
        created_at=base + timedelta(minutes=2),
    )
    second_event_id = await enqueue(
        service,
        event_type="comment:add",
        card_id=card_id,
        comment_id=second_id,
        created_at=base + timedelta(minutes=2),
    )

    assert await service.run_due_batch() == 1
    deferred = await storage.get_event(second_event_id)
    restored = await storage.get_event_by_key(f"comment:add:{first_id}")
    assert deferred is not None and deferred.status is EventStatus.RETRY
    assert deferred.error_code == "comment_predecessor_restored"
    assert restored is not None and restored.status is EventStatus.PENDING
    assert receiver.calls == []

    assert await service.run_due_batch() == 1
    assert await service.run_due_batch() == 1
    assert [item["comment_id"] for item in receiver.calls] == [first_id, second_id]
    assert "FIRST-REMOTE-COMMENT" in receiver.calls[0]["prompt"]
    assert "SECOND-WEBHOOK-COMMENT" in receiver.calls[1]["prompt"]


@pytest.mark.asyncio
async def test_new_comment_corrects_rejected_initial_without_reusing_old_content(
    service: BridgeService,
    storage: SQLiteStorage,
    kaiten: FakeKaiten,
    receiver: FakeReceiver,
) -> None:
    card_id = 202
    comment_id = 2021
    bad_message = "Старый файл OLD-REJECTED-FILE.pdf отклонён."
    corrected_file = attachment("corrected.pdf", file_id="corrected", comment_id=comment_id)
    kaiten.file_contents["corrected"] = b"good"
    kaiten.cards[card_id] = {"id": card_id, "board_id": BOARD, "column_id": QUEUE}
    kaiten.initial_results[(card_id, None)] = DataRejectedError("invalid_files", bad_message)
    kaiten.initial_results[(card_id, comment_id)] = card_snapshot(
        card_id,
        comments=("CORRECTED-COMMENT",),
        attachments=(corrected_file,),
    )
    initial_id = await enqueue(service, event_type="card:add", card_id=card_id)
    assert await service.run_due_batch() == 1
    rejected = await storage.get_event(initial_id)
    assert rejected is not None and rejected.status is EventStatus.REJECTED

    correction_id = await enqueue(
        service,
        event_type="comment:add",
        card_id=card_id,
        comment_id=comment_id,
    )
    assert await service.run_due_batch() == 1

    corrected = await storage.get_event(correction_id)
    state = await storage.get_card_job(card_id)
    assert corrected is not None and corrected.status is EventStatus.SUCCESS
    assert state is not None and state.job_id == f"job-{card_id}"
    assert kaiten.card_snapshot_calls == [(card_id, None), (card_id, comment_id)]
    assert len(receiver.calls) == 1
    payload = receiver.calls[0]
    assert payload["event_type"] == "initial"
    assert payload["idempotency_key"] == f"card:add:{card_id}"
    assert payload["comment_id"] is None and payload["ai_job_id"] is None
    assert "CORRECTED-COMMENT" in payload["prompt"]
    assert "OLD-REJECTED-FILE" not in payload["prompt"]
    assert [item["name"] for item in payload["files"]] == ["corrected.pdf"]
    assert kaiten.public_comments == [(card_id, bad_message)]
    assert kaiten.move_attempts == [(card_id, WORK)]


@pytest.mark.asyncio
async def test_ignored_comment_does_not_extend_rejected_initial_activity(
    service: BridgeService,
    storage: SQLiteStorage,
    kaiten: FakeKaiten,
    receiver: FakeReceiver,
) -> None:
    card_id = 203
    comment_id = 2031
    base = datetime.now(UTC) - timedelta(days=5)
    await storage.ensure_card_state(card_id=card_id, last_human_activity_at=base)
    kaiten.cards[card_id] = {
        "id": card_id,
        "space_id": SPACE,
        "board_id": BOARD,
        "column_id": QUEUE,
    }
    kaiten.initial_results[(card_id, comment_id)] = DataRejectedError(
        "initial_service_comment",
        "Исходный комментарий Service Desk уже входит в первоначальную заявку.",
    )
    event_id = await enqueue(
        service,
        event_type="comment:add",
        card_id=card_id,
        comment_id=comment_id,
        created_at=datetime.now(UTC),
    )

    assert await service.run_due_batch() == 1

    event = await storage.get_event(event_id)
    state = await storage.get_card_job(card_id)
    assert event is not None and event.status is EventStatus.IGNORED
    assert state is not None
    assert state.last_human_activity_at.startswith(base.isoformat(timespec="seconds")[:19])
    assert receiver.calls == []


def callback(result_id: str, job_id: str, text: str) -> CallbackRequest:
    return CallbackRequest(
        schema_version=1,
        result_id=result_id,
        job_id=job_id,
        status="completed",
        result={"text": text},
        created_at=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_callback_reservation_duplicate_busy_and_release_on_publish_failure(
    service: BridgeService,
    storage: SQLiteStorage,
    kaiten: FakeKaiten,
) -> None:
    card_id = 301
    await storage.ensure_card_state(card_id=card_id, last_human_activity_at=datetime.now(UTC))
    await storage.bind_job(card_id=card_id, job_id="job-callback")

    published = await service.handle_callback(callback("result-ok", "job-callback", "AI-RESULT"))
    duplicate = await service.handle_callback(callback("result-ok", "job-callback", "AI-RESULT"))
    assert published.outcome == "published" and published.card_id == card_id
    assert duplicate.outcome == "duplicate" and duplicate.card_id == card_id
    assert kaiten.public_comments == [(card_id, "AI-RESULT")]

    await storage.reserve_callback(result_id="result-busy", job_id="job-callback")
    with pytest.raises(CallbackBusyError):
        await service.handle_callback(callback("result-busy", "job-callback", "BUSY"))
    assert await storage.release_callback("result-busy")

    kaiten.public_comment_failures.append(KaitenUnavailableError())
    with pytest.raises(KaitenUnavailableError):
        await service.handle_callback(callback("result-retry", "job-callback", "RETRY-RESULT"))
    cursor = await storage.connection.execute(
        "SELECT COUNT(*) FROM callback_keys WHERE result_id = 'result-retry'"
    )
    assert (await cursor.fetchone())[0] == 0

    retried = await service.handle_callback(
        callback("result-retry", "job-callback", "RETRY-RESULT")
    )
    assert retried.outcome == "published"
    assert kaiten.public_comments[-1] == (card_id, "RETRY-RESULT")


@pytest.mark.asyncio
async def test_reconciliation_recovers_lost_card_comment_and_technical_error(
    service: BridgeService,
    storage: SQLiteStorage,
    kaiten: FakeKaiten,
) -> None:
    now = datetime.now(UTC)
    lost_card = 401
    comment_card = 402
    error_card = 403
    kaiten.target_cards = [
        {"id": lost_card, "board_id": BOARD, "column_id": QUEUE, "created": now},
        {"id": comment_card, "board_id": BOARD, "column_id": WORK, "created": now},
        {"id": error_card, "board_id": BOARD, "column_id": WORK, "created": now},
    ]
    kaiten.cards = {int(item["id"]): copy.deepcopy(item) for item in kaiten.target_cards}
    await storage.ensure_card_state(
        card_id=comment_card, last_human_activity_at=now - timedelta(hours=1)
    )
    await storage.bind_job(card_id=comment_card, job_id="job-402")
    await storage.ensure_card_state(
        card_id=error_card, last_human_activity_at=now - timedelta(hours=1)
    )
    await storage.bind_job(card_id=error_card, job_id="job-403")
    kaiten.comments[comment_card] = [
        {"id": 4021, "created": now, "author_id": 50},
        {"id": 4022, "created": now, "author_id": 51, "internal": True},
        {"id": 4023, "created": now, "author": {"id": TECHNICAL_AUTHOR}},
        {"id": 4024, "created": now, "author": {"id": CURRENT_USER}},
        {"id": 4025, "created": now, "author_id": 52, "deleted": True},
    ]
    error = await storage.enqueue_event(
        event_type="comment:add",
        card_id=error_card,
        comment_id=4031,
        event_key="comment:add:4031",
        source_created_at=now,
    )
    [claimed] = await storage.claim_due_events()
    assert claimed.id == error.event.id
    await storage.mark_error(claimed.id, error_code="kaiten_unavailable")

    await service.reconcile_once()

    restored_card = await storage.get_event_by_key(f"card:add:{lost_card}")
    restored_comment = await storage.get_event_by_key("comment:add:4021")
    restored_error = await storage.get_event(error.event.id)
    assert restored_card is not None and restored_card.status is EventStatus.PENDING
    assert restored_comment is not None and restored_comment.status is EventStatus.PENDING
    for ignored_id in (4022, 4023, 4024, 4025):
        assert await storage.get_event_by_key(f"comment:add:{ignored_id}") is None
    assert restored_error is not None and restored_error.status is EventStatus.RETRY
    assert restored_error.notification_sent
    assert kaiten.public_comments == [(error_card, TECHNICAL_ERROR_MESSAGE)]
    status = await storage.get_service_status("last_reconciliation_result")
    assert status is not None and status.value == "ok"


@pytest.mark.asyncio
async def test_cleanup_warns_once_expires_all_state_and_ignores_later_comments(
    service: BridgeService,
    storage: SQLiteStorage,
    kaiten: FakeKaiten,
    receiver: FakeReceiver,
) -> None:
    now = datetime.now(UTC)
    warning_card = 501
    expired_card = 502
    for card_id, activity in (
        (warning_card, now - timedelta(days=29, hours=1)),
        (expired_card, now - timedelta(days=31)),
    ):
        await storage.ensure_card_state(card_id=card_id, last_human_activity_at=activity)
        await storage.bind_job(card_id=card_id, job_id=f"job-{card_id}")
        kaiten.cards[card_id] = {"id": card_id, "board_id": BOARD, "column_id": WORK}

    old_event = await storage.enqueue_event(
        event_type="comment:add",
        card_id=expired_card,
        comment_id=5020,
        event_key="comment:add:5020",
        source_created_at=now - timedelta(days=31),
    )
    await storage.reserve_callback(result_id="expired-result", job_id=f"job-{expired_card}")
    await storage.mark_callback_success("expired-result")

    await service.cleanup_once()
    await service.cleanup_once()

    warned = await storage.get_card_job(warning_card)
    assert warned is not None and warned.warning_sent
    assert kaiten.public_comments == [(warning_card, INACTIVITY_WARNING_MESSAGE)]
    assert await storage.get_card_job(expired_card) is None
    assert await storage.get_event(old_event.event.id) is None
    assert kaiten.move_attempts == [(expired_card, DONE)]
    cursor = await storage.connection.execute(
        "SELECT COUNT(*) FROM callback_keys WHERE result_id = 'expired-result'"
    )
    assert (await cursor.fetchone())[0] == 0

    later_id = await enqueue(
        service,
        event_type="comment:add",
        card_id=expired_card,
        comment_id=5021,
        created_at=now,
    )
    assert await service.run_due_batch() == 1
    later = await storage.get_event(later_id)
    assert later is not None and later.status is EventStatus.IGNORED
    assert later.error_code == "card_closed"
    assert receiver.calls == []
    assert kaiten.public_comments == [(warning_card, INACTIVITY_WARNING_MESSAGE)]
    assert await storage.get_event_by_key(f"card:add:{expired_card}") is None


@pytest.mark.asyncio
async def test_cleanup_warning_uses_configured_hours_with_russian_plural(
    tmp_path: Path,
    storage: SQLiteStorage,
    kaiten: FakeKaiten,
    receiver: FakeReceiver,
) -> None:
    custom_settings = configured(tmp_path, warning_before_close_hours=21)
    custom_service = BridgeService(
        custom_settings,
        storage,
        kaiten,
        receiver,  # type: ignore[arg-type]
    )
    card_id = 503
    await storage.ensure_card_state(
        card_id=card_id,
        last_human_activity_at=datetime.now(UTC) - timedelta(days=29, hours=4),
    )
    await storage.bind_job(card_id=card_id, job_id="job-503")
    kaiten.cards[card_id] = {"id": card_id, "board_id": BOARD, "column_id": WORK}

    await custom_service.cleanup_once()

    assert len(kaiten.public_comments) == 1
    assert "через 21 час из-за" in kaiten.public_comments[0][1]


@pytest.mark.asyncio
async def test_cleanup_removes_only_old_rotated_application_logs(
    service: BridgeService,
    settings: Settings,
) -> None:
    settings.log_dir.mkdir(parents=True)
    old_rotated = settings.log_dir / "bridge.log.2026-08-20"
    new_rotated = settings.log_dir / "bridge.log.2026-09-01"
    active = settings.log_dir / "bridge.log"
    unrelated = settings.log_dir / "keep.txt"
    for path in (old_rotated, new_rotated, active, unrelated):
        path.touch()
    old_timestamp = (datetime.now(UTC) - timedelta(days=4)).timestamp()
    os.utime(old_rotated, (old_timestamp, old_timestamp))

    await service.cleanup_once()

    assert not old_rotated.exists()
    assert new_rotated.exists()
    assert active.exists()
    assert unrelated.exists()


@pytest.mark.asyncio
async def test_cleanup_processes_more_than_100_due_rows_past_individual_failures(
    service: BridgeService,
    storage: SQLiteStorage,
    kaiten: FakeKaiten,
) -> None:
    now = datetime.now(UTC)
    warning_cards = list(range(5_100, 5_202))
    expired_cards = list(range(5_300, 5_402))
    for card_id in warning_cards:
        await storage.ensure_card_state(
            card_id=card_id,
            last_human_activity_at=now - timedelta(days=29, hours=1),
        )
        await storage.bind_job(card_id=card_id, job_id=f"job-{card_id}")
        kaiten.cards[card_id] = {"id": card_id, "board_id": BOARD, "column_id": WORK}
    for card_id in expired_cards:
        await storage.ensure_card_state(
            card_id=card_id, last_human_activity_at=now - timedelta(days=31)
        )
        await storage.bind_job(card_id=card_id, job_id=f"job-{card_id}")
        kaiten.cards[card_id] = {"id": card_id, "board_id": BOARD, "column_id": WORK}
    kaiten.public_comment_failures.append(KaitenUnavailableError())
    kaiten.move_failures.append(KaitenUnavailableError())

    await service.cleanup_once()

    first_warning = await storage.get_card_job(warning_cards[0])
    last_warning = await storage.get_card_job(warning_cards[-1])
    first_expired = await storage.get_card_job(expired_cards[0])
    last_expired = await storage.get_card_job(expired_cards[-1])
    assert first_warning is not None and not first_warning.warning_sent
    assert last_warning is not None and last_warning.warning_sent
    assert first_expired is not None
    assert last_expired is None
    assert len(kaiten.public_comments) == len(warning_cards) - 1
    assert len(kaiten.move_attempts) == len(expired_cards)


@pytest.mark.asyncio
async def test_user_prompts_files_and_callback_bodies_never_enter_sqlite(
    service: BridgeService,
    storage: SQLiteStorage,
    kaiten: FakeKaiten,
    receiver: FakeReceiver,
) -> None:
    card_id = 601
    markers = {
        "PRIVATE-TITLE",
        "PRIVATE-COMMENT",
        "PRIVATE-FILENAME.pdf",
        "PRIVATE-FILE-CONTENT",
        "private-person@example.test",
        "PRIVATE-CALLBACK-BODY",
    }
    private_file = attachment("PRIVATE-FILENAME.pdf", file_id="private")
    kaiten.file_contents["private"] = b"PRIVATE-FILE-CONTENT"
    kaiten.cards[card_id] = {"id": card_id, "board_id": BOARD, "column_id": QUEUE}
    kaiten.initial_results[(card_id, None)] = card_snapshot(
        card_id,
        title="PRIVATE-TITLE private-person@example.test",
        comments=("PRIVATE-COMMENT",),
        attachments=(private_file,),
    )
    await enqueue(service, event_type="card:add", card_id=card_id)
    assert await service.run_due_batch() == 1
    await service.handle_callback(
        callback("private-result-id", f"job-{card_id}", "PRIVATE-CALLBACK-BODY")
    )
    assert any("PRIVATE-COMMENT" in call["prompt"] for call in receiver.calls)

    cursor = await storage.connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    )
    tables = [row["name"] for row in await cursor.fetchall()]
    persisted = ""
    for table in tables:
        rows = await (await storage.connection.execute(f'SELECT * FROM "{table}"')).fetchall()
        persisted += repr([dict(row) for row in rows])
    for marker in markers:
        assert marker not in persisted


@pytest.mark.asyncio
async def test_startup_gate_waits_for_reconciliation_and_cleanup_before_claiming(
    tmp_path: Path,
    storage: SQLiteStorage,
    kaiten: FakeKaiten,
    receiver: FakeReceiver,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_settings = configured(tmp_path, reconcile_on_startup=True, cleanup_on_startup=True)
    startup_service = BridgeService(
        startup_settings,
        storage,
        kaiten,
        receiver,  # type: ignore[arg-type]
    )
    card_id = 701
    kaiten.cards[card_id] = {"id": card_id, "board_id": BOARD, "column_id": QUEUE}
    kaiten.initial_results[(card_id, None)] = card_snapshot(card_id)
    event_id = await enqueue(startup_service, event_type="card:add", card_id=card_id)
    reconcile_entered = asyncio.Event()
    reconcile_release = asyncio.Event()
    cleanup_entered = asyncio.Event()
    cleanup_release = asyncio.Event()

    async def blocked_reconciliation() -> None:
        reconcile_entered.set()
        await reconcile_release.wait()

    async def blocked_cleanup() -> None:
        cleanup_entered.set()
        await cleanup_release.wait()

    monkeypatch.setattr(startup_service, "reconcile_once", blocked_reconciliation)
    monkeypatch.setattr(startup_service, "cleanup_once", blocked_cleanup)
    await startup_service.start()
    try:
        await asyncio.wait_for(reconcile_entered.wait(), timeout=1)
        during_reconciliation = await storage.get_event(event_id)
        assert during_reconciliation is not None
        assert during_reconciliation.status is EventStatus.PENDING
        assert receiver.calls == []

        reconcile_release.set()
        await asyncio.wait_for(cleanup_entered.wait(), timeout=1)
        during_cleanup = await storage.get_event(event_id)
        assert during_cleanup is not None and during_cleanup.status is EventStatus.PENDING
        assert receiver.calls == []

        cleanup_release.set()
        await asyncio.wait_for(
            wait_for_event_status(storage, event_id, EventStatus.SUCCESS), timeout=1
        )
        assert len(receiver.calls) == 1
    finally:
        await startup_service.stop()


@pytest.mark.asyncio
async def test_startup_failures_are_logged_but_always_open_worker_gate(
    tmp_path: Path,
    storage: SQLiteStorage,
    kaiten: FakeKaiten,
    receiver: FakeReceiver,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    startup_settings = configured(tmp_path, reconcile_on_startup=True, cleanup_on_startup=True)
    startup_service = BridgeService(
        startup_settings,
        storage,
        kaiten,
        receiver,  # type: ignore[arg-type]
    )
    card_id = 702
    kaiten.cards[card_id] = {"id": card_id, "board_id": BOARD, "column_id": QUEUE}
    kaiten.initial_results[(card_id, None)] = card_snapshot(card_id)
    event_id = await enqueue(startup_service, event_type="card:add", card_id=card_id)

    async def failed_reconciliation() -> None:
        raise RuntimeError("reconciliation unavailable")

    async def failed_cleanup() -> None:
        raise RuntimeError("cleanup unavailable")

    monkeypatch.setattr(startup_service, "reconcile_once", failed_reconciliation)
    monkeypatch.setattr(startup_service, "cleanup_once", failed_cleanup)
    await startup_service.start()
    try:
        await asyncio.wait_for(
            wait_for_event_status(storage, event_id, EventStatus.SUCCESS), timeout=1
        )
    finally:
        await startup_service.stop()

    assert "startup_reconciliation_failed" in caplog.messages
    assert "startup_cleanup_failed" in caplog.messages
