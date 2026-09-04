from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio

from kaiten_ai_bridge.storage import (
    CallbackOutcome,
    CardJobConflict,
    EventStatus,
    InvalidTransition,
    SQLiteStorage,
    utc_iso,
)

BASE = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


def moment(seconds: int = 0) -> datetime:
    return BASE + timedelta(seconds=seconds)


@pytest_asyncio.fixture
async def storage(tmp_path):
    database = SQLiteStorage(tmp_path / "bridge.sqlite3", recover_on_initialize=False)
    await database.initialize()
    try:
        yield database
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_pragmas_and_schema_contain_only_minimal_technical_state(storage):
    pragmas = await storage.pragma_settings()
    assert str(pragmas["journal_mode"]).lower() == "wal"
    assert pragmas["foreign_keys"] == 1
    assert pragmas["busy_timeout"] == 5_000

    expected_columns = {
        "id",
        "event_type",
        "card_id",
        "comment_id",
        "event_key",
        "status",
        "error_code",
        "attempts",
        "next_attempt_at",
        "notification_sent",
        "source_created_at",
        "created_at",
        "updated_at",
        "claimed_at",
        "success_at",
    }
    cursor = await storage.connection.execute("PRAGMA table_info(events)")
    columns = {row["name"] for row in await cursor.fetchall()}
    assert columns == expected_columns

    cursor = await storage.connection.execute("PRAGMA table_info(callback_keys)")
    callback_columns = {row["name"] for row in await cursor.fetchall()}
    assert callback_columns == {
        "result_id",
        "job_id",
        "status",
        "reserved_at",
        "updated_at",
        "success_at",
    }

    all_columns: set[str] = set()
    for table in ("events", "card_jobs", "callback_keys", "service_status"):
        cursor = await storage.connection.execute(f"PRAGMA table_info({table})")
        all_columns.update(row["name"] for row in await cursor.fetchall())
    assert all_columns.isdisjoint(
        {
            "webhook_body",
            "callback_body",
            "payload",
            "prompt",
            "comment_text",
            "file_name",
            "file_content",
            "base64",
            "email",
        }
    )


@pytest.mark.asyncio
async def test_atomic_enqueue_deduplicates_concurrent_webhooks(storage):
    async def enqueue():
        return await storage.enqueue_event(
            event_type="comment:add",
            card_id=17,
            comment_id=91,
            event_key="comment:add:91",
            source_created_at="2026-09-01T15:01:02+03:00",
            now=moment(),
        )

    results = await asyncio.gather(*(enqueue() for _ in range(12)))
    assert sum(result.inserted for result in results) == 1
    assert {result.event.id for result in results} == {results[0].event.id}
    assert results[0].event.source_created_at == "2026-09-01T12:01:02.000000Z"
    assert results[0].event.attempts == 0
    assert results[0].event.status is EventStatus.PENDING

    cursor = await storage.connection.execute("SELECT COUNT(*) FROM events")
    assert (await cursor.fetchone())[0] == 1


@pytest.mark.asyncio
async def test_claim_preserves_source_order_per_card_and_parallelizes_cards(storage):
    late = await storage.enqueue_event(
        event_type="comment:add",
        card_id=1,
        comment_id=102,
        event_key="comment:add:102",
        source_created_at=moment(20),
        now=moment(1),
    )
    early = await storage.enqueue_event(
        event_type="card:add",
        card_id=1,
        event_key="card:add:1",
        source_created_at=moment(10),
        now=moment(2),
    )
    other_card = await storage.enqueue_event(
        event_type="card:add",
        card_id=2,
        event_key="card:add:2",
        source_created_at=moment(11),
        now=moment(3),
    )

    listed = await storage.list_due_events(limit=10, now=moment(30))
    assert [event.id for event in listed] == [early.event.id, other_card.event.id]

    claimed = await storage.claim_due_events(limit=10, now=moment(30))
    assert [event.id for event in claimed] == [early.event.id, other_card.event.id]
    assert all(event.status is EventStatus.IN_PROGRESS for event in claimed)
    assert all(event.attempts == 1 for event in claimed)
    assert await storage.list_due_events(limit=10, now=moment(30)) == []

    await storage.mark_success(early.event.id, now=moment(31))
    due = await storage.list_due_events(limit=10, now=moment(31))
    assert [event.id for event in due] == [late.event.id]


@pytest.mark.asyncio
async def test_card_add_precedes_comment_even_when_comment_webhook_arrives_first(storage):
    comment = await storage.enqueue_event(
        event_type="comment:add",
        card_id=3,
        comment_id=301,
        event_key="comment:add:301",
        source_created_at=moment(10),
        now=moment(),
    )
    initial = await storage.enqueue_event(
        event_type="card:add",
        card_id=3,
        event_key="card:add:3",
        # Defensive case: even an inconsistent later source timestamp must not
        # allow a comment to run before the card has an AI job.
        source_created_at=moment(11),
        now=moment(1),
    )

    ordered = await storage.list_card_events(3)
    assert [event.id for event in ordered] == [initial.event.id, comment.event.id]
    [claimed_initial] = await storage.claim_due_events(now=moment(20))
    assert claimed_initial.id == initial.event.id
    await storage.mark_success(claimed_initial.id, now=moment(21))
    [claimed_comment] = await storage.claim_due_events(now=moment(22))
    assert claimed_comment.id == comment.event.id


@pytest.mark.asyncio
async def test_claim_is_atomic_across_independent_connections(tmp_path):
    path = tmp_path / "shared.sqlite3"
    first = SQLiteStorage(path, recover_on_initialize=False)
    second = SQLiteStorage(path, recover_on_initialize=False)
    await first.initialize()
    await second.initialize()
    try:
        for card_id in (7, 8):
            await first.enqueue_event(
                event_type="card:add",
                card_id=card_id,
                event_key=f"card:add:{card_id}",
                now=moment(),
            )
        claims = await asyncio.gather(
            first.claim_due_events(limit=1, now=moment()),
            second.claim_due_events(limit=1, now=moment()),
        )
        claimed = [event for batch in claims for event in batch]
        assert len(claimed) == 2
        assert {event.card_id for event in claimed} == {7, 8}
        assert all(event.attempts == 1 for event in claimed)
    finally:
        await first.close()
        await second.close()


@pytest.mark.asyncio
async def test_retry_and_technical_error_keep_later_card_event_blocked(storage):
    first = await storage.enqueue_event(
        event_type="card:add",
        card_id=10,
        event_key="card:add:10",
        source_created_at=moment(),
        now=moment(),
    )
    second = await storage.enqueue_event(
        event_type="comment:add",
        card_id=10,
        comment_id=1001,
        event_key="comment:add:1001",
        source_created_at=moment(1),
        now=moment(1),
    )

    [claimed] = await storage.claim_due_events(now=moment(2))
    assert claimed.id == first.event.id
    retried = await storage.mark_retry(
        claimed.id,
        next_attempt_at=moment(60),
        error_code="kaiten_unavailable",
        now=moment(3),
    )
    assert retried.status is EventStatus.RETRY
    assert await storage.list_due_events(now=moment(59)) == []

    [claimed_again] = await storage.claim_due_events(now=moment(60))
    assert claimed_again.id == first.event.id
    assert claimed_again.attempts == 2
    failed = await storage.mark_error(
        claimed_again.id, error_code="attempts_exhausted", now=moment(61)
    )
    assert failed.status is EventStatus.TECHNICAL_ERROR
    assert await storage.list_due_events(now=moment(600)) == []

    await storage.requeue_error(first.event.id, next_attempt_at=moment(601), now=moment(600))
    [recovered] = await storage.claim_due_events(now=moment(601))
    assert recovered.id == first.event.id
    await storage.mark_rejected(recovered.id, error_code="invalid_file", now=moment(602))

    [next_event] = await storage.claim_due_events(now=moment(603))
    assert next_event.id == second.event.id


@pytest.mark.asyncio
async def test_terminal_transitions_are_idempotent_but_not_replaceable(storage):
    result = await storage.enqueue_event(
        event_type="card:add", card_id=11, event_key="card:add:11", now=moment()
    )
    [claimed] = await storage.claim_due_events(now=moment())
    completed = await storage.mark_success(claimed.id, now=moment(1))
    repeated = await storage.mark_success(claimed.id, now=moment(2))
    assert repeated == completed
    assert repeated.success_at == utc_iso(moment(1))
    with pytest.raises(InvalidTransition):
        await storage.mark_ignored(result.event.id, now=moment(3))


@pytest.mark.asyncio
async def test_restart_recovers_claims_and_callback_reservations(tmp_path):
    path = tmp_path / "restart.sqlite3"
    first_process = SQLiteStorage(path, recover_on_initialize=False)
    await first_process.initialize()
    event = await first_process.enqueue_event(
        event_type="card:add", card_id=22, event_key="card:add:22", now=moment()
    )
    await first_process.claim_due_events(now=moment())
    await first_process.upsert_card_job(
        card_id=22,
        job_id="job-22",
        last_human_activity_at=moment(),
        now=moment(),
    )
    reservation = await first_process.reserve_callback(
        result_id="result-interrupted", job_id="job-22", now=moment()
    )
    assert reservation.acquired
    await first_process.close()

    second_process = SQLiteStorage(path)
    await second_process.initialize()
    try:
        recovered = await second_process.get_event(event.event.id)
        assert recovered is not None
        assert recovered.status is EventStatus.RETRY
        assert recovered.error_code == "process_restarted"
        assert recovered.claimed_at is None
        [claimed_again] = await second_process.claim_due_events(now=moment(10_000_000))
        assert claimed_again.id == event.event.id
        assert claimed_again.attempts == 2

        reservation_again = await second_process.reserve_callback(
            result_id="result-interrupted", job_id="job-22", now=moment(2)
        )
        assert reservation_again.outcome is CallbackOutcome.RESERVED
    finally:
        await second_process.close()


@pytest.mark.asyncio
async def test_card_job_activity_warning_expiry_and_conflicts(storage):
    job = await storage.upsert_card_job(
        card_id=31,
        job_id="job-31",
        last_human_activity_at=moment(),
        now=moment(),
    )
    assert job.warning_sent is False
    assert await storage.get_card_job_by_job_id("job-31") == job

    due = await storage.list_jobs_due_warning(inactive_since=moment())
    assert [item.card_id for item in due] == [31]
    assert await storage.mark_warning_sent(31, now=moment(1)) is True
    assert await storage.mark_warning_sent(31, now=moment(2)) is False
    assert await storage.list_jobs_due_warning(inactive_since=moment(100)) == []

    touched = await storage.touch_human_activity(31, activity_at=moment(200), now=moment(201))
    assert touched is not None
    assert touched.last_human_activity_at == utc_iso(moment(200))
    assert touched.warning_sent is False
    assert await storage.list_expired_jobs(inactive_since=moment(199)) == []
    assert [
        item.card_id for item in await storage.list_expired_jobs(inactive_since=moment(200))
    ] == [31]

    # Older late-arriving activity must not move the clock backwards.
    unchanged = await storage.touch_human_activity(31, activity_at=moment(100), now=moment(202))
    assert unchanged is not None
    assert unchanged.last_human_activity_at == utc_iso(moment(200))

    with pytest.raises(CardJobConflict):
        await storage.upsert_card_job(
            card_id=31,
            job_id="different-job",
            last_human_activity_at=moment(300),
            now=moment(300),
        )
    with pytest.raises(CardJobConflict):
        await storage.upsert_card_job(
            card_id=32,
            job_id="job-31",
            last_human_activity_at=moment(300),
            now=moment(300),
        )


@pytest.mark.asyncio
async def test_card_state_exists_before_202_and_job_is_bound_later(storage):
    state = await storage.ensure_card_state(
        card_id=35,
        last_human_activity_at=moment(),
        now=moment(),
    )
    assert state.job_id is None
    assert await storage.get_card_job_by_job_id("job-35") is None
    assert [
        item.card_id for item in await storage.list_jobs_due_warning(inactive_since=moment())
    ] == [35]

    touched = await storage.touch_human_activity(35, activity_at=moment(10), now=moment(10))
    assert touched is not None and touched.job_id is None
    bound = await storage.bind_job(card_id=35, job_id="job-35", now=moment(11))
    assert bound.job_id == "job-35"
    assert bound.last_human_activity_at == utc_iso(moment(10))
    assert await storage.get_card_job_by_job_id("job-35") == bound

    # Binding is idempotent, but neither side may be rebound differently.
    assert await storage.bind_job(card_id=35, job_id="job-35", now=moment(12))
    with pytest.raises(CardJobConflict):
        await storage.bind_job(card_id=35, job_id="other", now=moment(13))
    with pytest.raises(KeyError):
        await storage.bind_job(card_id=999, job_id="job-999", now=moment(13))


@pytest.mark.asyncio
async def test_callback_reserve_success_release_and_unknown_job(storage):
    unknown = await storage.reserve_callback(
        result_id="unknown-result", job_id="missing-job", now=moment()
    )
    assert unknown.outcome is CallbackOutcome.UNKNOWN_JOB
    cursor = await storage.connection.execute("SELECT COUNT(*) FROM callback_keys")
    assert (await cursor.fetchone())[0] == 0

    for card_id in (41, 42):
        await storage.upsert_card_job(
            card_id=card_id,
            job_id=f"job-{card_id}",
            last_human_activity_at=moment(),
            now=moment(),
        )

    reserved = await storage.reserve_callback(result_id="result-1", job_id="job-41", now=moment())
    assert reserved.outcome is CallbackOutcome.RESERVED
    assert reserved.card_id == 41
    assert (
        await storage.reserve_callback(result_id="result-1", job_id="job-41", now=moment(1))
    ).outcome is CallbackOutcome.BUSY
    assert (
        await storage.reserve_callback(result_id="result-1", job_id="job-42", now=moment(1))
    ).outcome is CallbackOutcome.CONFLICT

    assert await storage.release_callback("result-1") is True
    assert await storage.release_callback("result-1") is False
    assert (
        await storage.reserve_callback(result_id="result-1", job_id="job-41", now=moment(2))
    ).acquired
    assert await storage.mark_callback_success("result-1", now=moment(3)) is True
    assert await storage.mark_callback_success("result-1", now=moment(4)) is True
    assert await storage.release_callback("result-1") is False
    assert (
        await storage.reserve_callback(result_id="result-1", job_id="job-41", now=moment(5))
    ).outcome is CallbackOutcome.DUPLICATE


@pytest.mark.asyncio
async def test_stale_callback_reservation_can_be_taken_over(storage):
    await storage.upsert_card_job(
        card_id=43,
        job_id="job-43",
        last_human_activity_at=moment(),
        now=moment(),
    )
    await storage.reserve_callback(result_id="stale", job_id="job-43", now=moment())
    takeover = await storage.reserve_callback(
        result_id="stale",
        job_id="job-43",
        stale_before=moment(1),
        now=moment(10),
    )
    assert takeover.outcome is CallbackOutcome.RESERVED


@pytest.mark.asyncio
async def test_notification_service_status_counts_and_health(storage):
    pending = await storage.enqueue_event(
        event_type="card:add", card_id=51, event_key="card:add:51", now=moment()
    )
    failed = await storage.enqueue_event(
        event_type="card:add", card_id=52, event_key="card:add:52", now=moment()
    )
    [first, second] = await storage.claim_due_events(limit=2, now=moment())
    assert {first.id, second.id} == {pending.event.id, failed.event.id}
    await storage.mark_success(pending.event.id, now=moment(1))
    await storage.mark_error(failed.event.id, error_code="network", now=moment(1))

    assert await storage.mark_notification_sent(failed.event.id, now=moment(2)) is True
    assert await storage.mark_notification_sent(failed.event.id, now=moment(3)) is False
    saved = await storage.get_event(failed.event.id)
    assert saved is not None and saved.notification_sent

    status = await storage.set_service_status("last_webhook_at", utc_iso(moment(4)), now=moment(4))
    assert await storage.get_service_status("last_webhook_at") == status
    counts = await storage.queue_counts()
    assert counts["success"] == 1
    assert counts["technical_error"] == 1
    assert counts["errors"] == 1
    assert counts["queue_size"] == 1
    assert counts["total"] == 2
    health = await storage.health_snapshot()
    assert health["queue"] == counts
    assert health["service"]["last_webhook_at"]["value"] == utc_iso(moment(4))


@pytest.mark.asyncio
async def test_retention_cleanup_preserves_unfinished_and_card_mapping(storage):
    old_success = await storage.enqueue_event(
        event_type="card:add", card_id=61, event_key="card:add:61", now=moment()
    )
    [claimed] = await storage.claim_due_events(now=moment())
    await storage.mark_success(claimed.id, now=moment(1))
    unfinished = await storage.enqueue_event(
        event_type="comment:add",
        card_id=61,
        comment_id=610,
        event_key="comment:add:610",
        now=moment(2),
    )
    ignored = await storage.enqueue_event(
        event_type="comment:add",
        card_id=62,
        comment_id=620,
        event_key="comment:add:620",
        now=moment(),
    )
    claimed_ignored = await storage.claim_due_events(now=moment())
    ignored_event = next(item for item in claimed_ignored if item.id == ignored.event.id)
    await storage.mark_ignored(ignored_event.id, error_code="card_closed", now=moment(1))
    await storage.upsert_card_job(
        card_id=61,
        job_id="job-61",
        last_human_activity_at=moment(),
        now=moment(),
    )
    await storage.reserve_callback(result_id="result-old", job_id="job-61", now=moment())
    await storage.mark_callback_success("result-old", now=moment(1))

    result = await storage.cleanup_retention(event_before=moment(10))
    assert result.events_deleted == 2
    assert result.callbacks_deleted == 1
    assert await storage.get_event(old_success.event.id) is None
    assert await storage.get_event(ignored.event.id) is None
    assert await storage.get_event(unfinished.event.id) is not None
    assert await storage.get_card_job(61) is not None


@pytest.mark.asyncio
async def test_delete_card_state_removes_events_job_and_callback_keys(storage):
    event = await storage.enqueue_event(
        event_type="card:add", card_id=71, event_key="card:add:71", now=moment()
    )
    await storage.upsert_card_job(
        card_id=71,
        job_id="job-71",
        last_human_activity_at=moment(),
        now=moment(),
    )
    await storage.reserve_callback(result_id="result-71", job_id="job-71", now=moment())
    await storage.mark_callback_success("result-71", now=moment(1))

    deleted = await storage.delete_card_state(71)
    assert deleted.events_deleted == 1
    assert deleted.callbacks_deleted == 1
    assert deleted.jobs_deleted == 1
    assert await storage.get_event(event.event.id) is None
    assert await storage.get_card_job(71) is None
    assert (
        await storage.reserve_callback(result_id="result-71", job_id="job-71", now=moment(2))
    ).outcome is CallbackOutcome.UNKNOWN_JOB


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-09-01T12:00:00Z", "2026-09-01T12:00:00.000000Z"),
        ("2026-09-01T15:00:00+03:00", "2026-09-01T12:00:00.000000Z"),
        (datetime(2026, 9, 1, 12, 0), "2026-09-01T12:00:00.000000Z"),
    ],
)
def test_utc_iso_is_canonical(value, expected):
    assert utc_iso(value) == expected
