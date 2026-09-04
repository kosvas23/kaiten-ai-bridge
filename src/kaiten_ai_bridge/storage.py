"""Persistent, content-free state for the Kaiten/AI bridge.

The database deliberately stores only technical identifiers, state flags, counters,
and timestamps.  User-authored webhook/callback bodies, prompts, comments, fields,
and file metadata do not belong in this module's schema or API.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import aiosqlite


class EventStatus(StrEnum):
    """Lifecycle states for a queued Kaiten event."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RETRY = "retry"
    SUCCESS = "success"
    TECHNICAL_ERROR = "technical_error"
    ERROR = "technical_error"  # Convenient alias used by callers.
    REJECTED = "rejected"
    IGNORED = "ignored"


class CallbackStatus(StrEnum):
    RESERVED = "reserved"
    SUCCESS = "success"


class CallbackOutcome(StrEnum):
    RESERVED = "reserved"
    DUPLICATE = "duplicate"
    BUSY = "busy"
    UNKNOWN_JOB = "unknown_job"
    CONFLICT = "conflict"


TERMINAL_EVENT_STATUSES: Final[frozenset[str]] = frozenset(
    {
        EventStatus.SUCCESS.value,
        EventStatus.REJECTED.value,
        EventStatus.IGNORED.value,
    }
)
BLOCKING_EVENT_STATUSES: Final[frozenset[str]] = frozenset(
    {
        EventStatus.PENDING.value,
        EventStatus.IN_PROGRESS.value,
        EventStatus.RETRY.value,
        EventStatus.TECHNICAL_ERROR.value,
    }
)


class StorageError(RuntimeError):
    """Base exception for local-state failures."""


class StorageNotInitialized(StorageError):
    """Raised when an operation is attempted before :meth:`initialize`."""


class InvalidTransition(StorageError):
    """Raised when an event is moved from a terminal/incompatible state."""


class CardJobConflict(StorageError):
    """Raised when a card or AI job is already mapped differently."""


Timestamp = datetime | str


def utc_now() -> datetime:
    """Return an aware current UTC timestamp."""

    return datetime.now(UTC)


def utc_iso(value: Timestamp | None = None) -> str:
    """Return a canonical, lexically sortable UTC ISO-8601 timestamp."""

    if value is None:
        parsed = utc_now()
    elif isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            raise ValueError("timestamp must not be empty")
        if candidate.endswith(("Z", "z")):
            candidate = f"{candidate[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError as exc:
            raise ValueError(f"invalid ISO-8601 timestamp: {value!r}") from exc
    else:  # pragma: no cover - guarded by typing, retained for runtime callers.
        raise TypeError("timestamp must be a datetime, ISO string, or None")

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _status_value(status: EventStatus | str) -> str:
    return status.value if isinstance(status, EventStatus) else status


@dataclass(frozen=True, slots=True)
class EventRecord:
    id: int
    event_type: str
    card_id: int
    comment_id: int | None
    event_key: str
    status: EventStatus
    error_code: str | None
    attempts: int
    next_attempt_at: str | None
    notification_sent: bool
    source_created_at: str | None
    created_at: str
    updated_at: str
    claimed_at: str | None
    success_at: str | None


@dataclass(frozen=True, slots=True)
class EnqueueResult:
    event: EventRecord
    inserted: bool

    def __bool__(self) -> bool:
        return self.inserted


@dataclass(frozen=True, slots=True)
class CardJobRecord:
    card_id: int
    job_id: str | None
    last_human_activity_at: str
    warning_sent: bool
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class CallbackReservation:
    outcome: CallbackOutcome
    result_id: str
    job_id: str
    card_id: int | None

    @property
    def acquired(self) -> bool:
        return self.outcome is CallbackOutcome.RESERVED


@dataclass(frozen=True, slots=True)
class ServiceStatusRecord:
    key: str
    value: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class CleanupResult:
    events_deleted: int
    callbacks_deleted: int


@dataclass(frozen=True, slots=True)
class CardStateDeletion:
    events_deleted: int
    callbacks_deleted: int
    jobs_deleted: int


_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type          TEXT NOT NULL CHECK (length(event_type) > 0),
    card_id             INTEGER NOT NULL CHECK (card_id > 0),
    comment_id          INTEGER CHECK (comment_id IS NULL OR comment_id > 0),
    event_key           TEXT NOT NULL UNIQUE CHECK (length(event_key) > 0),
    status              TEXT NOT NULL DEFAULT 'pending' CHECK (
        status IN (
            'pending', 'in_progress', 'retry', 'success',
            'technical_error', 'rejected', 'ignored'
        )
    ),
    error_code          TEXT CHECK (error_code IS NULL OR length(error_code) <= 128),
    attempts            INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    next_attempt_at     TEXT,
    notification_sent   INTEGER NOT NULL DEFAULT 0 CHECK (notification_sent IN (0, 1)),
    source_created_at   TEXT,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    claimed_at          TEXT,
    success_at          TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_due
    ON events(status, next_attempt_at, card_id);
CREATE INDEX IF NOT EXISTS idx_events_card_order
    ON events(card_id, source_created_at, created_at, id);

CREATE TABLE IF NOT EXISTS card_jobs (
    card_id                    INTEGER PRIMARY KEY CHECK (card_id > 0),
    job_id                     TEXT UNIQUE CHECK (job_id IS NULL OR length(job_id) > 0),
    last_human_activity_at     TEXT NOT NULL,
    warning_sent               INTEGER NOT NULL DEFAULT 0 CHECK (warning_sent IN (0, 1)),
    created_at                 TEXT NOT NULL,
    updated_at                 TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_card_jobs_activity
    ON card_jobs(last_human_activity_at, warning_sent);

CREATE TABLE IF NOT EXISTS callback_keys (
    result_id       TEXT PRIMARY KEY CHECK (length(result_id) > 0),
    job_id          TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('reserved', 'success')),
    reserved_at     TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    success_at      TEXT,
    FOREIGN KEY (job_id) REFERENCES card_jobs(job_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_callback_keys_success
    ON callback_keys(status, success_at);

CREATE TABLE IF NOT EXISTS service_status (
    key             TEXT PRIMARY KEY CHECK (length(key) > 0),
    value           TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
"""


_EVENT_COLUMNS = """
id, event_type, card_id, comment_id, event_key, status, error_code,
attempts, next_attempt_at, notification_sent, source_created_at,
created_at, updated_at, claimed_at, success_at
"""

_CARD_JOB_COLUMNS = """
card_id, job_id, last_human_activity_at, warning_sent, created_at, updated_at
"""


class SQLiteStorage:
    """Async SQLite queue and minimal bridge state.

    One instance owns one ``aiosqlite`` connection.  Claims are serialized with a
    short ``BEGIN IMMEDIATE`` transaction; workers can process the returned events
    concurrently because at most one event per card is returned per claim.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        busy_timeout_ms: int = 5_000,
        recover_on_initialize: bool = True,
    ) -> None:
        if busy_timeout_ms < 0:
            raise ValueError("busy_timeout_ms must be non-negative")
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self.busy_timeout_ms = int(busy_timeout_ms)
        self.recover_on_initialize = recover_on_initialize
        self._connection: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()

    @property
    def connection(self) -> aiosqlite.Connection:
        connection = self._connection
        if connection is None:
            raise StorageNotInitialized("initialize() must be awaited first")
        return connection

    async def initialize(self) -> SQLiteStorage:
        """Open the database, create its schema, and recover interrupted work."""

        if self._connection is not None:
            return self

        connection = await aiosqlite.connect(self.path, isolation_level=None)
        connection.row_factory = aiosqlite.Row
        try:
            await connection.execute("PRAGMA foreign_keys = ON")
            await connection.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms:d}")
            await connection.execute("PRAGMA journal_mode = WAL")
            await connection.execute("PRAGMA synchronous = NORMAL")
            await connection.executescript(_SCHEMA)
        except BaseException:
            await connection.close()
            raise
        self._connection = connection

        if self.recover_on_initialize:
            await self.recover_after_restart()
        return self

    open = initialize

    async def close(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            await connection.close()

    async def __aenter__(self) -> SQLiteStorage:
        return await self.initialize()

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.close()

    async def pragma_settings(self) -> dict[str, Any]:
        """Expose effective durability settings for diagnostics/tests."""

        connection = self.connection
        settings: dict[str, Any] = {}
        for name in ("journal_mode", "foreign_keys", "busy_timeout", "synchronous"):
            cursor = await connection.execute(f"PRAGMA {name}")
            row = await cursor.fetchone()
            settings[name] = row[0] if row is not None else None
        return settings

    async def enqueue_event(
        self,
        *,
        event_type: str,
        card_id: int,
        event_key: str,
        comment_id: int | None = None,
        source_created_at: Timestamp | None = None,
        now: Timestamp | None = None,
    ) -> EnqueueResult:
        """Atomically enqueue an event, returning the existing row on a duplicate."""

        if not event_type:
            raise ValueError("event_type must not be empty")
        if not event_key:
            raise ValueError("event_key must not be empty")
        if card_id <= 0:
            raise ValueError("card_id must be positive")
        if comment_id is not None and comment_id <= 0:
            raise ValueError("comment_id must be positive")
        timestamp = utc_iso(now)
        source_timestamp = utc_iso(source_created_at) if source_created_at is not None else None

        async with self._write_lock:
            connection = self.connection
            await connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = await connection.execute(
                    """
                    INSERT INTO events (
                        event_type, card_id, comment_id, event_key, status,
                        attempts, next_attempt_at, notification_sent,
                        source_created_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 'pending', 0, ?, 0, ?, ?, ?)
                    ON CONFLICT(event_key) DO NOTHING
                    """,
                    (
                        event_type,
                        card_id,
                        comment_id,
                        event_key,
                        timestamp,
                        source_timestamp,
                        timestamp,
                        timestamp,
                    ),
                )
                inserted = cursor.rowcount == 1
                cursor = await connection.execute(
                    f"SELECT {_EVENT_COLUMNS} FROM events WHERE event_key = ?",
                    (event_key,),
                )
                row = await cursor.fetchone()
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise

        if row is None:  # pragma: no cover - protected by the transaction above.
            raise StorageError("event disappeared during enqueue")
        return EnqueueResult(event=_event_from_row(row), inserted=inserted)

    async def get_event(self, event_id: int) -> EventRecord | None:
        cursor = await self.connection.execute(
            f"SELECT {_EVENT_COLUMNS} FROM events WHERE id = ?", (event_id,)
        )
        return _optional_event(await cursor.fetchone())

    async def get_event_by_key(self, event_key: str) -> EventRecord | None:
        cursor = await self.connection.execute(
            f"SELECT {_EVENT_COLUMNS} FROM events WHERE event_key = ?", (event_key,)
        )
        return _optional_event(await cursor.fetchone())

    async def list_card_events(self, card_id: int) -> list[EventRecord]:
        cursor = await self.connection.execute(
            f"""
            SELECT {_EVENT_COLUMNS}
             FROM events
             WHERE card_id = ?
             ORDER BY CASE WHEN event_type = 'card:add' THEN 0 ELSE 1 END,
                      COALESCE(source_created_at, created_at),
                      id
            """,
            (card_id,),
        )
        return [_event_from_row(row) for row in await cursor.fetchall()]

    async def list_due_events(
        self, *, limit: int = 100, now: Timestamp | None = None
    ) -> list[EventRecord]:
        """List due card-head events without claiming them.

        A pending/in-progress/retry/technical-error predecessor blocks every later
        event for the same card.  Terminal predecessors do not.  Consequently this
        method yields no more than one event per card.
        """

        if limit <= 0:
            return []
        rows = await self._select_due_rows(limit=limit, now_iso=utc_iso(now))
        return [_event_from_row(row) for row in rows]

    async def claim_due_events(
        self, *, limit: int = 100, now: Timestamp | None = None
    ) -> list[EventRecord]:
        """Atomically claim due card-head events for concurrent worker processing."""

        if limit <= 0:
            return []
        timestamp = utc_iso(now)
        async with self._write_lock:
            connection = self.connection
            await connection.execute("BEGIN IMMEDIATE")
            try:
                rows = await self._select_due_rows(
                    limit=limit, now_iso=timestamp, connection=connection
                )
                ids = [int(row["id"]) for row in rows]
                if ids:
                    placeholders = ",".join("?" for _ in ids)
                    await connection.execute(
                        f"""
                        UPDATE events
                           SET status = 'in_progress',
                               attempts = attempts + 1,
                               claimed_at = ?,
                               next_attempt_at = NULL,
                               updated_at = ?
                         WHERE id IN ({placeholders})
                           AND status IN ('pending', 'retry')
                        """,
                        (timestamp, timestamp, *ids),
                    )
                    cursor = await connection.execute(
                        f"SELECT {_EVENT_COLUMNS} FROM events WHERE id IN ({placeholders})",
                        ids,
                    )
                    claimed_by_id = {
                        int(row["id"]): _event_from_row(row) for row in await cursor.fetchall()
                    }
                    claimed = [claimed_by_id[event_id] for event_id in ids]
                else:
                    claimed = []
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        return claimed

    claim_events = claim_due_events

    async def _select_due_rows(
        self,
        *,
        limit: int,
        now_iso: str,
        connection: aiosqlite.Connection | None = None,
    ) -> list[aiosqlite.Row]:
        connection = connection or self.connection
        blockers = ",".join("?" for _ in BLOCKING_EVENT_STATUSES)
        cursor = await connection.execute(
            f"""
            SELECT {_EVENT_COLUMNS}
              FROM events AS candidate
             WHERE candidate.status IN ('pending', 'retry')
               AND candidate.next_attempt_at <= ?
               AND NOT EXISTS (
                    SELECT 1
                      FROM events AS previous
                     WHERE previous.card_id = candidate.card_id
                       AND previous.status IN ({blockers})
                       AND (
                            CASE WHEN previous.event_type = 'card:add' THEN 0 ELSE 1 END
                                < CASE WHEN candidate.event_type = 'card:add' THEN 0 ELSE 1 END
                            OR (
                                CASE WHEN previous.event_type = 'card:add' THEN 0 ELSE 1 END
                                    = CASE WHEN candidate.event_type = 'card:add' THEN 0 ELSE 1 END
                                AND (
                                    COALESCE(previous.source_created_at, previous.created_at)
                                        < COALESCE(candidate.source_created_at, candidate.created_at)
                                    OR (
                                        COALESCE(previous.source_created_at, previous.created_at)
                                            = COALESCE(candidate.source_created_at, candidate.created_at)
                                        AND previous.id < candidate.id
                                    )
                                )
                            )
                       )
               )
             ORDER BY COALESCE(candidate.source_created_at, candidate.created_at),
                      candidate.id
             LIMIT ?
            """,
            (now_iso, *sorted(BLOCKING_EVENT_STATUSES), int(limit)),
        )
        return list(await cursor.fetchall())

    async def mark_success(self, event_id: int, *, now: Timestamp | None = None) -> EventRecord:
        return await self._transition_event(
            event_id,
            target=EventStatus.SUCCESS,
            allowed=(EventStatus.IN_PROGRESS,),
            now=now,
            error_code=None,
            next_attempt_at=None,
            set_success_at=True,
        )

    succeed_event = mark_success

    async def mark_retry(
        self,
        event_id: int,
        *,
        next_attempt_at: Timestamp,
        error_code: str | None = None,
        now: Timestamp | None = None,
    ) -> EventRecord:
        return await self._transition_event(
            event_id,
            target=EventStatus.RETRY,
            allowed=(EventStatus.IN_PROGRESS, EventStatus.TECHNICAL_ERROR),
            now=now,
            error_code=error_code,
            next_attempt_at=utc_iso(next_attempt_at),
        )

    retry_event = mark_retry

    async def mark_error(
        self,
        event_id: int,
        *,
        error_code: str | None = None,
        now: Timestamp | None = None,
    ) -> EventRecord:
        return await self._transition_event(
            event_id,
            target=EventStatus.TECHNICAL_ERROR,
            allowed=(EventStatus.IN_PROGRESS, EventStatus.RETRY, EventStatus.PENDING),
            now=now,
            error_code=error_code,
            next_attempt_at=None,
        )

    fail_event = mark_error

    async def mark_rejected(
        self,
        event_id: int,
        *,
        error_code: str | None = None,
        now: Timestamp | None = None,
    ) -> EventRecord:
        return await self._transition_event(
            event_id,
            target=EventStatus.REJECTED,
            allowed=(
                EventStatus.PENDING,
                EventStatus.IN_PROGRESS,
                EventStatus.RETRY,
                EventStatus.TECHNICAL_ERROR,
            ),
            now=now,
            error_code=error_code,
            next_attempt_at=None,
        )

    reject_event = mark_rejected

    async def mark_ignored(
        self,
        event_id: int,
        *,
        error_code: str | None = None,
        now: Timestamp | None = None,
    ) -> EventRecord:
        return await self._transition_event(
            event_id,
            target=EventStatus.IGNORED,
            allowed=(
                EventStatus.PENDING,
                EventStatus.IN_PROGRESS,
                EventStatus.RETRY,
                EventStatus.TECHNICAL_ERROR,
            ),
            now=now,
            error_code=error_code,
            next_attempt_at=None,
        )

    ignore_event = mark_ignored

    async def requeue_error(
        self,
        event_id: int,
        *,
        next_attempt_at: Timestamp | None = None,
        now: Timestamp | None = None,
    ) -> EventRecord:
        timestamp = utc_iso(now)
        return await self._transition_event(
            event_id,
            target=EventStatus.RETRY,
            allowed=(EventStatus.TECHNICAL_ERROR,),
            now=timestamp,
            error_code=None,
            next_attempt_at=utc_iso(next_attempt_at or timestamp),
        )

    async def _transition_event(
        self,
        event_id: int,
        *,
        target: EventStatus,
        allowed: Iterable[EventStatus],
        now: Timestamp | None,
        error_code: str | None,
        next_attempt_at: str | None,
        set_success_at: bool = False,
    ) -> EventRecord:
        if error_code is not None and len(error_code) > 128:
            raise ValueError("error_code must be at most 128 characters")
        timestamp = utc_iso(now)
        allowed_values = tuple(dict.fromkeys(_status_value(item) for item in allowed))
        placeholders = ",".join("?" for _ in allowed_values)

        async with self._write_lock:
            connection = self.connection
            await connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = await connection.execute(
                    f"""
                    UPDATE events
                       SET status = ?, error_code = ?, next_attempt_at = ?,
                           claimed_at = NULL, updated_at = ?,
                           success_at = CASE WHEN ? THEN ? ELSE success_at END
                     WHERE id = ? AND status IN ({placeholders})
                    """,
                    (
                        target.value,
                        error_code,
                        next_attempt_at,
                        timestamp,
                        int(set_success_at),
                        timestamp,
                        event_id,
                        *allowed_values,
                    ),
                )
                if cursor.rowcount != 1:
                    existing_cursor = await connection.execute(
                        f"SELECT {_EVENT_COLUMNS} FROM events WHERE id = ?", (event_id,)
                    )
                    existing = await existing_cursor.fetchone()
                    if existing is None:
                        raise KeyError(f"unknown event id: {event_id}")
                    if existing["status"] == target.value:
                        await connection.commit()
                        return _event_from_row(existing)
                    raise InvalidTransition(
                        f"cannot move event {event_id} from {existing['status']} to {target.value}"
                    )
                cursor = await connection.execute(
                    f"SELECT {_EVENT_COLUMNS} FROM events WHERE id = ?", (event_id,)
                )
                row = await cursor.fetchone()
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        if row is None:  # pragma: no cover
            raise StorageError("event disappeared during transition")
        return _event_from_row(row)

    async def mark_notification_sent(self, event_id: int, *, now: Timestamp | None = None) -> bool:
        timestamp = utc_iso(now)
        async with self._write_lock:
            cursor = await self.connection.execute(
                """
                UPDATE events
                   SET notification_sent = 1, updated_at = ?
                 WHERE id = ? AND notification_sent = 0
                """,
                (timestamp, event_id),
            )
        return cursor.rowcount == 1

    async def recover_in_progress(self, *, now: Timestamp | None = None) -> int:
        """Return interrupted queue claims to retryable state."""

        timestamp = utc_iso(now)
        async with self._write_lock:
            cursor = await self.connection.execute(
                """
                UPDATE events
                   SET status = 'retry', next_attempt_at = ?, claimed_at = NULL,
                       updated_at = ?,
                       error_code = COALESCE(error_code, 'process_restarted')
                 WHERE status = 'in_progress'
                """,
                (timestamp, timestamp),
            )
        return cursor.rowcount

    async def recover_callback_reservations(self) -> int:
        """Release body-less callback reservations after a process restart."""

        async with self._write_lock:
            cursor = await self.connection.execute(
                "DELETE FROM callback_keys WHERE status = 'reserved'"
            )
        return cursor.rowcount

    async def recover_after_restart(self, *, now: Timestamp | None = None) -> dict[str, int]:
        events = await self.recover_in_progress(now=now)
        callbacks = await self.recover_callback_reservations()
        return {"events_requeued": events, "callbacks_released": callbacks}

    async def upsert_card_job(
        self,
        *,
        card_id: int,
        job_id: str | None = None,
        last_human_activity_at: Timestamp,
        now: Timestamp | None = None,
    ) -> CardJobRecord:
        """Create/refresh card state, optionally with its accepted AI job.

        A row with ``job_id=None`` tracks activity while the initial request is
        pending or data-rejected.  Once a receiver returns ``202`` plus a job ID,
        :meth:`bind_job` fills the mapping without losing that activity history.
        """

        if card_id <= 0:
            raise ValueError("card_id must be positive")
        if job_id == "":
            raise ValueError("job_id must not be empty")
        timestamp = utc_iso(now)
        activity = utc_iso(last_human_activity_at)

        async with self._write_lock:
            connection = self.connection
            await connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = await connection.execute(
                    f"""
                    SELECT {_CARD_JOB_COLUMNS}
                      FROM card_jobs
                     WHERE card_id = ? OR (? IS NOT NULL AND job_id = ?)
                    """,
                    (card_id, job_id, job_id),
                )
                existing_rows = list(await cursor.fetchall())
                if any(int(row["card_id"]) != card_id for row in existing_rows):
                    raise CardJobConflict(
                        f"card {card_id} or job {job_id!r} is already mapped differently"
                    )
                current = next(
                    (row for row in existing_rows if int(row["card_id"]) == card_id),
                    None,
                )
                if (
                    current is not None
                    and current["job_id"] is not None
                    and job_id is not None
                    and current["job_id"] != job_id
                ):
                    raise CardJobConflict(
                        f"card {card_id} is already mapped to {current['job_id']!r}"
                    )

                await connection.execute(
                    """
                    INSERT INTO card_jobs (
                        card_id, job_id, last_human_activity_at, warning_sent,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, 0, ?, ?)
                    ON CONFLICT(card_id) DO UPDATE SET
                        job_id = COALESCE(card_jobs.job_id, excluded.job_id),
                        last_human_activity_at = CASE
                            WHEN excluded.last_human_activity_at
                                 > card_jobs.last_human_activity_at
                            THEN excluded.last_human_activity_at
                            ELSE card_jobs.last_human_activity_at
                        END,
                        warning_sent = CASE
                            WHEN excluded.last_human_activity_at
                                 > card_jobs.last_human_activity_at
                            THEN 0 ELSE card_jobs.warning_sent
                        END,
                        updated_at = excluded.updated_at
                    """,
                    (card_id, job_id, activity, timestamp, timestamp),
                )
                cursor = await connection.execute(
                    f"SELECT {_CARD_JOB_COLUMNS} FROM card_jobs WHERE card_id = ?",
                    (card_id,),
                )
                row = await cursor.fetchone()
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        if row is None:  # pragma: no cover
            raise StorageError("card job disappeared during upsert")
        return _card_job_from_row(row)

    save_card_job = upsert_card_job

    async def ensure_card_state(
        self,
        *,
        card_id: int,
        last_human_activity_at: Timestamp,
        now: Timestamp | None = None,
    ) -> CardJobRecord:
        """Ensure lifecycle state exists before an AI job has been accepted."""

        return await self.upsert_card_job(
            card_id=card_id,
            job_id=None,
            last_human_activity_at=last_human_activity_at,
            now=now,
        )

    async def bind_job(
        self,
        *,
        card_id: int,
        job_id: str,
        now: Timestamp | None = None,
    ) -> CardJobRecord:
        """Bind a receiver job ID after confirmed acceptance of the initial request."""

        if not job_id:
            raise ValueError("job_id must not be empty")
        timestamp = utc_iso(now)
        async with self._write_lock:
            connection = self.connection
            await connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = await connection.execute(
                    f"SELECT {_CARD_JOB_COLUMNS} FROM card_jobs WHERE card_id = ?",
                    (card_id,),
                )
                current = await cursor.fetchone()
                if current is None:
                    raise KeyError(f"unknown card state: {card_id}")
                if current["job_id"] is not None and current["job_id"] != job_id:
                    raise CardJobConflict(
                        f"card {card_id} is already mapped to {current['job_id']!r}"
                    )
                cursor = await connection.execute(
                    "SELECT card_id FROM card_jobs WHERE job_id = ? AND card_id <> ?",
                    (job_id, card_id),
                )
                conflict = await cursor.fetchone()
                if conflict is not None:
                    raise CardJobConflict(
                        f"job {job_id!r} is already mapped to card {conflict['card_id']}"
                    )
                await connection.execute(
                    """
                    UPDATE card_jobs
                       SET job_id = ?, updated_at = ?
                     WHERE card_id = ?
                    """,
                    (job_id, timestamp, card_id),
                )
                cursor = await connection.execute(
                    f"SELECT {_CARD_JOB_COLUMNS} FROM card_jobs WHERE card_id = ?",
                    (card_id,),
                )
                row = await cursor.fetchone()
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        if row is None:  # pragma: no cover
            raise StorageError("card state disappeared while binding its job")
        return _card_job_from_row(row)

    async def get_card_job(self, card_id: int) -> CardJobRecord | None:
        cursor = await self.connection.execute(
            f"SELECT {_CARD_JOB_COLUMNS} FROM card_jobs WHERE card_id = ?", (card_id,)
        )
        row = await cursor.fetchone()
        return _card_job_from_row(row) if row is not None else None

    async def get_card_job_by_job_id(self, job_id: str) -> CardJobRecord | None:
        cursor = await self.connection.execute(
            f"SELECT {_CARD_JOB_COLUMNS} FROM card_jobs WHERE job_id = ?", (job_id,)
        )
        row = await cursor.fetchone()
        return _card_job_from_row(row) if row is not None else None

    async def touch_human_activity(
        self,
        card_id: int,
        *,
        activity_at: Timestamp,
        now: Timestamp | None = None,
    ) -> CardJobRecord | None:
        """Advance activity monotonically and reset a prior expiry warning."""

        activity = utc_iso(activity_at)
        timestamp = utc_iso(now)
        async with self._write_lock:
            connection = self.connection
            await connection.execute(
                """
                UPDATE card_jobs
                   SET last_human_activity_at = ?, warning_sent = 0, updated_at = ?
                 WHERE card_id = ? AND last_human_activity_at < ?
                """,
                (activity, timestamp, card_id, activity),
            )
            cursor = await connection.execute(
                f"SELECT {_CARD_JOB_COLUMNS} FROM card_jobs WHERE card_id = ?", (card_id,)
            )
            row = await cursor.fetchone()
        return _card_job_from_row(row) if row is not None else None

    async def mark_warning_sent(self, card_id: int, *, now: Timestamp | None = None) -> bool:
        timestamp = utc_iso(now)
        async with self._write_lock:
            cursor = await self.connection.execute(
                """
                UPDATE card_jobs
                   SET warning_sent = 1, updated_at = ?
                 WHERE card_id = ? AND warning_sent = 0
                """,
                (timestamp, card_id),
            )
        return cursor.rowcount == 1

    async def list_jobs_due_warning(
        self, *, inactive_since: Timestamp, limit: int = 100
    ) -> list[CardJobRecord]:
        if limit <= 0:
            return []
        cursor = await self.connection.execute(
            f"""
            SELECT {_CARD_JOB_COLUMNS}
              FROM card_jobs
             WHERE warning_sent = 0 AND last_human_activity_at <= ?
             ORDER BY last_human_activity_at, card_id
             LIMIT ?
            """,
            (utc_iso(inactive_since), int(limit)),
        )
        return [_card_job_from_row(row) for row in await cursor.fetchall()]

    async def list_expired_jobs(
        self, *, inactive_since: Timestamp, limit: int = 100
    ) -> list[CardJobRecord]:
        if limit <= 0:
            return []
        cursor = await self.connection.execute(
            f"""
            SELECT {_CARD_JOB_COLUMNS}
              FROM card_jobs
             WHERE last_human_activity_at <= ?
             ORDER BY last_human_activity_at, card_id
             LIMIT ?
            """,
            (utc_iso(inactive_since), int(limit)),
        )
        return [_card_job_from_row(row) for row in await cursor.fetchall()]

    async def reserve_callback(
        self,
        *,
        result_id: str,
        job_id: str,
        now: Timestamp | None = None,
        stale_before: Timestamp | None = None,
    ) -> CallbackReservation:
        """Reserve a callback key without persisting any callback body.

        Unknown jobs are reported without writing a key.  A released reservation is
        deleted, so the sender's retry can reserve it afresh.  Optionally, callers may
        take over a reservation older than ``stale_before``.
        """

        if not result_id or not job_id:
            raise ValueError("result_id and job_id must not be empty")
        timestamp = utc_iso(now)
        stale_timestamp = utc_iso(stale_before) if stale_before is not None else None

        async with self._write_lock:
            connection = self.connection
            await connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = await connection.execute(
                    "SELECT card_id FROM card_jobs WHERE job_id = ?", (job_id,)
                )
                job = await cursor.fetchone()
                if job is None:
                    await connection.commit()
                    return CallbackReservation(CallbackOutcome.UNKNOWN_JOB, result_id, job_id, None)
                card_id = int(job["card_id"])

                cursor = await connection.execute(
                    """
                    SELECT result_id, job_id, status, reserved_at
                      FROM callback_keys
                     WHERE result_id = ?
                    """,
                    (result_id,),
                )
                existing = await cursor.fetchone()
                if existing is None:
                    await connection.execute(
                        """
                        INSERT INTO callback_keys (
                            result_id, job_id, status, reserved_at, updated_at
                        ) VALUES (?, ?, 'reserved', ?, ?)
                        """,
                        (result_id, job_id, timestamp, timestamp),
                    )
                    outcome = CallbackOutcome.RESERVED
                elif existing["job_id"] != job_id:
                    outcome = CallbackOutcome.CONFLICT
                elif existing["status"] == CallbackStatus.SUCCESS.value:
                    outcome = CallbackOutcome.DUPLICATE
                elif stale_timestamp is not None and existing["reserved_at"] <= stale_timestamp:
                    await connection.execute(
                        """
                        UPDATE callback_keys
                           SET reserved_at = ?, updated_at = ?
                         WHERE result_id = ? AND status = 'reserved'
                        """,
                        (timestamp, timestamp, result_id),
                    )
                    outcome = CallbackOutcome.RESERVED
                else:
                    outcome = CallbackOutcome.BUSY
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        return CallbackReservation(outcome, result_id, job_id, card_id)

    async def mark_callback_success(self, result_id: str, *, now: Timestamp | None = None) -> bool:
        timestamp = utc_iso(now)
        async with self._write_lock:
            cursor = await self.connection.execute(
                """
                UPDATE callback_keys
                   SET status = 'success', success_at = ?, updated_at = ?
                 WHERE result_id = ? AND status = 'reserved'
                """,
                (timestamp, timestamp, result_id),
            )
            if cursor.rowcount == 1:
                return True
            cursor = await self.connection.execute(
                "SELECT status FROM callback_keys WHERE result_id = ?", (result_id,)
            )
            row = await cursor.fetchone()
        return row is not None and row["status"] == CallbackStatus.SUCCESS.value

    complete_callback = mark_callback_success

    async def release_callback(self, result_id: str) -> bool:
        async with self._write_lock:
            cursor = await self.connection.execute(
                "DELETE FROM callback_keys WHERE result_id = ? AND status = 'reserved'",
                (result_id,),
            )
        return cursor.rowcount == 1

    async def set_service_status(
        self,
        key: str,
        value: str,
        *,
        now: Timestamp | None = None,
    ) -> ServiceStatusRecord:
        if not key:
            raise ValueError("status key must not be empty")
        if not isinstance(value, str):
            raise TypeError("service status values must be strings")
        timestamp = utc_iso(now)
        async with self._write_lock:
            await self.connection.execute(
                """
                INSERT INTO service_status (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (key, value, timestamp),
            )
        return ServiceStatusRecord(key, value, timestamp)

    async def get_service_status(self, key: str) -> ServiceStatusRecord | None:
        cursor = await self.connection.execute(
            "SELECT key, value, updated_at FROM service_status WHERE key = ?", (key,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return ServiceStatusRecord(row["key"], row["value"], row["updated_at"])

    async def list_service_status(self) -> dict[str, ServiceStatusRecord]:
        cursor = await self.connection.execute(
            "SELECT key, value, updated_at FROM service_status ORDER BY key"
        )
        return {
            row["key"]: ServiceStatusRecord(
                key=row["key"], value=row["value"], updated_at=row["updated_at"]
            )
            for row in await cursor.fetchall()
        }

    async def queue_counts(self) -> dict[str, int]:
        cursor = await self.connection.execute(
            "SELECT status, COUNT(*) AS count FROM events GROUP BY status"
        )
        counts = {status.value: 0 for status in EventStatus}
        for row in await cursor.fetchall():
            counts[row["status"]] = int(row["count"])
        counts["queue_size"] = sum(counts[item] for item in BLOCKING_EVENT_STATUSES)
        counts["waiting"] = counts[EventStatus.PENDING.value] + counts[EventStatus.RETRY.value]
        counts["errors"] = counts[EventStatus.TECHNICAL_ERROR.value]
        counts["total"] = sum(
            count for key, count in counts.items() if key in {item.value for item in EventStatus}
        )
        return counts

    async def health_snapshot(self) -> dict[str, Any]:
        statuses = await self.list_service_status()
        return {
            "queue": await self.queue_counts(),
            "service": {
                key: {"value": record.value, "updated_at": record.updated_at}
                for key, record in statuses.items()
            },
        }

    async def cleanup_retention(
        self,
        *,
        event_before: Timestamp,
        callback_before: Timestamp | None = None,
    ) -> CleanupResult:
        """Delete retained terminal keys older than their configured cutoffs."""

        event_cutoff = utc_iso(event_before)
        callback_cutoff = utc_iso(callback_before or event_before)
        async with self._write_lock:
            connection = self.connection
            await connection.execute("BEGIN IMMEDIATE")
            try:
                event_cursor = await connection.execute(
                    """
                    DELETE FROM events
                     WHERE (status = 'success' AND success_at < ?)
                        OR (status = 'ignored' AND updated_at < ?)
                    """,
                    (event_cutoff, event_cutoff),
                )
                callback_cursor = await connection.execute(
                    """
                    DELETE FROM callback_keys
                     WHERE status = 'success' AND success_at < ?
                    """,
                    (callback_cutoff,),
                )
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        return CleanupResult(event_cursor.rowcount, callback_cursor.rowcount)

    cleanup = cleanup_retention

    async def delete_card_state(self, card_id: int) -> CardStateDeletion:
        """Remove all technical state when a card reaches its 30-day expiry."""

        async with self._write_lock:
            connection = self.connection
            await connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = await connection.execute(
                    "SELECT job_id FROM card_jobs WHERE card_id = ?", (card_id,)
                )
                jobs = list(await cursor.fetchall())
                callback_count = 0
                if jobs and jobs[0]["job_id"] is not None:
                    callback_cursor = await connection.execute(
                        "DELETE FROM callback_keys WHERE job_id = ?", (jobs[0]["job_id"],)
                    )
                    callback_count = callback_cursor.rowcount
                event_cursor = await connection.execute(
                    "DELETE FROM events WHERE card_id = ?", (card_id,)
                )
                job_cursor = await connection.execute(
                    "DELETE FROM card_jobs WHERE card_id = ?", (card_id,)
                )
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        return CardStateDeletion(
            events_deleted=event_cursor.rowcount,
            callbacks_deleted=callback_count,
            jobs_deleted=job_cursor.rowcount,
        )


Storage = SQLiteStorage


def _event_from_row(row: Mapping[str, Any]) -> EventRecord:
    return EventRecord(
        id=int(row["id"]),
        event_type=row["event_type"],
        card_id=int(row["card_id"]),
        comment_id=int(row["comment_id"]) if row["comment_id"] is not None else None,
        event_key=row["event_key"],
        status=EventStatus(row["status"]),
        error_code=row["error_code"],
        attempts=int(row["attempts"]),
        next_attempt_at=row["next_attempt_at"],
        notification_sent=bool(row["notification_sent"]),
        source_created_at=row["source_created_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        claimed_at=row["claimed_at"],
        success_at=row["success_at"],
    )


def _optional_event(row: Mapping[str, Any] | None) -> EventRecord | None:
    return _event_from_row(row) if row is not None else None


def _card_job_from_row(row: Mapping[str, Any]) -> CardJobRecord:
    return CardJobRecord(
        card_id=int(row["card_id"]),
        job_id=row["job_id"],
        last_human_activity_at=row["last_human_activity_at"],
        warning_sent=bool(row["warning_sent"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


__all__ = [
    "BLOCKING_EVENT_STATUSES",
    "TERMINAL_EVENT_STATUSES",
    "CallbackOutcome",
    "CallbackReservation",
    "CallbackStatus",
    "CardJobConflict",
    "CardJobRecord",
    "CardStateDeletion",
    "CleanupResult",
    "EnqueueResult",
    "EventRecord",
    "EventStatus",
    "InvalidTransition",
    "SQLiteStorage",
    "ServiceStatusRecord",
    "Storage",
    "StorageError",
    "StorageNotInitialized",
    "utc_iso",
    "utc_now",
]
