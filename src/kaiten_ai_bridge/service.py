"""Application orchestration for durable events, callbacks, reconciliation and cleanup."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .config import Settings
from .domain import (
    AttachmentRef,
    PreparedFile,
    build_comment_prompt,
    build_initial_prompt,
    build_outgoing_payload,
    deduplicate_prepared,
    safe_json,
    validate_file_refs,
    validate_prepared_files,
)
from .errors import BridgeError, DataRejectedError, UnsupportedResultFilesError
from .kaiten import KaitenClient
from .receiver import ReceiverClient
from .schemas import CallbackRequest, Trigger
from .storage import (
    CallbackOutcome,
    EventRecord,
    EventStatus,
    SQLiteStorage,
    utc_iso,
)

TECHNICAL_ERROR_MESSAGE = (
    "Не удалось обработать заявку из-за временной технической ошибки.\n"
    "Повторная проверка будет выполнена автоматически."
)

_ALL_SQLITE_ROWS = 2**63 - 1

_IGNORED_REJECTIONS = {
    "card_outside_space",
    "card_outside_target",
    "card_outside_service",
    "card_not_active",
    "card_closed",
    "internal_comment",
    "technical_comment",
    "deleted_comment",
    "initial_service_comment",
}


def _parse_time(value: str | datetime | None, *, fallback: datetime | None = None) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    elif fallback is not None:
        parsed = fallback
    else:
        parsed = datetime.now(UTC)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _comment_author_id(comment: Mapping[str, Any]) -> int | None:
    direct = _positive_int(comment.get("author_id"))
    if direct is not None:
        return direct
    author = comment.get("author")
    if isinstance(author, Mapping):
        return _positive_int(author.get("id"))
    return None


def _inactivity_warning_message(hours: int) -> str:
    last_two = hours % 100
    last = hours % 10
    if last == 1 and last_two != 11:
        unit = "час"
    elif last in {2, 3, 4} and not 12 <= last_two <= 14:
        unit = "часа"
    else:
        unit = "часов"
    return (
        f"Эта заявка будет автоматически завершена через {hours} {unit} "
        "из-за отсутствия активности.\n"
        "Для новой работы с ИИ после завершения создайте новую заявку в Service Desk."
    )


INACTIVITY_WARNING_MESSAGE = _inactivity_warning_message(24)


@dataclass(frozen=True, slots=True)
class CallbackHandlingResult:
    outcome: str
    card_id: int | None = None


class CallbackBusyError(RuntimeError):
    pass


class CallbackUnknownJobError(RuntimeError):
    pass


class CallbackConflictError(RuntimeError):
    pass


class InvalidCallbackError(RuntimeError):
    pass


class _CommentPredecessorRestored(RuntimeError):
    """Internal control flow: an earlier remote comment was durably queued."""


class BridgeService:
    def __init__(
        self,
        settings: Settings,
        storage: SQLiteStorage,
        kaiten: KaitenClient,
        receiver: ReceiverClient,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.kaiten = kaiten
        self.receiver = receiver
        self.log = logging.getLogger("kaiten_ai_bridge.service")
        self._stop = asyncio.Event()
        self._startup_ready = asyncio.Event()
        self._tasks: list[asyncio.Task[None]] = []
        self._reconcile_lock = asyncio.Lock()
        self._cleanup_lock = asyncio.Lock()

    async def start(self) -> None:
        self._stop.clear()
        self._startup_ready.clear()
        self._tasks = [
            asyncio.create_task(self._startup_maintenance(), name="bridge-startup"),
            asyncio.create_task(self._worker_loop(), name="bridge-worker"),
            asyncio.create_task(self._reconciliation_loop(), name="bridge-reconciliation"),
            asyncio.create_task(self._cleanup_loop(), name="bridge-cleanup"),
        ]

    async def stop(self) -> None:
        self._stop.set()
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def _startup_maintenance(self) -> None:
        """Finish recovery checks before persistent events may be claimed.

        Reconciliation must see the pre-downtime activity boundary before a newer
        queued comment can advance it.  Cleanup must also get the first chance to
        close expired cards.  Either maintenance action may fail independently;
        a transient dependency failure must not leave the worker blocked forever.
        """

        try:
            if self.settings.reconcile_on_startup:
                try:
                    await self.reconcile_once()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    self.log.exception("startup_reconciliation_failed")
            if self.settings.cleanup_on_startup:
                try:
                    await self.cleanup_once()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    self.log.exception("startup_cleanup_failed")
        finally:
            self._startup_ready.set()

    async def enqueue_trigger(self, trigger: Trigger) -> tuple[EventRecord, bool]:
        result = await self.storage.enqueue_event(
            event_type=trigger.event_type,
            card_id=trigger.card_id,
            comment_id=trigger.comment_id,
            event_key=trigger.event_key,
            source_created_at=trigger.source_created_at,
        )
        await self.storage.set_service_status("last_webhook", "accepted")
        self.log.info(
            "webhook_enqueued",
            extra={
                "event_type": trigger.event_type,
                "card_id": trigger.card_id,
                "comment_id": trigger.comment_id,
                "event_id": result.event.id,
                "duplicate": not result.inserted,
            },
        )
        return result.event, result.inserted

    async def _worker_loop(self) -> None:
        await self._startup_ready.wait()
        while not self._stop.is_set():
            try:
                processed = await self.run_due_batch()
            except asyncio.CancelledError:
                raise
            except Exception:
                self.log.exception("worker_batch_failed")
                processed = 0
            if processed == 0:
                try:
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=self.settings.worker_poll_seconds
                    )
                except TimeoutError:
                    pass

    async def run_due_batch(self) -> int:
        events = await self.storage.claim_due_events(limit=self.settings.card_parallelism)
        if not events:
            return 0
        await asyncio.gather(*(self._process_event(event) for event in events))
        return len(events)

    async def _process_event(self, event: EventRecord) -> None:
        started = asyncio.get_running_loop().time()
        try:
            if event.event_type == "card:add":
                await self._process_initial(event)
            elif event.event_type == "comment:add" and event.comment_id is not None:
                await self._process_comment(event)
            else:
                await self.storage.mark_ignored(event.id, error_code="unsupported_event")
                return
            await self.storage.mark_success(event.id)
            self.log.info(
                "event_succeeded",
                extra={
                    "event_id": event.id,
                    "event_type": event.event_type,
                    "card_id": event.card_id,
                    "comment_id": event.comment_id,
                    "attempt": event.attempts,
                    "duration_ms": int((asyncio.get_running_loop().time() - started) * 1000),
                },
            )
        except DataRejectedError as exc:
            await self._handle_data_rejection(event, exc)
        except _CommentPredecessorRestored:
            await self.storage.mark_retry(
                event.id,
                next_attempt_at=datetime.now(UTC),
                error_code="comment_predecessor_restored",
            )
        except asyncio.CancelledError:
            raise
        except BridgeError as exc:
            await self._schedule_technical_failure(event, exc.code)
        except Exception:
            self.log.exception(
                "event_unexpected_failure",
                extra={"event_id": event.id, "card_id": event.card_id},
            )
            await self._schedule_technical_failure(event, "unexpected_error")

    async def _prepare_files(self, references: Sequence[AttachmentRef]) -> list[PreparedFile]:
        validate_file_refs(
            references,
            max_bytes=self.settings.max_file_bytes,
            allowed_extensions=self.settings.allowed_extensions,
        )
        prepared: list[PreparedFile] = []
        for reference in references:
            async with self.kaiten.downloaded_file(
                reference, max_bytes=self.settings.max_file_bytes
            ) as item:
                prepared.append(item)
        validate_prepared_files(
            prepared,
            max_bytes=self.settings.max_file_bytes,
            allowed_extensions=self.settings.allowed_extensions,
        )
        return deduplicate_prepared(prepared)

    async def _process_initial(
        self, event: EventRecord, *, correction_comment_id: int | None = None
    ) -> None:
        state = await self.storage.get_card_job(event.card_id)
        if state is not None and state.job_id:
            card = await self.kaiten.get_card(event.card_id)
            column_id = _positive_int(card.get("column_id"))
            if column_id == self.settings.kaiten_queue_column_id:
                await self.kaiten.move_card(event.card_id, self.settings.kaiten_work_column_id)
            return

        snapshot = await self.kaiten.card_snapshot(
            event.card_id, correction_comment_id=correction_comment_id
        )
        if snapshot.column_id != self.settings.kaiten_queue_column_id:
            raise DataRejectedError("card_not_active", "Карточка не находится в колонке запуска.")
        await self.storage.ensure_card_state(
            card_id=event.card_id,
            last_human_activity_at=event.source_created_at or event.created_at,
        )
        prepared = await self._prepare_files(snapshot.attachments)
        prompt = build_initial_prompt(snapshot, [item.reference for item in prepared])
        payload = build_outgoing_payload(
            schema_version=self.settings.schema_version,
            event_type="initial",
            idempotency_key=f"card:add:{event.card_id}",
            card_id=event.card_id,
            card_url=snapshot.url,
            request_type=snapshot.request_type,
            prompt=prompt,
            files=prepared,
            created_at=_parse_time(event.source_created_at or event.created_at),
        )
        job_id = await self.receiver.submit(payload)
        await self.storage.ensure_card_state(
            card_id=event.card_id,
            last_human_activity_at=event.source_created_at or event.created_at,
        )
        await self.storage.bind_job(card_id=event.card_id, job_id=job_id)
        await self.kaiten.move_card(event.card_id, self.settings.kaiten_work_column_id)

    async def _process_comment(self, event: EventRecord) -> None:
        assert event.comment_id is not None
        state = await self.storage.get_card_job(event.card_id)
        if state is None:
            # A lost card:add must be restored ahead of this already-claimed comment.
            card = await self.kaiten.get_card(event.card_id)
            if (
                "space_id" in card
                and _positive_int(card.get("space_id")) != self.settings.kaiten_space_id
            ):
                raise DataRejectedError(
                    "card_outside_space",
                    "Карточка не относится к настроенному пространству обработки.",
                )
            if _positive_int(card.get("board_id")) != self.settings.kaiten_board_id:
                raise DataRejectedError(
                    "card_outside_target",
                    "Карточка не относится к настроенной доске обработки.",
                )
            if (
                self.settings.kaiten_service_id is not None
                and _positive_int(card.get("service_id")) != self.settings.kaiten_service_id
            ):
                raise DataRejectedError(
                    "card_outside_service",
                    "Карточка не относится к настроенному сервису.",
                )
            column_id = _positive_int(card.get("column_id"))
            if column_id == self.settings.kaiten_done_column_id:
                raise DataRejectedError("card_closed", "Завершённая карточка не обрабатывается.")
            if column_id != self.settings.kaiten_queue_column_id:
                raise DataRejectedError(
                    "card_not_active", "Карточка не находится в колонке запуска."
                )
            created = card.get("created") or event.created_at
            await self.storage.enqueue_event(
                event_type="card:add",
                card_id=event.card_id,
                event_key=f"card:add:{event.card_id}",
                source_created_at=created,
            )
            raise BridgeError("initial_event_required")

        if await self._restore_missing_comment_predecessors(event, state.last_human_activity_at):
            raise _CommentPredecessorRestored()
        if not state.job_id:
            await self._process_initial(event, correction_comment_id=event.comment_id)
            return

        snapshot, request_type = await self.kaiten.comment_snapshot(event.card_id, event.comment_id)
        await self.storage.touch_human_activity(event.card_id, activity_at=snapshot.created_at)
        prepared = await self._prepare_files(snapshot.attachments)
        prompt = build_comment_prompt(snapshot, [item.reference for item in prepared])
        payload = build_outgoing_payload(
            schema_version=self.settings.schema_version,
            event_type="comment",
            idempotency_key=event.event_key,
            card_id=event.card_id,
            card_url=snapshot.card_url,
            request_type=request_type,
            prompt=prompt,
            files=prepared,
            comment_id=event.comment_id,
            ai_job_id=state.job_id,
            created_at=snapshot.created_at,
        )
        accepted_job_id = await self.receiver.submit(payload)
        if accepted_job_id != state.job_id:
            raise BridgeError("receiver_job_id_mismatch")

    async def _restore_missing_comment_predecessors(
        self,
        event: EventRecord,
        last_human_activity_at: str,
    ) -> bool:
        """Queue eligible remote comments that precede the claimed webhook event.

        Webhooks can be delivered out of order. The durable queue can order only
        events it already knows, so a claimed later comment performs this narrow
        read-through before it is allowed to reach the receiver.
        """

        assert event.comment_id is not None
        comments = await self.kaiten.list_comments(event.card_id)
        target = next(
            (
                comment
                for comment in comments
                if _positive_int(comment.get("id")) == event.comment_id
            ),
            None,
        )
        if target is None:
            return False
        target_raw_created = (
            target.get("created")
            or target.get("created_at")
            or target.get("updated")
            or target.get("update")
            or event.source_created_at
        )
        try:
            target_order = (_parse_time(target_raw_created), event.comment_id)
        except (TypeError, ValueError):
            return False

        technical_authors = set(self.settings.kaiten_technical_author_ids)
        current_user = await self.kaiten.current_user_id()
        if current_user:
            technical_authors.add(current_user)
        known_keys = {item.event_key for item in await self.storage.list_card_events(event.card_id)}
        last_activity = _parse_time(last_human_activity_at)
        recent_cutoff = datetime.now(UTC) - timedelta(days=self.settings.dedup_retention_days)
        inserted = False
        ordered: list[tuple[datetime, int, Mapping[str, Any]]] = []
        for comment in comments:
            comment_id = _positive_int(comment.get("id"))
            if comment_id is None:
                continue
            raw_created = (
                comment.get("created")
                or comment.get("created_at")
                or comment.get("updated")
                or comment.get("update")
            )
            try:
                created = _parse_time(raw_created)
            except (TypeError, ValueError):
                continue
            ordered.append((created, comment_id, comment))

        for created, comment_id, comment in sorted(ordered, key=lambda item: item[:2]):
            if (created, comment_id) >= target_order:
                break
            if (
                comment.get("deleted") is True
                or comment.get("internal") is True
                or comment.get("sd_description") is True
                or _comment_author_id(comment) in technical_authors
            ):
                continue
            key = f"comment:add:{comment_id}"
            if key in known_keys:
                continue
            # Successful keys are intentionally retained only three days. Avoid
            # recreating old history after that cleanup boundary while still
            # restoring a genuinely missing recent predecessor.
            if created <= last_activity and created < recent_cutoff:
                continue
            result = await self.storage.enqueue_event(
                event_type="comment:add",
                card_id=event.card_id,
                comment_id=comment_id,
                event_key=key,
                source_created_at=created,
            )
            known_keys.add(key)
            inserted = inserted or result.inserted
        return inserted

    async def _handle_data_rejection(
        self, event: EventRecord, rejection: DataRejectedError
    ) -> None:
        if rejection.code in _IGNORED_REJECTIONS:
            await self.storage.mark_ignored(event.id, error_code=rejection.code)
            self.log.info(
                "event_ignored",
                extra={"event_id": event.id, "card_id": event.card_id, "code": rejection.code},
            )
            return

        activity_at = event.source_created_at or event.created_at
        await self.storage.ensure_card_state(
            card_id=event.card_id, last_human_activity_at=activity_at
        )
        await self.storage.touch_human_activity(event.card_id, activity_at=activity_at)
        try:
            if not event.notification_sent:
                await self.kaiten.add_public_comment(
                    event.card_id, self._fit_user_message(rejection.user_message)
                )
                await self.storage.mark_notification_sent(event.id)
        except BridgeError as exc:
            await self._schedule_technical_failure(event, f"notify_{rejection.code}_{exc.code}")
            return
        await self.storage.mark_rejected(event.id, error_code=rejection.code)
        self.log.info(
            "event_rejected",
            extra={"event_id": event.id, "card_id": event.card_id, "code": rejection.code},
        )

    @staticmethod
    def _fit_user_message(message: str) -> str:
        if len(message) <= 4096:
            return message
        return message[:4080].rstrip() + "…"

    async def _schedule_technical_failure(self, event: EventRecord, code: str) -> None:
        safe_code = code[:128]
        if event.attempts < len(self.settings.retry_schedule_seconds):
            created = _parse_time(event.created_at)
            target = created + timedelta(
                seconds=self.settings.retry_schedule_seconds[event.attempts]
            )
            now = datetime.now(UTC)
            await self.storage.mark_retry(
                event.id,
                next_attempt_at=max(target, now),
                error_code=safe_code,
            )
            self.log.warning(
                "event_retry_scheduled",
                extra={
                    "event_id": event.id,
                    "card_id": event.card_id,
                    "attempt": event.attempts,
                    "code": safe_code,
                    "next_attempt_at": utc_iso(max(target, now)),
                },
            )
            return

        try:
            if not event.notification_sent:
                await self.kaiten.add_public_comment(event.card_id, TECHNICAL_ERROR_MESSAGE)
                await self.storage.mark_notification_sent(event.id)
        except BridgeError:
            pass
        await self.storage.mark_error(event.id, error_code=safe_code)
        self.log.error(
            "event_attempts_exhausted",
            extra={
                "event_id": event.id,
                "card_id": event.card_id,
                "attempt": event.attempts,
                "code": safe_code,
            },
        )

    async def handle_callback(self, callback: CallbackRequest) -> CallbackHandlingResult:
        if callback.schema_version != self.settings.schema_version:
            raise InvalidCallbackError("unsupported_schema_version")
        if callback.status not in set(self.settings.callback_success_statuses):
            raise InvalidCallbackError("unsupported_callback_status")
        if callback.has_result_files():
            raise UnsupportedResultFilesError()
        text = self._callback_text(callback)
        if not text.strip():
            raise InvalidCallbackError("empty_callback_result")
        if len(text) > 4096:
            raise InvalidCallbackError("callback_text_exceeds_kaiten_limit")

        now = datetime.now(UTC)
        reservation = await self.storage.reserve_callback(
            result_id=callback.result_id,
            job_id=callback.job_id,
            stale_before=now
            - timedelta(seconds=self.settings.callback_reservation_timeout_seconds),
        )
        if reservation.outcome is CallbackOutcome.DUPLICATE:
            return CallbackHandlingResult("duplicate", reservation.card_id)
        if reservation.outcome is CallbackOutcome.BUSY:
            raise CallbackBusyError()
        if reservation.outcome is CallbackOutcome.UNKNOWN_JOB:
            raise CallbackUnknownJobError()
        if reservation.outcome is CallbackOutcome.CONFLICT:
            raise CallbackConflictError()
        assert reservation.card_id is not None
        try:
            await self.kaiten.add_public_comment(reservation.card_id, text)
        except Exception:
            await self.storage.release_callback(callback.result_id)
            raise
        await self.storage.mark_callback_success(callback.result_id)
        self.log.info(
            "callback_published",
            extra={"result_id": callback.result_id, "card_id": reservation.card_id},
        )
        return CallbackHandlingResult("published", reservation.card_id)

    @staticmethod
    def _callback_text(callback: CallbackRequest) -> str:
        text = callback.result_text()
        if text is not None:
            return text
        result = callback.result
        if hasattr(result, "model_dump"):
            value = result.model_dump(exclude={"files"}, exclude_none=True)
        elif isinstance(result, Mapping):
            value = {key: item for key, item in result.items() if key != "files"}
        else:
            value = result
        if value in (None, {}, []):
            return ""
        return f"```json\n{safe_json(value)}\n```"

    async def _reconciliation_loop(self) -> None:
        await self._startup_ready.wait()
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=self.settings.reconciliation_interval_seconds,
                )
            except TimeoutError:
                with suppress(Exception):
                    await self.reconcile_once()

    async def reconcile_once(self) -> None:
        if self._reconcile_lock.locked():
            return
        async with self._reconcile_lock:
            started = datetime.now(UTC)
            await self.storage.set_service_status("last_reconciliation_result", "running")
            try:
                cards = await self.kaiten.list_target_cards()
                technical_authors = set(self.settings.kaiten_technical_author_ids)
                current_user = await self.kaiten.current_user_id()
                if current_user:
                    technical_authors.add(current_user)
                recent_cutoff = started - timedelta(days=self.settings.dedup_retention_days)
                for card in cards:
                    card_id = _positive_int(card.get("id"))
                    column_id = _positive_int(card.get("column_id"))
                    if not card_id:
                        continue
                    state = await self.storage.get_card_job(card_id)
                    events = await self.storage.list_card_events(card_id)
                    by_key = {event.event_key: event for event in events}

                    for event in events:
                        if event.status is not EventStatus.TECHNICAL_ERROR:
                            continue
                        if not event.notification_sent:
                            try:
                                await self.kaiten.add_public_comment(
                                    card_id, TECHNICAL_ERROR_MESSAGE
                                )
                                await self.storage.mark_notification_sent(event.id)
                            except BridgeError:
                                pass
                        await self.storage.requeue_error(event.id)

                    if column_id == self.settings.kaiten_queue_column_id:
                        if state is not None and state.job_id:
                            await self.kaiten.move_card(
                                card_id, self.settings.kaiten_work_column_id
                            )
                        elif state is None:
                            await self.storage.enqueue_event(
                                event_type="card:add",
                                card_id=card_id,
                                event_key=f"card:add:{card_id}",
                                source_created_at=card.get("created"),
                            )

                    if state is None:
                        if column_id != self.settings.kaiten_queue_column_id:
                            continue
                        last_activity = _parse_time(card.get("created"))
                    else:
                        last_activity = _parse_time(state.last_human_activity_at)
                    comments = await self.kaiten.list_comments(card_id)
                    ordered = sorted(
                        comments,
                        key=lambda item: str(
                            item.get("created")
                            or item.get("created_at")
                            or item.get("updated")
                            or item.get("update")
                            or ""
                        ),
                    )
                    for comment in ordered:
                        comment_id = _positive_int(comment.get("id"))
                        author_id = _comment_author_id(comment)
                        if (
                            not comment_id
                            or comment.get("deleted") is True
                            or comment.get("internal") is True
                            or comment.get("sd_description") is True
                            or author_id in technical_authors
                        ):
                            continue
                        raw_created = (
                            comment.get("created")
                            or comment.get("created_at")
                            or comment.get("updated")
                            or comment.get("update")
                        )
                        try:
                            created = _parse_time(raw_created)
                        except (TypeError, ValueError):
                            continue
                        key = f"comment:add:{comment_id}"
                        if key in by_key:
                            continue
                        if created <= last_activity and created < recent_cutoff:
                            continue
                        await self.storage.enqueue_event(
                            event_type="comment:add",
                            card_id=card_id,
                            comment_id=comment_id,
                            event_key=key,
                            source_created_at=created,
                        )
                await self.storage.set_service_status("last_reconciliation_result", "ok")
                self.log.info(
                    "reconciliation_finished",
                    extra={
                        "cards_seen": len(cards),
                        "duration_ms": int((datetime.now(UTC) - started).total_seconds() * 1000),
                    },
                )
            except Exception:
                await self.storage.set_service_status("last_reconciliation_result", "error")
                self.log.exception("reconciliation_failed")
                raise

    async def _cleanup_loop(self) -> None:
        await self._startup_ready.wait()
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self.settings.cleanup_interval_seconds
                )
            except TimeoutError:
                with suppress(Exception):
                    await self.cleanup_once()

    async def cleanup_once(self) -> None:
        if self._cleanup_lock.locked():
            return
        async with self._cleanup_lock:
            now = datetime.now(UTC)
            expiry_cutoff = now - timedelta(days=self.settings.activity_retention_days)
            warning_age = timedelta(days=self.settings.activity_retention_days) - timedelta(
                hours=self.settings.warning_before_close_hours
            )
            warning_cutoff = now - warning_age

            warning_states = await self.storage.list_jobs_due_warning(
                inactive_since=warning_cutoff, limit=_ALL_SQLITE_ROWS
            )
            warning_message = _inactivity_warning_message(self.settings.warning_before_close_hours)
            for state in warning_states:
                if _parse_time(state.last_human_activity_at) <= expiry_cutoff:
                    continue
                try:
                    await self.kaiten.add_public_comment(state.card_id, warning_message)
                except BridgeError:
                    continue
                await self.storage.mark_warning_sent(state.card_id)

            expired_states = await self.storage.list_expired_jobs(
                inactive_since=expiry_cutoff, limit=_ALL_SQLITE_ROWS
            )
            for state in expired_states:
                try:
                    card = await self.kaiten.get_card(state.card_id)
                    if _positive_int(card.get("column_id")) != self.settings.kaiten_done_column_id:
                        await self.kaiten.move_card(
                            state.card_id, self.settings.kaiten_done_column_id
                        )
                except BridgeError:
                    continue
                await self.storage.delete_card_state(state.card_id)

            retention_cutoff = now - timedelta(days=self.settings.dedup_retention_days)
            result = await self.storage.cleanup_retention(
                event_before=retention_cutoff, callback_before=retention_cutoff
            )
            temp_deleted = self._delete_old_files(
                self.settings.temp_dir,
                older_than=now - timedelta(days=self.settings.temp_retention_days),
                prefixes=("bridge-",),
            )
            log_files_deleted = self._delete_old_files(
                self.settings.log_dir,
                older_than=now - timedelta(days=self.settings.log_retention_days),
                prefixes=("bridge.log.",),
            )
            await self.storage.set_service_status("last_cleanup", "ok")
            self.log.info(
                "cleanup_finished",
                extra={
                    "events_deleted": result.events_deleted,
                    "callbacks_deleted": result.callbacks_deleted,
                    "temp_files_deleted": temp_deleted,
                    "log_files_deleted": log_files_deleted,
                },
            )

    @staticmethod
    def _delete_old_files(
        directory: Path, *, older_than: datetime, prefixes: tuple[str, ...]
    ) -> int:
        if not directory.exists():
            return 0
        root = directory.resolve()
        deleted = 0
        for path in directory.iterdir():
            try:
                resolved = path.resolve()
                if resolved.parent != root or not resolved.is_file():
                    continue
                if not resolved.name.startswith(prefixes):
                    continue
                modified = datetime.fromtimestamp(resolved.stat().st_mtime, tz=UTC)
                if modified < older_than:
                    resolved.unlink()
                    deleted += 1
            except OSError:
                continue
        return deleted

    async def health(self) -> dict[str, Any]:
        async def bounded(check: Any) -> bool:
            try:
                return bool(
                    await asyncio.wait_for(
                        check(), timeout=self.settings.health_probe_timeout_seconds
                    )
                )
            except Exception:
                return False

        kaiten_ok, receiver_ok = await asyncio.gather(
            bounded(self.kaiten.healthcheck), bounded(self.receiver.healthcheck)
        )
        snapshot = await self.storage.health_snapshot()
        service = snapshot["service"]
        queue = snapshot["queue"]

        def updated(key: str) -> str | None:
            item = service.get(key)
            return item.get("updated_at") if item else None

        return {
            "status": "ok" if kaiten_ok and receiver_ok else "degraded",
            "process": "running",
            "kaiten_api": "available" if kaiten_ok else "unavailable",
            "receiver": "available" if receiver_ok else "unavailable",
            "last_webhook_at": updated("last_webhook"),
            "last_reconciliation_at": updated("last_reconciliation_result"),
            "last_reconciliation_result": (
                service.get("last_reconciliation_result", {}).get("value")
            ),
            "pending_events": queue["waiting"],
            "error_events": queue["errors"],
            "queue_size": queue["queue_size"],
            "last_cleanup_at": updated("last_cleanup"),
        }
