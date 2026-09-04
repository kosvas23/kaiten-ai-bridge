from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from kaiten_ai_bridge.config import Settings
from kaiten_ai_bridge.domain import AttachmentRef, CardSnapshot, PreparedFile
from kaiten_ai_bridge.probe import export_initial_payload


def configured(tmp_path: Path) -> Settings:
    return Settings(
        environment="test",
        temp_dir=tmp_path / "tmp",
        kaiten_base_url="https://tagat.example.test",
        kaiten_token="kaiten-secret",
        kaiten_space_id=834_556,
        kaiten_board_id=1_868_751,
        kaiten_queue_column_id=6_467_479,
        kaiten_work_column_id=6_467_480,
        kaiten_done_column_id=6_467_481,
        ai_bearer_token="unused",
        allowed_extensions=[".xlsx"],
    )


class FakeReadOnlyKaiten:
    def __init__(self, snapshot: CardSnapshot, content: bytes) -> None:
        self.snapshot = snapshot
        self.content = content
        self.snapshot_calls: list[int] = []
        self.download_calls: list[str | None] = []

    async def card_snapshot(self, card_id: int) -> CardSnapshot:
        self.snapshot_calls.append(card_id)
        return self.snapshot

    @asynccontextmanager
    async def downloaded_file(
        self,
        reference: AttachmentRef,
        *,
        max_bytes: int,
    ) -> AsyncIterator[PreparedFile]:
        assert len(self.content) <= max_bytes
        self.download_calls.append(reference.file_id)
        yield PreparedFile(reference=reference, content=self.content)


@pytest.mark.asyncio
async def test_probe_writes_exact_initial_json_without_receiver_or_mutation(tmp_path: Path) -> None:
    reference = AttachmentRef(
        file_id="file-1",
        name="example.xlsx",
        size=4,
        mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        download_url="https://files.example.test/example.xlsx",
        source="кастомное поле «Документация»",
    )
    snapshot = CardSnapshot(
        card_id=17,
        url="https://tagat.example.test/space/834556/boards/card/17",
        title="Тестовая заявка",
        request_type="Новый тип",
        space="ИИ",
        board="Доска агента",
        column="Очередь",
        column_id=6_467_479,
        author="Иван Иванов",
        initial_comments=("Нужен расчёт",),
        attachments=(reference,),
    )
    client = FakeReadOnlyKaiten(snapshot, b"xlsx")
    output = tmp_path / "probe" / "card-17.json"

    summary = await export_initial_payload(
        configured(tmp_path),
        card_id=17,
        output_path=output,
        kaiten=client,  # type: ignore[arg-type]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["event_type"] == "initial"
    assert payload["idempotency_key"] == "card:add:17"
    assert payload["comment_id"] is None and payload["ai_job_id"] is None
    assert payload["request_type"] == "Новый тип"
    assert "Нужен расчёт" in payload["prompt"]
    assert base64.b64decode(payload["files"][0]["content_base64"]) == b"xlsx"
    assert client.snapshot_calls == [17]
    assert client.download_calls == ["file-1"]
    assert summary.card_id == 17
    assert summary.file_count == 1
    assert summary.source_file_bytes == 4
    assert summary.json_bytes == output.stat().st_size
    assert len(summary.output_sha256) == 64
