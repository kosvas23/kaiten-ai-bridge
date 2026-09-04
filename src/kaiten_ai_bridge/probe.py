"""Read-only diagnostic export of one Kaiten card into the outgoing JSON shape."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import tempfile
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import Settings
from .domain import (
    PreparedFile,
    build_initial_prompt,
    build_outgoing_payload,
    deduplicate_prepared,
    encoded_json_size,
    validate_file_refs,
    validate_prepared_files,
)
from .errors import DataRejectedError
from .kaiten import KaitenClient


@dataclass(frozen=True, slots=True)
class ProbeSummary:
    card_id: int
    output_path: str
    json_bytes: int
    file_count: int
    source_file_bytes: int
    output_sha256: str


def _write_private_json(path: Path, content: bytes) -> Path:
    """Atomically write a requested sensitive diagnostic artifact with mode 0600."""

    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.chmod(temporary, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
        os.chmod(target, 0o600)
        return target
    except BaseException:
        if descriptor >= 0:
            with suppress(OSError):
                os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


async def export_initial_payload(
    settings: Settings,
    *,
    card_id: int,
    output_path: Path,
    kaiten: KaitenClient | None = None,
) -> ProbeSummary:
    """Fetch one card and files using GET only, then write the exact initial payload."""

    if card_id <= 0:
        raise ValueError("card_id must be positive")
    owned_client = kaiten is None
    client = kaiten or KaitenClient(settings)
    try:
        snapshot = await client.card_snapshot(card_id)
        validate_file_refs(
            snapshot.attachments,
            max_bytes=settings.max_file_bytes,
            allowed_extensions=settings.allowed_extensions,
        )
        prepared: list[PreparedFile] = []
        for reference in snapshot.attachments:
            async with client.downloaded_file(
                reference,
                max_bytes=settings.max_file_bytes,
            ) as downloaded:
                prepared.append(downloaded)
        prepared = deduplicate_prepared(prepared)
        validate_prepared_files(
            prepared,
            max_bytes=settings.max_file_bytes,
            allowed_extensions=settings.allowed_extensions,
        )
        prompt = build_initial_prompt(snapshot, [item.reference for item in prepared])
        payload = build_outgoing_payload(
            schema_version=settings.schema_version,
            event_type="initial",
            idempotency_key=f"card:add:{card_id}",
            card_id=card_id,
            card_url=snapshot.url,
            request_type=snapshot.request_type,
            prompt=prompt,
            files=prepared,
        )
        compact_size = encoded_json_size(payload)
        maximum = settings.max_outgoing_body_bytes
        if maximum is not None and compact_size > maximum:
            raise DataRejectedError(
                "outgoing_body_too_large",
                "Сформированный JSON превышает настроенный общий лимит передачи.",
            )
        content = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        resolved = _write_private_json(output_path, content)
        return ProbeSummary(
            card_id=card_id,
            output_path=str(resolved),
            json_bytes=len(content),
            file_count=len(prepared),
            source_file_bytes=sum(len(item.content) for item in prepared),
            output_sha256=hashlib.sha256(content).hexdigest(),
        )
    finally:
        if owned_client:
            await client.close()


def _probe_settings() -> Settings:
    # The receiver is deliberately absent from this read-only command. Supplying
    # a process-local placeholder lets the shared settings model validate all
    # Kaiten and file-policy values without requiring an AI credential.
    return Settings(environment="test", ai_bearer_token="probe-command-unused")  # type: ignore[call-arg]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read one Kaiten card and write the would-be initial JSON without mutations."
    )
    parser.add_argument("--card-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    summary = asyncio.run(
        export_initial_payload(
            _probe_settings(),
            card_id=arguments.card_id,
            output_path=arguments.output,
        )
    )
    safe_result: dict[str, Any] = asdict(summary)
    print(json.dumps(safe_result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
