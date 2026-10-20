"""Incremental JSONL reading with byte-offset checkpoints.

Every JSONL source (Codex rollouts, Claude Code transcripts) is read through
this module so that the awkward parts are solved once:

* files are read from a byte offset, never fully re-parsed on append;
* a fingerprint of the file head detects replaced/rotated files and resets
  the offset safely (upserts keep re-reads idempotent);
* a truncated file (size shrank) is re-read from the start;
* a partially written trailing line is held back — its bytes stay unread
  until the writer finishes the line;
* malformed lines are reported by byte offset (never content) and skipped.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

#: How many head bytes identify a file across rotations/replacements.
FINGERPRINT_BYTES = 256


@dataclass(slots=True)
class JsonlBatch:
    records: list[dict] = field(default_factory=list)
    malformed: list[tuple[int, str]] = field(default_factory=list)  # (byte offset, reason)
    start_offset: int = 0
    end_offset: int = 0
    reset: bool = False  # True when the file was replaced or truncated
    fingerprint: str = ""


def file_fingerprint(path: Path, *, head_bytes: int = FINGERPRINT_BYTES) -> str:
    with path.open("rb") as handle:
        head = handle.read(head_bytes)
    return hashlib.sha256(head).hexdigest()[:32]


def read_jsonl_batch(
    path: Path,
    *,
    offset: int = 0,
    fingerprint: str | None = None,
    head_bytes: int = FINGERPRINT_BYTES,
) -> JsonlBatch:
    """Read complete JSONL records from ``offset`` to the last complete line."""
    current_fingerprint = file_fingerprint(path, head_bytes=head_bytes)
    size = path.stat().st_size
    reset = False
    if offset and (current_fingerprint != fingerprint or size < offset):
        offset = 0
        reset = True
    elif offset > size:
        offset = 0
        reset = True

    with path.open("rb") as handle:
        handle.seek(offset)
        payload = handle.read()

    consumed = payload.rfind(b"\n") + 1  # hold back a partial trailing line
    complete = payload[:consumed]

    batch = JsonlBatch(
        start_offset=offset,
        end_offset=offset + consumed,
        reset=reset,
        fingerprint=current_fingerprint,
    )

    position = offset
    for line in complete.split(b"\n"):
        line_start = position
        position += len(line) + 1  # +1 for the newline byte
        if not line.strip():
            continue
        if offset == 0 and line_start == 0 and line.startswith(b"\xef\xbb\xbf"):
            line = line[3:]  # strip UTF-8 BOM
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            batch.malformed.append((line_start, type(exc).__name__))
            continue
        if isinstance(obj, dict):
            batch.records.append(obj)
        else:
            batch.malformed.append((line_start, "not a JSON object"))
    return batch
