"""Atomic result saving, post-save verification, archiving (never silent overwrite)."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from .utils import canonical_json, sha256_text, utc_stamp

SCHEMA_VERSION = 1


def raw_path(results_dir: Path, app_id: str, model_id: str, strategy: str, run: int) -> Path:
    return Path(results_dir) / "raw" / app_id / model_id / strategy / f"run_{run:02d}.json"


def archive_path(results_dir: Path, path: Path) -> Path:
    rel = Path(path).relative_to(Path(results_dir))
    return Path(results_dir) / "archive" / rel.parent / f"{rel.stem}__{utc_stamp()}{rel.suffix}"


def compute_integrity(record: dict) -> dict:
    body = {k: v for k, v in record.items() if k != "integrity"}
    return {"algorithm": "sha256-canonical-json", "sha256": sha256_text(canonical_json(body))}


def atomic_write_text(path: Path, text: str) -> None:
    """Write to a temp file in the same directory, fsync, then os.replace (atomic on POSIX/NTFS)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_write_json(path: Path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_result(results_dir: Path, path: Path, record: dict, allow_replace_existing: bool = False) -> Optional[Path]:
    """Atomically save a run record.

    * If `path` already exists it is NEVER overwritten in place: the old file is first moved to
      results/archive/ (only allowed when allow_replace_existing=True; otherwise FileExistsError).
    * The integrity hash is added just before writing.
    Returns the archive path if an old file was archived.
    """
    path = Path(path)
    archived = None
    if path.exists():
        if not allow_replace_existing:
            raise FileExistsError(f"Refusing to overwrite existing result: {path}")
        archived = archive_path(results_dir, path)
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(archived))
    record = dict(record)
    record["integrity"] = compute_integrity(record)
    atomic_write_json(path, record)
    return archived


def verify_saved(path: Path, record_expected: dict | None = None) -> list:
    """Re-read the file from disk and check parse + integrity hash (+ equality with in-memory record)."""
    from .validation import validate_record
    problems = []
    try:
        on_disk = load_json(path)
    except Exception as e:
        return [f"cannot re-read saved file: {e}"]
    problems += validate_record(on_disk, path=path)
    if record_expected is not None:
        exp = dict(record_expected)
        exp["integrity"] = compute_integrity(exp)
        if json.loads(json.dumps(exp, ensure_ascii=False, default=str)) != on_disk:
            problems.append("saved content differs from in-memory record")
    return problems


def append_event(results_dir: Path, event: dict) -> None:
    """Append one JSON line to results/logs/events.jsonl (append-only audit trail)."""
    p = Path(results_dir) / "logs" / "events.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())
