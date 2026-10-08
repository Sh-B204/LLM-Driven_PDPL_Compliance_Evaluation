from __future__ import annotations
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat(timespec="seconds")


def utc_stamp() -> str:
    """Filesystem-safe UTC timestamp, e.g. 20261005T134501Z."""
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def fingerprint(obj) -> str:
    return sha256_text(canonical_json(obj))


def parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def safe_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def setup_logging(log_dir: str | Path, name: str, verbose: bool = True) -> Path:
    """Console + file logging. Returns the log file path."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}_{utc_stamp()}.log"
    root = logging.getLogger("pdpl")
    root.setLevel(logging.DEBUG)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%dT%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO if verbose else logging.WARNING)
    ch.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(ch)
    root.propagate = False
    return log_path


def get_logger(suffix: str = "") -> logging.Logger:
    return logging.getLogger("pdpl" + (f".{suffix}" if suffix else ""))
