from __future__ import annotations
import json
import os
import shutil
import tempfile
from uuid import uuid4
from pathlib import Path
from .inputs import canonical_json, sha256_text, utc_stamp
from typing import List, Optional
from .predictions import aggregate
from .config import STRATEGY_NAMES

SCHEMA_VERSION = 1


def raw_path(results_dir: Path, app_id: str, model_id: str, strategy: str, run: int) -> Path:
    return Path(results_dir) / "raw" / app_id / model_id / strategy / f"run_{run:02d}.json"


def archive_path(results_dir: Path, path: Path) -> Path:
    rel = Path(path).relative_to(Path(results_dir))
    return Path(results_dir) / "archive" / rel.parent / f"{rel.stem}__{utc_stamp()}__{uuid4().hex}{rel.suffix}"


def compute_integrity(record: dict) -> dict:
    body = {k: v for k, v in record.items() if k != "integrity"}
    return {"algorithm": "sha256-canonical-json", "sha256": sha256_text(canonical_json(body))}


def atomic_write_text(path: Path, text: str) -> None:
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
    p = Path(results_dir) / "logs" / "events.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


STATUSES = ("success", "failed", "parse_failed")
REQUIRED_KEYS = ("schema_version", "application", "model", "strategy", "run", "status", "started_at",
                 "finished_at", "runtime_seconds", "parameters", "rubric", "integrity")


def validate_record(rec: dict, path: Optional[Path] = None) -> List[str]:
    p: List[str] = []
    for k in REQUIRED_KEYS:
        if k not in rec:
            p.append(f"missing key '{k}'")
    if p:
        return p
    if rec["schema_version"] != SCHEMA_VERSION:
        p.append(f"unsupported schema_version {rec['schema_version']}")
    if rec["integrity"].get("sha256") != compute_integrity(rec)["sha256"]:
        p.append("integrity hash mismatch (file edited or corrupted)")
    if rec["strategy"] not in STRATEGY_NAMES:
        p.append(f"unknown strategy '{rec['strategy']}'")
    if rec["status"] not in STATUSES:
        p.append(f"unknown status '{rec['status']}'")
    if path is not None:
        parts = Path(path).parts
        try:
            app, model, strat, fn = parts[-4:]
            if (app, model, strat, fn) != (rec["application"]["id"], rec["model"]["id"], rec["strategy"],
                                           f"run_{int(rec['run']):02d}.json"):
                p.append("record identity does not match its file path")
        except ValueError:
            pass
    items = rec["rubric"]["items"]
    codes = rec["rubric"].get("codes")
    if not isinstance(items, list) or not items or any(not isinstance(i, str) for i in items) \
            or len(set(items)) != len(items) or codes != [i.split(". ")[0] for i in items]:
        p.append("rubric items/codes are missing, duplicated or inconsistent")
        return p
    if rec["status"] == "success":
        pred = rec.get("prediction")
        if not isinstance(pred, dict) or list(pred.keys()) != items:
            p.append("prediction keys do not match rubric items")
        else:
            if any(type(v) is not int or v not in (0, 1) for v in pred.values()):
                p.append("success requires a valid integer 0/1 prediction for every rubric item")
        if rec["strategy"].startswith("section"):
            secs = rec.get("section_predictions")
            n = rec.get("policy", {}).get("n_sections")
            if not secs or len(secs) != n:
                p.append("section_predictions count != policy.n_sections")
            else:
                vecs = []
                for s in secs:
                    v = s.get("parsed_vector")
                    if s.get("status") != "success" or not isinstance(v, list) or len(v) != len(items) \
                            or any(type(x) is not int or x not in (0, 1) for x in v) or not isinstance(s.get("raw_response"), str) or not s["raw_response"].strip():
                        p.append(f"section {s.get('section_index')} invalid/incomplete")
                    else:
                        vecs.append(v)
                if len(vecs) == len(secs) and isinstance(pred, dict):
                    final = aggregate(vecs, rec.get("aggregation", {}).get("rule", "or"))
                    if final != [pred[i] for i in items]:
                        p.append("final prediction != aggregation of section predictions")
        else:
            if not isinstance(rec.get("raw_response"), str) or not rec["raw_response"].strip():
                p.append("full-policy success record without raw_response")
    elif rec["status"] in ("failed", "parse_failed") and not rec.get("error"):
        p.append("failed record without error information")
    return p


def check_file(path: Path) -> tuple:
    try:
        rec = load_json(path)
    except Exception as e:
        return None, [f"unreadable JSON: {e}"]
    try:
        return rec, validate_record(rec, path=path)
    except (TypeError, AttributeError, KeyError, ValueError) as e:
        return rec, [f"malformed record: {e}"]


def is_valid_completed(path: Path) -> bool:
    if not Path(path).exists():
        return False
    rec, problems = check_file(path)
    return rec is not None and not problems and rec["status"] == "success"


def validate_tree(results_dir: Path) -> dict:
    raw = Path(results_dir) / "raw"
    files = sorted(raw.rglob("run_*.json")) if raw.exists() else []
    out = {"n_files": len(files), "valid_success": 0, "valid_failed": 0, "invalid": [], "stray_tmp": [],
           "duplicates": []}
    seen = {}
    for f in files:
        rec, problems = check_file(f)
        if problems:
            out["invalid"].append({"path": str(f), "problems": problems})
            continue
        key = (rec["application"]["id"], rec["model"]["id"], rec["strategy"], rec["run"])
        if key in seen:
            out["duplicates"].append(str(f))
        seen[key] = f
        out["valid_success" if rec["status"] == "success" else "valid_failed"] += 1
    if raw.exists():
        out["stray_tmp"] = [str(x) for x in raw.rglob("*.tmp")]
    return out
