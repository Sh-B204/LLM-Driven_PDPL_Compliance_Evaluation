"""Validation of saved run records and of the whole results tree."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .aggregation import aggregate
from .config import STRATEGY_NAMES
from .storage import SCHEMA_VERSION, compute_integrity, load_json

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
    if rec["status"] == "success":
        pred = rec.get("prediction")
        if not isinstance(pred, dict) or list(pred.keys()) != items:
            p.append("prediction keys do not match rubric items")
        else:
            if any(v not in (0, 1, None) for v in pred.values()):
                p.append("prediction contains values other than 0/1/null")
            if all(v is None for v in pred.values()):
                p.append("success record without any parsed prediction")
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
                            or any(x not in (0, 1) for x in v) or not isinstance(s.get("raw_response"), str):
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
    elif rec["status"] in ("failed", "parse_failed") and not rec.get("error") and rec["status"] == "failed":
        p.append("failed record without error information")
    return p


def check_file(path: Path) -> tuple:
    """Return (record or None, problems)."""
    try:
        rec = load_json(path)
    except Exception as e:
        return None, [f"unreadable JSON: {e}"]
    return rec, validate_record(rec, path=path)


def is_valid_completed(path: Path) -> bool:
    if not Path(path).exists():
        return False
    rec, problems = check_file(path)
    return rec is not None and not problems and rec["status"] == "success"


def validate_tree(results_dir: Path) -> dict:
    """Validate every results/raw/**/run_*.json. Also reports stray temp files."""
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
