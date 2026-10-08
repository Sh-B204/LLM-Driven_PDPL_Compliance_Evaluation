from __future__ import annotations
import csv
import io
import json
from pathlib import Path

from .storage import atomic_write_text
from .validation import check_file

PRED_FIELDS = ["application_id", "application", "model_id", "model_label", "model_revision", "strategy", "run",
               "rubric_code", "rubric_criterion", "prediction", "status", "aggregation_rule", "seed",
               "started_at", "finished_at", "runtime_seconds", "raw_file"]
SECTION_FIELDS = ["application_id", "model_id", "strategy", "run", "section_index", "section_title",
                  "rubric_code", "rubric_criterion", "prediction", "section_status", "raw_response",
                  "prompt_tokens", "n_retrieved", "raw_file"]
RUN_FIELDS = ["application_id", "model_id", "model_label", "strategy", "run", "status", "n_sections",
              "prompt_tokens", "runtime_seconds", "started_at", "finished_at", "error_type", "error_message", "raw_file"]


def _csv_text(fields, rows) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})
    return buf.getvalue()


def _jsonl_text(rows) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def consolidate(results_dir: Path, include_invalid: bool = False, cfg=None) -> dict:
    results_dir = Path(results_dir)
    files = sorted((results_dir / "raw").rglob("run_*.json")) if (results_dir / "raw").exists() else []
    preds, secs, runs, skipped_invalid = [], [], [], []
    for f in files:
        rec, problems = check_file(f)
        if problems:
            skipped_invalid.append({"path": str(f), "problems": problems})
            continue
        if cfg is not None and rec["status"] == "success":
            from .runner import experiment_fingerprint, can_reuse_result
            from .rubric import load_rubric
            from .prompts import load_prompt
            try:
                app = next(a for a in cfg.applications if a.id == rec["application"]["id"])
                model = next(m for m in cfg.models if m.id == rec["model"]["id"])
                spec = cfg.strategies[rec["strategy"]]
                expected = experiment_fingerprint(cfg, app, model, spec, load_rubric(cfg.rubric_path), load_prompt(spec.name, spec.prompt_path))
                if not can_reuse_result(f, expected, model.revision, cfg.retrieval.get("embedding_revision")):
                    raise ValueError("stale result or unpinned revision")
            except Exception as exc:
                skipped_invalid.append({"path": str(f), "problems": [str(exc)]})
                continue
        rel = str(f.relative_to(results_dir))
        app, model = rec["application"], rec["model"]
        err = rec.get("error") or {}
        toks = rec.get("prompt_tokens") or (rec.get("token_lengths") or {}).get("prompt_tokens")
        runs.append({"application_id": app["id"], "model_id": model["id"], "model_label": model["label"],
                     "strategy": rec["strategy"], "run": rec["run"], "status": rec["status"],
                     "n_sections": rec["policy"]["n_sections"] if rec["unit"] == "section" else "",
                     "prompt_tokens": toks, "runtime_seconds": rec["runtime_seconds"],
                     "started_at": rec["started_at"], "finished_at": rec["finished_at"],
                     "error_type": err.get("type"), "error_message": err.get("message"), "raw_file": rel})
        if rec["status"] == "success":
            for item, code in zip(rec["rubric"]["items"], rec["rubric"]["codes"]):
                preds.append({"application_id": app["id"], "application": app["name"], "model_id": model["id"],
                              "model_label": model["label"], "model_revision": model.get("revision_resolved"),
                              "strategy": rec["strategy"], "run": rec["run"], "rubric_code": code,
                              "rubric_criterion": item, "prediction": rec["prediction"][item], "status": "success",
                              "aggregation_rule": (rec.get("aggregation") or {}).get("rule"),
                              "seed": rec["parameters"].get("seed"), "started_at": rec["started_at"],
                              "finished_at": rec["finished_at"], "runtime_seconds": rec["runtime_seconds"],
                              "raw_file": rel})
            for s in rec.get("section_predictions") or []:
                for j, (item, code) in enumerate(zip(rec["rubric"]["items"], rec["rubric"]["codes"])):
                    secs.append({"application_id": app["id"], "model_id": model["id"], "strategy": rec["strategy"],
                                 "run": rec["run"], "section_index": s["section_index"], "section_title": s["title"],
                                 "rubric_code": code, "rubric_criterion": item,
                                 "prediction": s["parsed_vector"][j], "section_status": s["status"],
                                 "raw_response": s.get("raw_response"), "prompt_tokens": s.get("prompt_tokens"),
                                 "n_retrieved": len((s.get("retrieval") or {}).get("retrieved", [])), "raw_file": rel})
    out = results_dir / "consolidated"
    atomic_write_text(out / "all_predictions.jsonl", _jsonl_text(preds))
    atomic_write_text(out / "all_predictions.csv", _csv_text(PRED_FIELDS, preds))
    atomic_write_text(out / "all_section_predictions.jsonl", _jsonl_text(secs))
    atomic_write_text(out / "all_section_predictions.csv", _csv_text(SECTION_FIELDS, secs))
    atomic_write_text(out / "all_runs.csv", _csv_text(RUN_FIELDS, runs))
    return {"raw_files": len(files), "runs": len(runs), "prediction_rows": len(preds),
            "section_prediction_rows": len(secs), "skipped_invalid": skipped_invalid, "output_dir": str(out)}
