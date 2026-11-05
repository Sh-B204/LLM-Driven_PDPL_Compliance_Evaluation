from __future__ import annotations
from pathlib import Path
from typing import Optional
from .config import Config, Selection
from .storage import atomic_write_json, atomic_write_text, raw_path
from .utils import utc_now_iso, utc_stamp
from .validation import check_file
from .runner import experiment_fingerprint, can_reuse_result
from .rubric import load_rubric
from .prompts import load_prompt


def build_completion_report(cfg: Config, sel: Selection, run_summary: Optional[dict] = None) -> dict:
    rd = cfg.results_dir
    rubric = load_rubric(cfg.rubric_path)
    rows, counts = [], {"completed": 0, "failed": 0, "parse_failed": 0, "invalid": 0, "missing": 0}
    for m in sel.models:
        for a in sel.applications:
            for s in sel.strategies:
                for r in sel.run_numbers:
                    p = raw_path(rd, a.id, m.id, s.name, r)
                    if not p.exists():
                        state, detail = "missing", None
                    else:
                        rec, problems = check_file(p)
                        if problems:
                            state, detail = "invalid", problems
                        elif rec["status"] == "success":
                            try:
                                expected = experiment_fingerprint(cfg, a, m, s, rubric, load_prompt(s.name, s.prompt_path))
                                reusable = can_reuse_result(p, expected, m.revision, cfg.retrieval.get("embedding_revision"))
                                state, detail = ("completed", None) if reusable else ("invalid", "settings mismatch or unpinned revision")
                            except Exception as exc:
                                state, detail = "invalid", str(exc)
                        else:
                            state, detail = rec["status"], rec.get("error")
                    counts[state] += 1
                    rows.append({"application": a.id, "model": m.id, "strategy": s.name, "run": r,
                                 "state": state, "detail": detail})
    rep = {"generated_at": utc_now_iso(), "expected_runs": len(rows), "counts": counts,
           "this_invocation": ({k: (len(v) if isinstance(v, list) else v) for k, v in run_summary.items()}
                               if run_summary else None),
           "skipped_this_invocation": run_summary["skipped"] if run_summary else None,
           "blocked_this_invocation": run_summary["blocked"] if run_summary else None,
           "not_completed": [r for r in rows if r["state"] != "completed"]}
    return rep


def write_completion_report(cfg: Config, sel: Selection, run_summary: Optional[dict] = None) -> Path:
    rep = build_completion_report(cfg, sel, run_summary)
    logs = cfg.results_dir / "logs"
    stamp = utc_stamp()
    atomic_write_json(logs / f"completion_report_{stamp}.json", rep)
    c = rep["counts"]
    lines = [f"# Completion report ({rep['generated_at']})", "",
             f"Expected runs: **{rep['expected_runs']}**", "",
             "| completed | failed | parse_failed | invalid | missing |", "|---|---|---|---|---|",
             f"| {c['completed']} | {c['failed']} | {c['parse_failed']} | {c['invalid']} | {c['missing']} |", ""]
    if run_summary:
        t = rep["this_invocation"]
        lines += ["This invocation: " + ", ".join(f"{k}={v}" for k, v in t.items()), ""]
    if rep["not_completed"]:
        lines += ["## Not completed", "", "| application | model | strategy | run | state |", "|---|---|---|---|---|"]
        lines += [f"| {r['application']} | {r['model']} | {r['strategy']} | {r['run']} | {r['state']} |"
                  for r in rep["not_completed"][:500]]
        if len(rep["not_completed"]) > 500:
            lines.append(f"\n(... {len(rep['not_completed']) - 500} more in the JSON report)")
    if rep["blocked_this_invocation"]:
        reasons = sorted({b["reason"] for b in rep["blocked_this_invocation"]})
        lines += ["", "## Blocked (not attempted) - reasons", ""] + [f"- {x}" for x in reasons]
    path = logs / f"completion_report_{stamp}.md"
    atomic_write_text(path, "\n".join(lines) + "\n")
    return path
