#!/usr/bin/env python3
"""Smoke test: 1 application, 1 model, 1 simple strategy (full_norag), 1 run.
Checks: loading, prompting, model response, parsing, atomic saving, validation, resume detection.

  python scripts/run_smoke_test.py                 # real model (GPU), first configured app/model with a policy file
  python scripts/run_smoke_test.py --app circlys --model qwen-2.5-7b
  python scripts/run_smoke_test.py --mock          # plumbing only, no GPU
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json

from pdpl_eval.cli import add_common_args, fail, load_cfg
from pdpl_eval.config import ConfigError, select
from pdpl_eval.runner import run_experiments
from pdpl_eval.storage import raw_path
from pdpl_eval.utils import sha256_file, setup_logging, utc_stamp
from pdpl_eval.validation import check_file, is_valid_completed

checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, with_selection=False)
    ap.add_argument("--app"); ap.add_argument("--model")
    ap.add_argument("--strategy", default="full_norag", choices=["full_norag", "section_norag"])
    args = ap.parse_args()
    try:
        cfg = load_cfg(args)
        app = next((a for a in cfg.applications if args.app in (a.id, a.name)), None) if args.app else \
            next((a for a in cfg.applications if a.policy_path.exists()), None)
        if args.app and app is None:
            fail(f"unknown application {args.app}")
        if app is None or not app.policy_path.exists():
            fail("no policy file found. Place policy DOCX files as described in data/README.md "
                 "(or run with --mock to test the plumbing).")
        sel = select(cfg, [app.id], [args.model] if args.model else [cfg.models[0].id], [args.strategy], runs=1)
    except ConfigError as e:
        fail(str(e))
    out = (cfg.results_dir if args.results_dir else cfg.root / "results" / "smoke_test") / utc_stamp()
    cfg = cfg.with_results_dir(out)
    setup_logging(cfg.results_dir / "logs", "smoke_test")
    model = sel.models[0]
    print(f"Smoke test: app={app.id} model={model.id} strategy={args.strategy} run=1 -> {cfg.results_dir}")

    s1 = run_experiments(cfg, sel)
    check("model loaded + prompt sent + response received + saved + verified", len(s1["completed"]) == 1,
          f"summary={ {k: len(v) for k, v in s1.items() if isinstance(v, list)} }")
    p = raw_path(cfg.results_dir, app.id, model.id, args.strategy, 1)
    check("result file exists at expected path", p.exists(), str(p.relative_to(cfg.results_dir)) if p.exists() else str(p))
    rec, problems = check_file(p) if p.exists() else (None, ["missing"])
    check("record passes validation (schema, integrity hash, rubric keys)", not problems, "; ".join(problems))
    if rec:
        check("raw response is non-empty", bool(rec.get("raw_response", "").strip()))
        n = sum(v is not None for v in (rec.get("prediction") or {}).values())
        check("parsed predictions present", n > 0, f"{n}/{len(rec['rubric']['items'])} rubric items parsed")
        check("parameters, timestamps and runtime recorded", all(k in rec for k in ("parameters", "started_at", "finished_at", "runtime_seconds")))
    check("resume detection: completed run recognised as valid", is_valid_completed(p))
    before = sha256_file(p) if p.exists() else None
    s2 = run_experiments(cfg, sel)
    check("second invocation skipped the run (no re-execution)", len(s2["skipped"]) == 1 and not s2["completed"])
    check("existing result left byte-identical (no silent overwrite)", p.exists() and sha256_file(p) == before)
    ok = all(c[1] for c in checks)
    print("\nSMOKE TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
