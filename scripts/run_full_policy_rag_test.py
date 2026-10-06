#!/usr/bin/env python3
"""Full-policy + RAG test: 1 application (default: longest policy), 1 model, 1 run, strategy full_rag.
Measures policy / retrieved-context / prompt / total token lengths and checks the model's context limit
BEFORE any weights are loaded. Never truncates: if the request does not fit it stops and explains why (exit 3).

  python scripts/run_full_policy_rag_test.py
  python scripts/run_full_policy_rag_test.py --app circlys --model llama-3.1-8b
  python scripts/run_full_policy_rag_test.py --mock
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json

from pdpl_eval.cli import add_common_args, fail, load_cfg
from pdpl_eval.config import ConfigError, select
from pdpl_eval.preflight import longest_policy, preflight_full_rag
from pdpl_eval.runner import run_experiments
from pdpl_eval.storage import atomic_write_json, raw_path
from pdpl_eval.utils import setup_logging, utc_stamp
from pdpl_eval.validation import check_file


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, with_selection=False)
    ap.add_argument("--app", help="application id/name (default: longest available policy)")
    ap.add_argument("--model", help="model id/label (default: first configured model)")
    args = ap.parse_args()
    try:
        cfg = load_cfg(args)
        app = next((a for a in cfg.applications if args.app in (a.id, a.name)), None) if args.app else longest_policy(cfg)
        if app is None or not app.policy_path.exists():
            fail("no policy file found (see data/README.md), or use --mock")
        sel = select(cfg, [app.id], [args.model] if args.model else [cfg.models[0].id], ["full_rag"], runs=1)
    except ConfigError as e:
        fail(str(e))
    out = (cfg.results_dir if args.results_dir else cfg.root / "results" / "full_policy_rag_test") / utc_stamp()
    cfg = cfg.with_results_dir(out)
    setup_logging(cfg.results_dir / "logs", "full_policy_rag_test")
    model = sel.models[0]
    print(f"Full-policy+RAG test: app={app.id} (longest available policy: {not args.app}) model={model.id}")
    try:
        pf = preflight_full_rag(cfg, app, model)
    except Exception as e:
        fail(f"preflight failed: {type(e).__name__}: {e}")
    atomic_write_json(cfg.results_dir / "preflight_report.json", pf)
    t = pf["token_lengths"]
    print(f"  policy tokens           : {t['policy_tokens']}  ({pf['policy_chars']} chars)")
    print(f"  retrieved-context tokens: {t['retrieved_context_tokens']}  ({pf['retrieved_context_chars']} chars)")
    print(f"  full prompt tokens      : {t['prompt_tokens']}")
    print(f"  + max_new_tokens        : {t['max_new_tokens']}")
    print(f"  total tokens needed     : {t['total_tokens']}   |  model context limit: {pf['context_limit_tokens']}")
    if not pf["fits_context"]:
        print(f"\nSTOPPED (nothing truncated, nothing sent): {pf['reason']}")
        print(f"Report: {cfg.results_dir / 'preflight_report.json'}")
        return 3
    s = run_experiments(cfg, sel)
    p = raw_path(cfg.results_dir, app.id, model.id, "full_rag", 1)
    rec, problems = check_file(p) if p.exists() else (None, ["result file missing"])
    ok = len(s["completed"]) == 1 and not problems
    print(f"  run status: {rec['status'] if rec else 'n/a'} | validation: {'OK' if not problems else problems}")
    if rec and rec.get("retrieval"):
        print("  retrieved chunks:", [h["title"] for h in rec["retrieval"]["retrieved"]])
    print("\nFULL-POLICY+RAG TEST", "PASSED" if ok else "FAILED", f"-> {p}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
