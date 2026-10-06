#!/usr/bin/env python3
"""Run the full experiment matrix sequentially (model -> application -> strategy -> run), resumable.

Examples:
  python scripts/run_experiments.py --resume
  python scripts/run_experiments.py --app circlys --model llama-3.1-8b --strategy full_rag --runs 2
  python scripts/run_experiments.py --dry-run
  python scripts/run_experiments.py --force --app circlys --strategy full_norag --run-numbers 3
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse

from pdpl_eval.cli import add_common_args, fail, load_cfg_and_selection
from pdpl_eval.config import ConfigError
from pdpl_eval.report import write_completion_report
from pdpl_eval.runner import plan, run_experiments
from pdpl_eval.utils import setup_logging


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--resume", action="store_true",
                    help="resume mode (this is always the behaviour: valid completed runs are skipped)")
    ap.add_argument("--force", action="store_true",
                    help="re-run even valid completed runs; the old file is moved to results/archive/ first")
    ap.add_argument("--dry-run", action="store_true", help="list what would run, load nothing")
    args = ap.parse_args()
    try:
        cfg, sel = load_cfg_and_selection(args)
    except ConfigError as e:
        fail(str(e))
    log_path = setup_logging(cfg.results_dir / "logs", "run_experiments")
    total = len(plan(cfg, sel))
    print(f"Plan: {len(sel.applications)} apps x {len(sel.models)} models x {len(sel.strategies)} strategies "
          f"x {len(sel.run_numbers)} runs = {total} runs | results: {cfg.results_dir} | log: {log_path}")
    summary = run_experiments(cfg, sel, force=args.force, dry_run=args.dry_run)
    if args.dry_run:
        return 0
    rep = write_completion_report(cfg, sel, summary)
    print("\nThis invocation:", {k: (len(v) if isinstance(v, list) else v) for k, v in summary.items()})
    print("Completion report:", rep)
    bad = summary["failed"] or summary["parse_failed"] or summary["blocked"] or summary["save_errors"]
    return 1 if bad or summary["interrupted"] else 0


if __name__ == "__main__":
    sys.exit(main())
