"""Pilot: one application, all configured models, all four strategies, one run.
Saves results and stops for review; never starts the full experiment.

  python scripts/run_smoke_test.py --app circlys
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import argparse

from pdpl_eval.cli import add_common_args, fail, load_cfg
from pdpl_eval.config import ConfigError, STRATEGY_NAMES, select
from pdpl_eval.report import build_completion_report, write_completion_report
from pdpl_eval.runner import run_experiments
from pdpl_eval.utils import setup_logging


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, with_selection=False)
    ap.add_argument("--app", required=True, help="one application id or name")
    args = ap.parse_args()
    try:
        cfg = load_cfg(args)
        sel = select(cfg, [args.app], strategies=list(STRATEGY_NAMES), runs=1)
        if not sel.applications[0].policy_path.exists():
            fail(f"policy file not found: {sel.applications[0].policy_path}")
        if not sel.models:
            fail("no models configured")
    except ConfigError as e:
        fail(str(e))
    setup_logging(cfg.results_dir / "logs", "smoke_test")
    expected = len(sel.models) * len(sel.strategies)
    print(f"Pilot: app={sel.applications[0].id}, {len(sel.models)} models, "
          f"{len(sel.strategies)} strategies, run=1 -> {expected} records")
    summary = run_experiments(cfg, sel)
    report_path = write_completion_report(cfg, sel, summary)
    report = build_completion_report(cfg, sel, summary)
    ok = report["counts"]["completed"] == expected and not summary["interrupted"] \
        and not any(summary[k] for k in ("failed", "parse_failed", "blocked", "save_errors"))
    print("Completion report:", report_path)
    print("Results:", cfg.results_dir / "raw" / sel.applications[0].id)
    print("PILOT", "PASSED" if ok else "FAILED")
    print("Stopped. Review the saved results before starting the full experiment separately.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
