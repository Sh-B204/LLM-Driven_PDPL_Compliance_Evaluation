"""Consolidate saved results into CSV/JSONL; optionally evaluate against expert labels."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import argparse

from pdpl_eval.cli import add_common_args, load_cfg
from pdpl_eval.consolidate import consolidate
from pdpl_eval.evaluation import analyze_results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap, with_selection=False)
    ap.add_argument("--evaluate", action="store_true", help="also evaluate saved predictions against expert labels")
    ap.add_argument("--allow-partial", action="store_true", help="allow incomplete diagnostic evaluation")
    args = ap.parse_args()
    cfg = load_cfg(args)
    r = consolidate(cfg.results_dir, cfg=cfg)
    print({k: v for k, v in r.items() if k != "skipped_invalid"})
    for s in r["skipped_invalid"]:
        print("  skipped invalid:", s["path"], s["problems"])
    if args.evaluate:
        print(analyze_results(cfg, allow_partial=args.allow_partial))
    return 1 if r["skipped_invalid"] and not args.allow_partial else 0


if __name__ == "__main__":
    sys.exit(main())
