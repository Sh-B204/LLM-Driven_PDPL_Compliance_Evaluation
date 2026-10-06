#!/usr/bin/env python3
"""Consolidate raw run files into results/consolidated/*.jsonl|csv (long format; no statistics)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse

from pdpl_eval.cli import add_common_args, load_cfg
from pdpl_eval.consolidate import consolidate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap, with_selection=False)
    ap.add_argument("--include-invalid", action="store_true", help="also consolidate records that fail validation")
    args = ap.parse_args()
    cfg = load_cfg(args)
    r = consolidate(cfg.results_dir, include_invalid=args.include_invalid)
    print({k: v for k, v in r.items() if k != "skipped_invalid"})
    for s in r["skipped_invalid"]:
        print("  skipped invalid:", s["path"], s["problems"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
