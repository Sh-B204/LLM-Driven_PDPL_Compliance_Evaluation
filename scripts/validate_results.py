#!/usr/bin/env python3
"""Validate every saved run record (schema, integrity hash, rubric keys, aggregation consistency, path identity)
and print a completion summary for the configured matrix.  Exit code 1 if anything is invalid."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse

from pdpl_eval.cli import add_common_args, fail, load_cfg_and_selection
from pdpl_eval.config import ConfigError
from pdpl_eval.report import build_completion_report
from pdpl_eval.validation import validate_tree


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--missing-ok", action="store_true", help="do not fail when runs are missing/failed")
    args = ap.parse_args()
    try:
        cfg, sel = load_cfg_and_selection(args)
    except ConfigError as e:
        fail(str(e))
    tree = validate_tree(cfg.results_dir)
    rep = build_completion_report(cfg, sel)
    print(f"Results dir: {cfg.results_dir}")
    print(f"Files: {tree['n_files']} | valid success: {tree['valid_success']} | valid failed records: {tree['valid_failed']} "
          f"| invalid: {len(tree['invalid'])} | stray tmp: {len(tree['stray_tmp'])} | duplicates: {len(tree['duplicates'])}")
    for bad in tree["invalid"][:50]:
        print("  INVALID", bad["path"], "->", "; ".join(bad["problems"]))
    print("Matrix:", rep["expected_runs"], "expected |", rep["counts"])
    bad = tree["invalid"] or tree["stray_tmp"] or tree["duplicates"]
    incomplete = rep["counts"]["completed"] != rep["expected_runs"]
    return 1 if bad or (incomplete and not args.missing_ok) else 0


if __name__ == "__main__":
    sys.exit(main())
