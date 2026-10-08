"""Shared command-line helpers for scripts/*.py."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from .config import REPO_ROOT, Config, ConfigError, Selection, load_config, select


def add_common_args(p: argparse.ArgumentParser, with_selection: bool = True):
    p.add_argument("--config", default="configs/experiment.yaml", help="experiment YAML (default: %(default)s)")
    p.add_argument("--results-dir", default=None, help="override output.results_dir")
    if with_selection:
        p.add_argument("--app", action="append", help="application id or name (repeatable)")
        p.add_argument("--model", action="append", help="model id or label (repeatable)")
        p.add_argument("--strategy", action="append", help="section_rag | full_rag | section_norag | full_norag (repeatable)")
        p.add_argument("--runs", type=int, default=None, help="number of runs per configuration (default: from config)")
        p.add_argument("--run-numbers", type=int, nargs="+", default=None, help="explicit run numbers, e.g. 3 4")


def load_cfg(args) -> Config:
    cfg = load_config(args.config)
    if args.results_dir:
        cfg = cfg.with_results_dir(args.results_dir)
    return cfg


def load_cfg_and_selection(args) -> tuple[Config, Selection]:
    cfg = load_cfg(args)
    sel = select(cfg, args.app, args.model, args.strategy, args.runs, args.run_numbers)
    return cfg, sel


def fail(msg: str, code: int = 2):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


__all__ = ["add_common_args", "load_cfg", "load_cfg_and_selection", "fail", "ConfigError", "Path"]
