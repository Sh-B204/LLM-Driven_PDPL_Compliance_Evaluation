#!/usr/bin/env python3
"""Optional: pre-download tokenizers/weights for the configured models (requires HF_TOKEN for gated models)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import os

from pdpl_eval.cli import add_common_args, load_cfg_and_selection


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--tokenizer-only", action="store_true")
    args = ap.parse_args()
    cfg, sel = load_cfg_and_selection(args)
    from huggingface_hub import login, snapshot_download
    tok = os.environ.get(cfg.output.get("hf_token_env", "HF_TOKEN"))
    if tok:
        login(tok)
    for m in sel.models:
        if m.type != "hf_local":
            continue
        print("Downloading", m.hf_id)
        kw = {"allow_patterns": ["*.json", "*.model", "*.txt", "tokenizer*"]} if args.tokenizer_only else {}
        snapshot_download(m.hf_id, cache_dir=m.cache_dir, revision=m.revision, **kw)


if __name__ == "__main__":
    main()
