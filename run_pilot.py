import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from pdpl.config import add_common_args, fail, load_cfg, select, ConfigError
from pdpl.runner import preflight_full_rag, run_experiments
from pdpl.records import atomic_write_json, raw_path, check_file
from pdpl.inputs import setup_logging

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); add_common_args(ap, with_selection=False)
    ap.add_argument("--app", required=True, help="application id or name"); ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--strategies", nargs="+", default=["full_norag", "full_rag"], choices=["full_norag", "full_rag"], help="default: both")
    a = ap.parse_args()
    try:
        cfg = load_cfg(a); app = next((x for x in cfg.applications if a.app in (x.id, x.name)), None)
        if app is None: fail(f"unknown application {a.app}")
        sel = select(cfg, [app.id], ["allam-7b"], a.strategies, runs=1)
    except ConfigError as e: fail(str(e))
    setup_logging(cfg.results_dir / "logs", "pilot"); model = sel.models[0]
    print(f"response allowance: {cfg.generation['max_new_tokens']} tokens | results: {cfg.results_dir}")
    if "full_rag" in a.strategies:
        pf = preflight_full_rag(cfg, app, model); atomic_write_json(cfg.results_dir / "preflight" / f"{app.id}__{model.id}.json", pf)
        t = pf["token_lengths"]; print(f"full_rag rendered prompt: {t['prompt_tokens']} tokens + {t['max_new_tokens']} allowance vs limit {pf['context_limit_tokens']} -> fits={pf['fits_context']}")
    if a.preflight_only: return 0
    s = run_experiments(cfg, sel)           
    bad = False
    for strat in a.strategies:
        p = raw_path(cfg.results_dir, app.id, model.id, strat, 1); rec, prob = check_file(p) if p.exists() else (None, ["missing"])
        ok = rec is not None and not prob and rec["status"] == "success"; bad |= not ok
        print(f"{strat}: status={rec['status'] if rec else 'n/a'} validation={'OK' if not prob else prob} error={(rec or {}).get('error', {}) and rec['error'].get('type')} -> {p}")
    bad |= bool(s["failed"] or s["parse_failed"] or s["blocked"] or s["save_errors"] or s["interrupted"])
    print("Pilot finished with FAILURES (see above)." if bad else "Pilot finished OK.", "The full experiment was NOT started."); return 1 if bad else 0

if __name__ == "__main__": sys.exit(main())
