import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from pdpl.config import add_common_args, fail, load_cfg_and_selection, ConfigError
from pdpl.runner import check_inputs, pin_model_revisions, plan, run_experiments, write_completion_report
from pdpl.inputs import setup_logging

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); add_common_args(ap)
    ap.add_argument("--resume", action="store_true", help="accepted for clarity: skipping valid completed records is always on")
    ap.add_argument("--force", action="store_true", help="re-run valid records; old file moved to results_v2/archive first")
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--check-inputs", action="store_true"); ap.add_argument("--pin-revisions", action="store_true")
    a = ap.parse_args()
    try: cfg, sel = load_cfg_and_selection(a)
    except ConfigError as e: fail(str(e))
    if a.check_inputs: print(check_inputs(cfg)); return 0
    if a.pin_revisions: pin_model_revisions(cfg); return 0
    log = setup_logging(cfg.results_dir / "logs", "run_experiments")
    print(f"Plan: {len(sel.applications)} apps x {len(sel.models)} models x {len(sel.strategies)} strategies x {len(sel.run_numbers)} runs = {len(plan(cfg, sel))} | results: {cfg.results_dir} | log: {log}")
    s = run_experiments(cfg, sel, force=a.force, dry_run=a.dry_run)
    if a.dry_run: return 1 if s["blocked"] else 0
    print("This invocation:", {k: (len(v) if isinstance(v, list) else v) for k, v in s.items()}); print("Report:", write_completion_report(cfg, sel, s))
    return 1 if (s["failed"] or s["parse_failed"] or s["blocked"] or s["save_errors"] or s["interrupted"]) else 0

if __name__ == "__main__": sys.exit(main())
