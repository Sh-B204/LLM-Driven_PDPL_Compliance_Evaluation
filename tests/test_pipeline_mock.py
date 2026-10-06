import json
import shutil

from pdpl_eval.config import select
from pdpl_eval.consolidate import consolidate
from pdpl_eval.models import MockModel
from pdpl_eval.preflight import longest_policy, preflight_full_rag
from pdpl_eval.report import build_completion_report
from pdpl_eval.runner import run_experiments
from pdpl_eval.storage import raw_path
from pdpl_eval.validation import validate_tree
from conftest import ROOT


def test_full_matrix_resume_force_and_consolidation(cfg):
    sel = select(cfg, runs=2)                                  # 2 apps x 2 models x 4 strategies x 2 runs
    s = run_experiments(cfg, sel)
    assert len(s["completed"]) == 32 and not s["failed"] and not s["blocked"] and not s["save_errors"]
    assert validate_tree(cfg.results_dir)["valid_success"] == 32
    # resume: nothing re-executed
    s2 = run_experiments(cfg, sel)
    assert len(s2["skipped"]) == 32 and not s2["completed"]
    # interrupted-run simulation: delete one result -> only that one re-runs
    p = raw_path(cfg.results_dir, "demo_long", "mock-b", "full_rag", 2)
    p.unlink()
    s3 = run_experiments(cfg, sel)
    assert len(s3["completed"]) == 1 and len(s3["skipped"]) == 31
    # force archives the old file instead of overwriting
    s4 = run_experiments(cfg, select(cfg, ["demo_short"], ["mock-a"], ["full_norag"], runs=1), force=True)
    assert len(s4["completed"]) == 1
    assert len(list((cfg.results_dir / "archive").rglob("run_01__*.json"))) == 1
    rep = build_completion_report(cfg, sel)
    assert rep["counts"]["completed"] == 32 and rep["counts"]["missing"] == 0
    c = consolidate(cfg.results_dir)
    assert c["runs"] == 32 and c["prediction_rows"] == 32 * 17
    # sections are retained for later OR/AND/majority/threshold: short=4, long=15 sections
    assert c["section_prediction_rows"] == 17 * 2 * 2 * 2 * (4 + 15)   # models x section strategies x runs x sections
    assert (cfg.results_dir / "consolidated" / "all_predictions.csv").exists()


def test_only_enabled_strategies_and_filters(cfg):
    sel = select(cfg, ["demo_short"], ["mock-a"], ["section_norag", "full_rag"], run_numbers=[3])
    s = run_experiments(cfg, sel)
    assert len(s["completed"]) == 2
    assert raw_path(cfg.results_dir, "demo_short", "mock-a", "full_rag", 3).exists()


def test_transient_failure_is_retried(cfg):
    sel = select(cfg, ["demo_short"], ["mock-a"], ["full_norag"], runs=1)
    s = run_experiments(cfg, sel, model_factory=lambda sp, r: MockModel(sp, r.codes, fail_first_n=2))
    assert len(s["completed"]) == 1
    rec = json.load(open(raw_path(cfg.results_dir, "demo_short", "mock-a", "full_norag", 1), encoding="utf-8"))
    assert rec["attempts"] == 3


def test_permanent_failure_recorded_and_pipeline_continues(cfg):
    sel = select(cfg, ["demo_short"], None, ["full_norag"], runs=1)
    def factory(sp, r):
        return MockModel(sp, r.codes, permanent_fail=(sp.id == "mock-a"))
    s = run_experiments(cfg, sel, model_factory=factory)
    assert len(s["failed"]) == 1 and len(s["completed"]) == 1          # mock-b still ran
    rec = json.load(open(raw_path(cfg.results_dir, "demo_short", "mock-a", "full_norag", 1), encoding="utf-8"))
    assert rec["status"] == "failed" and rec["error"]["attempts"] == 3
    # failed record is not 'completed': resume retries it (old record archived)
    s2 = run_experiments(cfg, sel)
    assert len(s2["completed"]) == 1 and len(s2["skipped"]) == 1
    assert list((cfg.results_dir / "archive").rglob("run_01__*.json"))


def test_unparseable_output_is_parse_failed(cfg):
    sel = select(cfg, ["demo_short"], ["mock-a"], ["full_norag"], runs=1)
    s = run_experiments(cfg, sel, model_factory=lambda sp, r: MockModel(sp, r.codes, garbage=True))
    assert len(s["parse_failed"]) == 1


def test_context_limit_never_truncates(cfg):
    from dataclasses import replace
    cfg.models = [replace(m, context_limit_tokens=500) for m in cfg.models]
    sel = select(cfg, ["demo_long"], ["mock-a"], ["full_rag"], runs=1)
    s = run_experiments(cfg, sel)
    rec = json.load(open(raw_path(cfg.results_dir, "demo_long", "mock-a", "full_rag", 1), encoding="utf-8"))
    assert rec["status"] == "failed" and rec["error"]["type"] == "ContextLimitExceeded"
    assert rec["error"]["attempts"] == 1                                 # not retried
    pf = preflight_full_rag(cfg, cfg.applications[1], cfg.models[0])
    assert pf["fits_context"] is False and "truncated" in pf["reason"]


def test_missing_policy_blocks_but_continues(cfg):
    from pdpl_eval.config import Application
    cfg.applications = [Application("Ghost", "ghost", cfg.applications[0].policy_path.parent / "nope.docx"), cfg.applications[0]]
    sel = select(cfg, None, ["mock-a"], ["full_norag"], runs=1)
    s = run_experiments(cfg, sel)
    assert len(s["blocked"]) == 1 and len(s["completed"]) == 1


def test_longest_policy(cfg):
    assert longest_policy(cfg).id == "demo_long"
