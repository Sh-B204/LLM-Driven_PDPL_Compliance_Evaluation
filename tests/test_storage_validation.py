import json
import pytest

from pdpl_eval.config import load_config, select
from pdpl_eval.runner import run_experiments
from pdpl_eval.storage import atomic_write_json, raw_path, save_result
from pdpl_eval.validation import check_file, validate_tree
from conftest import ROOT


def _one(cfg, strategy="full_norag"):
    sel = select(cfg, ["demo_short"], ["mock-a"], [strategy], runs=1)
    s = run_experiments(cfg, sel)
    return sel, s, raw_path(cfg.results_dir, "demo_short", "mock-a", strategy, 1)


def test_atomic_write_leaves_no_tmp(tmp_path):
    atomic_write_json(tmp_path / "a" / "x.json", {"k": "\u0639\u0631\u0628\u064a"})
    assert json.load(open(tmp_path / "a" / "x.json", encoding="utf-8")) == {"k": "\u0639\u0631\u0628\u064a"}
    assert not list(tmp_path.rglob("*.tmp"))


def test_never_overwrites_without_flag(cfg):
    sel, s, p = _one(cfg)
    rec = json.load(open(p, encoding="utf-8"))
    with pytest.raises(FileExistsError):
        save_result(cfg.results_dir, p, rec)


def test_tamper_detected(cfg):
    sel, s, p = _one(cfg)
    rec = json.load(open(p, encoding="utf-8"))
    rec["prediction"][list(rec["prediction"])[0]] = 1 - rec["prediction"][list(rec["prediction"])[0]]
    p.write_text(json.dumps(rec), encoding="utf-8")
    _, problems = check_file(p)
    assert any("integrity" in x for x in problems)
    assert validate_tree(cfg.results_dir)["invalid"]


def test_section_record_has_every_section_and_consistent_or(cfg):
    sel, s, p = _one(cfg, "section_rag")
    rec = json.load(open(p, encoding="utf-8"))
    assert len(rec["section_predictions"]) == rec["policy"]["n_sections"] == 4
    assert all(sp["retrieval"]["top_k"] == 3 and len(sp["retrieval"]["retrieved"]) == 3 for sp in rec["section_predictions"])
    assert check_file(p)[1] == []
