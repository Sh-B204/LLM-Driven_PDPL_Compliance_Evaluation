"""Prompts, parsers and corpus parsing must match the notebook's own code (executed from the notebook)."""
import json
import re

import pytest

from conftest import ROOT
from pdpl_eval.parsing import parse_llm_response
from pdpl_eval.prompts import load_prompt, render_full_prompt, render_section_prompt
from pdpl_eval.rubric import load_rubric

NB = ROOT / "notebooks" / "PDPL_Evaluate_with_Arabic.ipynb"
pytestmark = pytest.mark.skipif(not NB.exists(), reason="notebook not present")


@pytest.fixture(scope="module")
def nb_ns():
    cells = json.load(open(NB, encoding="utf-8"))["cells"]
    src = lambda i: "".join(cells[i]["source"])
    ns = {"re": re, "Dict": dict, "List": list}
    exec(src(6).split("# ── Rubric ──")[1], ns)                      # rubric_items, rubric_text
    s10 = src(10)
    exec(s10[s10.index("def parse_llm_response"):s10.index("def evaluate")], ns)
    exec(s10[s10.index("def build_section_query"):s10.index('print("✓ Helpers ready")')], ns)
    s91 = src(91)
    exec(s91[s91.index("def build_full_policy_grounded_prompt"):s91.index("if 'exp1_results'")], ns)
    return ns


def test_rubric_identical(nb_ns):
    r = load_rubric(ROOT / "rubric/rubric_items.txt")
    assert r.items == nb_ns["rubric_items"] and r.text == nb_ns["rubric_text"]


def test_prompts_identical(nb_ns):
    r = load_rubric(ROOT / "rubric/rubric_items.txt")
    sec = {"title": "Data sharing", "content": " We share data. \u0645\u062b\u0627\u0644"}
    t = load_prompt("section_rag", ROOT / "prompts/section_rag.txt")
    assert render_section_prompt(t, r.text, sec["title"], sec["content"], "PDPL-X") == nb_ns["build_section_query"](sec, "PDPL-X")
    t = load_prompt("full_norag", ROOT / "prompts/full_policy.txt")
    assert render_full_prompt(t, r.text, "FULL TEXT") == nb_ns["build_full_policy_prompt"]("FULL TEXT")
    t = load_prompt("full_rag", ROOT / "prompts/full_policy_rag.txt")
    assert render_full_prompt(t, r.text, "FULL TEXT", "CTX") == nb_ns["build_full_policy_grounded_prompt"]("FULL TEXT", "CTX")


def test_section_norag_prompt_is_rag_prompt_minus_excerpts_block(nb_ns):
    r = load_rubric(ROOT / "rubric/rubric_items.txt")
    sec = {"title": "T", "content": "C"}
    rag = nb_ns["build_section_query"](sec, "EXCERPT")
    norag = render_section_prompt(load_prompt("n", ROOT / "prompts/section_no_rag.txt"), r.text, "T", "C")
    assert norag == rag.replace("PDPL Excerpts:\nEXCERPT\n\n", "") and "PDPL Excerpts" not in norag


def test_parser_identical(nb_ns):
    r = load_rubric(ROOT / "rubric/rubric_items.txt")
    for txt in ["A1: 1\nA2: 0\nB7 - 1\nD2 = 0", "**C1**: 1\nc2 -> 0\nA10: 1", "nothing", "A1 maybe 0 or 1"]:
        assert parse_llm_response(txt, r.items) == nb_ns["parse_llm_response"](txt)
