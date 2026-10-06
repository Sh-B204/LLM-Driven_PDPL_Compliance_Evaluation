from pdpl_eval.aggregation import aggregate
from pdpl_eval.parsing import parse_full_response, parse_llm_response, parse_section_vector
from pdpl_eval.rubric import load_rubric
from conftest import ROOT

R = load_rubric(ROOT / "rubric/rubric_items.txt")


def test_rubric_is_notebook_rubric():
    assert len(R) == 17 and R.items[0] == "A1. Right to access data" and R.items[-1].startswith("D2.")
    assert R.codes[:2] == ["A1", "A2"]


def test_code_anchored_formats():
    txt = "**A1**: 1\nA2 - 0\nA3=1\nA4 → 0\nB1: 1\nA10 1"
    p = parse_llm_response(txt, R.items)
    assert p[R.items[0]] == 1 and p[R.items[1]] == 0 and p[R.items[2]] == 1 and p[R.items[3]] == 0
    assert p[R.items[5]] == 1 and p[R.items[4]] is None          # A5 absent -> None


def test_section_vector_pad_and_truncate():
    v, m = parse_section_vector("A1: 1\nA2: 0", 17)
    assert len(v) == 17 and v[:2] == [1, 0] and m["padded_zeros"] == 15
    v, m = parse_section_vector("\n".join(f"x: 1" for _ in range(20)), 17)
    assert len(v) == 17 and m["truncated_values"] == 3


def test_full_fallback_last_n_and_unparsed():
    raw = "thinking 1 then 0\n" + "\n".join(["1", "0"] * 9)
    preds, meta = parse_full_response(raw, R.items)
    assert meta["parse_method"] == "fallback_last_n_values" and list(preds.values()) == ([1, 0] * 9)[-17:]
    preds, meta = parse_full_response("no numbers here", R.items)
    assert preds is None and meta["parse_method"] == "unparsed"


def test_aggregations():
    v = [[1, 0, 0], [0, 0, 1], [1, 0, 1]]
    assert aggregate(v, "or") == [1, 0, 1]
    assert aggregate(v, "and") == [0, 0, 0]
    assert aggregate(v, "majority") == [1, 0, 1]
    assert aggregate(v, "threshold", 0.34) == [1, 0, 1] and aggregate(v, "threshold", 1.0) == [0, 0, 0]
