"""Response parsers - logic copied unchanged from the notebook (cell 10 / cell 15)."""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


def parse_llm_response(text: str, rubric_items: List[str]) -> Dict[str, Optional[int]]:
    """Notebook parse_llm_response (cell 10 definition, which overrides cell 8's).

    Code-anchored extraction first; falls back to first isolated 0/1 on a line mentioning the code.
    """
    predictions: Dict[str, Optional[int]] = {}
    lines = text.strip().split("\n")
    for item in rubric_items:
        code = item.split(". ")[0]
        predictions[item] = None
        for line in lines:
            if not re.search(rf"(?<![A-Za-z0-9]){re.escape(code)}(?![A-Za-z0-9])", line, re.IGNORECASE):
                continue
            m = re.search(
                rf"(?<![A-Za-z0-9]){re.escape(code)}(?![A-Za-z0-9])[^\n]*?\b([01])\b",
                line, re.IGNORECASE,
            )
            if m:
                predictions[item] = int(m.group(1))
                break
            m2 = re.search(r"\b([01])\b", line)
            if m2:
                predictions[item] = int(m2.group(1))
                break
    return predictions


def parse_section_vector(raw: str, n_items: int) -> Tuple[List[int], dict]:
    """Section-by-section parser (notebook cell 15): first isolated 0/1 on each line, in order;
    pad with 0 if too few values, truncate if too many. Returns (vector, parse_meta)."""
    vals: List[int] = []
    for line in raw.splitlines():
        match = re.search(r"\b([01])\b", line)
        if match:
            vals.append(int(match.group(1)))
    n_found = len(vals)
    padded = truncated = 0
    if len(vals) < n_items:
        padded = n_items - len(vals)
        vals += [0] * padded
    elif len(vals) > n_items:
        truncated = len(vals) - n_items
        vals = vals[:n_items]
    return vals, {"n_values_found": n_found, "padded_zeros": padded, "truncated_values": truncated}


def parse_full_response(raw: str, rubric_items: List[str]) -> Tuple[Optional[Dict[str, Optional[int]]], dict]:
    """Full-policy parser (notebook cell 15). Returns (predictions or None, parse_meta).

    1. parse_llm_response. 2. If nothing parsed: take the LAST n 0/1 values of the output.
    3. Else unparsed (notebook skipped these results; here the run is recorded as 'parse_failed').
    """
    n = len(rubric_items)
    preds = parse_llm_response(raw, rubric_items)
    parsed_count = sum(1 for v in preds.values() if v is not None)
    if parsed_count > 0:
        return preds, {"parse_method": "code_anchored", "n_parsed": parsed_count}
    vals = [int(m.group(1)) for line in raw.splitlines() if (m := re.search(r"\b([01])\b", line))]
    if len(vals) >= n:
        used = vals[-n:]
        return ({item: used[i] for i, item in enumerate(rubric_items)},
                {"parse_method": "fallback_last_n_values", "n_parsed": n, "n_values_found": len(vals)})
    return None, {"parse_method": "unparsed", "n_parsed": 0, "n_values_found": len(vals)}
