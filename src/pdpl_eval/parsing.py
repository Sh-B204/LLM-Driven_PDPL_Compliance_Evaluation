from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple


def parse_llm_response(text: str, rubric_items: List[str]) -> Dict[str, Optional[int]]:
    """parse_llm_response Code-anchored extraction first; falls back to first isolated 0/1 on a line mentioning the code.
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


def parse_section_vector(raw: str, n_items: int) -> Tuple[Optional[List[int]], dict]:
    """line-based extraction; require exactly n_items values, without padding or truncation."""
    vals: List[int] = []
    for line in raw.splitlines():
        match = re.search(r"\b([01])\b", line)
        if match:
            vals.append(int(match.group(1)))
    meta = {"n_values_found": len(vals), "expected": n_items, "padded_zeros": 0, "truncated_values": 0}
    return (vals if len(vals) == n_items else None), meta


def parse_full_response(raw: str, rubric_items: List[str]) -> Tuple[Optional[Dict[str, Optional[int]]], dict]:
    """code-anchored extraction; accept only complete predictions or an exact-length fallback."""
    n = len(rubric_items)
    preds = parse_llm_response(raw, rubric_items)
    parsed_count = sum(1 for v in preds.values() if v is not None)
    if parsed_count > 0:
        missing = [item.split(". ")[0] for item, value in preds.items() if value is None]
        return (preds if parsed_count == n else None), {"parse_method": "code_anchored", "n_parsed": parsed_count,
                                                       "expected": n, "missing_codes": missing, "partial_predictions": preds}
    vals = [int(m.group(1)) for line in raw.splitlines() if (m := re.search(r"\b([01])\b", line))]
    if len(vals) == n:
        return ({item: vals[i] for i, item in enumerate(rubric_items)},
                {"parse_method": "fallback_exact_n_values", "n_parsed": n, "n_values_found": len(vals), "expected": n})
    return None, {"parse_method": "unparsed", "n_parsed": 0, "n_values_found": len(vals), "expected": n}
