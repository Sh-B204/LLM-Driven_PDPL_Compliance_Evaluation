from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple


def parse_llm_response(text: str, rubric_items: List[str]) -> Dict[str, Optional[int]]:
    """Extract one binary answer per criterion code; reject duplicate or invalid answers."""
    predictions: Dict[str, Optional[int]] = {}
    codes = [item.split(". ")[0] for item in rubric_items]
    pattern = r"(?<![A-Za-z0-9])(" + "|".join(re.escape(code) for code in codes) + r")(?![A-Za-z0-9])"
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    for item, code in zip(rubric_items, codes):
        answers = []
        for i, match in enumerate(matches):
            if match.group(1).upper() != code.upper():
                continue
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            line = text[match.end():end].split("\n")[0].strip()
            value = re.match(r"^(?:[:=\-]\s*|\.\s*[^\n]*?\s[-:=]\s*)([01])(?![\w.])", line)
            answers.append(int(value.group(1)) if value else None)
        predictions[item] = answers[0] if len(answers) == 1 else None
    return predictions


def parse_section_vector(raw: str, rubric_items: List[str]) -> Tuple[Optional[List[int]], dict]:
    """Return predictions in rubric order, without padding or truncation."""
    predictions, meta = parse_full_response(raw, rubric_items)
    meta.update({"n_values_found": meta["n_parsed"], "padded_zeros": 0, "truncated_values": 0})
    return ([predictions[item] for item in rubric_items] if predictions is not None else None), meta


def parse_full_response(raw: str, rubric_items: List[str]) -> Tuple[Optional[Dict[str, Optional[int]]], dict]:
    """Accept only one valid binary answer for every supplied criterion."""
    predictions = parse_llm_response(raw, rubric_items)
    missing = [item.split(". ")[0] for item, value in predictions.items() if value is None]
    meta = {"parse_method": "code_anchored", "n_parsed": len(rubric_items) - len(missing),
            "expected": len(rubric_items), "missing_or_invalid_codes": missing, "partial_predictions": predictions}
    return (predictions if not missing else None), meta
