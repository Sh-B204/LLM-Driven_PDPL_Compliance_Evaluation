from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple 


class BinaryRubricSchema:
    protocol = "binary_rubric_fsm_v1"

    def __init__(self, tokenizer, codes):
        self.codes = list(codes)
        if not self.codes or len(set(self.codes)) != len(self.codes) or any(
                not re.fullmatch(r"[A-Z][1-9][0-9]*", c) for c in self.codes):
            raise ValueError("Output schema requires unique rubric codes")
        self.eos_token_id = tokenizer.eos_token_id
        if not isinstance(self.eos_token_id, int):
            raise ValueError("Output schema requires a tokenizer EOS token")
        self.lines = []
        for code in self.codes:
            alternatives = [tuple(tokenizer.encode(f"{code}: {value}\n", add_special_tokens=False))
                            for value in (0, 1)]
            if any(not a or self.eos_token_id in a for a in alternatives):
                raise ValueError(f"Invalid schema tokenization for {code}")
            a, b = alternatives
            if a == b or a == b[:len(a)] or b == a[:len(b)]:
                raise ValueError(f"Binary alternatives must be distinct and prefix-free for {code}")
            self.lines.append(alternatives)
        self.max_tokens = sum(max(map(len, line)) for line in self.lines) + 1

    def allowed_tokens(self, generated):
        remaining = tuple(generated)
        for alternatives in self.lines:
            completed = next((a for a in alternatives if remaining[:len(a)] == a), None)
            if completed is not None:
                remaining = remaining[len(completed):]
                continue
            compatible = [a for a in alternatives if a[:len(remaining)] == remaining]
            if not compatible:
                raise ValueError("Generated tokens do not match the rubric output schema")
            return sorted({a[len(remaining)] for a in compatible})
        if remaining:
            raise ValueError("Unexpected tokens after the final rubric answer")
        return [self.eos_token_id]

    def callback(self, prompt_length):
        return lambda batch_id, input_ids: self.allowed_tokens(input_ids[prompt_length:].tolist())

    def is_complete(self, generated):
        tokens = list(generated)
        if tokens and tokens[-1] == self.eos_token_id:
            tokens.pop()
        return self.allowed_tokens(tokens) == [self.eos_token_id]


def parse_llm_response(text: str, rubric_items: List[str]) -> Dict[str, Optional[int]]:
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
    predictions, meta = parse_full_response(raw, rubric_items)
    meta.update({"n_values_found": meta["n_parsed"], "padded_zeros": 0, "truncated_values": 0})
    return ([predictions[item] for item in rubric_items] if predictions is not None else None), meta


def parse_full_response(raw: str, rubric_items: List[str]) -> Tuple[Optional[Dict[str, Optional[int]]], dict]:
    predictions = parse_llm_response(raw, rubric_items)
    missing = [item.split(". ")[0] for item, value in predictions.items() if value is None]
    meta = {"parse_method": "code_anchored", "n_parsed": len(rubric_items) - len(missing),
            "expected": len(rubric_items), "missing_or_invalid_codes": missing, "partial_predictions": predictions}
    return (predictions if not missing else None), meta


def aggregate_or(vectors: List[List[int]]) -> List[int]:
    return [int(max(col)) for col in zip(*vectors)]


def aggregate_and(vectors: List[List[int]]) -> List[int]:
    return [int(min(col)) for col in zip(*vectors)]


def aggregate_majority(vectors: List[List[int]]) -> List[int]:
    n = len(vectors)
    return [int(2 * sum(col) > n) for col in zip(*vectors)]


def aggregate_threshold(vectors: List[List[int]], threshold: float) -> List[int]:
    n = len(vectors)
    return [int(sum(col) / n >= threshold) for col in zip(*vectors)]


def aggregate(vectors: List[List[int]], rule: str = "or", threshold: Optional[float] = None) -> List[int]:
    if not vectors:
        raise ValueError("No section vectors to aggregate")
    n = len(vectors[0])
    if not n or any(len(v) != n or any(type(x) is not int or x not in (0, 1) for x in v) for v in vectors):
        raise ValueError("Section vectors must have equal lengths and contain only integer 0/1 values")
    if rule == "or":
        return aggregate_or(vectors)
    if rule == "and":
        return aggregate_and(vectors)
    if rule == "majority":
        return aggregate_majority(vectors)
    if rule == "threshold":
        if threshold is None or not 0 < threshold <= 1:
            raise ValueError("threshold must satisfy 0 < threshold <= 1")
        return aggregate_threshold(vectors, threshold)
    raise ValueError(f"Unknown aggregation rule: {rule}")
