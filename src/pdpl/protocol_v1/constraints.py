"""A finite-state output schema: rubric labels are fixed; binary values are chosen by the model."""
from __future__ import annotations

import re


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
        """Accept a valid prefix; merge the two line branches before the next criterion."""
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
