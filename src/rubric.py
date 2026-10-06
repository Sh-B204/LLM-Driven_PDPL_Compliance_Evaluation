"""Rubric loading. The rubric text and labels are read verbatim from rubric/rubric_items.txt."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from .utils import sha256_text


@dataclass(frozen=True)
class Rubric:
    items: List[str]          # e.g. "A1. Right to access data"
    text: str                 # "\n".join(items)  (notebook: rubric_text)
    sha256: str

    @property
    def codes(self) -> List[str]:
        return [i.split(". ")[0] for i in self.items]     # notebook: item.split(". ")[0]

    def __len__(self) -> int:
        return len(self.items)


def load_rubric(path: str | Path) -> Rubric:
    raw = Path(path).read_text(encoding="utf-8")
    items = [ln.rstrip("\r") for ln in raw.split("\n") if ln.strip()]
    if not items:
        raise ValueError(f"Rubric file is empty: {path}")
    for it in items:
        if ". " not in it:
            raise ValueError(f"Rubric item without 'CODE. description' form: {it!r}")
    text = "\n".join(items)
    return Rubric(items=items, text=text, sha256=sha256_text(text))
