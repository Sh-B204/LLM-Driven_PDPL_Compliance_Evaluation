from __future__ import annotations
from typing import List, Optional


def aggregate_or(vectors: List[List[int]]) -> List[int]:
    return [int(max(col)) for col in zip(*vectors)]


def aggregate_and(vectors: List[List[int]]) -> List[int]:
    return [int(min(col)) for col in zip(*vectors)]


def aggregate_majority(vectors: List[List[int]]) -> List[int]:
    """1 when strictly more than half of the sections predict 1."""
    n = len(vectors)
    return [int(2 * sum(col) > n) for col in zip(*vectors)]


def aggregate_threshold(vectors: List[List[int]], threshold: float) -> List[int]:
    """1 when the fraction of sections predicting 1 is >= threshold (0 < threshold <= 1)."""
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
