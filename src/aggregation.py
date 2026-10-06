"""Aggregation of per-section prediction vectors. The pipeline's primary rule is 'or'
(the notebook's rule: numpy max over sections). Others are provided so they can be computed
later from saved per-section predictions without re-running any model."""
from __future__ import annotations

from typing import List, Optional


def aggregate_or(vectors: List[List[int]]) -> List[int]:
    return [int(max(col)) for col in zip(*vectors)]          # == np.array(...).max(axis=0)


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
    if rule == "or":
        return aggregate_or(vectors)
    if rule == "and":
        return aggregate_and(vectors)
    if rule == "majority":
        return aggregate_majority(vectors)
    if rule == "threshold":
        if threshold is None:
            raise ValueError("threshold rule needs a threshold value")
        return aggregate_threshold(vectors, threshold)
    raise ValueError(f"Unknown aggregation rule: {rule}")
