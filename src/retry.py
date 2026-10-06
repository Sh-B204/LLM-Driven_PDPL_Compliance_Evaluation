"""Retry with exponential backoff. Permanent errors are never retried."""
from __future__ import annotations

import random
import time
from typing import Callable, Tuple


class PermanentError(Exception):
    """Failure that retrying cannot fix (e.g. request exceeds the context window)."""


class TransientError(Exception):
    """Temporary failure (OOM, I/O, network); retried with backoff."""


def is_transient(exc: BaseException) -> bool:
    if isinstance(exc, PermanentError):
        return False
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        return False
    return True      # unknown errors get the retry budget, then are recorded as permanent failures


def backoff_delay(attempt: int, cfg: dict) -> float:
    """attempt is 1-based (the attempt that just failed)."""
    delay = cfg["base_delay_seconds"] * (cfg["backoff_factor"] ** (attempt - 1))
    delay = min(delay, cfg["max_delay_seconds"])
    jitter = cfg.get("jitter_seconds", 0) or 0
    return delay + (random.uniform(0, jitter) if jitter else 0.0)


def retry_call(fn: Callable, cfg: dict, sleep: Callable[[float], None] = time.sleep,
               on_retry: Callable[[int, BaseException, float], None] | None = None) -> Tuple[object, int]:
    """Call fn() up to cfg['max_attempts'] times. Returns (result, attempts_used); raises the last error."""
    max_attempts = int(cfg["max_attempts"])
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn(), attempt
        except BaseException as exc:      # noqa: BLE001 - classified below
            if not is_transient(exc) or attempt >= max_attempts:
                exc.attempts = attempt        # type: ignore[attr-defined]
                raise
            delay = backoff_delay(attempt, cfg)
            if on_retry:
                on_retry(attempt, exc, delay)
            sleep(delay)
