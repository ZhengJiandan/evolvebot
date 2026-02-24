"""Simple in-memory rate limiter (token bucket)."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class TokenBucket:
    rate_per_sec: float
    capacity: float
    tokens: float
    updated_at: float
    last_seen: float

    @staticmethod
    def create(rate_per_min: int, burst: int | None = None) -> "TokenBucket":
        rate_per_min = max(1, int(rate_per_min))
        rate_per_sec = rate_per_min / 60.0
        capacity = float(burst if burst is not None else rate_per_min)
        now = time.monotonic()
        return TokenBucket(rate_per_sec, capacity, capacity, now, now)

    def allow(self, cost: float = 1.0) -> bool:
        now = time.monotonic()
        elapsed = now - self.updated_at
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_per_sec)
            self.updated_at = now
        self.last_seen = now
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False


class RateLimiter:
    def __init__(self, rate_per_min: int, burst: int | None = None, idle_ttl_s: int = 300) -> None:
        self.rate_per_min = max(1, int(rate_per_min))
        self.burst = burst if burst is not None else self.rate_per_min
        self.idle_ttl_s = max(60, int(idle_ttl_s))
        self._buckets: dict[str, TokenBucket] = {}
        self._last_cleanup = time.monotonic()

    def allow(self, key: str, cost: float = 1.0) -> bool:
        self._cleanup_if_needed()
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = TokenBucket.create(self.rate_per_min, self.burst)
            self._buckets[key] = bucket
        return bucket.allow(cost)

    def _cleanup_if_needed(self) -> None:
        now = time.monotonic()
        if now - self._last_cleanup < self.idle_ttl_s:
            return
        self._last_cleanup = now
        cutoff = now - self.idle_ttl_s
        dead = [k for k, b in self._buckets.items() if b.last_seen < cutoff]
        for k in dead:
            self._buckets.pop(k, None)
