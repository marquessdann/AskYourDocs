import hashlib
import threading
import time
from collections import OrderedDict, deque

from app.core.exceptions import RateLimitExceededError
from app.schemas import ChatResponse


def _normalize(question: str) -> str:
    return " ".join(question.strip().lower().split())


def _cache_key(question: str) -> str:
    return hashlib.sha256(_normalize(question).encode("utf-8")).hexdigest()


class TTLCache:
    """Small in-memory LRU+TTL cache for repeated questions, so an identical prompt
    doesn't re-spend embedding + LLM tokens. Good enough for a single-process demo;
    a multi-instance deployment would swap this for Redis without touching call sites."""

    def __init__(self, max_size: int, ttl_seconds: int):
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._store: OrderedDict[str, tuple[float, ChatResponse]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, question: str) -> ChatResponse | None:
        key = _cache_key(question)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, response = entry
            if time.monotonic() > expires_at:
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return response

    def set(self, question: str, response: ChatResponse) -> None:
        key = _cache_key(question)
        with self._lock:
            self._store[key] = (time.monotonic() + self._ttl_seconds, response)
            self._store.move_to_end(key)
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)


class SlidingWindowRateLimiter:
    """Caps requests per client key (e.g. IP) within a rolling time window."""

    def __init__(self, max_requests: int, window_seconds: int):
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, client_key: str) -> None:
        now = time.monotonic()
        with self._lock:
            hits = self._hits.setdefault(client_key, deque())
            while hits and now - hits[0] > self._window_seconds:
                hits.popleft()

            if len(hits) >= self._max_requests:
                retry_after = self._window_seconds - (now - hits[0])
                raise RateLimitExceededError(retry_after_seconds=max(retry_after, 0))

            hits.append(now)
