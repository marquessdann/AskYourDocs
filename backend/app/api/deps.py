from functools import lru_cache

from fastapi import Request

from app.cache.query_cache import SlidingWindowRateLimiter
from app.config import settings


@lru_cache
def get_rate_limiter() -> SlidingWindowRateLimiter:
    return SlidingWindowRateLimiter(
        max_requests=settings.rate_limit_requests,
        window_seconds=settings.rate_limit_window_seconds,
    )


def enforce_rate_limit(request: Request) -> None:
    client_key = request.client.host if request.client else "unknown"
    get_rate_limiter().check(client_key)
