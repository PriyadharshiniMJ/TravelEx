"""Retrying calls that failed for a passing reason (rate limit, server error, network). Nodes that fail soft catch
every exception to keep the trip going, so LangGraph's RetryPolicy never sees these: the retry has to happen inside,
around the model or search call, before the node gives up."""
import os
import time
from typing import Callable, TypeVar

T = TypeVar("T")
# Rate limits: Tavily raises UsageLimitExceededError for HTTP 429 only (running out of plan credits is a different
# error, and is not retried); OpenAI raises RateLimitError. They clear in seconds, so they wait longer.
RATE_LIMIT_NAMES = {"RateLimitError", "UsageLimitExceededError"}
TRANSIENT_NAMES = {"APIConnectionError", "APITimeoutError", "ConnectError", "ConnectTimeout", "ReadTimeout",
                   "RemoteProtocolError", "InternalServerError", "TimeoutError", *RATE_LIMIT_NAMES}
ATTEMPTS = 3
FIRST_WAIT_SECONDS = 1.0
RATE_LIMIT_WAIT_SECONDS = 5.0  # then 10 s: long enough for a per-minute limit to ease


def is_transient(exc: BaseException) -> bool:
    """Retry rate limits, 5xx and network errors; never 4xx, validation errors or bugs."""
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    return isinstance(exc, (ConnectionError, TimeoutError)) or type(exc).__name__ in TRANSIENT_NAMES


def call_with_retry(fn: Callable[[], T], attempts: int = ATTEMPTS) -> T:
    """fn(), tried again after 1 s, 2 s… when it fails for a passing reason. Anything else is raised at once."""
    scale = os.getenv("TRAVELEX_RETRY_WAIT")  # tests set 0: retries happen at once
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            if attempt == attempts - 1 or not is_transient(e):
                raise
            first = RATE_LIMIT_WAIT_SECONDS if type(e).__name__ in RATE_LIMIT_NAMES else FIRST_WAIT_SECONDS
            time.sleep((float(scale) if scale is not None else first) * 2 ** attempt)
    raise AssertionError("unreachable")
