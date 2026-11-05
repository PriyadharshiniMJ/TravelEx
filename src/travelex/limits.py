"""Run-wide limits: one trip's research shares a budget of searches, model tokens and time across every parallel
branch and nested subgraph, so a looping agent or a huge party can't run up an unbounded bill. When a limit is
reached, research stops (searches fail soft and agents stop asking for more); writing the plan carries on."""
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langgraph.config import get_config

KEPT_RUNS = 200  # limits are kept per thread, so a resumed run (after an interrupt) keeps counting


class LimitReached(RuntimeError):
    pass


@dataclass
class RunLimits:
    max_searches: int = 150
    max_tokens: int = 3_000_000
    max_seconds: float = 1200
    searches: int = 0
    tokens: int = 0
    started: float = field(default_factory=time.monotonic)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def from_env(cls) -> "RunLimits":
        env = lambda name, default, kind: kind(os.getenv(name) or default)
        return cls(max_searches=env("TRAVELEX_MAX_SEARCHES", cls.max_searches, int),
                   max_tokens=env("TRAVELEX_MAX_TOKENS", cls.max_tokens, int),
                   max_seconds=env("TRAVELEX_MAX_SECONDS", cls.max_seconds, float))

    def reached(self) -> str | None:
        """Which limit the run has reached, in words, or None."""
        if self.searches >= self.max_searches:
            return f"{self.max_searches} searches"
        if self.tokens >= self.max_tokens:
            return f"{self.max_tokens:,} tokens"
        if time.monotonic() - self.started >= self.max_seconds:
            return f"{self.max_seconds:.0f} seconds"
        return None

    def spend_search(self) -> None:
        """Count one search or page fetch, or raise LimitReached if the run can't afford another."""
        with self._lock:
            if why := self.reached():
                raise LimitReached(f"Run limit reached ({why}): research stopped.")
            self.searches += 1

    def add_tokens(self, n: int) -> None:
        with self._lock:
            self.tokens += n


_runs: "OrderedDict[str, RunLimits]" = OrderedDict()
_runs_lock = threading.Lock()


def limits_for(thread_id: str) -> RunLimits:
    """The limits shared by every step of one run (one checkpoint thread)."""
    with _runs_lock:
        if thread_id not in _runs:
            _runs[thread_id] = RunLimits.from_env()
            while len(_runs) > KEPT_RUNS:
                _runs.popitem(last=False)
        _runs.move_to_end(thread_id)
        return _runs[thread_id]


def current() -> RunLimits | None:
    """The limits of the run this code is part of; None outside a graph run (a unit test, a script)."""
    try:
        limits = get_config().get("configurable", {}).get("limits")
    except RuntimeError:
        return None
    return limits if isinstance(limits, RunLimits) else None


def research_stopped() -> str | None:
    """Why research must stop now, or None to go on."""
    limits = current()
    return limits.reached() if limits else None


class TokenCounter(BaseCallbackHandler):
    """Adds every model call's token usage to the run's limits."""

    def __init__(self, limits: RunLimits):
        self.limits = limits

    def on_llm_end(self, response: Any, **kwargs) -> None:
        for generations in response.generations:
            for g in generations:
                usage = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
                self.limits.add_tokens(usage.get("total_tokens", 0))
