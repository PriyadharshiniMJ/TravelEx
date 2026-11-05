"""Tavily wrappers: trimmed results, cached on disk under .cache/tavily/.

Cached pages expire: a search or page older than TRAVELEX_SEARCH_TTL_DAYS (default 7) is fetched again, so a visa or
fee rule that expires is rebuilt from fresh pages, not the same old ones. TRAVELEX_REFRESH=1 ignores the cache."""
import hashlib
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from langchain_core.runnables.config import ContextThreadPoolExecutor
from langchain_core.tools import tool
from langsmith import traceable
from tavily import TavilyClient

from travelex.limits import current as current_limits
from travelex.retry import call_with_retry

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = Path(os.getenv("TRAVELEX_CACHE_DIR", PROJECT_ROOT / ".cache" / "tavily"))
SNIPPET_CHARS = 1200
EXTRACT_CHARS = 4000
MAX_RESULTS = 5
MAX_EXTRACT_URLS = 5
SEARCH_TTL_DAYS = 7

_client: TavilyClient | None = None


class SearchConfigError(RuntimeError):
    pass


def _tavily() -> TavilyClient:
    global _client
    if _client is None:
        key = os.getenv("TAVILY_API_KEY", "").strip()
        if not key:
            raise SearchConfigError("TAVILY_API_KEY is not set. Add it to .env.")
        _client = TavilyClient(api_key=key)
    return _client


def ttl_days() -> float:
    return float(os.getenv("TRAVELEX_SEARCH_TTL_DAYS") or SEARCH_TTL_DAYS)


def fresh(data: dict, max_age_days: float | None) -> bool:
    """Fetched within max_age_days (None: never expires)."""
    if max_age_days is None:
        return True
    fetched = datetime.fromisoformat(data["retrieved_at"])
    fetched = fetched if fetched.tzinfo else fetched.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - fetched <= timedelta(days=max_age_days)


def _cached(kind: str, params: dict, call, max_age_days: float | None = None) -> dict:
    """Return {"retrieved_at", "response"}, calling Tavily on a cache miss, when the cached copy is older than
    max_age_days, or always when TRAVELEX_REFRESH is set."""
    key = hashlib.sha1(json.dumps([kind, params], sort_keys=True).encode()).hexdigest()
    path = CACHE_DIR / f"{key}.json"
    if path.exists() and not os.getenv("TRAVELEX_REFRESH"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if fresh(data, max_age_days):
            return data
    if limits := current_limits():  # a cached page costs nothing; a new fetch counts against the run's limit
        limits.spend_search()
    data = {"retrieved_at": datetime.now(timezone.utc).isoformat(), "response": call_with_retry(call)}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def _trim(item: dict, kind: str, retrieved_at: str, limit: int) -> dict:
    text = item.get("content") or item.get("raw_content") or ""
    return {
        "kind": kind,
        "url": item["url"],
        "title": (item.get("title") or item["url"]).strip(),
        "published": item.get("published_date"),
        "retrieved_at": retrieved_at,
        "text": " ".join(text.split())[:limit],
    }


@traceable(run_type="tool")
def search(query: str, max_results: int = MAX_RESULTS) -> list[dict]:
    params = {"query": query, "max_results": max_results, "search_depth": "basic"}
    data = _cached("search", params, lambda: _tavily().search(**params), ttl_days())
    return [_trim(r, "web", data["retrieved_at"], SNIPPET_CHARS) for r in data["response"]["results"]]


@traceable(run_type="tool")
def news(query: str, days: int = 30, max_results: int = MAX_RESULTS) -> list[dict]:
    params = {"query": query, "topic": "news", "days": days, "max_results": max_results}
    # Today's date is in the cache key, so news is re-fetched daily instead of cached forever.
    data = _cached(f"news:{date.today()}", params, lambda: _tavily().search(**params))
    return [_trim(r, "news", data["retrieved_at"], SNIPPET_CHARS) for r in data["response"]["results"]]


@traceable(run_type="tool")
def extract(urls: list[str]) -> list[dict]:
    params = {"urls": sorted(set(urls))[:MAX_EXTRACT_URLS]}
    data = _cached("extract", params, lambda: _tavily().extract(**params), ttl_days())
    return [_trim(r, "page_extract", data["retrieved_at"], EXTRACT_CHARS) for r in data["response"]["results"]]


def search_all(queries: list[str]) -> list[dict]:
    """Run several searches at once (results in query order). Each search is mostly waiting on the network,
    so 4 searches take about as long as 1. The context-preserving pool keeps LangSmith traces nested."""
    if len(queries) <= 1:
        return [r for q in queries for r in search(q)]
    with ContextThreadPoolExecutor(max_workers=len(queries)) as pool:
        return [r for results in pool.map(lambda q: search(q), queries) for r in results]


@tool("web_search")
def web_search_tool(query: str) -> list[dict]:
    """Search the web. Use for routes, fares, fees, visa rules, hotels and restaurants."""
    return search(query)


@tool("news_search")
def news_search_tool(query: str, days: int = 30) -> list[dict]:
    """Search recent news articles from the last `days` days."""
    return news(query, days)


@tool("extract_pages")
def extract_pages_tool(urls: list[str]) -> list[dict]:
    """Fetch the full text of up to 5 pages, e.g. an airline schedule or an official visa page found by web_search."""
    return extract(urls)


TOOLS = [web_search_tool, news_search_tool, extract_pages_tool]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}