import os

import pytest

# Before travelex loads .env: load_dotenv never overrides a variable that is already set,
# so tests never upload traces to LangSmith or fetch live exchange rates.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["TRAVELEX_FX_OFFLINE"] = "1"
os.environ["TRAVELEX_RETRY_WAIT"] = "0"  # retries of passing errors happen at once in tests


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Blank the keys .env loaded: an LLM or search call a test forgot to fake fails loudly instead of
    spending money and time on a real API."""
    for name in ("LLM_MODEL", "LLM_MODEL_FAST", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TAVILY_API_KEY"):
        monkeypatch.setenv(name, "")
    import travelex.tools.search as search_module
    monkeypatch.setattr(search_module, "_client", None)


@pytest.fixture(autouse=True)
def rules_dir(tmp_path, monkeypatch):
    """Each test learns rules into its own empty cache."""
    monkeypatch.setenv("TRAVELEX_RULES_DIR", str(tmp_path / "rules"))
    return tmp_path / "rules"


@pytest.fixture(autouse=True)
def search_cache(tmp_path, monkeypatch):
    """Each test starts with an empty search cache: a test must never pass on real pages cached on this machine."""
    import travelex.tools.search as search_module
    monkeypatch.setattr(search_module, "CACHE_DIR", tmp_path / "tavily")
    return tmp_path / "tavily"


@pytest.fixture
def use_fake_llm(monkeypatch):
    """use_fake_llm(module, *replies, responder=None) patches module.get_llm to return one FakeLLM."""
    from fakes import FakeLLM

    def install(module, *responses, responder=None):
        llm = FakeLLM(responses=list(responses), responder=responder)
        monkeypatch.setattr(module, "get_llm", lambda role, **kwargs: llm)
        return llm
    return install


@pytest.fixture
def fake_search(monkeypatch):
    """Patch Tavily. Every search call returns the pages appended to the returned list."""
    import travelex.tools.search as search_module

    pages: list[dict] = []
    monkeypatch.setattr(search_module, "search", lambda query, *args, **kwargs: [dict(p) for p in pages])
    return pages


@pytest.fixture
def search_calls(monkeypatch, fake_search):
    """The queries sent to the patched search, in order."""
    import travelex.tools.search as search_module

    calls: list[str] = []
    inner = search_module.search

    def counting(query, *args, **kwargs):
        calls.append(query)
        return inner(query, *args, **kwargs)

    monkeypatch.setattr(search_module, "search", counting)
    return calls
