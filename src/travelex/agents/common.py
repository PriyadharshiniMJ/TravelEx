"""Helpers shared by every agent node: trace events, numbering sources and the ReAct tools node."""
import re
import time

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables.config import ContextThreadPoolExecutor
from langchain_core.tools import BaseTool

from travelex.schemas import Citation, Evidence, Source
from travelex.tools.search import SearchConfigError
from travelex.tools.visa_rules import SCHENGEN


def event(agent: str, action: str, detail: str = "",
          queries: list[str] | None = None, started: float | None = None) -> dict:
    """One row of the agent trace shown in the CLI and the UI."""
    ms = round((time.perf_counter() - started) * 1000) if started is not None else None
    return {"agent": agent, "action": action, "detail": detail, "queries": queries or [], "ms": ms}


def next_number(prefix: str, sources: dict[str, Source]) -> int:
    pattern = re.compile(rf"{re.escape(prefix)}(\d+)")
    return 1 + max((int(m.group(1)) for sid in sources if (m := pattern.fullmatch(sid))), default=0)


def register(results: list[dict], prefix: str, sources: dict[str, Source],
             taken: dict[str, Source] | None = None) -> tuple[dict[str, Source], list[Source]]:
    """Number search results as Sources ("S1", "G1.4"…), reusing the ID of a result this branch saw before.
    taken: IDs earlier research in the run already gave out (a city that was first on the last version of the
    plan): new sources are numbered after them, so an ID never comes to mean a different page.
    Returns (the new sources, every distinct source for these results in order)."""
    own = re.compile(rf"{re.escape(prefix)}\d+")
    known = {(s.url, s.text): s for sid, s in sources.items() if own.fullmatch(sid)}
    n = next_number(prefix, {**(taken or {}), **sources})
    new, ordered = {}, []
    for result in results:
        source = known.get((result["url"], result["text"]))
        if source is None:
            source = Source(id=f"{prefix}{n}", **result)
            n += 1
            known[(source.url, source.text)] = new[source.id] = source
        if source not in ordered:  # two searches often return the same page
            ordered.append(source)
    return new, ordered


def squash(text: str) -> str:
    """Compare text ignoring case and runs of whitespace."""
    return " ".join(text.split()).casefold()


# Government domains: .gov, .gov.bt, gov.uk, .gouv.fr, .gob.mx, .go.jp, .govt.nz, europa.eu, *.mofa.*, *.mfa.*, and
# government.<country> (government.is, government.nl, government.se)
OFFICIAL = re.compile(r"(^|\.)(gov|govt|gouv|gob|go|gc|mofa|mfa)\.([a-z]{2,3}\.)?[a-z]{2,3}$|(^|\.)gov$|"
                      r"(^|\.)government\.[a-z]{2}$|(^|\.)europa\.eu$|(^|\.)canada\.ca$", re.IGNORECASE)


US_GOVERNMENT = re.compile(r"(^|\.)(gov|usembassy\.gov|state\.gov)$", re.IGNORECASE)
TLD_COUNTRY = {"uk": "GB", "eu": "EU"}
CONTEXT_CHARS = 250


def host_of(url: str) -> str:
    return re.sub(r"^https?://", "", url).split("/")[0].split(":")[0].removeprefix("www.").lower()


def official(url: str) -> bool:
    """A government site, judged by its domain."""
    return bool(OFFICIAL.search(host_of(url)))


def government_of(url: str) -> str | None:
    """The ISO code of the government that runs a site ("EU" for europa.eu), or None for any other site."""
    host = host_of(url)
    if not OFFICIAL.search(host):
        return None
    if US_GOVERNMENT.search(host):
        return "US"
    if host.endswith("canada.ca"):
        return "CA"
    tld = host.rsplit(".", 1)[-1]
    return TLD_COUNTRY.get(tld, tld.upper())


def source_kind(url: str, passport: str, country: str, zone: str | None) -> str:
    """Whose page this is, relative to the trip. The destination's government states its own entry rules;
    the traveller's home government writes for its own citizens; any other government writes for ITS citizens."""
    government = government_of(url)
    if government is None:
        return "third_party"
    if government == country or (zone == "Schengen" and (government == "EU" or government in SCHENGEN)):
        return "destination_government"  # any Schengen government speaks for the Schengen visa; the judge drops
                                         # what is only about its own country ("visit France as your main destination")
    return "home_government" if government == passport else "other_government"


def context_of(quote: str, text: str) -> str:
    """The text around a quote on its page, so a reader (or a judge) sees what the passage is about."""
    flat = " ".join(text.split())
    at = flat.casefold().find(squash(quote))
    if at < 0:
        return ""
    return flat[max(0, at - CONTEXT_CHARS): at + len(quote) + CONTEXT_CHARS]


def verify_evidence(evidence: list[Evidence], sources: dict[str, Source]) -> tuple[list[Citation], int]:
    """Keep the quotes found word for word in the source they cite, as Citations, official sites first.
    Returns (kept, dropped)."""
    kept: dict[tuple[str, str], Citation] = {}
    dropped = 0
    for e in evidence:
        source, quote = sources.get(e.source_id), squash(e.quote)
        if source is None or not quote or quote not in squash(source.text):
            dropped += 1
            continue
        kept.setdefault((source.url, quote), Citation(url=source.url, title=source.title, quote=" ".join(e.quote.split()),
                                                      supports=e.supports, retrieved_at=source.retrieved_at,
                                                      context=context_of(e.quote, source.text)))
    return sorted(kept.values(), key=lambda c: not official(c.url)), dropped


def taken_ids(state: dict, prefix: str) -> dict[str, Source]:
    """The run's sources already numbered under this prefix: a new research branch numbers after them."""
    return {sid: s for sid, s in (state.get("sources") or {}).items() if sid.startswith(prefix)}


def own_sources(sources: dict[str, Source], prefix: str) -> list[Source]:
    own = re.compile(rf"{re.escape(prefix)}\d+")
    return [s for sid, s in sources.items() if own.fullmatch(sid)]


def render(sources: list[Source]) -> str:
    return "\n\n".join(s.render() for s in sources) or "(no sources found)"


def run_tool_calls(message: AIMessage, tools: list[BaseTool], prefix: str, sources: dict[str, Source],
                   taken: dict[str, Source] | None = None) -> tuple[list[ToolMessage], dict[str, Source], list[str]]:
    """Run the tool calls, register every result as a Source, and render them for the LLM."""
    by_name = {t.name: t for t in tools}

    def run(call) -> list[dict] | Exception:
        if call["name"] not in by_name:
            return KeyError(f"unknown tool {call['name']}")
        try:
            return by_name[call["name"]].invoke(call["args"])
        except SearchConfigError:
            raise
        except Exception as e:  # one failed search shouldn't end the loop; the model sees the error (and a
            return e            # LimitReached tells it to stop searching and answer with what it has)

    calls = message.tool_calls
    with ContextThreadPoolExecutor(max_workers=max(len(calls), 1)) as pool:  # searches in one turn run at once
        outcomes = list(pool.map(run, calls))
    seen = dict(sources)
    replies, new, queries = [], {}, []
    for call, results in zip(calls, outcomes):  # numbered in call order, so IDs don't depend on timing
        args = call["args"]
        queries.append(f'{call["name"]}({args.get("query") or ", ".join(args.get("urls", []))})')
        if isinstance(results, Exception):
            replies.append(ToolMessage(f"Tool error: {type(results).__name__}: {results}", tool_call_id=call["id"]))
            continue
        added, found = register(results, prefix, seen, taken)
        seen |= added
        new |= added
        replies.append(ToolMessage(render(found) if found else "No results.", tool_call_id=call["id"]))
    return replies, new, queries


def tools_node(agent: str, messages_key: str, tools: list[BaseTool], who=lambda state: ""):
    """The 'tools' half of a hand-built ReAct loop; the prefix comes from state['source_prefix']."""
    def node(state: dict) -> dict:
        started = time.perf_counter()
        replies, new, queries = run_tool_calls(state[messages_key][-1], tools, state["source_prefix"],
                                               state.get("sources") or {}, state.get("taken"))
        return {messages_key: replies, "sources": new,
                "events": [event(agent, "searched", f"{who(state)}{len(new)} new sources", queries, started)]}
    return node
