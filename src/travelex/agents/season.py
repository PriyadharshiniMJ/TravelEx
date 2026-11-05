"""Season check: a verdict for every stop. Seeded or cached profiles cost nothing; a new place is researched once."""
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.config import ContextThreadPoolExecutor

from travelex.agents.common import event, register
from travelex.llm import get_llm, structured
from travelex.schemas import ClimateProfile, ClimateResearch, SeasonVerdict, Source, Stop
from travelex.state import TripState
from travelex.tools import climate, search
from travelex.trip import has_vulnerable, trip_months

PREFIX = "S"
SEVERITY = {"good": 0, "unknown": 1, "ok_with_caveats": 2, "avoid": 3}
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
               "October", "November", "December"]

PROFILE_PROMPT = """You write a month-by-month travel climate profile for one place, using only the numbered sources.
- months: one entry for each month 1-12.
  - "good": comfortable weather and the main sights are open.
  - "ok_with_caveats": travel works but with rain, heat, cold, crowds or partial closures; say which in reason.
  - "avoid": monsoon, hurricanes, extreme heat or cold, or the main sights or trails closed; say why.
- vulnerable_avoid: months that are fine for fit adults but should be avoided with young children or
  seniors (extreme cold, heat or altitude), with the reason. Often empty.
- vulnerable_note: only when the whole place is a real risk for children or seniors all year (e.g. above
  2,500 m altitude, or extreme heat). Otherwise "". Don't mention mountains a visitor can simply avoid.
- Reasons are short plain sentences for a traveller. Never put source IDs in them.
- source_ids: the IDs of the sources you used.
If the sources don't cover a month, use "ok_with_caveats" with reason "no data". Never invent facts."""


def queries_for(stop: Stop) -> list[str]:
    where = f"{stop.place} {', '.join(stop.cities)}".strip()
    return [f"best time to visit {where} month by month weather",
            f"{where} rainy season monsoon closures when to avoid travel"]


def research_profile(stop: Stop, sources: dict[str, Source], prefix: str = PREFIX,
                     taken: dict[str, Source] | None = None) -> tuple[ClimateProfile, dict[str, Source], list[str]]:
    """Two searches, then one extraction. The profile is cached so the next trip here skips all of this."""
    queries = queries_for(stop)
    results = search.search_all(queries)
    new, found = register(results, prefix, sources, taken)
    rendered = "\n\n".join(s.render() for s in found) or "(no sources found)"
    messages = [SystemMessage(PROFILE_PROMPT),
                HumanMessage(f"Place: {stop.place} (country {stop.country})\n\nSources:\n{rendered}")]
    research = structured(get_llm("season"), ClimateResearch, messages)
    known = {s.id for s in found}
    profile = research.to_profile(stop.place)
    profile = profile.model_copy(update={"source_ids": [i for i in profile.source_ids if i in known]})
    used = [s for s in found if s.id in profile.source_ids] or found
    climate.remember(profile, evidence_at=min((s.retrieved_at for s in used), default=None, key=str))
    return profile, new, queries


def worst(verdicts: list[SeasonVerdict]) -> SeasonVerdict:
    """A trip that spans two months is as risky as its worse month."""
    return max(verdicts, key=lambda v: SEVERITY[v.verdict])


def stop_verdict(profile: ClimateProfile, months: list[int], vulnerable: bool) -> SeasonVerdict:
    if not months:
        return SeasonVerdict(place=profile.place, verdict="unknown", reasons=["The travel month is not known yet."])
    verdict = worst([climate.season_verdict(profile, m, vulnerable) for m in months])
    if verdict.month is not None and len(months) > 1:
        verdict = verdict.model_copy(update={"reasons": [f"{MONTH_NAMES[verdict.month - 1]}: {r}" for r in verdict.reasons]})
    return verdict


def check_stop(i: int, stop: Stop, months: list[int], vulnerable: bool, namespace: str,
               taken: dict[str, Source] | None = None) -> tuple[SeasonVerdict, dict[str, Source], dict, str | None]:
    """One stop's verdict: (verdict, new sources, trace event, error). Seeded or cached profiles cost nothing."""
    started = time.perf_counter()
    profile = next(filter(None, map(climate.climate_for, [stop.place, *stop.cities])), None)
    new, queries = {}, []
    if profile is None:
        try:
            profile, new, queries = research_profile(stop, {}, f"{namespace}{PREFIX}{i}.", taken)  # own IDs per stop
        except search.SearchConfigError:
            raise
        except Exception as e:  # fail soft (bad extraction, network): one place must not sink the trip
            verdict = SeasonVerdict(place=stop.place, verdict="unknown",
                                    reasons=["Season research failed; check the weather yourself."])
            return (verdict, {}, event("season_check", "failed", f"{stop.place}: {e}", started=started),
                    f"season_check ({stop.place}): {type(e).__name__}: {e}")
    verdict = stop_verdict(profile, months, vulnerable)
    how = "researched" if queries else f"{profile.provenance} profile"
    why = f": {'; '.join(verdict.reasons)}" if verdict.reasons else ""
    return verdict, new, event("season_check", verdict.verdict, f"{stop.place} ({how}){why}", queries, started), None


def season_check(state: TripState) -> dict:
    """Every stop at once. This runs as one research branch beside transport, visas and costs: nothing there
    needs the season, which only matters later, when the budget check decides on alternatives."""
    request = state["request"]
    months, vulnerable = trip_months(request), has_vulnerable(request)
    namespace = state.get("namespace", "")
    with ContextThreadPoolExecutor(max_workers=max(len(request.stops), 1)) as pool:
        results = list(pool.map(lambda pair: check_stop(pair[0], pair[1], months, vulnerable, namespace,
                                                        state.get("taken")),
                                 list(enumerate(request.stops, 1))))
    sources = {sid: s for _, new, _, _ in results for sid, s in new.items()}
    return {"season": {stop.place: verdict for stop, (verdict, _, _, _) in zip(request.stops, results)},
            "sources": sources, "events": [e for _, _, e, _ in results],
            "errors": [err for _, _, _, err in results if err]}
