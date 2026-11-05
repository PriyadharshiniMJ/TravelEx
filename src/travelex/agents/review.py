"""Review: after the plan is saved, the traveller can comment on it (interrupt). Changes to the days go back to the
itinerary planner; changes to the trip itself (destination, month, length, hotels, budget) re-run the research
they affect first. Unchanged research is reused, because route_research skips results whose key still matches."""
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import Overwrite, interrupt

from travelex.agents.alternatives import apply_changes, checked, describe_changes
from travelex.agents.common import event
from travelex.agents.intake import describe
from travelex.llm import get_llm, structured
from travelex.schemas import PlanComment, Stop, TripChanges, TripRequest
from travelex.state import TripState
from travelex.tools.countries import country_name
from travelex.trip import parse_duration, resolve_nights

MAX_REVISIONS = 3
DONE_WORDS = {"no", "nope", "done", "finish", "finished", "ok", "okay", "yes", "yep", "fine", "good", "great",
              "perfect", "awesome", "lovely", "looks", "it", "it's", "that's", "is", "this", "all", "thanks", "thank",
              "you"}
NOTHING = re.compile(r"\b(no|nothing|none)( more| else)?( changes?| to change)?( needed)?\b")


def finished(comment: str) -> bool:
    """"looks good, thanks", "no changes needed", or nothing at all: the traveller is happy with the plan.
    "change the plan" is not: every word must say done."""
    words = re.findall(r"[a-z']+", NOTHING.sub(" done ", comment.lower()))
    return all(w in DONE_WORDS for w in words)


REVIEW_PROMPT = """A traveller has commented on their finished day-by-day plan. Today is {today}.
- summary: the change they ask for, in a few words.
- trip_changes: fill only what changes the trip itself: new, dropped or different stops or cities (place and
  ISO country code, e.g. Jakarta instead of Bali), with every stop the trip keeps,
  a different month ("YYYY-MM", on or after today), a different length (duration_text, e.g. "5 days"), a hotel
  level (lodging_style), or a bigger budget. Leave everything else empty.
- Changes to the days need no trip_changes: pace, rest, adding or removing activities, the order of days, and
  how the nights are split between towns in a country already on the trip (e.g. "2 nights in Bergen")."""


def norm(name: str) -> str:
    return " ".join(name.casefold().split())


def same_places(new: list[Stop], old: list[Stop], towns: list[str]) -> bool:
    """The same destinations: every new stop and city is a place the trip already has (a stop, a city or a town in
    the plan), and every old stop and city is still there. "2 nights in Bergen" on a Norway trip through Oslo and
    Bergen is a change to the days; Bali to Jakarta, adding Tromsø or dropping Bergen needs new research."""
    countries = lambda stops: list(dict.fromkeys(s.country for s in stops))
    if countries(new) != countries(old):
        return False
    known = {norm(s.place) for s in old} | {norm(c) for s in old for c in s.cities} | {norm(t) for t in towns}
    whole = lambda s, others: norm(s.place) == norm(country_name(s.country)) and s.country in {o.country for o in others}
    named = {norm(s.place) for s in new} | {norm(c) for s in new for c in s.cities}
    return (all(norm(s.place) in known or whole(s, old) for s in new)
            and all(norm(c) in known for s in new for c in s.cities)
            and all(norm(s.place) in named or whole(s, new) for s in old)
            and all(norm(c) in named for s in old for c in s.cities))  # dropping Bergen is a change to the trip


def real_changes(changes: TripChanges, request: TripRequest, towns: list[str] = ()) -> TripChanges:
    """Drop "changes" that change nothing: the same places (moving nights between towns already in the plan is a
    change to the days), or the month, length or hotel level the trip already has."""
    nights, _ = resolve_nights(request)
    days, new_nights = parse_duration(changes.duration_text)
    month = request.month or (f"{request.start_date:%Y-%m}" if request.start_date else None)
    same = {"stops": bool(changes.stops) and same_places(changes.stops, request.stops, list(towns)),
            "month": changes.month is not None and changes.month == month,
            "duration_text": changes.duration_text is not None and (new_nights or (days[1] - 1 if days else None)) == nights,
            "lodging_style": changes.lodging_style is not None and changes.lodging_style == request.lodging_style}
    return changes.model_copy(update={field: None for field, unchanged in same.items() if unchanged})


def review_plan(state: TripState, config: RunnableConfig) -> dict:
    """Pause on the finished plan. Resume with "" (or "done") to finish, or with a comment to rework it."""
    rounds = len(state.get("revisions") or [])
    if config.get("configurable", {}).get("auto_approve") or rounds >= MAX_REVISIONS:
        return {"plan_comment": None}
    answer = interrupt({"kind": "review", "revision": rounds, "max_revisions": MAX_REVISIONS,
                        "report_dir": state.get("report_dir")})
    comment = answer.strip() if isinstance(answer, str) else ""
    if finished(comment):
        return {"plan_comment": None, "events": [event("review_plan", "finished", "the traveller is happy")]}

    started = time.perf_counter()
    request, today = state["request"], state["today"]
    towns = list(dict.fromkeys(d.city for d in state["itinerary"].days)) if state.get("itinerary") else []
    sorted_comment = structured(get_llm("intake"), PlanComment, [
        SystemMessage(REVIEW_PROMPT.format(today=today.isoformat())),
        HumanMessage(f"Trip: {describe(request)}\nTowns in the plan: {', '.join(towns) or 'none yet'}\n"
                     f"Comment: {comment}")])
    sorted_comment.trip_changes = real_changes(sorted_comment.trip_changes, request, towns)
    update = {"plan_comment": comment, "qa_round": 0,
              "revisions": [*(state.get("revisions") or []), sorted_comment.summary]}
    if not sorted_comment.changes_trip():
        return update | {"replan": False,
                         "events": [event("review_plan", "reworking the days", comment, started=started)]}

    changes = checked(sorted_comment.trip_changes, request, today)
    new = apply_changes(request, changes, today)
    places, groups = {s.place for s in new.stops}, {g.id for g in new.groups}
    keep = lambda d, keys: Overwrite({k: v for k, v in (d or {}).items() if k in keys})  # drop removed stops' research
    return update | {"request": new, "replan": True, "season": keep(state.get("season"), places),
                     "costs": keep(state.get("costs"), places),
                     "group_results": keep(state.get("group_results"), groups),
                     "events": [event("review_plan", "changing the trip", describe_changes(changes), started=started)]}


def route_after_review(state: TripState) -> str:
    """Finished; or the days change; or the trip changes, so its research runs again (unchanged parts are reused)."""
    if not state.get("plan_comment"):
        return END
    return "plan_research" if state.get("replan") else "itinerary_planner"
