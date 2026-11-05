"""Itinerary: a day-by-day plan from everything researched so far. Code numbers and dates the days."""
import re
import time
from datetime import timedelta

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Send

from travelex.agents.common import event, taken_ids
from travelex.agents.intake import describe
from travelex.llm import get_llm, structured
from travelex.schemas import Itinerary, TripRequest
from travelex.state import TripState
from travelex.trip import allocate_nights, has_vulnerable, party, resolve_nights

ITINERARY_PROMPT = """You plan a realistic day-by-day itinerary from the research below and your knowledge of
the places.
- Exactly {days} days. Day 1 is arrival (keep it light), day {days} is departure (transfer home).
- Visit the stops in order with the nights given for each. Put each transfer between stops on the day it
  happens, with how and roughly how long.
- city: the town where that day mostly happens and where you sleep (e.g. "Vík", not "South Coast Region").
  Never a country or region name.
- activities: 1-4 a day. strenuous true for hikes, long climbs, very long walks or drives over 4 hours.
- With children or seniors: at most one strenuous activity a day, an easy day after a strenuous one, a
  rest day (rest_day true) on trips of 5 nights or more, and a kid_option for every strenuous activity.
- Above 2,500 m keep the first two days easy, to acclimatise.
- Follow the season notes (e.g. rain: indoor options) and the travellers' interests.
- Match the spending money in the research: paid tours, cruises and experiences when there is plenty,
  mostly free sights, walks and markets when it is tight.
- trimmed_variant: if the travellers allowed a shorter trip, one sentence on which day to drop; else "".
- notes: practical tips: what to book ahead, what to pack, local customs. Never state visa rules, fees or
  prices: those come from cited research elsewhere in the plan.
- If the travellers commented on a previous version, make the changes they asked for and keep the rest of
  that plan as it was, unless the trip itself changed."""


def cruise_brief(request: TripRequest) -> str:
    """What the planner must know about the cruise it doesn't plan."""
    board, length = request.stops[0].place, f"{request.cruise_nights}-night " if request.cruise_nights else ""
    if len(request.stops) == 2:
        land = (f"Plan only the days on land: in {board} before boarding (that stay's last day ends with boarding "
                f"the ship), then in {request.stops[1].place} after leaving it (the first day there starts with "
                f"leaving the ship).")
    else:
        land = (f"Plan only the days in {board} before boarding; the last day ends with boarding the ship, and "
                f"the travellers fly home when the cruise returns.")
    return (f"The travellers take their own {length}cruise from {board}, which you don't plan: no days at sea, "
            f"no ports of call. {land}")


def research_brief(state: TripState) -> str:
    request = state["request"]
    nights, trimmed = resolve_nights(request)
    lines = [f"Trip: {describe(request)}", f"Party: {party(request)}",
             f"Interests: {', '.join(request.interests) or 'none given'}. Diet: {', '.join(request.diet) or 'none'}."]
    if trimmed:
        lines.append(f"A shorter trip of {trimmed} nights is also acceptable.")
    if request.trip_type == "around_cruise" and request.stops:
        lines.append(cruise_brief(request))
    costs, season = state.get("costs") or {}, state.get("season") or {}
    for stop, n in allocate_nights(request):
        info = [f"Stop: {stop.place} ({stop.country}), {n} nights"]
        if stop.cities:
            info.append(f"cities named: {', '.join(stop.cities)}")
        if (v := season.get(stop.place)) and v.reasons:
            info.append(f"season {v.verdict}: {'; '.join(v.reasons)}")
        if (c := costs.get(stop.place)):
            if c.guide_required:
                info.append("a licensed guide is required")
            if c.onward:
                info.append(f"onward: {c.onward.route} by {c.onward.mode}")
            info += c.notes[:2]
        lines.append(" · ".join(info))
    groups = {g.id: g for g in request.groups}
    for gid, result in (state.get("group_results") or {}).items():
        if (o := result.chosen_option()) and gid in groups:
            lines.append(f"{groups[gid].origin} group arrives by {o.route} ({o.mode}).")
    if meeting := state.get("meeting"):
        lines.append(f"Meeting point: {meeting.meeting_point}. {meeting.options[0].note}")
    if (f := state.get("feasibility")) and f.budget is not None:
        lines.append(spending(f, request))
    if notes := state.get("traveller_notes"):
        lines.append("The travellers told us: " + "; ".join(notes))
    if comment := state.get("plan_comment"):
        lines.append(f"The travellers' comment on the previous version: {comment}")
        if previous := state.get("itinerary"):
            lines.append("The previous version:")
            lines += [f"- Day {d.day} · {d.city}: {d.title} ({'; '.join(a.name for a in d.activities)})"
                      for d in previous.days]
    if (qa := state.get("qa_report")) and qa.issues:
        lines.append("Fix these problems found in your previous plan: " + "; ".join(qa.issues))
    return "\n".join(lines)


def spending(f, request: TripRequest) -> str:
    """The budget left after hotels, food, transport and fees, so the plan's activities fit it."""
    hotels = f"Hotels: {f.lodging_style}" + (" (chosen to fit the budget)." if f.lodging_chosen else ".")
    if f.spare <= 0:
        return f"{hotels} The budget is used up: favour free sights and walks."
    nights, _ = resolve_nights(request)
    per_day = f.spare / max(len(request.travellers), 1) / ((nights or 0) + 1)
    return (f"{hotels} Spending money left for activities and extras: about {f.spare:,.0f} {f.currency} "
            f"(about {per_day:,.0f} {f.currency} per person per day).")


def first_day(state: TripState):
    meeting = state.get("meeting")
    return (meeting.day1 if meeting else None) or state["request"].start_date


def itinerary_planner(state: TripState) -> dict:
    started = time.perf_counter()
    request = state["request"]
    nights, _ = resolve_nights(request)
    days = (nights or 4) + 1
    prompt = ITINERARY_PROMPT.format(days=days)
    plan = structured(get_llm("itinerary"), Itinerary, [SystemMessage(prompt), HumanMessage(research_brief(state))])
    plan = tidy_plan(number_days(plan, first_day(state)), request)
    return {"itinerary": plan,
            "events": [event("itinerary_planner", "planned", f"{len(plan.days)} days: "
                             + " → ".join(dict.fromkeys(d.city for d in plan.days)), started=started)]}


def number_days(plan: Itinerary, day1) -> Itinerary:
    days = [d.model_copy(update={"day": i, "date": day1 + timedelta(days=i - 1) if day1 else None})
            for i, d in enumerate(plan.days, 1)]
    return plan.model_copy(update={"days": days})


FACT_WORDS = re.compile(r"\b(visa|permit|fee|sdf|levy|tax|price|cost|usd|inr|eur|\$|₹|€)\b", re.IGNORECASE)


def tidy_plan(plan: Itinerary, request: TripRequest) -> Itinerary:
    """Code-side fixes: no kid options when nobody needs them, no shorter version nobody asked for, and no
    notes that state visa, fee or price facts (the model invents them; cited research covers them)."""
    vulnerable = has_vulnerable(request)
    days = [d.model_copy(update={"activities": [a.model_copy(update={"kid_option": a.kid_option if vulnerable else ""})
                                                for a in d.activities]}) for d in plan.days]
    trimmed = plan.trimmed_variant if resolve_nights(request)[1] else ""
    notes = [n for n in plan.notes if not FACT_WORDS.search(n)]
    return plan.model_copy(update={"days": days, "trimmed_variant": trimmed, "notes": notes})


def same_city(a: str, b: str) -> bool:
    """"Kyoto" and "Kyoto city", "Paro" and "Paro Valley": the planner and the user name places a little differently."""
    a, b = a.strip().casefold(), b.strip().casefold()
    return a == b or a in b or b in a


def guide_for(city: str, food: dict) -> object | None:
    """The food guide for a city, matching names tolerantly; None if there is none."""
    return food.get(city) or next((g for name, g in food.items() if same_city(name, city)), None)


def country_of(request: TripRequest, city: str, plan: Itinerary | None = None) -> str | None:
    """The country a food city is in (ISO code): the stop it is named in, the trip's only country, or the stop the
    itinerary is at on the city's first day. Food research needs it: "South Coast" alone finds California."""
    stops = request.stops
    if named := next((s for s in stops if same_city(city, s.place) or any(same_city(city, c) for c in s.cities)),
                     None):
        return named.country
    if len({s.country for s in stops}) == 1:
        return stops[0].country
    day = next((d.day for d in plan.days if d.city == city), None) if plan else None
    if day is not None:
        slept = 0
        for stop, nights in allocate_nights(request):
            slept += nights
            if day <= slept:
                return stop.country
        return stops[-1].country if stops else None
    return None


def named_cities(request: TripRequest) -> list[str]:
    """Cities the user named, known before any itinerary exists, so their food research can start early."""
    seen: list[str] = []
    for city in (c for s in request.stops for c in s.cities):
        if not any(same_city(city, other) for other in seen):
            seen.append(city)
    return seen


def cities(plan: Itinerary) -> list[str]:
    """Every city on the itinerary, in order. Food guides cover only these: the destination, not home."""
    return list(dict.fromkeys(d.city for d in plan.days))


def food_sends(state: TripState, only: set[str] | None = None) -> list[Send]:
    request, plan, have = state["request"], state["itinerary"], state.get("food") or {}
    namespace = state.get("namespace", "")
    return [Send("food_agent", {"city": city, "country": country_of(request, city, plan),
                                "source_prefix": f"{namespace}F{i}.", "request": request,
                                "taken": taken_ids(state, f"{namespace}F{i}.")})
            for i, city in enumerate(cities(plan), 1)
            if (only is not None and city in only) or (only is None and guide_for(city, have) is None)]


def route_after_itinerary(state: TripState) -> list[Send] | str:
    """Food guides for cities that don't have one yet (named cities were researched early), then the check."""
    return food_sends(state) or "qa"
