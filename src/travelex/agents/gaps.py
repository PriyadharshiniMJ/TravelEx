"""Gap filling: after research, code lists what is weak (prices no source states, fares from one source, routes not
found), and an agent decides which gaps to chase, what to search and read, and when to give up. It records a price only
with words a source contains; code checks every record before it changes a cost. Runs on the main trip, not inside
an alternative's research."""
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import Overwrite

from travelex.agents.common import event, run_tool_calls
from travelex.agents.evidence import (UNIT_OF, about, currency_agrees, item_named, stated, unit_fits,  # noqa: F401
                                     verbatim)
from travelex.agents.transport import MAX_FARE_USD, MIN_FLIGHT_FARE_USD
from travelex.llm import get_llm
from travelex.limits import research_stopped
from travelex.retry import call_with_retry
from travelex.schemas import CostEvidence, Gap, LodgingRate, Money, RouteOption, Source, TripRequest
from travelex.state import TripState, tool_rounds
from travelex.tools import fx
from travelex.tools.countries import country_name
from travelex.tools.search import extract_pages_tool, web_search_tool
from travelex.trip import next_stop, style_for

MAX_ROUNDS = 4
MAX_GAPS = 8
STYLES = ("budget", "mid-range", "luxury")
# What a believable price looks like, in USD, per unit of each item: anything outside is a misread
PLAUSIBLE_USD = {"hotel": (3, 3000), "food": (2, 500), "local transport": (0.5, 300), "vehicle": (10, 1500),
                 "guide": (10, 1500), "fare": (MIN_FLIGHT_FARE_USD, MAX_FARE_USD), "onward": (1, MAX_FARE_USD)}
NOT_FOUND_IN_TIME = "No source found within the search limit."
FAR_FROM_ESTIMATE = 4  # a sourced price over 4× the research's estimate, or under a quarter of it, is another item

GAPS_PROMPT = """You fill gaps in a trip's research. Each gap is a price we only estimated or found in one source, or
a journey with no route found. Today is {today}.
- Work through the gaps, most useful first. Search with specific queries (place, item, travel month), read a page
  in full when the snippets don't show prices, and let one search fill several gaps when it can.
- record_price: a price for a gap, quoting only the source's own words that state it, copied exactly and
  including the number, without the [R.n] label or words of your own. For a fare, say whether it is one way.
  record_route: a way home → destination for a journey with no route.
- Record every price you can in the same turn. If a record is rejected, fix it as the reply says.
- If no source states a price after a fair try, call give_up for that gap with a short reason. Never invent a
  price, a source or a quote.
- When every gap is filled or given up, reply with a one-line summary and no tool call."""


@tool
def record_price(gap_id: str, amount: float, currency: str, source_id: str, quote: str, one_way: bool = False) -> str:
    """Record the price that fills a gap: per room per night for a hotel, per person per day for food and local
    transport, per day for a vehicle or guide, per adult for a fare. quote: the source's exact words, with the number."""
    return ""


@tool
def record_route(gap_id: str, route: str, mode: str, amount: float, currency: str, source_id: str, quote: str,
                 one_way: bool = False) -> str:
    """Record a way to make a journey with no route: route as IATA codes or places joined by "-" (where you start
    first: home, or the stop you leave), mode "flight", "flight+overland" or "surface", the fare per adult, and the
    source's exact words with the price."""
    return ""


@tool
def give_up(gap_id: str, reason: str) -> str:
    """Stop working on a gap no source can fill, with a short reason."""
    return ""


SEARCH_TOOLS = [web_search_tool, extract_pages_tool]
TOOLS = SEARCH_TOOLS + [record_price, record_route, give_up]


# ---------- finding the gaps (code) ----------

def find_gaps(state: TripState) -> list[Gap]:
    """The weak spots, most useful first: routes not found, fares from one source, then each stop's unsourced prices."""
    request, costs = state["request"], state.get("costs") or {}
    groups = {g.id: g for g in request.groups}
    found: list[Gap] = []
    for gid, result in (state.get("group_results") or {}).items():
        group = groups.get(gid)
        if group is None:
            continue
        option = result.chosen_option()
        to = request.stops[0].place if request.stops else "the destination"
        if option is None:
            found.append(Gap(id="", kind="route", key=gid, item="fare",
                             text=f"No route found from {group.origin} to {to}."))
        elif option.mode == "flight" and not any(w.startswith("Typical return fare") for w in option.warnings):
            found.append(Gap(id="", kind="fare", key=gid, item="fare",
                             estimate_usd=usd(option.adult_fare.amount, option.adult_fare.currency),
                             text=f"Return flight {group.origin} to {to} ({option.route}): {option.adult_fare.amount:,.0f} "
                                  f"{option.adult_fare.currency} per adult, not yet confirmed by a second fare source."))
    for i, stop in enumerate(request.stops):  # a journey between stops with no price at all
        if (nxt := next_stop(request, i)) and (c := costs.get(stop.place)) is not None and c.onward is None:
            found.append(Gap(id="", kind="route", key=stop.place, item="onward",
                             text=f"No route found from {stop.place} to {nxt.place} (one way, between stops)."))
    for stop in request.stops:
        c = costs.get(stop.place)
        if c is None:
            continue
        have = {x.item for x in c.evidence}
        levels = [request.lodging_style] if request.lodging_style else list(STYLES)
        for style in levels:
            if f"{style} hotel" not in have:
                found.append(Gap(id="", kind="cost", key=stop.place, item=f"{style} hotel",
                                 estimate_usd=usd(c.room_rate(style), c.currency),
                                 text=f"{stop.place}: a {style} hotel room per night (now {c.room_rate(style):,.0f} "
                                      f"{c.currency}, an estimate)."))
        wanted = [("food", c.food_per_person_day, "food per person per day")]
        if c.vehicle_per_day and (c.vehicle_needed or c.guide_required):
            wanted.append(("vehicle", c.vehicle_per_day, "a car with driver per day"))
        elif c.local_transport_per_person_day:
            wanted.append(("local transport", c.local_transport_per_person_day, "local transport per person per day"))
        if c.guide_per_day:
            wanted.append(("guide", c.guide_per_day, "a licensed guide per day"))
        for item, amount, what in wanted:
            if item not in have:
                found.append(Gap(id="", kind="cost", key=stop.place, item=item, estimate_usd=usd(amount, c.currency),
                                 text=f"{stop.place}: {what} (now {amount:,.0f} {c.currency}, an estimate)."))
    return [g.model_copy(update={"id": f"gap{i}"}) for i, g in enumerate(found[:MAX_GAPS], 1)]


def usd(amount: float, currency: str) -> float | None:
    try:
        return fx.convert(amount, currency, "USD") if amount else None
    except fx.FxError:
        return None


def plan_gap_filling(state: TripState) -> dict:
    """List the gaps and start the agent's conversation; with no gaps, the budget check runs straight away."""
    started = time.perf_counter()
    gaps = find_gaps(state)
    if not gaps:
        return {"gaps": [], "gap_messages": Overwrite([]),
                "events": [event("fill_gaps", "nothing to fill", "every price has a source")]}
    task = f"Trip: {state['request'].destination}, {travel_month(state['request'])}.\nGaps:\n" + \
        "\n".join(f"- {g.id}: {g.text}" for g in gaps)
    return {"gaps": gaps, "gap_messages": Overwrite([SystemMessage(GAPS_PROMPT.format(today=state["today"])),
                                                    HumanMessage(task)]),
            "events": [event("fill_gaps", "found gaps", "; ".join(g.text for g in gaps), started=started)]}


def travel_month(request: TripRequest) -> str:
    return (f"{request.start_date:%B %Y}" if request.start_date else request.month or "month not set")


# ---------- the agent (LLM) ----------

def gap_agent(state: TripState) -> dict:
    """Decide the next searches and records, or stop. A failing model ends gap filling, never the trip."""
    started = time.perf_counter()
    convo = state["gap_messages"]
    can_act = (tool_rounds(convo) < MAX_ROUNDS and any(g.status == "open" for g in state["gaps"])
               and not research_stopped())
    try:
        llm = get_llm("gaps")
        reply = call_with_retry(lambda: (llm.bind_tools(TOOLS) if can_act else llm).invoke(convo))
    except Exception as e:  # the trip carries on with the research it has; open gaps go to the traveller
        return {"gap_messages": [AIMessage(f"Stopped: {e}")], "gaps": closed(state["gaps"]),
                "errors": [f"fill_gaps: {type(e).__name__}: {e}"],
                "events": [event("fill_gaps", "stopped", str(e), started=started)]}
    if reply.tool_calls and can_act:
        names = ", ".join(c["name"] for c in reply.tool_calls)
        return {"gap_messages": [reply], "events": [event("fill_gaps", "decided", names, started=started)]}
    final = AIMessage(str(reply.content) or "Done.") if reply.tool_calls else reply
    filled = [g.id for g in state["gaps"] if g.status == "filled"]
    gaps = closed(state["gaps"])
    left = [g.id for g in gaps if g.status == "gave_up"]
    detail = f"{len(filled)} of {len(gaps)} gaps filled" + (f"; left for the traveller: {', '.join(left)}" if left else "")
    return {"gap_messages": [final], "gaps": gaps, "events": [event("fill_gaps", "finished", detail, started=started)]}


def closed(gaps: list[Gap]) -> list[Gap]:
    """Whatever is still open when the agent stops goes to the traveller."""
    return [g if g.status != "open" else g.model_copy(update={"status": "gave_up", "note": NOT_FOUND_IN_TIME})
            for g in gaps]


def route_gaps(state: TripState) -> str:
    """More tool calls, or done: what the agent couldn't settle goes to the traveller (who may not be asked)."""
    last = state["gap_messages"][-1] if state.get("gap_messages") else None
    return "gap_tools" if isinstance(last, AIMessage) and last.tool_calls else "ask_traveller"


def route_after_planning(state: TripState) -> str:
    return "gap_agent" if state.get("gaps") else "ask_traveller"


# ---------- the tools (code checks every record) ----------

def gap_tools(state: TripState) -> dict:
    """Run searches like any research tool; check each record and apply it to the costs or routes it fills."""
    started = time.perf_counter()
    message = state["gap_messages"][-1]
    searches = [c for c in message.tool_calls if c["name"] in {t.name for t in SEARCH_TOOLS}]
    others = [c for c in message.tool_calls if c not in searches]
    prefix = f"{state.get('namespace', '')}R."
    replies, new, queries = run_tool_calls(message.model_copy(update={"tool_calls": searches}), SEARCH_TOOLS,
                                           prefix, state.get("sources") or {})
    sources = {**(state.get("sources") or {}), **new}
    gaps = {g.id: g for g in state["gaps"]}
    costs, results = dict(state.get("costs") or {}), dict(state.get("group_results") or {})
    for call in others:
        outcome = apply(call, gaps, costs, results, sources, state["request"])
        replies.append(ToolMessage(outcome, tool_call_id=call["id"]))
        queries.append(f"{call['name']}({call['args'].get('gap_id')}): {outcome}")
    changed_costs = {k: v for k, v in costs.items() if v is not (state.get("costs") or {}).get(k)}
    changed_routes = {k: v for k, v in results.items() if v is not (state.get("group_results") or {}).get(k)}
    return {"gap_messages": replies, "sources": new, "gaps": list(gaps.values()),
            "costs": changed_costs, "group_results": changed_routes,
            "events": [event("fill_gaps", "worked on gaps", f"{len(new)} new sources", queries, started)]}


def apply(call: dict, gaps: dict[str, Gap], costs: dict, results: dict, sources: dict[str, Source],
          request: TripRequest) -> str:
    """Check one record or give-up and apply it; the reply tells the agent what happened, so it can try again."""
    args, name = call["args"], call["name"]
    gap = gaps.get(args.get("gap_id", ""))
    if gap is None or gap.status != "open":
        return f"Rejected: {args.get('gap_id')} is not an open gap."
    if name == "give_up":
        gaps[gap.id] = gap.model_copy(update={"status": "gave_up", "note": str(args.get("reason", ""))[:200]})
        return f"{gap.id} left open for the traveller: {args.get('reason', '')}"
    problem, quote = check(args, gap, sources, place_names(request, gap.key))
    if problem:
        return f"Rejected: {problem}"
    args = {**args, "quote": quote}
    amount = float(args["amount"]) * (2 if args.get("one_way") and gap.item == "fare" else 1)
    source = sources[args["source_id"]]
    if gap.kind == "cost":
        costs[gap.key] = fill_cost(costs[gap.key], gap.item, amount, args["currency"].upper(), args["source_id"],
                                   args["quote"], request)
    elif gap.kind == "fare":
        results[gap.key] = fill_fare(results[gap.key], amount, args["currency"].upper(), args["source_id"], args["quote"])
    else:
        onward = gap.item == "onward"  # between two stops: a one-way fare, kept with the stop it leaves from
        option = RouteOption(mode=args.get("mode") if args.get("mode") in ("flight", "flight+overland", "surface") else "flight",
                             route=str(args.get("route") or "?"), fare_basis="quoted",
                             fare_covers="one_way" if onward else "return",
                             adult_fare=Money(amount=amount, currency=args["currency"].upper()),
                             evidence=args["quote"][:300], source_ids=[args["source_id"]])
        if onward:
            costs[gap.key] = costs[gap.key].model_copy(update={"onward": option})
        else:
            results[gap.key] = results[gap.key].model_copy(update={"options": [option], "chosen": 0})
    gaps[gap.id] = gap.model_copy(update={"status": "filled", "note": f"{source.title} ({source.url})"})
    return f"Accepted: {gap.id} is now {amount:,.2f} {args['currency'].upper()}, citing {args['source_id']}."


def place_names(request: TripRequest, place: str) -> list[str]:
    """Ways a page can name a stop: its place, its cities, its country."""
    stop = next((s for s in request.stops if s.place == place), None)
    return [place, *stop.cities, country_name(stop.country)] if stop else [place]


def check(args: dict, gap: Gap, sources: dict[str, Source], names: list[str] = ()) -> tuple[str, str]:
    """(why a record can't be accepted, or "" when it can; the source's exact words): a real source, its exact
    words, the number in them, in that currency and unit, for this place, and a believable amount."""
    source = sources.get(str(args.get("source_id")))
    if source is None:
        return f"there is no source {args.get('source_id')}; cite one returned by a search.", ""
    quote = verbatim(str(args.get("quote") or ""), source.text)
    if not quote:
        return f"the quote is not in {source.id}; copy only the source's words, exactly.", ""
    try:
        amount = float(args["amount"])
    except (KeyError, TypeError, ValueError):
        return "amount must be a number.", ""
    if not stated(amount, quote):
        return f"{amount:g} is not stated in the quote.", ""
    if not currency_agrees(str(args.get("currency", "")), quote):
        return f"the quote is not in {args.get('currency')}: record the currency the source uses.", ""
    if gap.kind == "cost":  # a price for this item and place, in the unit asked for
        if not item_named(gap.item, quote):
            return f"the quote doesn't say it prices {gap.item}: quote the words that name it.", ""
        if not unit_fits(UNIT_OF[gap.item], quote):
            return f"the quote is not a price {UNIT_OF[gap.item]}.", ""
        if not about(source, quote, list(names) or [gap.key]):
            return f"the source doesn't say this price is for {gap.key}.", ""
    try:
        found_usd = fx.convert(amount, str(args.get("currency", "")).upper(), "USD")
    except fx.FxError:
        return f"unknown currency {args.get('currency')}.", ""
    found_usd *= 2 if args.get("one_way") and gap.item == "fare" else 1
    low, high = PLAUSIBLE_USD["hotel" if gap.item.endswith("hotel") else gap.item]
    if not low <= found_usd <= high:
        return f"{found_usd:,.0f} USD is not a believable price for {gap.item}.", ""
    if gap.estimate_usd and not gap.estimate_usd / FAR_FROM_ESTIMATE <= found_usd <= gap.estimate_usd * FAR_FROM_ESTIMATE:
        return (f"{found_usd:,.2f} USD is far from the estimate of {gap.estimate_usd:,.2f} USD: check the price is for "
                f"{gap.item} in this place, in the unit asked (e.g. per person per day), not something else.", "")
    return "", quote


def fill_cost(costs, item: str, amount: float, currency: str, source_id: str, quote: str, request: TripRequest):
    """The stop's rate for that item, now from a source, in the stop's currency."""
    value = round(fx.convert(amount, currency, costs.currency), 2)
    update = {"evidence": [*costs.evidence, CostEvidence(item=item, source_id=source_id, quote=quote[:200],
                                                         amount=amount, currency=currency, unit=UNIT_OF[item])]}
    if item.endswith("hotel"):
        style = item.removesuffix(" hotel")
        update["lodging_rates"] = [r for r in costs.lodging_rates if r.style != style] + \
                                  [LodgingRate(style=style, per_room_night=value)]
        if style == style_for(request):
            update["lodging_per_room_night"] = value
    else:
        field = {"food": "food_per_person_day", "local transport": "local_transport_per_person_day",
                 "vehicle": "vehicle_per_day", "guide": "guide_per_day"}[item]
        update[field] = value
    return costs.model_copy(update=update)


def fill_fare(result, amount: float, currency: str, source_id: str, quote: str):
    """A second source for the chosen flight: the fare becomes the middle of the two, citing both."""
    option = result.chosen_option()
    before = fx.convert(option.adult_fare.amount, option.adult_fare.currency, "USD")
    now = fx.convert(amount, currency, "USD")
    low, high = sorted((before, now))
    note = f"Typical return fare from 2 sources: {low:,.0f} to {high:,.0f} USD."
    updated = option.model_copy(update={
        "adult_fare": Money(amount=round((low + high) / 2), currency="USD"), "fare_basis": "quoted",
        "source_ids": list(dict.fromkeys([*option.source_ids, source_id])), "evidence": f"{option.evidence} | {quote}"[:300],
        "warnings": [*option.warnings, note]})
    options = list(result.options)
    options[result.chosen] = updated
    return result.model_copy(update={"options": options})
