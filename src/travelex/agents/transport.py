"""Transport: a ReAct loop per travel group finds 2-3 ways there and back; code picks the one that fits the group.
Fixed fare searches run first, and every price the sources state is pooled into a typical fare, so one sale banner
can't set the flight cost."""
import math
import re
import statistics
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from travelex.agents.common import event, own_sources, register, render, squash, tools_node
from travelex.agents.season import MONTH_NAMES
from travelex.limits import research_stopped
from travelex.llm import get_llm, structured
from travelex.schemas import FareQuote, Money, RouteList, RouteOption, Source, Traveller, TravelGroup, TripRequest
from travelex.state import GroupState, tool_rounds
from travelex.tools import fx, search
from travelex.tools.countries import country_name
from travelex.tools.search import extract_pages_tool, web_search_tool
from travelex.trip import is_vulnerable, resolve_nights

MAX_ROUNDS = 4
MAX_OPTIONS = 3
MAX_LAYOVER_VULNERABLE = 4  # hours, with children or seniors
OUTLIER = 3  # an option costing more than 3× the group's cheapest is almost always a bad extraction
MAX_FARE_USD = 5000  # per adult, economy, there and back: anything above is a bad extraction
MIN_FLIGHT_FARE_USD = 30  # below this a "fare" is a sale banner ("from as low as US$1.50") or a misread
NONSTOP_UNCONFIRMED = "No source confirms a non-stop flight; you may need to connect."
NONSTOP = re.compile(r"\b(direct|non-?stop)\b", re.IGNORECASE)
NEGATED = re.compile(r"\b(no|not|n't|without|lack|unavailable|isn't|aren't)\b", re.IGNORECASE)
ONE_WAY = re.compile(r"\bone[- ]way\b", re.IGNORECASE)
RETURN = re.compile(r"\b(round[- ]?trip|return)\b", re.IGNORECASE)
ONE_WAY_DOUBLED = "The source gives a one-way fare: doubled here to include the trip home."
FARE_UNCLEAR = "The source doesn't say if this fare is one-way or return: the trip home may cost extra."
# "from $350", "as low as US$1.50", "starting at INR 51,315": the lowest price ever advertised, not a typical fare
TEASER = re.compile(r"\b(?:from|as low as|starting (?:at|from))\s*(?:US\$|[A-Z]{3}\s?|[$€£₹¥]|Rs\.?\s?)\s?[\d,]+",
                    re.IGNORECASE)
TEASER_NOTE = "This is the lowest advertised price (\"from …\"): expect to pay more."
RAISED = "The price found looked like a sale or teaser fare, so the typical fare from the sources is used."
TYPICAL_SHARE = 0.7  # a flight fare below 70% of the typical one is almost always a teaser
FARE_SITES = ("google.com/travel", "kayak.", "skyscanner.", "momondo.", "cheapflights.", "expedia.", "cleartrip.",
              "makemytrip.", "ixigo.", "goibibo.", "wego.", "trip.com", "edreams.", "opodo.", "travelocity.")
MAX_FARE_PAGES = 3
TOOLS = [web_search_tool, extract_pages_tool]

TRANSPORT_PROMPT = """You find how one group of travellers gets to their trip and back home. Today is {today}.
Search for realistic ways from their home city to the first stop, and back home from the last stop:
- direct and 1-2 stop flights (name the airports and airlines);
- flight + overland when the nearest airport is far away or across a border, or flights are scarce;
- train, bus or ferry when that is practical for the distance.
Look up typical round-trip economy fares for the travel month, and make every search specific to this route.
Children or seniors need short connections (under 4 hours) and no overnight bus. A wheelchair user needs
few changes and airport assistance.
When you have 2-3 options with prices, reply with a one-line summary and no tool call."""

EXTRACT_PROMPT = """List this group's 2-3 best options, cheapest first, using only the numbered sources.
- route: the outbound journey only, from home to the FIRST stop, as IATA airport codes or place names for
  overland stops joined by "-", e.g. "MAA-CCU-PBH" or "DEL-IXB-Phuentsholing". Travel between stops is
  planned separately: never include it. A route with one flight means a non-stop flight: list it only if a
  source says that non-stop service exists.
- mode: "flight", "flight+overland" or "surface".
- legs: the outbound legs in order. depart and arrive only when a source gives them; overnight true for a
  night bus, train or ferry.
- adult_fare: for one adult, in the currency the source uses, exactly as the source gives it: never double or
  halve it. fare_covers: "return" when the source says round trip or return (out to the first stop and home
  from the last), "one_way" when it says one-way, "unclear" when it doesn't say. fare_basis "quoted" only when a
  source states this price for this route; otherwise give your best estimate from the sources, "estimated".
- child_fare only if a source quotes one. duration_hours: one way. max_layover_hours: the longest connection.
- kid_note: one short line about the journey for children, only if the group has children; else "".
  accessibility_note: likewise, only if someone has access needs. Never mention visas: they are checked
  separately.
- evidence: words copied exactly, character for character, from one source, showing the route exists and,
  for a quoted fare, its price. For a non-stop flight it must say the flight is direct or non-stop.
- source_ids: the sources behind each option. Never invent a route the sources don't support.
- fare_quotes: every economy flight price any source states for flying between the group's home city and the
  first stop (any airline, any connection), one entry per price: source_id, amount, currency, covers ("return",
  "one_way" or "unclear"), and words: the words around the price, copied exactly."""


def group_task(request: TripRequest, group: TravelGroup) -> str:
    by_label = {t.label: t for t in request.travellers}
    people = []
    for t in (by_label[label] for label in group.travellers):
        age = f" aged {t.age}" if t.age is not None else ""
        needs = f" ({', '.join(t.needs)})" if t.needs else ""
        people.append(f"{t.type}{age}{needs}")
    first, last = request.stops[0], request.stops[-1]
    going = f"{first.place}, {country_name(first.country)}" + (f" ({', '.join(first.cities)})" if first.cities else "")
    back = f"; home again from {last.place}, {country_name(last.country)}" if last is not first else ""
    when = (f"{request.start_date} to {request.end_date}" if request.start_date else request.month or "dates not set")
    nights, _ = resolve_nights(request)
    return (f"Group travelling from {group.origin}: {', '.join(people)}.\n"
            f"Going to: {going}{back}.\nWhen: {when}, {nights or '?'} nights.")


def fare_queries(request: TripRequest, group: TravelGroup) -> list[str]:
    """The same searches for every route, aimed at fare sites rather than airline sale pages."""
    first = request.stops[0]
    to = first.cities[0] if first.cities else first.place
    when = (f"{MONTH_NAMES[request.start_date.month - 1]} {request.start_date.year}" if request.start_date
            else f"{MONTH_NAMES[int(request.month[5:]) - 1]} {request.month[:4]}" if request.month else "")
    return [f"{group.origin} to {to} round trip flight price {when}".strip(),
            f"cheap flights {group.origin} to {to} {when} kayak skyscanner google flights".strip()]


def fare_research(request: TripRequest, group: TravelGroup, prefix: str, known: dict[str, Source],
                  taken: dict[str, Source] | None = None) -> dict[str, Source]:
    """Run the fixed fare searches and read the top fare-site pages in full: their fare tables ("typical price",
    "cheapest in March") are worth more than a search snippet."""
    results = search.search_all(fare_queries(request, group))
    pages = list(dict.fromkeys(r["url"] for r in results if any(site in r["url"] for site in FARE_SITES)))
    if pages:
        try:
            results += search.extract(pages[:MAX_FARE_PAGES])
        except Exception:  # fail soft: the snippets are still there
            pass
    new, _ = register(results, prefix, known, taken)
    return new


def transport_agent(state: GroupState) -> dict:
    started = time.perf_counter()
    group = state["group"]
    history = state.get("messages") or []
    prompt = TRANSPORT_PROMPT.format(today=state["today"].isoformat())
    fares = {} if history else fare_research(state["request"], group, state["source_prefix"],
                                             state.get("sources") or {}, state.get("taken"))
    task = group_task(state["request"], group) + ("\nFare pages for the flight have already been fetched: search for "
                                                  "the routes, connections and overland options." if fares else "")
    new = [] if history else [SystemMessage(prompt), HumanMessage(task)]
    convo = history + new
    llm = get_llm("transport")
    can_search = tool_rounds(convo) < MAX_ROUNDS
    reply = (llm.bind_tools(TOOLS) if can_search else llm).invoke(convo)
    if reply.tool_calls and can_search:
        return {"messages": new + [reply], "sources": fares,
                "events": [event("transport", "decided to search", f"{group.origin}: {len(reply.tool_calls)} tool calls",
                                 fare_queries(state["request"], group) if fares else None, started=started)]}
    final = AIMessage(str(reply.content) or "Step limit reached.") if reply.tool_calls else reply
    return {"messages": new + [final], "sources": fares,
            "events": [event("transport", "finished research", f"{group.origin}: {tool_rounds(convo)} rounds",
                             started=started)]}


transport_tools = tools_node("transport", "messages", TOOLS, who=lambda s: f"{s['group'].origin}: ")


def route_transport(state: GroupState) -> str:
    """More searching, or extract the routes: also when the run's research limit is reached."""
    last = state["messages"][-1]
    wants_tools = isinstance(last, AIMessage) and last.tool_calls
    return "transport_tools" if wants_tools and not research_stopped() else "extract_routes"


def extract_routes(state: GroupState) -> dict:
    started = time.perf_counter()
    group = state["group"]
    found = own_sources(state.get("sources") or {}, state["source_prefix"])
    messages = [SystemMessage(EXTRACT_PROMPT),
                HumanMessage(f"{group_task(state['request'], group)}\n\nSources:\n{render(found)}")]
    try:
        routes = structured(get_llm("transport"), RouteList, messages)
    except Exception as e:  # fail soft: no options means feasibility reports the gap
        return {"options": [], "errors": [f"transport ({group.origin}): {type(e).__name__}: {e}"],
                "events": [event("transport", "extraction failed", f"{group.origin}: {e}", started=started)]}
    sources = {s.id: s for s in found}
    options, dropped = clean(routes.options, state["request"].exact_dates, sources, routes.home_airports, group.origin)
    typical = typical_fare(routes.fare_quotes, sources)
    if typical:
        options = with_typical(options, typical)
    options = for_this_group(options[:MAX_OPTIONS], state["request"], group)
    summary = "; ".join(f"{o.route} {o.adult_fare.amount:,.0f} {o.adult_fare.currency}" for o in options)
    note = f" (dropped {'; '.join(dropped)})" if dropped else ""
    pooled = f"; typical return fare {typical[0]:,.0f} USD from {typical[3]} sources" if typical else ""
    return {"options": options,
            "events": [event("transport", "found options", f"{group.origin}: {summary or 'none'}{note}{pooled}",
                             started=started)]}


def typical_fare(quotes: list[FareQuote], sources: dict[str, Source]) -> tuple[float, float, float, int, list[str]] | None:
    """(middle, lowest, highest, how many sources, their IDs) in USD for a return economy flight, from every price the
    sources state: only words found verbatim, one-way prices doubled, teasers left out when at least two sources
    give real prices, one price per source. None with fewer than two sources."""
    found = []
    for q in quotes:
        source = sources.get(q.source_id)
        if source is None or squash(q.words) not in squash(source.text) or \
                f"{q.amount:.0f}" not in re.sub(r"\D", "", q.words):
            continue
        try:
            amount = fx.convert(q.amount, q.currency, "USD")
        except fx.FxError:
            continue
        one_way = q.covers == "one_way" or (q.covers == "unclear" and ONE_WAY.search(q.words) and not RETURN.search(q.words))
        amount *= 2 if one_way else 1
        if MIN_FLIGHT_FARE_USD <= amount <= MAX_FARE_USD:
            found.append((q.source_id, amount, TEASER.search(q.words) is not None))
    real = [f for f in found if not f[2]]
    use = real if len({f[0] for f in real}) >= 2 else found
    by_source: dict[str, list[float]] = {}
    for sid, amount, _ in use:
        by_source.setdefault(sid, []).append(amount)
    prices = sorted(statistics.median(v) for v in by_source.values())
    if len(prices) < 2:
        return None
    return statistics.median(prices), prices[0], prices[-1], len(prices), list(by_source)


def with_typical(options: list[RouteOption], typical: tuple[float, float, float, int, list[str]]) -> list[RouteOption]:
    """Flights below the typical fare that look like teasers (or far below it) take the typical fare instead, and
    cite the sources it was pooled from; every flight shows the range the sources gave."""
    middle, low, high, n, pooled = typical
    note = f"Typical return fare from {n} sources: {low:,.0f} to {high:,.0f} USD."
    out = []
    for o in options:
        if o.mode != "flight":
            out.append(o)
            continue
        warnings = list(o.warnings)
        if usd(o) < middle and (TEASER.search(o.evidence) or usd(o) < TYPICAL_SHARE * middle):
            o = o.model_copy(update={"adult_fare": Money(amount=round(middle), currency="USD"), "child_fare": None,
                                     "fare_basis": "estimated", "fare_covers": "return", "source_ids": pooled,
                                     "evidence": note})
            warnings = [w for w in warnings if w not in (TEASER_NOTE, FARE_UNCLEAR, ONE_WAY_DOUBLED)] + [RAISED]
        out.append(o.model_copy(update={"warnings": [*warnings, note]}))
    return out



def mode_of(option: RouteOption) -> str:
    """The legs decide the mode, not the model's label."""
    modes = {leg.mode for leg in option.legs}
    if not modes:
        return option.mode
    return "flight" if modes == {"flight"} else "flight+overland" if "flight" in modes else "surface"


def is_nonstop_flight(o: RouteOption) -> bool:
    return o.mode == "flight" and len(o.route.split("-")) == 2


def evidence_found(o: RouteOption, sources: dict[str, Source]) -> bool:
    """The evidence must appear word for word in one of the option's cited sources."""
    quote = squash(o.evidence)
    return bool(quote) and any(quote in squash(sources[i].text) for i in o.source_ids if i in sources)


def nonstop_supported(o: RouteOption, sources: dict[str, Source]) -> bool:
    """The model likes to turn "Dubai to Paro flights" into a non-stop flight, and a source saying "direct
    flights are not available" fools a keyword search. So: verbatim evidence that itself says direct or
    non-stop, with no negation."""
    return evidence_found(o, sources) and NONSTOP.search(o.evidence) is not None and not NEGATED.search(o.evidence)


def fare_in_evidence(o: RouteOption) -> bool:
    digits = re.sub(r"\D", "", o.evidence)
    return f"{o.adult_fare.amount:.0f}" in digits


def starts_at_home(o: RouteOption, home: list[str], origin: str) -> bool:
    first = o.route.split("-")[0].strip().casefold()
    return not home or first in {h.casefold() for h in home} or first == origin.casefold()


def covers(o: RouteOption) -> str:
    """What the fare covers, as its evidence says; the model's label when the evidence says neither or both."""
    one_way, back = ONE_WAY.search(o.evidence), RETURN.search(o.evidence)
    if one_way and not back:
        return "one_way"
    if back and not one_way:
        return "return"
    return o.fare_covers


def as_return(o: RouteOption) -> RouteOption:
    """A one-way price pays for half the journey: double it, so the budget includes getting home."""
    def double(m: Money | None) -> Money | None:
        return m.model_copy(update={"amount": m.amount * 2}) if m else None
    return o.model_copy(update={"adult_fare": double(o.adult_fare), "child_fare": double(o.child_fare),
                                "fare_basis": "estimated", "fare_covers": "return"})


def clean(options: list[RouteOption], exact_dates: bool, sources: dict[str, Source],
          home: list[str] = (), origin: str = "") -> tuple[list[RouteOption], list[str]]:
    """Code-side fixes for what the model gets wrong: unknown source IDs, a mode that contradicts the legs,
    clock times on a month-only trip, "quoted" fares the evidence doesn't show, one-way fares (doubled) and
    fares that may be one-way (kept, with a warning), a non-stop claim no source makes (kept, with a warning),
    routes that don't start from home (a hop between stops), impossible fares, repeated routes, and fares
    more than OUTLIER × the group's cheapest.
    Returns (kept, "route: why" for each dropped)."""
    fixed, dropped = {}, []
    for o in options:
        legs = o.legs if exact_dates else [leg.model_copy(update={"depart": None, "arrive": None}) for leg in o.legs]
        o = o.model_copy(update={"legs": legs, "source_ids": [i for i in o.source_ids if i in sources]})
        o = o.model_copy(update={"mode": mode_of(o), "fare_covers": covers(o)})
        warnings = []
        if o.fare_basis == "quoted" and not (evidence_found(o, sources) and fare_in_evidence(o)):
            o = o.model_copy(update={"fare_basis": "estimated"})
        if o.fare_covers == "one_way":
            o, warnings = as_return(o), [ONE_WAY_DOUBLED]
        elif o.fare_covers == "unclear":
            warnings.append(FARE_UNCLEAR)
        if TEASER.search(o.evidence):
            warnings.append(TEASER_NOTE)
        if is_nonstop_flight(o) and not nonstop_supported(o, sources):
            warnings.append(NONSTOP_UNCONFIRMED)
        o = o.model_copy(update={"warnings": warnings})
        if not starts_at_home(o, home, origin):
            dropped.append(f"{o.route}: does not start from home")
        elif usd(o) > MAX_FARE_USD or ("flight" in o.mode and usd(o) < MIN_FLIGHT_FARE_USD):
            fare = f"{o.adult_fare.amount:,.0f}" if o.adult_fare.amount >= 10 else f"{o.adult_fare.amount:.2f}"
            dropped.append(f"{o.route}: fare of {fare} {o.adult_fare.currency} is not plausible")
        elif o.route not in fixed or usd(o) < usd(fixed[o.route]):
            fixed[o.route] = o
    cheapest = min((usd(o) for o in fixed.values()), default=math.inf)
    kept = [o for o in fixed.values() if usd(o) <= OUTLIER * cheapest]
    dropped += [f"{o.route}: fare far above the others" for o in fixed.values() if o not in kept]
    return kept, dropped


def for_this_group(options: list[RouteOption], request: TripRequest, group: TravelGroup) -> list[RouteOption]:
    """No kid notes for a group without children, no access notes without access needs: the model adds them anyway."""
    people = [t for t in request.travellers if t.label in group.travellers]
    kids = any(t.type != "adult" for t in people)
    access = any(n != "senior" for t in people for n in t.needs)
    return [o.model_copy(update={"kid_note": o.kid_note if kids else "",
                                 "accessibility_note": o.accessibility_note if access else ""}) for o in options]


def fits(option: RouteOption, vulnerable: bool) -> bool:
    if not vulnerable:
        return True
    long_layover = (option.max_layover_hours or 0) > MAX_LAYOVER_VULNERABLE
    night_bus = any(leg.overnight and leg.mode == "bus" for leg in option.legs)
    return not (long_layover or night_bus)


def usd(option: RouteOption) -> float:
    try:
        return fx.convert(option.adult_fare.amount, option.adult_fare.currency, "USD")
    except fx.FxError:
        return math.inf


def choose_option(options: list[RouteOption], travellers: list[Traveller]) -> int:
    """The cheapest option that suits the group; with a wheelchair user, the fewest legs first."""
    if not options:
        return 0
    vulnerable = any(is_vulnerable(t) for t in travellers)
    wheelchair = any("wheelchair" in t.needs for t in travellers)
    candidates = [i for i, o in enumerate(options) if fits(o, vulnerable)] or list(range(len(options)))
    if wheelchair:
        return min(candidates, key=lambda i: (len(options[i].legs) or 1, usd(options[i])))
    return min(candidates, key=lambda i: usd(options[i]))
