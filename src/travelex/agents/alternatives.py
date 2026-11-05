"""Alternatives: when the budget or the season doesn't work, propose 2-3 changed trips, re-run only the research
that the change affects, and let the user pick one, keep their plan, or say what they want instead (interrupt)."""
import re
import time
from datetime import datetime, timedelta

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import ContextThreadPoolExecutor
from langgraph.types import Overwrite, interrupt

from travelex.agents.common import event
from travelex.agents.intake import describe, tidy
from travelex.agents.itinerary import named_cities, same_city
from travelex.llm import get_llm, structured
from travelex.schemas import Alternative, AlternativeIdea, AlternativeIdeas, TripChanges, TripRequest
from travelex.state import TripState
from travelex.tools import fx
from travelex.trip import parse_duration, party, resolve_nights, style_for

MAX_ALT_ROUNDS = 2
MAX_SHOWN = 3  # alternatives offered at once, after a second try
RESEARCH_STEPS = 100

ALTERNATIVES_PROMPT = """You help travellers whose trip plan doesn't work. Today is {today}.
Propose 2-3 genuinely different fixes, each changing as little as possible:
- Over budget: look at the biggest costs and attack those: fewer nights, cheaper hotels (lodging_style
  "budget"), a cheaper month, or a similar destination that costs less (e.g. one without a daily tourist fee).
- A season to avoid: a better month for the same place, or a similar place that is good in that month.
- Keep their wishes in mind: interests, diet, children, seniors, accessibility, and keep it reachable from
  their home cities.
- For a new destination give stops with place and ISO country code.
- changes must contain everything the title promises: "budget hotels" needs lodging_style "budget",
  "fewer nights" needs duration_text. Only what is in changes is applied.
- A new month is "YYYY-MM" on or after today: next year's June if this June has passed.
- Use the cost numbers given: when the fixed costs alone are over the budget, fewer nights or cheaper hotels
  cannot fix it; look for a destination with cheaper flights and fees instead.
- Each idea changes something different: never two cheaper versions of the same trip, and never what the trip
  already has (budget hotels when it already uses budget hotels).
- why: what it fixes and what the travellers give up (their destination, month or nights). Code works out
  whether it really fits the budget: never claim it does.
- Raise the budget only if nothing else can work, and say so in why. Never lower it.
- Never repeat an idea the travellers already turned down."""

PER_NIGHT = ("lodging", "food", "guide", "vehicle", "local_transport", "destination_fee")


def problems(state: TripState) -> list[str]:
    """What the user should know is wrong: over budget, or a stop in a season to avoid."""
    request, f = state["request"], state.get("feasibility")
    found = []
    if f and f.budget is not None and f.gap > 0:
        found.append(f"Over budget by {f.gap:,.0f} {f.currency} ({f.counted:,.0f} vs {f.budget:,.0f}).")
    season = state.get("season") or {}
    for stop in request.stops:
        verdict = season.get(stop.place)
        if verdict and verdict.verdict == "avoid":
            found.append(f"{stop.place} is best avoided then: {'; '.join(verdict.reasons)}.")
    return found


def route_after_feasibility(state: TripState) -> str:
    if problems(state) and state.get("alt_round", 0) < MAX_ALT_ROUNDS:
        return "alternatives_agent"
    return "itinerary_planner"


def cost_breakdown(state: TripState) -> str:
    f = state.get("feasibility")
    if not f:
        return "no costs yet"
    totals: dict[str, float] = {}
    for line in f.lines:
        totals[line.category] = totals.get(line.category, 0) + line.amount_report
    return ", ".join(f"{cat} {amount:,.0f}" for cat, amount in sorted(totals.items(), key=lambda kv: -kv[1])) + \
        f" (total {f.total:,.0f} {f.currency})"


def by_category(f) -> dict[str, float]:
    by: dict[str, float] = {}
    for line in f.lines:
        by[line.category] = by.get(line.category, 0) + line.amount_report
    return by


def cost_split(f, nights: int) -> tuple[float, float]:
    """(about what a night costs, what is paid however long the trip is), in the budget currency."""
    nightly = sum(by_category(f).get(c, 0) for c in PER_NIGHT) / nights
    return nightly, max(f.counted - nightly * nights, 0)


def destination_can_fit(state: TripState) -> bool:
    """Whether the trip's own destination could fit the budget for a real trip: at least half the nights asked
    for, and two. If not (3 adults to Bhutan on $800), only another destination can fix it."""
    f, nights = state.get("feasibility"), resolve_nights(state["request"])[0]
    if not f or f.budget is None or f.gap <= 0 or not nights:
        return True  # a season problem: a better month keeps the place
    nightly, fixed = cost_split(f, nights)
    return fixed + nightly * max(2, nights // 2) <= f.budget


def levers(state: TripState) -> list[str]:
    """How the costs scale, so the model can see which fixes can work: what a night costs, what is paid however
    long the trip is, and how many nights would fit. Live, it kept offering a shorter Bhutan trip when the flights
    and the daily fee for one night already broke the budget."""
    f, request = state.get("feasibility"), state["request"]
    nights, _ = resolve_nights(request)
    if not f or not nights:
        return []
    by = by_category(f)
    nightly, fixed = cost_split(f, nights)
    cur = f.currency
    out = [f"Each night costs about {nightly:,.0f} {cur} (hotels, food, guides, vehicles, daily fees); about "
           f"{fixed:,.0f} {cur} (flights, visas, one-off fees) is paid however long the trip is."]
    if f.budget is not None and f.gap > 0:
        room = f.budget - fixed
        if room <= 0:
            out.append(f"The fixed costs alone are over the {f.budget:,.0f} {cur} budget: fewer nights or cheaper "
                       f"hotels cannot fit it.")
        elif room < nightly:
            out.append("Even one night is over the budget: fewer nights cannot fit it.")
        elif (fit := int(room // nightly)) < nights:
            out.append(f"Fewer nights alone fits at {fit} nights or fewer (now {nights}).")
    if by.get("lodging"):
        out.append(f"Hotels ({style_for(request)}) are {by['lodging']:,.0f} {cur} of the total.")
    return out


def month_name(month: str | None) -> str:
    try:
        return datetime.strptime(month, "%Y-%m").strftime("%B %Y")
    except (TypeError, ValueError):
        return month or "the dates asked"


def trip_month(request: TripRequest) -> str | None:
    return request.month or (f"{request.start_date:%Y-%m}" if request.start_date else None)


def trade_offs(old: TripRequest, new: TripRequest) -> list[str]:
    """What an alternative changes, worked out from the two trips rather than the model's words."""
    out = []
    before, after = [s.place for s in old.stops], [s.place for s in new.stops]
    if [p.casefold() for p in before] != [p.casefold() for p in after]:
        out.append(f"goes to {' → '.join(after)} instead of {' → '.join(before)}")
    if trip_month(new) != trip_month(old):
        out.append(f"travels in {month_name(trip_month(new))} instead of {month_name(trip_month(old))}")
    was, now = resolve_nights(old)[0], resolve_nights(new)[0]
    if was and now and now != was:
        n = abs(was - now)
        out.append(f"{n} {'fewer' if now < was else 'more'} night{'s' * (n != 1)} ({now} instead of {was})")
    if style_for(new) != style_for(old):
        out.append(f"{style_for(new)} hotels instead of {style_for(old)}")
    if new.budget and old.budget and new.budget.amount != old.budget.amount:
        out.append(f"a budget of {new.budget.amount:,.0f} {new.budget.currency} instead of "
                   f"{old.budget.amount:,.0f} {old.budget.currency}")
    return out


def still_wrong(a: Alternative) -> list[str]:
    f = a.feasibility
    if f is None:
        return []
    out = [f"still over budget by {f.gap:,.0f} {f.currency}"] if f.budget is not None and f.gap > 0 else []
    return out + [f"{place} is still best avoided then" for place, v in a.season.items() if v.verdict == "avoid"]


def ranked(alternatives: list[Alternative]) -> list[Alternative]:
    """The ones that fix everything first, then the ones closest to the budget."""
    gap = lambda a: a.feasibility.gap if a.feasibility else float("inf")
    return sorted(alternatives, key=lambda a: (not a.fixes(), gap(a)))


def upcoming(month: str, today) -> str:
    """The next time that month comes round: "go in June" proposed in October means next June."""
    try:
        year, m = (int(x) for x in month.split("-"))
    except ValueError:
        return month
    return f"{year + (f'{year}-{m:02d}' < f'{today:%Y-%m}'):04d}-{m:02d}" if 1 <= m <= 12 else month


def checked(changes: TripChanges, request: TripRequest, today) -> TripChanges:
    """Code-side fixes: a month already past moves to next year, and the budget can only go up
    (the model has "fixed" a $5000 trip by setting the budget to $1500)."""
    update = {}
    if changes.month:
        update["month"] = upcoming(changes.month, today)
    if changes.budget and request.budget and fx.convert(
            changes.budget.amount, changes.budget.currency, request.budget.currency) <= request.budget.amount:
        update["budget"] = None
    return changes.model_copy(update=update) if update else changes


def apply_changes(request: TripRequest, changes: TripChanges, today) -> TripRequest:
    update: dict = {}
    if changes.stops:
        update["stops"] = changes.stops
    if changes.month:
        update |= {"month": changes.month, "start_date": None, "end_date": None, "exact_dates": False}
    if changes.duration_text:
        days, nights = parse_duration(changes.duration_text)
        if days or nights:
            update |= {"duration_text": changes.duration_text, "duration_range": days, "nights": nights}
            if request.start_date and "start_date" not in update:  # keep the start day, move the end
                update["end_date"] = request.start_date + timedelta(days=nights or days[1] - 1)
    if changes.lodging_style:
        update["lodging_style"] = changes.lodging_style
    if changes.budget:
        update["budget"] = changes.budget
    return tidy(request.model_copy(update=update), today)


def research_config(config: RunnableConfig, title: str) -> dict:
    """The config for one alternative's research: the run's tracing callbacks, and its shared limits and
    concurrency, so costing alternatives draws on the same budget of searches, tokens and time as the trip."""
    child = {"callbacks": config.get("callbacks"), "recursion_limit": RESEARCH_STEPS, "run_name": f"alternative: {title}",
             "configurable": {"limits": config.get("configurable", {}).get("limits")}}
    if config.get("max_concurrency"):
        child["max_concurrency"] = config["max_concurrency"]
    return child


def make_alternatives_agent(research):
    """research: the compiled season → transport/visas/costs → meeting → feasibility graph."""

    def evaluate(idea: AlternativeIdea, namespace: str, state: TripState, config: RunnableConfig) -> Alternative | None:
        """The idea applied and costed; None when it changes nothing (budget hotels on a budget-hotel trip)."""
        idea = idea.model_copy(update={"changes": checked(idea.changes, state["request"], state["today"])})
        request = apply_changes(state["request"], idea.changes, state["today"])
        changes = trade_offs(state["request"], request)
        if not changes:
            return None
        seed = {"raw_query": state["raw_query"], "today": state["today"], "namespace": namespace,
                "request": request, "group_results": state.get("group_results") or {},
                "costs": state.get("costs") or {}}
        child = research_config(config, idea.title)
        try:
            out = research.invoke(seed, child)
        except Exception as e:  # fail soft: show the idea without numbers rather than drop it
            return Alternative(title=idea.title, why=idea.why, changes=idea.changes, request=request,
                               error=f"{type(e).__name__}: {e}", trade_offs=changes)
        places, groups = {s.place for s in request.stops}, {g.id for g in request.groups}
        only = lambda d, keep: {k: v for k, v in (d or {}).items() if k in keep}  # drop the reused old trip's
        alternative = Alternative(title=idea.title, why=idea.why, changes=idea.changes, request=request,
                                  feasibility=out.get("feasibility"), meeting=out.get("meeting"),
                                  season=only(out.get("season"), places),
                                  group_results=only(out.get("group_results"), groups),
                                  costs=only(out.get("costs"), places), sources=out.get("sources") or {})
        return alternative.model_copy(update={"trade_offs": changes + still_wrong(alternative)})

    def evaluate_all(ideas: list[AlternativeIdea], base: str, first: int, state: TripState,
                     config: RunnableConfig) -> list[Alternative]:
        if not ideas:
            return []
        with ContextThreadPoolExecutor(max_workers=len(ideas)) as pool:  # keeps the run's context in each thread
            done = list(pool.map(lambda pair: evaluate(pair[1], f"{base}{pair[0]}.", state, config),
                                 list(enumerate(ideas, first))))
        return [a for a in done if a is not None]

    def alternatives_agent(state: TripState, config: RunnableConfig) -> dict:
        started = time.perf_counter()
        rounds = state.get("alt_round", 0) + 1
        request = state["request"]
        offered = [a.title for a in state.get("alternatives") or []]
        lines = [f"Trip: {describe(request)}", f"Party: {party(request)}",
                 f"Interests: {', '.join(request.interests) or 'none given'}; diet: {', '.join(request.diet) or 'none'}",
                 "Problems:", *[f"- {p}" for p in problems(state)], f"Costs: {cost_breakdown(state)}", *levers(state)]
        if offered:
            lines.append(f"Already offered and turned down: {'; '.join(offered)}")
        if state.get("alt_feedback"):
            lines.append(f"What the travellers said: {state['alt_feedback']}")
        prompt = ALTERNATIVES_PROMPT.format(today=state["today"].isoformat())
        propose = lambda text: structured(get_llm("alternatives"), AlternativeIdeas,
                                          [SystemMessage(prompt), HumanMessage("\n".join(text))]).ideas
        base = f"{state.get('namespace', '')}A{rounds}."
        ideas = propose(lines)
        alternatives = evaluate_all(ideas, base, 1, state, config)
        events = []
        if not any(a.fixes() for a in alternatives):  # one more try, told what each idea cost and why it fell short
            tried = [f"- {summary(a)}" for a in alternatives] or ["- nothing that changes the trip"]
            more = propose(lines + ["Tried, and none of these fixes it:", *tried,
                                    "Propose different fixes that do, using the cost numbers above. If only a bigger "
                                    "budget can work, include that and say so in why."])
            alternatives = ranked(alternatives + evaluate_all(more, base, len(ideas) + 1, state, config))[:MAX_SHOWN]
            events.append(event("alternatives", "tried again", "no first idea fixed the problems"))
        events += [event("alternatives", "proposed", summary(a)) for a in alternatives]
        events.append(event("alternatives", f"round {rounds}", f"{len(alternatives)} alternatives", started=started))
        return {"alternatives": alternatives, "alt_round": rounds, "alt_feedback": None, "events": events,
                "errors": [f"alternative ({a.title}): {a.error}" for a in alternatives if a.error]}

    return alternatives_agent


def describe_changes(changes: TripChanges) -> str:
    """What the alternative really changes, from the structured changes, not the model's title (which can
    promise "budget hotels" while only the length changed)."""
    parts = []
    if changes.stops:
        parts.append("go to " + " → ".join(s.place for s in changes.stops))
    if changes.month:
        parts.append(f"travel in {changes.month}")
    if changes.duration_text:
        parts.append(changes.duration_text)
    if changes.lodging_style:
        parts.append(f"{changes.lodging_style} hotels")
    if changes.budget:
        parts.append(f"budget {changes.budget.amount:,.0f} {changes.budget.currency}")
    return ", ".join(parts) or "no change"


def summary(a: Alternative) -> str:
    what = f"{a.title} [{describe_changes(a.changes)}]"
    if a.feasibility is None:
        return f"{what}: could not be costed ({a.error or 'no result'})"
    f = a.feasibility
    fits = f.status()
    season = ", ".join(f"{p} {v.verdict}" for p, v in a.season.items())
    return f"{what}: {f.counted:,.0f} {f.currency}, {fits}" + (f"; season: {season}" if season else "") + \
        (f"; {'; '.join(a.trade_offs)}" if a.trade_offs else "")


def moves(a: Alternative) -> bool:
    return any(t.startswith("goes to ") for t in a.trade_offs)


def raises_budget(a: Alternative) -> bool:
    return any(t.startswith("a budget of ") for t in a.trade_offs)


def best(alternatives: list[Alternative], keep_destination: bool = False, spend_more: bool = True) -> int:
    """The cheapest one that fixes everything, else the one closest to the budget. Picking for the user (auto
    mode): never a bigger budget nobody agreed to (its gap is measured against that bigger budget, so it looks
    closest), and, when the destination could fit, only among those that keep it, so "see Prague and Vienna" is
    never swapped for Slovakia without asking."""
    def total(i):
        f = alternatives[i].feasibility
        return f.counted if f else float("inf")

    every = list(range(len(alternatives)))
    affordable = [i for i in every if spend_more or not raises_budget(alternatives[i])] or every
    pool = [i for i in affordable if not (keep_destination and moves(alternatives[i]))] or affordable
    fixing = [i for i in pool if alternatives[i].fixes()]
    if fixing:
        return min(fixing, key=total)
    return min(pool, key=lambda i: (alternatives[i].feasibility.gap
                                    if alternatives[i].feasibility else float("inf")))


def parse_choice(answer, alternatives: list[Alternative]) -> int | str | None:
    """An index (0-based), "keep" to keep the current plan, or None when the answer is feedback."""
    if isinstance(answer, dict):
        answer = answer.get("choice", answer.get("feedback", ""))
    if isinstance(answer, int):
        return answer - 1 if 1 <= answer <= len(alternatives) else "keep" if answer == 0 else None
    text = str(answer).strip().lower()
    if text in ("", "0", "keep", "keep mine", "original", "keep the original"):
        return "keep"
    if re.search(r"\bcheapest\b", text) and not re.search(r"\b(none|not|cheaper)\b", text):
        return best(alternatives)
    if re.fullmatch(r"(?:the )?best(?: one)?|pick (?:one )?for me|you (?:choose|pick|decide)", text):
        return "auto"
    match = re.fullmatch(r"(?:option|choice|#)?\s*(\d)", text)
    if match and 1 <= int(match.group(1)) <= len(alternatives):
        return int(match.group(1)) - 1
    return None


def switch_to(alternative: Alternative, how: str, food: dict | None = None) -> dict:
    """Adopt an alternative: its research replaces the old results instead of merging into them, and food guides
    started early survive only for cities the new trip still names."""
    named = named_cities(alternative.request)
    kept_food = {c: g for c, g in (food or {}).items() if any(same_city(c, n) for n in named)}
    return {"request": alternative.request, "feasibility": alternative.feasibility, "meeting": alternative.meeting,
            "food": Overwrite(kept_food),
            "season": Overwrite(alternative.season), "group_results": Overwrite(alternative.group_results),
            "costs": Overwrite(alternative.costs), "sources": alternative.sources,
            "chosen_alternative": alternative.title, "alt_feedback": None,
            "events": [event("choose_alternative", how, summary(alternative))]}


def choose_alternative(state: TripState, config: RunnableConfig) -> dict:
    """Resume with an option number, "keep", or any other text as feedback for another round."""
    alternatives = [a for a in state.get("alternatives") or [] if a.feasibility is not None]
    if not alternatives:
        return {"events": [event("choose_alternative", "kept the plan", "no alternative could be costed")]}
    auto = lambda: best(alternatives, keep_destination=destination_can_fit(state), spend_more=False)
    if config.get("configurable", {}).get("auto_approve"):
        return switch_to(alternatives[auto()], "auto-picked", state.get("food"))

    answer = interrupt({"kind": "alternatives", "problems": problems(state),
                        "current": describe(state["request"]),
                        "alternatives": [{"n": i, "title": a.title, "why": a.why, "summary": summary(a),
                                          "trade_offs": a.trade_offs, "fixes": a.fixes()}
                                         for i, a in enumerate(alternatives, 1)]})
    choice = parse_choice(answer, alternatives)
    if choice == "auto":
        return switch_to(alternatives[auto()], "picked for the user", state.get("food"))
    if choice == "keep":
        return {"alt_round": MAX_ALT_ROUNDS, "events": [event("choose_alternative", "kept the plan", "")]}
    if choice is None:
        return {"alt_feedback": str(answer.get("feedback") if isinstance(answer, dict) else answer),
                "events": [event("choose_alternative", "asked for others", str(answer))]}
    return switch_to(alternatives[choice], "user picked", state.get("food"))


def route_after_choice(state: TripState) -> str:
    if state.get("alt_feedback") and state.get("alt_round", 0) < MAX_ALT_ROUNDS:
        return "alternatives_agent"
    return "itinerary_planner"
