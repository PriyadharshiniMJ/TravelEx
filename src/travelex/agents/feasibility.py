"""Budget check: pure Python over the research results. The LLM never adds up money."""
from langsmith import traceable

from travelex.agents.common import event
from travelex.agents.transport import ONE_WAY_DOUBLED
from travelex.schemas import (CostLine, Feasibility, GroupResult, LodgingCost, MeetingPlan, Money, RouteOption,
                              StopCosts, Traveller, TripRequest)
from travelex.tools import fx
from travelex.tools.fees import fee_lines
from travelex.trip import allocate_nights, next_stop, resolve_nights, rooms_needed, style_for

DEFAULT_CURRENCY = "USD"
# Seed: share of the adult fare when the source quotes no child fare.
FARE_SHARE = {"adult": 1.0, "child": 0.75, "infant": 0.1}


def traveller_fare(t: Traveller, option: RouteOption) -> Money:
    if t.type == "child" and option.child_fare:
        return option.child_fare
    return Money(amount=option.adult_fare.amount * FARE_SHARE[t.type], currency=option.adult_fare.currency)


def transport_lines(request: TripRequest, results: list[GroupResult]) -> tuple[list[tuple], list[str]]:
    """The fares to and from home, and the journeys from home that have no price."""
    by_label = {t.label: t for t in request.travellers}
    by_group = {r.group_id: r for r in results}
    lines, unpriced = [], []
    for group in request.groups:
        option = by_group[group.id].chosen_option() if group.id in by_group else None
        if option is None:
            unpriced.append(f"No transport found from {group.origin}.")
            continue
        out, home = journeys(request, group.origin, option)
        said = {"ids": option.source_ids, "quote": option.evidence}
        for label in group.travellers:
            fare = traveller_fare(by_label[label], option)
            lines += [("transport", out, label, fare.amount / 2, fare.currency, said),
                      ("transport", home, label, fare.amount / 2, fare.currency, said)]
    return lines, unpriced


def journeys(request: TripRequest, origin: str, option: RouteOption) -> tuple[str, str]:
    """Cost-line labels for the two halves of a return fare: the way out, and home from the last stop."""
    how = f"{option.mode}, " + ("one-way fare" if ONE_WAY_DOUBLED in option.warnings else "half the return fare")
    back = ("-".join(reversed(option.route.split("-"))) if len(request.stops) < 2
            else f"{request.stops[-1].place} to {origin}")
    return f"Getting there: {option.route} ({how})", f"Getting home: {back} ({how})"


def visa_lines(results: list[GroupResult]) -> list[tuple]:
    """One fee per traveller per country, or per zone: one Schengen visa covers France, Italy and Switzerland."""
    lines, seen = [], set()
    for v in (v for r in results for v in r.visas):
        if not (v.rule and v.rule.fee):
            continue
        key = (v.traveller, v.rule.zone or v.rule.country)
        if key not in seen:
            seen.add(key)
            fee_quotes = [c for c in v.rule.citations if c.supports == "fee"]
            said = {"urls": [c.url for c in fee_quotes], "quote": fee_quotes[0].quote if fee_quotes else ""}
            unconfirmed = ", fee unconfirmed" if "fee" in v.rule.unconfirmed else ""
            lines.append(("visa", f"{v.rule.zone or v.country} {v.rule.document}{unconfirmed}", v.traveller,
                          v.rule.fee, v.rule.fee_currency, said if not unconfirmed else {}))
    return lines


def people(n: int) -> str:
    return f"{n} {'person' if n == 1 else 'people'}"


def said(costs: StopCosts, item: str) -> dict:
    """The checked words behind one of a stop's amounts, if a source states it."""
    found = [x for x in costs.evidence if x.item == item]
    return {"ids": list(dict.fromkeys(x.source_id for x in found)), "quote": found[0].quote} if found else {}


def stay_lines(request: TripRequest, costs: StopCosts, nights: int, days: int, style: str) -> list[tuple]:
    paying = sum(t.type != "infant" for t in request.travellers)  # infants eat and ride free
    cur, where = costs.currency, costs.place
    lines = [("lodging", f"{where}: {style} hotel, {costs.room_config}, {nights} nights", None,
              costs.room_rate(style) * costs.rooms * nights, cur, said(costs, f"{style} hotel")),
             ("food", f"{where}: food, {people(paying)} × {days} days", None,
              costs.food_per_person_day * paying * days, cur, said(costs, "food"))]
    if costs.guide_per_day:
        lines.append(("guide", f"{where}: guide × {days} days", None, costs.guide_per_day * days, cur,
                      said(costs, "guide")))
    if costs.vehicle_per_day and (costs.vehicle_needed or costs.guide_required):  # a car nobody needs is not a cost
        lines.append(("vehicle", f"{where}: car with driver × {days} days", None, costs.vehicle_per_day * days, cur,
                      said(costs, "vehicle")))
    elif costs.local_transport_per_person_day:
        lines.append(("local_transport", f"{where}: local transport, {people(paying)} × {days} days", None,
                      costs.local_transport_per_person_day * paying * days, cur, said(costs, "local transport")))
    if costs.other_per_trip:
        lines.append(("other", f"{where}: other costs", None, costs.other_per_trip, cur, said(costs, "other")))
    return lines


def waiting_lines(request: TripRequest, meeting: MeetingPlan | None, costs: dict[str, StopCosts],
                  style: str) -> list[tuple]:
    """Groups that arrive before the others wait at the meeting point: their extra nights of hotel and food, at
    the first stop's rates."""
    first = costs.get(request.stops[0].place) if request.stops else None
    if meeting is None or first is None:
        return []
    by_label = {t.label: t for t in request.travellers}
    lines = []
    for group in request.groups:
        nights = meeting.buffer_nights.get(group.id, 0)
        if nights <= 0:
            continue
        members = [by_label[label] for label in group.travellers if label in by_label]
        paying, rooms = sum(t.type != "infant" for t in members), rooms_needed(members)
        where = f"Waiting for the others in {meeting.meeting_point} ({group.origin} group)"
        lines += [("lodging", f"{where}: {style} hotel, {rooms} room{'s' * (rooms > 1)}, {nights} night{'s' * (nights > 1)}",
                   None, first.room_rate(style) * rooms * nights, first.currency, said(first, f"{style} hotel")),
                  ("food", f"{where}: food, {people(paying)} × {nights} day{'s' * (nights > 1)}", None,
                   first.food_per_person_day * paying * nights, first.currency, said(first, "food"))]
    return lines


@traceable(name="feasibility.assess")
def assess(request: TripRequest, results: list[GroupResult], costs: dict[str, StopCosts],
           table: fx.Rates | None = None, style: str | None = None, meeting: MeetingPlan | None = None) -> Feasibility:
    """costs maps each stop's place to its researched rates. Every total is in the budget currency.
    style: the hotel level to price; the one the user asked for when None. meeting: prices the nights early
    arrivals wait for the others."""
    table = table or fx.rates()
    style = style or style_for(request)
    report = request.budget.currency if request.budget else DEFAULT_CURRENCY
    home, unpriced = transport_lines(request, results)
    reasons: list[str] = []
    raw = home + visa_lines(results)

    nights_total, _ = resolve_nights(request)
    plan = allocate_nights(request)
    fee_names: set[str] = set()
    charged_per_trip: set[tuple[str, str]] = set()  # (traveller, fee): a per-trip fee is paid once
    by_label = {t.label: t for t in request.travellers}
    for i, (stop, nights) in enumerate(plan):
        stop_costs = costs.get(stop.place)
        nxt = next_stop(request, i)  # the cruise between two ports is the travellers' own: not priced here
        if stop_costs is None:
            unpriced.append(f"Costs for {stop.place} are missing.")
            if nxt:
                unpriced.append(f"{stop.place} to {nxt.place}: no price found for this journey.")
            continue
        for f in fee_lines(request.travellers, stop_costs.fees, stop.place, nights, [stop.place, *stop.cities]):
            if f.basis == "per_trip":
                if (f.traveller, f.name) in charged_per_trip:
                    continue
                charged_per_trip.add((f.traveller, f.name))
            raw.append(("destination_fee", f"{f.name} ({f.place}), {f.band}" + ("" if f.confirmed else ", unconfirmed"),
                        f.traveller, f.total, f.currency, {"urls": [f.source_url] if f.source_url else []}))
            fee_names.add(f.name)
        days = nights + (1 if i == len(plan) - 1 else 0)  # the last stop also feeds you on departure day
        raw += stay_lines(request, stop_costs, nights, days, style)
        if nxt and stop_costs.onward is None:  # every leg between stops is priced, or said to be missing
            unpriced.append(f"{stop.place} to {nxt.place}: no price found for this journey.")
        elif nxt:
            for label, t in by_label.items():
                fare = traveller_fare(t, stop_costs.onward)
                raw.append(("transport", f"{stop_costs.onward.route} ({stop_costs.onward.mode}, one way)", label,
                            fare.amount, fare.currency,
                            {"ids": stop_costs.onward.source_ids, "quote": stop_costs.onward.evidence}))
    raw += waiting_lines(request, meeting, costs, style)
    if not plan:
        reasons.append("No destination has been chosen yet.")
    if not nights_total:
        reasons.append("The trip length is unknown.")

    solo = request.travellers[0].label if len(request.travellers) == 1 else None  # nobody to share rooms with
    lines = [CostLine(category=cat, label=label, traveller=who or solo, amount=round(amount, 2), currency=cur,
                      amount_report=round(fx.convert(amount, cur, report, table), 2),
                      source_ids=(cite[0] if cite else {}).get("ids", []), urls=(cite[0] if cite else {}).get("urls", []),
                      quote=(cite[0] if cite else {}).get("quote", ""))
             for cat, label, who, amount, cur, *cite in raw]
    total = round(sum(l.amount_report for l in lines), 2)
    # the first lines are the journeys to and from home: "$5000 excluding airfare" leaves them out of the budget
    excluded = 0.0 if request.flights_in_budget else round(sum(l.amount_report for l in lines[:len(home)]), 2)
    counted = round(total - excluded, 2)
    budget = request.budget.amount if request.budget else None
    gap = round(counted - budget, 2) if budget is not None else 0.0

    per_traveller: dict[str, float] = {}
    for line in lines:
        if line.traveller:
            per_traveller[line.traveller] = round(per_traveller.get(line.traveller, 0) + line.amount_report, 2)
    shared = round(sum(l.amount_report for l in lines if not l.traveller), 2)

    if gap > 0:
        reasons.append(f"Over budget by {gap:,.2f} {report} ({counted:,.2f} vs {budget:,.2f}).")
        fee_total = sum(l.amount_report for l in lines if l.category == "destination_fee")
        if fee_total > budget:
            reasons.append(f"Destination fees alone ({', '.join(sorted(fee_names))}) cost {fee_total:,.2f} "
                           f"{report}, more than the whole budget.")
        biggest = max(lines, key=lambda l: l.amount_report)
        reasons.append(f"Largest cost: {biggest.label} ({biggest.amount_report:,.2f} {report}).")

    conversions = sorted({f"{l.currency}->{report}" for l in lines if l.currency != report})
    return Feasibility(ok=not reasons and not unpriced, currency=report, total=total, budget=budget,
                       excluded=excluded, gap=gap, reasons=reasons, unpriced=unpriced,
                       lines=lines, per_traveller=per_traveller, shared=shared, conversions=conversions,
                       lodging_style=style,
                       fx_as_of=table.as_of if conversions else None,
                       fx_source=table.source if conversions else None)


STYLES = ("luxury", "mid-range", "budget")
SPENDING_SHARE = 0.2  # when choosing hotels, keep a fifth of the budget for activities and extras


@traceable(name="feasibility.fit_to_budget")
def fit_to_budget(request: TripRequest, results: list[GroupResult], costs: dict[str, StopCosts],
                  table: fx.Rates | None = None, meeting: MeetingPlan | None = None) -> Feasibility:
    """The plan follows the budget. When the user didn't choose a hotel level, take the most comfortable one
    that leaves SPENDING_SHARE of the budget to spend; else the most comfortable one that fits; else budget.
    Either way, the result carries the trip's cost at the other levels, so the traveller sees the trade-off."""
    if request.budget is None or not any(c.lodging_rates for c in costs.values()):
        return assess(request, results, costs, table, meeting=meeting)
    table = table or fx.rates()
    priced = {s: assess(request, results, costs, table, style=s, meeting=meeting) for s in STYLES}
    budget = request.budget.amount
    pick = request.lodging_style or (
        next((s for s in STYLES if priced[s].counted <= budget * (1 - SPENDING_SHARE)), None)
        or next((s for s in STYLES if priced[s].gap <= 0), "budget"))
    others = [LodgingCost(style=s, counted=priced[s].counted) for s in STYLES if s != pick]
    return priced[pick].model_copy(update={"lodging_chosen": not request.lodging_style, "lodging_others": others})


def feasibility_node(state: dict) -> dict:
    f = fit_to_budget(state["request"], list((state.get("group_results") or {}).values()), state.get("costs") or {},
                      meeting=state.get("meeting"))
    verdict = "; ".join(f.reasons) or f.status()
    detail = f"{f.counted:,.0f} {f.currency}" + (f" of {f.budget:,.0f}" if f.budget is not None else "") + f": {verdict}"
    if f.unpriced:
        detail += f" (not priced: {' '.join(f.unpriced)})"
    if f.excluded:
        detail += f" (flights from home, {f.excluded:,.0f}, are outside the budget)"
    action = "ok" if f.ok else "not ok" if f.reasons else "incomplete"
    return {"feasibility": f, "events": [event("feasibility", action, detail)]}
