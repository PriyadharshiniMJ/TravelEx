"""Destination costs: per stop, a few targeted searches and one extraction into StopCosts.
Known tourist fees (seeded or learned) are reused; unknown ones are researched and remembered."""
import time

from langchain_core.messages import HumanMessage, SystemMessage

from travelex.agents.common import event, register, render
from travelex.agents.evidence import (UNIT_OF, about, currency_agrees, fee_supported, price_problem, stated,
                                     verbatim)
from travelex.agents.season import MONTH_NAMES
from travelex.llm import get_llm, structured
from travelex.schemas import CostEvidence, FeeRule, Source, StopCosts
from travelex.state import StopInput
from travelex.tools import fees, search
from travelex.tools.countries import country_name
from travelex.trip import allocate_nights, party, rooms_needed, stop_key, style_for, trip_months

COSTS_PROMPT = """You estimate what one stop of a trip costs, using only the numbered sources.
- currency: one ISO code for every amount below (the local currency or USD, whichever the sources use).
- lodging_per_room_night: a typical {style} room in the travel month. lodging_rates: a typical room at each
  level (budget, mid-range, luxury) in the travel month. rooms and room_config: enough rooms for this party:
  at most 2 adults to a room (friends don't share one double); a family with children can share a family room;
  infants share a bed.
- food_per_person_day: typical meals for one traveller.
- guide_per_day and guide_required: only where a licensed guide is mandatory or clearly needed; else 0, false.
- vehicle_needed: true only where visitors can't get around by public transport, taxis or day tours, or
  must travel with a guide and driver (e.g. Bhutan). Then vehicle_per_day for a car or van with driver,
  vehicle_capacity, and vehicle_accessible when a wheelchair user travels. Otherwise false and 0.
- local_transport_per_person_day: buses, trains, metro and taxis for one traveller, when no vehicle is needed.
- other_per_trip: entry tickets or permits for the main sights, for the whole party.
- fees: {fees_rule}
- onward: {onward_rule}
- notes: short points a traveller should know about costs here.
- evidence: one entry for every amount a source states: each of the three hotel levels, food, local transport,
  and vehicle, guide or other when set. item, source_id, the words that state that price copied exactly
  (including the number), and the amount, currency and unit those words give. The amount you record for the
  item must be that price. Leave out amounts you estimated.
- source_ids: the sources you used. Estimate sensibly from the sources; never invent fees."""

FEES_KNOWN = "already known; return an empty list."
FEES_RESEARCH = """every per-person tourist fee or levy here (tourist tax, park or conservation fee, sustainable
  development fee…): basis "per_night" or "per_trip", the adult amount and currency, child age_bands, and
  passports that pay a different rate (a separate rule with passports set). Set place when a fee applies only
  to a region or city, not the whole country. For each, source_id and quote: the source's exact words that
  state the amount. An empty list if there are none."""
ONWARD = """the cheapest practical way from {here} to {next}, one way: mode, route, legs and the fare for one
  adult in adult_fare (fare_basis "quoted" only if a source states it)."""
LAST_STOP = "null: this is the last stop."


def oldest(sources: list[Source]):
    """When the oldest of these pages was fetched: a rule built from them is only as fresh as that."""
    return min((s.retrieved_at for s in sources if s.retrieved_at), default=None, key=str)


def rate_for(costs: StopCosts, item: str) -> float:
    """The amount recorded for an item, in the stop's currency."""
    if item.endswith(" hotel"):
        return costs.room_rate(item.removesuffix(" hotel"))
    return {"food": costs.food_per_person_day, "local transport": costs.local_transport_per_person_day,
            "vehicle": costs.vehicle_per_day, "guide": costs.guide_per_day, "other": costs.other_per_trip}[item]


def checked_evidence(costs: StopCosts, by_id: dict[str, Source], names: list[str]) -> list[CostEvidence]:
    """Evidence that backs the recorded amount: the source's own words, stating that amount in that currency and
    unit, about this place, and matching the rate recorded (a $15 quote can't back a $900 room)."""
    kept = []
    for x in costs.evidence:
        source = by_id.get(x.source_id)
        quote = verbatim(x.quote, source.text) if source else ""
        if not quote or x.unit != UNIT_OF[x.item]:
            continue
        if not price_problem(x.amount, x.currency, x.unit, quote, source, names, rate_for(costs, x.item), costs.currency,
                             item=x.item):
            kept.append(x.model_copy(update={"quote": quote}))
    return kept


def confirm_fee(fee: FeeRule, by_id: dict[str, Source], names: list[str]) -> FeeRule:
    """A researched fee is confirmed only when a source's exact words name a fee, levy or tax, state its amount
    and currency, say how it is charged (per night, or once per person), and are about here."""
    source = by_id.get(fee.source_id)
    quote = verbatim(fee.quote, source.text) if source else ""
    ok = bool(quote) and stated(fee.amount, quote) and currency_agrees(fee.currency, quote) \
        and fee_supported(fee.basis, quote) and about(source, quote, names)
    return fee.model_copy(update={"confirmed": ok, "quote": quote,
                                  "source_url": source.url if source else fee.source_url})


def destination_costs(state: StopInput) -> dict:
    started = time.perf_counter()
    stop, nxt, request = state["stop"], state["next_stop"], state["request"]
    where = f"{stop.place} {' '.join(stop.cities)}".strip()
    months = trip_months(request)
    when = f"{MONTH_NAMES[months[0] - 1]} {(request.month or str(request.start_date))[:4]}" if months else ""
    known_fees = fees.rules_for(stop.country, stop.place)
    style = style_for(request)

    family = "family room" if any(t.type != "adult" for t in request.travellers) else ""
    queries = [f"{where} {style} hotel price per night {when} {family}".strip(),
               f"{where} hotel prices per night budget vs mid-range vs luxury {when}",
               f"{where} daily travel costs food per day",
               f"{where} getting around guide driver car hire cost per day"]
    if known_fees is None:
        queries.append(f"{stop.place} {country_name(stop.country)} tourist tax entry fee levy for foreign visitors")
    if nxt:
        queries.append(f"{stop.place} to {nxt.place} train bus flight price")
    try:
        new, found = register(search.search_all(queries), state["source_prefix"], {}, state.get("taken"))
        nights = dict((s.place, n) for s, n in allocate_nights(request)).get(stop.place)
        prompt = COSTS_PROMPT.format(style=style, fees_rule=FEES_KNOWN if known_fees is not None else FEES_RESEARCH,
                                     onward_rule=ONWARD.format(here=stop.place, next=nxt.place) if nxt else LAST_STOP)
        task = (f"Stop: {where}, {country_name(stop.country)} ({stop.country}). Travel: {when or 'month not set'}, "
                f"{nights or '?'} nights here.\nParty: {party(request)}.\n\nSources:\n{render(found)}")
        costs = structured(get_llm("costs"), StopCosts, [SystemMessage(prompt), HumanMessage(task)])
    except search.SearchConfigError:
        raise
    except Exception as e:  # fail soft: feasibility will report this stop's costs as missing
        return {"errors": [f"destination_costs ({stop.place}): {type(e).__name__}: {e}"],
                "events": [event("destination_costs", "failed", f"{stop.place}: {e}", queries, started)]}

    ids, by_id = {s.id for s in found}, {s.id: s for s in found}
    names = [stop.place, *stop.cities, country_name(stop.country)]
    unconfirmed = 0
    if known_fees is None:
        found_fees = [confirm_fee(f.model_copy(update={"country": stop.country, "verified_on": None}), by_id, names)
                      for f in costs.fees]
        unconfirmed = sum(not f.confirmed for f in found_fees)
        if unconfirmed:  # used for this trip, labelled; never cached as if a source had said it
            known_fees = found_fees
        else:
            fees.remember(stop.country, found_fees, stop.place, evidence_at=oldest(found))
            known_fees = fees.rules_for(stop.country, stop.place) or []
    costs = costs.model_copy(update={"place": stop.place, "fees": known_fees, "key": stop_key(request, stop, nxt),
                                     "source_ids": [i for i in costs.source_ids if i in ids]})
    costs = costs.model_copy(update={"evidence": checked_evidence(costs, by_id, names)})
    if costs.rooms < (needed := rooms_needed(request)):  # the model has put 3 friends in "1 double room (sharing)"
        costs = costs.model_copy(update={"rooms": needed, "room_config": f"{needed} rooms (at most 2 adults each)"})
    fee_names = ", ".join(sorted({f.name for f in known_fees})) or "no tourist fees"
    if unconfirmed:
        fee_names += f" ({unconfirmed} unconfirmed)"
    detail = (f"{stop.place}: {costs.lodging_per_room_night:,.0f} {costs.currency}/room-night × {costs.rooms}, "
              f"food {costs.food_per_person_day:,.0f}/day, {fee_names}"
              + (f", onward {costs.onward.route}" if costs.onward else ""))
    return {"costs": {stop.place: costs}, "sources": new,
            "events": [event("destination_costs", "estimated", detail, queries, started)]}
