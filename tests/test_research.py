"""Step 3: transport per group, visas per traveller, costs per stop, and the graph that runs them in parallel."""
import re
from datetime import date

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

import travelex.agents.costs as costs_module
import travelex.agents.intake as intake_module
import travelex.agents.transport as transport_module
import travelex.agents.visa as visa_module
from builders import BHUTAN, option, person, result, stop_costs, trip
from fakes import page, tool_call
from travelex.agents.costs import destination_costs
from travelex.agents.feasibility import assess
from travelex.agents.transport import choose_option, clean
from travelex.agents.visa import visa_check
from travelex.graph import build_graph, build_group_subgraph, initial_state, run_config, serializer
from travelex.schemas import FeeRule, Leg, Money, Stop
from travelex.tools import fees, fx, visa_rules

TODAY = date(2026, 10, 27)
ROUTES = {"home_airports": ["MAA"], "options": [
    {"mode": "flight", "route": "MAA-CCU-PBH", "adult_fare": {"amount": 520, "currency": "USD"},
     "fare_basis": "quoted", "max_layover_hours": 3, "source_ids": ["G1.1", "G1.9"]},
    {"mode": "flight+overland", "route": "MAA-IXB-Phuentsholing", "adult_fare": {"amount": 30000, "currency": "INR"},
     "max_layover_hours": 2, "source_ids": ["G1.1"]},
]}
BHUTAN_COSTS = {"place": "anything", "currency": "USD", "lodging_per_room_night": 80, "rooms": 1,
                "room_config": "1 double", "food_per_person_day": 20, "guide_per_day": 40, "guide_required": True,
                "vehicle_per_day": 60, "source_ids": ["C1.1"]}


# ---------- choosing a route ----------

def test_choose_option_cheapest_after_currency_conversion():
    options = [option("MAA-CCU-PBH", 520), option("MAA-IXB-Phuentsholing", 30000, currency="INR")]
    assert choose_option(options, [person("A")]) == 1  # ₹30,000 is about $315


def test_choose_option_respects_kids_and_wheelchairs():
    long_layover = option("DXB-DEL-PBH", 300, max_layover_hours=9)
    night_bus = option("CCU-Phuentsholing", 80, mode="surface")
    night_bus.legs[0] = Leg(origin="CCU", destination="Phuentsholing", mode="bus", overnight=True)
    fine = option("DXB-CCU-PBH", 450, max_layover_hours=2)
    options = [long_layover, night_bus, fine]
    assert choose_option(options, [person("A")]) == 1
    assert choose_option(options, [person("A"), person("Kid", type="child", age=5)]) == 2
    direct, cheap_two_stop = option("ORD-MEX", 600), option("ORD-DFW-IAH-MEX", 350)
    assert choose_option([cheap_two_stop, direct], [person("A", needs=["wheelchair"])]) == 1


def test_choose_option_with_no_or_odd_options():
    assert choose_option([], [person("A")]) == 0
    assert choose_option([option("A-B", 5, currency="XYZ"), option("A-C", 900)], [person("A")]) == 1


@pytest.mark.parametrize("evidence, supported", [
    ("Drukair operates a direct flight from Delhi to Paro", True),
    ("Non-stop flights from Delhi to Paro run twice weekly", True),
    ("There are no direct flights from Dubai to Paro", False),
    ("Direct flights from Dubai to Paro are not available", False),
    ("Flights are not non-stop; connect in Delhi", False),
    ("Indirect routes via Kolkata are cheapest", False),
    ("Emirates flies direct from Dubai to Paro daily", False),  # not in the source: invented evidence
])
def test_nonstop_needs_verbatim_positive_evidence(evidence, supported):
    from datetime import datetime
    from travelex.agents.transport import nonstop_supported
    from travelex.schemas import Source
    text = ("Drukair operates a direct flight from Delhi to Paro. Non-stop flights from Delhi to Paro run twice "
            "weekly. There are no direct flights from Dubai to Paro. Direct flights from Dubai to Paro are not "
            "available. Flights are not non-stop; connect in Delhi. Indirect routes via Kolkata are cheapest.")
    src = {"G1.1": Source(id="G1.1", url="https://a", title="a", kind="web", retrieved_at=datetime(2026, 10, 27), text=text)}
    o = option("DXB-PBH", 500, source_ids=["G1.1"], evidence=evidence)
    assert nonstop_supported(o, src) is supported


def test_clean_fixes_what_the_model_gets_wrong():
    from datetime import datetime
    from travelex.agents.transport import NONSTOP_UNCONFIRMED
    from travelex.schemas import Source
    sources = {"G1.1": Source(id="G1.1", url="https://a", title="a", kind="web", retrieved_at=datetime(2026, 10, 27),
                              text="Fly Chennai to Paro via Kolkata on Drukair")}
    mislabelled = option("MAA-CCU-PBH", 80000, currency="INR", mode="flight+overland",
                         arrive=datetime(1201, 11, 25), source_ids=["G1.1", "G1.7"])
    same_route_dearer = option("MAA-CCU-PBH", 1500, source_ids=["G1.1"])
    overland = option("MAA-IXB-Phuentsholing", 300, mode="flight", fare_basis="quoted", source_ids=["G1.1"])
    overland.legs[1] = Leg(origin="IXB", destination="Phuentsholing", mode="road")
    unconfirmed_nonstop = option("MAA-PBH", 317, source_ids=["G1.1"])
    between_stops = option("CDG-ZRH", 34, mode="surface", source_ids=["G1.1"])
    absurd = option("MAA-BKK-PBH", 4260, source_ids=["G1.1"])
    impossible = option("MAA-SIN-PBH", 16485, source_ids=["G1.1"])
    kept, dropped = clean([mislabelled, same_route_dearer, overland, unconfirmed_nonstop, between_stops, absurd,
                           impossible], exact_dates=False, sources=sources, home=["MAA"], origin="Chennai")
    assert [(o.route, o.mode) for o in kept] == [("MAA-CCU-PBH", "flight"), ("MAA-IXB-Phuentsholing", "flight+overland"),
                                                 ("MAA-PBH", "flight")]
    assert kept[0].arrival() is None and kept[0].source_ids == ["G1.1"]
    assert kept[0].adult_fare.currency == "INR"  # the cheaper of the two MAA-CCU-PBH options
    assert kept[1].fare_basis == "estimated"  # "quoted", but no evidence shows the price
    assert kept[2].warnings == [NONSTOP_UNCONFIRMED] and not kept[0].warnings
    assert dropped == ["CDG-ZRH: does not start from home", "MAA-SIN-PBH: fare of 16,485 USD is not plausible",
                       "MAA-BKK-PBH: fare far above the others"]


# ---------- the group subgraph: transport loop, extraction, visas ----------

def transport_llm(messages):
    """Search once, then stop; then extract the routes."""
    system = messages[0].content
    if system.startswith("List this group's"):
        return tool_call("RouteList", ROUTES)
    if not any(isinstance(m, ToolMessage) for m in messages):
        return tool_call("web_search", {"query": "Chennai to Paro flights October"})
    return AIMessage("Two options found.")


def test_group_subgraph_finds_routes_and_checks_visas(use_fake_llm, fake_search):
    fake_search.append(page("https://www.drukair.com.bt", "Chennai - Kolkata - Paro from USD 520 return"))
    use_fake_llm(transport_module, responder=transport_llm)
    family = trip(person("Mum"), person("Kid", type="child", age=5), month="2027-10", nights=5)
    out = build_group_subgraph().invoke({"group": family.groups[0], "source_prefix": "G1.",
                                         "request": family, "today": TODAY})
    result = out["result"]
    assert [o.route for o in result.options] == ["MAA-CCU-PBH", "MAA-IXB-Phuentsholing"]
    assert result.options[0].source_ids == ["G1.1"]  # G1.9 was never a source
    assert list(out["sources"]) == ["G1.1"]
    assert result.chosen == 1
    assert [(v.traveller, v.rule.document) for v in result.visas] == [("Mum", "entry_permit"), ("Kid", "entry_permit")]
    assert not any(v.confirmed_by_search for v in result.visas)  # seed rules: no research


# ---------- visa research ----------

FRANCE_PAGE = ("Indian nationals need a Schengen visa for short stays. The visa fee is EUR 90 for adults. "
               "A short-stay visa allows up to 90 days in any 180-day period.")
SCHENGEN_RULE = {"passport": "IN", "country": "FR", "visa_required": True, "document": "visa", "fee": 90,
                 "fee_currency": "EUR", "max_stay_days": 90, "notes": "Apply at VFS.",
                 "source_url": "https://france-visas.gouv.fr"}
SCHENGEN_RESEARCH = {"rule": SCHENGEN_RULE, "evidence": [
    {"source_id": "G1.V1", "quote": "Indian nationals need a Schengen visa for short stays.", "supports": "requirement"},
    {"source_id": "G1.V1", "quote": "The visa fee is EUR 90 for adults.", "supports": "fee"},
    {"source_id": "G1.V1", "quote": "allows up to 90 days in any 180-day period", "supports": "stay"},
    {"source_id": "G1.V1", "quote": "Indians can get a visa on arrival in Paris.", "supports": "how_to_apply"},  # invented
]}


def visa_llm(research: dict, verdicts: dict[int, str] | None = None, calls: list | None = None):
    """The visa extraction answers `research`; the relevance judge answers `verdicts` (index → verdict,
    default "applies" for every quote it is shown)."""
    def respond(messages):
        system = messages[0].content
        if calls is not None:
            calls.append(system[:20])
        if system.startswith("Extract the entry rule"):
            return tool_call("VisaResearch", research)
        shown = [int(i) for i in re.findall(r"^\[(\d+)\]", messages[-1].content, re.MULTILINE)]
        return tool_call("RelevanceCheck", {"verdicts": [
            {"index": i, "verdict": (verdicts or {}).get(i, "applies"), "reason": "test"} for i in shown]})
    return respond


def visa_state(request):
    return {"group": request.groups[0], "source_prefix": "G1.", "request": request, "today": TODAY,
            "options": [], "sources": {}}


def france_trip():
    return trip(person("A", origin="Delhi"), stops=[Stop(place="France", country="FR")], month="2027-06")


def test_unknown_visa_is_researched_once_cited_and_covers_the_schengen_zone(use_fake_llm, fake_search):
    fake_search.append(page("https://france-visas.gouv.fr", FRANCE_PAGE, title="France-Visas"))
    calls = []
    use_fake_llm(visa_module, responder=visa_llm(SCHENGEN_RESEARCH, calls=calls))
    stops = [Stop(place="France", country="FR"), Stop(place="Italy", country="IT"), Stop(place="Switzerland", country="CH")]
    request = trip(person("A", origin="Delhi"), person("B", origin="Delhi"), stops=stops, month="2027-06", nights=9)
    out = visa_check(visa_state(request))
    visas = out["visas"]
    assert len(visas) == 6 and len(calls) == 4  # France: extraction + judge; Italy, Switzerland: re-judge France's quotes
    assert {(v.country, v.rule.zone, v.rule.fee) for v in visas} == {("FR", "Schengen", 90), ("IT", "Schengen", 90),
                                                                     ("CH", "Schengen", 90)}
    rule = visas[0].rule
    assert rule.confirmed and [c.supports for c in rule.citations] == ["requirement", "fee", "stay"]  # invented one dropped
    first = rule.citations[0]
    assert (first.url, first.title, first.source_kind, first.relevance) == (
        "https://france-visas.gouv.fr", "France-Visas", "destination_government", "applies")
    assert "Indian nationals need a Schengen visa" in first.context
    assert visas[0].confirmed_by_search and visas[0].source_ids == ["G1.V1"]
    euros_in_usd = fx.convert(90, "EUR", "USD")  # the trip's budget is in USD, so the fee is shown in USD
    assert visas[0].summary == f"Visa needed (Schengen visa, one covers the whole area), ≈ {euros_in_usd:,.0f} USD, up to 90 days"
    assert rule.fee == 90 and rule.fee_currency == "EUR"  # the rule itself keeps the official amount
    assert "dropped 1 quote" in out["events"][0]["detail"]
    cached = visa_rules.lookup("IN", "ES")  # the next Schengen trip needs no search, and keeps the citations
    assert cached.provenance == "learned" and len(cached.citations) == 3


US_EMBASSY = ("U.S. citizens do not need a visa for France for stays up to 90 days. Entry into France for short-term "
              "tourism requires that your passport be valid for at least three months beyond your departure.")


def test_a_page_written_for_another_nationality_cannot_confirm_the_rule(use_fake_llm, fake_search):
    """The live failure: a US embassy page cited for Indian travellers."""
    fake_search.append(page("https://fr.usembassy.gov/travel", US_EMBASSY, title="US Embassy France"))
    research = {"rule": {**SCHENGEN_RULE, "fee": None, "max_stay_days": None}, "evidence": [
        {"source_id": "G1.V1", "quote": "U.S. citizens do not need a visa for France for stays up to 90 days.",
         "supports": "requirement"},
        {"source_id": "G1.V1", "quote": "your passport be valid for at least three months beyond your departure",
         "supports": "how_to_apply"}]}
    # even if the judge slips and says "applies", a foreign government's page only counts for all visitors
    use_fake_llm(visa_module, responder=visa_llm(research, {0: "applies", 1: "all_visitors"}))
    rule = visa_check(visa_state(france_trip()))["visas"][0].rule
    assert [(c.supports, c.source_kind, c.relevance) for c in rule.citations] == [
        ("how_to_apply", "other_government", "all_visitors")]
    assert not rule.confirmed


def test_the_judge_drops_quotes_about_other_people_and_reports_contradictions(use_fake_llm, fake_search):
    fake_search.append(page("https://france-visas.gouv.fr", FRANCE_PAGE + " Students need a long-stay visa.",
                            title="France-Visas"))
    research = {"rule": SCHENGEN_RULE, "evidence": SCHENGEN_RESEARCH["evidence"][:3] + [
        {"source_id": "G1.V1", "quote": "Students need a long-stay visa.", "supports": "how_to_apply"}]}
    use_fake_llm(visa_module, responder=visa_llm(research, {0: "applies", 1: "contradicts", 2: "applies",
                                                            3: "other_purpose"}))
    rule = visa_check(visa_state(france_trip()))["visas"][0].rule
    assert [c.supports for c in rule.citations] == ["requirement", "stay"]
    assert rule.fee is None  # its only quote contradicted the rule
    assert "france-visas.gouv.fr says otherwise: “The visa fee is EUR 90 for adults.”" in rule.notes


def test_a_fee_quote_must_contain_the_fee(use_fake_llm, fake_search):
    fake_search.append(page("https://france-visas.gouv.fr", FRANCE_PAGE + " Fees are listed on the fees page."))
    research = {"rule": SCHENGEN_RULE, "evidence": [
        SCHENGEN_RESEARCH["evidence"][0],
        {"source_id": "G1.V1", "quote": "Fees are listed on the fees page.", "supports": "fee"}]}
    use_fake_llm(visa_module, responder=visa_llm(research))
    rule = visa_check(visa_state(france_trip()))["visas"][0].rule
    assert rule.confirmed and rule.fee is None and [c.supports for c in rule.citations] == ["requirement"]


def test_if_the_judge_fails_quotes_are_kept_but_confirm_nothing(use_fake_llm, fake_search):
    fake_search.append(page("https://france-visas.gouv.fr", FRANCE_PAGE))

    def respond(messages):
        if messages[0].content.startswith("Extract the entry rule"):
            return tool_call("VisaResearch", SCHENGEN_RESEARCH)
        raise RuntimeError("judge down")
    use_fake_llm(visa_module, responder=respond)
    visa = visa_check(visa_state(france_trip()))["visas"][0]
    assert {c.relevance for c in visa.rule.citations} == {"unchecked"}
    assert not visa.rule.confirmed and "not confirmed" in visa.summary


def test_claims_without_a_relevant_quote_are_removed():
    from travelex.agents.visa import ground, summarise
    from travelex.schemas import Citation, VisaRule
    rule = VisaRule(passport="BR", country="JP", visa_required=False, document="none", fee=30, max_stay_days=90)
    bare = ground(rule, [])
    assert (bare.fee, bare.max_stay_days, bare.confirmed) == (None, None, False)
    assert "not confirmed" in summarise(bare)
    unchecked = Citation(url="https://mofa.go.jp", quote="Brazil: visa exempt", supports="requirement")
    assert not ground(rule, [unchecked]).confirmed
    quoted = ground(rule, [unchecked.model_copy(update={"relevance": "applies"})])
    assert quoted.confirmed and quoted.fee is None and "not confirmed" not in summarise(quoted)


BHUTAN_PAGE = "Indian nationals do not require a visa to visit Bhutan; an entry permit is issued on arrival."


def bhutan_research(required: bool):
    return {"rule": {"passport": "IN", "country": "BT", "visa_required": required, "document": "entry_permit"},
            "evidence": [{"source_id": "G1.V1", "quote": BHUTAN_PAGE.split(";")[0], "supports": "requirement"}]}


def test_seed_rules_get_citations_once_and_reuse_them(use_fake_llm, fake_search):
    fake_search.append(page("https://immi.gov.bt", BHUTAN_PAGE, title="Department of Immigration"))
    calls = []
    use_fake_llm(visa_module, responder=visa_llm(bhutan_research(False), calls=calls))
    request = trip(person("A"), month="2027-10")
    rule = visa_check(visa_state(request))["visas"][0].rule
    assert rule.provenance == "seed" and rule.confirmed and rule.document == "entry_permit"
    assert rule.citations[0].quote == "Indian nationals do not require a visa to visit Bhutan"
    assert rule.citations[0].source_kind == "destination_government"
    again = visa_check(visa_state(request))["visas"][0]
    assert len(calls) == 2 and again.rule.citations == rule.citations and not again.confirmed_by_search


def test_seed_rule_contradicted_by_sources_is_flagged(use_fake_llm, fake_search):
    fake_search.append(page("https://example.org", BHUTAN_PAGE))
    use_fake_llm(visa_module, responder=visa_llm(bhutan_research(True)))
    visa = visa_check(visa_state(trip(person("A"), month="2027-10")))["visas"][0]
    assert not visa.rule.confirmed and "Unlike our stored rule, sources found now say a visa is needed" in visa.rule.notes
    assert "not confirmed" in visa.summary


def test_by_law_rules_need_no_research(use_fake_llm):
    use_fake_llm(visa_module, responder=lambda m: pytest.fail("no research for domestic or Schengen travel"))
    request = trip(person("A", passport="DE", origin="Berlin"), stops=[Stop(place="Prague", country="CZ"),
                                                                       Stop(place="Berlin", country="DE")])
    visas = visa_check({"group": request.groups[0], "source_prefix": "G1.", "request": request, "today": TODAY,
                        "options": [], "sources": {}})["visas"]
    assert [v.summary for v in visas] == ["No visa needed", "No visa needed. Domestic travel."]


def test_unknown_passport_and_failed_research_fail_soft(use_fake_llm, fake_search):
    use_fake_llm(visa_module, responder=lambda m: (_ for _ in ()).throw(RuntimeError("model down")))
    request = trip(person("A", passport=None), person("B", passport="BR"), stops=[Stop(place="Japan", country="JP")])
    out = visa_check({"group": request.groups[0], "source_prefix": "G1.", "request": request, "today": TODAY,
                      "options": [], "sources": {}})
    a, b = out["visas"]
    assert a.rule is None and "Passport not known" in a.summary
    assert b.rule is None and "Not known" in b.summary
    assert "model down" in out["errors"][0]


# ---------- destination costs ----------

def stop_input(request, i=0, prefix="C1."):
    stops = request.stops
    return {"stop": stops[i], "next_stop": stops[i + 1] if i + 1 < len(stops) else None,
            "source_prefix": prefix, "request": request, "today": TODAY}


def test_known_fees_are_not_researched(use_fake_llm, fake_search, search_calls):
    fake_search.append(page("https://example.org/bhutan-costs", "Hotels in Paro from $80"))
    made_up = {"country": "BT", "name": "Made-up fee", "basis": "per_trip", "amount": 999, "currency": "USD"}
    llm = use_fake_llm(costs_module, tool_call("StopCosts", {**BHUTAN_COSTS, "fees": [made_up]}))
    out = destination_costs(stop_input(trip(person("A"), month="2027-10", nights=5)))
    costs = out["costs"]["Bhutan"]
    assert costs.place == "Bhutan" and costs.guide_required
    assert costs.fees == fees.rules_for("BT")  # the seeded SDF, not whatever the model said
    assert not any("tourist tax" in q for q in search_calls)
    assert "already known" in llm.calls[0][0].content


def test_unknown_fees_are_researched_and_cached_per_stop(use_fake_llm, fake_search, search_calls):
    fake_search.append(page("https://example.org/bali-levy", "Bali tourist levy IDR 150,000 per person"))
    levy = {"country": "ID", "place": "Bali", "name": "Bali tourist levy", "basis": "per_trip", "amount": 150000,
            "currency": "IDR", "source_id": "C1.1", "quote": "Bali tourist levy IDR 150,000 per person"}
    onward = {"mode": "flight", "route": "DPS-CGK", "adult_fare": {"amount": 60, "currency": "USD"}}
    use_fake_llm(costs_module, tool_call("StopCosts", {**BHUTAN_COSTS, "currency": "IDR", "fees": [levy], "onward": onward}))
    stops = [Stop(place="Bali", country="ID"), Stop(place="Jakarta", country="ID")]
    request = trip(person("A", passport="AU", origin="Sydney"), stops=stops, month="2027-07", nights=6)
    costs = destination_costs(stop_input(request))["costs"]["Bali"]
    assert any("tourist tax" in q for q in search_calls) and any("Bali to Jakarta" in q for q in search_calls)
    assert [f.name for f in costs.fees] == ["Bali tourist levy"] and costs.onward.route == "DPS-CGK"
    assert fees.rules_for("ID", "Bali")[0].provenance == "learned"
    assert fees.rules_for("ID", "Jakarta") is None  # cached for this stop only


def test_costs_failure_is_reported_not_raised(use_fake_llm, fake_search):
    use_fake_llm(costs_module, responder=lambda m: (_ for _ in ()).throw(RuntimeError("model down")))
    out = destination_costs(stop_input(trip(person("A"), month="2027-10")))
    assert "costs" not in out and "model down" in out["errors"][0]


# ---------- budget rules added in step 3 ----------

def test_onward_legs_are_priced_per_traveller_and_per_trip_fees_charged_once():
    levy = FeeRule(country="ID", name="Indonesia levy", basis="per_trip", amount=10, currency="USD")
    stops = [Stop(place="Bali", country="ID"), Stop(place="Lombok", country="ID")]
    request = trip(person("A", passport="AU", origin="Sydney"), person("K", type="child", age=8, passport="AU",
                   origin="Sydney"), stops=stops, nights=6, budget=None)
    hop = option("DPS-LOP", 100)
    costs = {"Bali": stop_costs("Bali", fee_country=None, fees=[levy], onward=hop),
             "Lombok": stop_costs("Lombok", fee_country=None, fees=[levy], onward=hop)}  # last stop: onward ignored
    f = assess(request, [result("G1", option("SYD-DPS", 700))], costs)
    hops = [(l.traveller, l.amount) for l in f.lines if l.label.startswith("DPS-LOP")]
    assert hops == [("A", 100), ("K", 75)]
    assert len([l for l in f.lines if l.category == "destination_fee"]) == 2  # once each, not once per stop


# ---------- the whole graph ----------

def everything(messages):
    """One fake model for every agent, answering by the system prompt."""
    system = messages[0].content
    if system.startswith("You turn a traveller's"):
        return tool_call("TripRequest", {
            "stops": [{"place": "Bhutan", "country": "BT"}], "month": "2027-10", "duration_text": "6 days",
            "budget": {"amount": 4000, "currency": "USD"},
            "travellers": [{"label": "Mum", "type": "adult", "passport": "IN", "origin": "Chennai"},
                           {"label": "Kid", "type": "child", "age": 11, "passport": "IN", "origin": "Chennai"},
                           {"label": "Grandma", "type": "adult", "age": 68, "passport": "IN", "origin": "Dubai"}]})
    if system.startswith("You estimate"):
        return tool_call("StopCosts", BHUTAN_COSTS)
    if system.startswith("List this group's"):
        origin = "MAA" if "Chennai" in messages[1].content else "DXB"
        return tool_call("RouteList", {"home_airports": [origin], "options": [{"mode": "flight", "route": f"{origin}-CCU-PBH",
                                                    "adult_fare": {"amount": 500, "currency": "USD"},
                                                    "max_layover_hours": 2}]})
    return AIMessage("Done.")  # the transport loop: no searching in this test


def test_graph_runs_groups_and_stops_in_parallel_then_checks_the_budget(use_fake_llm, fake_search):
    for module in (intake_module, transport_module, costs_module):
        use_fake_llm(module, responder=everything)
    graph = build_graph(InMemorySaver(serde=serializer()), interrupt_before=["itinerary_planner"])
    config = run_config("e2e", auto_approve=True)
    for _ in graph.stream(initial_state("family + grandma to Bhutan", TODAY), config):
        pass
    values = graph.get_state(config).values
    assert set(values["group_results"]) == {"G1", "G2"}
    assert set(values["costs"]) == {"Bhutan"}
    assert [o.place for o in values["meeting"].options] == ["PBH", "CCU"]  # both groups connect in Kolkata
    f = values["feasibility"]
    assert f.ok and f.currency == "USD"
    assert {"Mum", "Kid", "Grandma"} <= set(f.per_traveller)
    agents = [e["agent"] for e in values["events"]]
    assert agents.index("feasibility") > agents.index("meeting_plan") > agents.index("season_check")


@pytest.mark.parametrize("url, expected", [
    ("https://www.immi.gov.bt/visa", True), ("https://www.gov.uk/foreign-travel-advice", True),
    ("https://france-visas.gouv.fr/en/", True), ("https://www.mofa.go.jp/j_info/visit/visa/", True),
    ("https://travel.state.gov/content/travel", True), ("https://home-affairs.ec.europa.eu/x", True),
    ("https://www.immigration.govt.nz/new-zealand-visas", True), ("https://www.canada.ca/en/immigration", True),
    ("https://bestbhutan.com/blog/bhutan-visa-guide", False), ("https://www.govisa.com", False),
    ("https://gov.example.com", False), ("https://www.government.is/topics/foreign-affairs/visa-to-iceland", True),
    ("https://www.government.nl/topics/visa", True), ("https://www.government.com/visas", False),
    ("https://my-government.is.example.org", False),
])
def test_official_sources(url, expected):
    from travelex.agents.common import official
    assert official(url) is expected


def test_a_government_site_speaks_for_its_own_country():
    from travelex.agents.common import government_of, source_kind
    assert government_of("https://www.government.is/topics/foreign-affairs/visa-to-iceland") == "IS"
    assert source_kind("https://www.government.is/x", "IN", "IS", "Schengen") == "destination_government"


# ---------- a Schengen answer borrowed from another country is re-checked (the Portugal report) ----------

FRANCE_ONLY = "People residing in India who wish to visit France as their sole or main destination, must go to any of the VFS Global Centres in India"
SCHENGEN_WIDE = "Indian nationals are required to obtain a Schengen visa"


def learned_france_rule(*quotes):
    from travelex.schemas import Citation, VisaRule
    citations = [Citation(url=url, title=url, quote=q, supports=supports, relevance="applies")
                 for url, q, supports in quotes]
    return VisaRule(passport="IN", country="FR", visa_required=True, document="visa", citations=citations,
                    confirmed=True, cited_for="FR")


def judge_by_text(calls: list):
    """A judge that marks quotes naming France as about another destination, and Schengen-wide ones as applying."""
    def respond(messages):
        calls.append(messages[0].content[:20])
        if messages[0].content.startswith("Extract the entry rule"):
            return tool_call("VisaResearch", {"rule": {**SCHENGEN_RULE, "country": "PT"}, "evidence": [
                {"source_id": "G1.V1", "quote": "Indian citizens need a Schengen visa to visit Portugal.",
                 "supports": "requirement"}]})
        blocks = re.findall(r"^\[(\d+)\].*?\nQuote: (.*)$", messages[-1].content, re.MULTILINE)
        return tool_call("RelevanceCheck", {"verdicts": [
            {"index": int(i), "verdict": "other_destination" if "France" in q else "applies", "reason": "t"}
            for i, q in blocks]})
    return respond


PORTUGAL = Stop(place="Portugal", country="PT")


def test_borrowed_schengen_quotes_about_another_country_are_dropped(use_fake_llm, fake_search):
    visa_rules.remember(learned_france_rule(("https://france-visas.gouv.fr/en/inde", FRANCE_ONLY, "how_to_apply"),
                                            ("https://www.airindia.com/visa", SCHENGEN_WIDE, "requirement")))
    calls = []
    use_fake_llm(visa_module, responder=judge_by_text(calls))
    request = trip(person("A", origin="Chennai"), stops=[PORTUGAL], month="2027-05")
    visa = visa_check(visa_state(request))["visas"][0]
    assert [c.quote for c in visa.rule.citations] == [SCHENGEN_WIDE]  # France's VFS instructions are gone
    assert visa.rule.confirmed and visa.rule.cited_for == "PT" and visa.rule.country == "PT"
    assert calls == ["You check whether qu"]  # one judge call, no new search or extraction
    assert visa_rules.lookup("IN", "PT").cited_for == "PT"  # Portugal now has its own cited entry


def test_when_nothing_borrowed_applies_the_country_is_researched(use_fake_llm, fake_search):
    fake_search.append(page("https://vistos.mne.gov.pt", "Indian citizens need a Schengen visa to visit Portugal."))
    visa_rules.remember(learned_france_rule(("https://france-visas.gouv.fr/en/inde", FRANCE_ONLY, "requirement")))
    calls = []
    use_fake_llm(visa_module, responder=judge_by_text(calls))
    request = trip(person("A", origin="Chennai"), stops=[PORTUGAL], month="2027-05")
    visa = visa_check(visa_state(request))["visas"][0]
    assert "Extract the entry ru" in calls  # researched Portugal itself
    assert [c.url for c in visa.rule.citations] == ["https://vistos.mne.gov.pt"]
    assert visa.rule.citations[0].source_kind == "destination_government" and visa.confirmed_by_search


def test_kid_and_access_notes_only_for_groups_that_need_them():
    from travelex.agents.transport import for_this_group
    noted = option("MAA-CCU-PBH", 520).model_copy(update={"kid_note": "Short layovers suit children",
                                                          "accessibility_note": "Book airport assistance"})
    adults = trip(person("A"), person("Grandpa", age=70, needs=["senior"]))
    assert [(o.kid_note, o.accessibility_note) for o in for_this_group([noted], adults, adults.groups[0])] == [("", "")]
    family = trip(person("A", needs=["wheelchair"]), person("Kid", type="child", age=6))
    assert [(o.kid_note, o.accessibility_note) for o in for_this_group([noted], family, family.groups[0])] == [
        ("Short layovers suit children", "Book airport assistance")]


def test_one_way_fares_are_doubled_and_unclear_ones_flagged():
    from datetime import datetime
    from travelex.agents.transport import FARE_UNCLEAR, ONE_WAY_DOUBLED, TEASER_NOTE
    from travelex.schemas import Source
    sources = {"G1.1": Source(id="G1.1", url="https://a", title="a", kind="web", retrieved_at=datetime(2026, 10, 27),
                              text="Flights from Chennai to Paro via Kolkata")}
    said_return = option("MAA-CCU-PBH", 500, fare_covers="one_way", evidence="Round-trip fares from $500",
                         source_ids=["G1.1"])  # the evidence beats the model's label
    one_way = option("MAA-DEL-PBH", 300, evidence="Chennai to Paro one-way from $300", source_ids=["G1.1"],
                     child_fare=Money(amount=200, currency="USD"))
    unclear = option("MAA-BKK-PBH", 450, fare_covers="unclear", evidence="Chennai to Paro from $450",
                     source_ids=["G1.1"])
    kept, _ = clean([said_return, one_way, unclear], exact_dates=False, sources=sources, home=["MAA"], origin="Chennai")
    fares = {o.route: (o.adult_fare.amount, o.child_fare and o.child_fare.amount, o.fare_covers, o.warnings) for o in kept}
    assert fares == {"MAA-CCU-PBH": (500, None, "return", [TEASER_NOTE]),  # every "from $…" is a teaser
                     "MAA-DEL-PBH": (600, 400, "return", [ONE_WAY_DOUBLED, TEASER_NOTE]),  # "one-way from $300"
                     "MAA-BKK-PBH": (450, None, "unclear", [FARE_UNCLEAR, TEASER_NOTE])}


def test_a_sale_banner_price_is_not_a_fare():
    from datetime import datetime
    from travelex.schemas import Source
    sources = {"G1.1": Source(id="G1.1", url="https://a", title="a", kind="web", retrieved_at=datetime(2026, 10, 27),
                              text="Book cheap Scoot flights from Chennai to Bali from as low as US$1.50.")}
    teaser = option("MAA-SIN-DPS", 1.5, source_ids=["G1.1"])
    bus = option("MAA-Pondicherry", 8, mode="surface", source_ids=["G1.1"])
    bus.legs[0] = Leg(origin="MAA", destination="Pondicherry", mode="bus")
    kept, dropped = clean([teaser, bus], exact_dates=False, sources=sources, home=["MAA"], origin="Chennai")
    assert [o.route for o in kept] == ["MAA-Pondicherry"]  # a cheap bus is fine
    assert dropped == ["MAA-SIN-DPS: fare of 1.50 USD is not plausible"]



# ---------- fares pooled from several sources ----------

def fare_sources():
    from datetime import datetime
    from travelex.schemas import Source
    texts = {"G1.1": "Round trip flights from Chennai to Bali typically cost $620 in March.",
             "G1.2": "Average round-trip fare Chennai to Denpasar: USD 580.",
             "G1.3": "Book cheap Scoot flights from Chennai to Bali from as low as US$1.50.",
             "G1.4": "Chennai to Bali deals from $250 with Scoot."}
    return {sid: Source(id=sid, url=f"https://fares.example/{sid}", title=sid, kind="web",
                        retrieved_at=datetime(2026, 10, 27), text=text) for sid, text in texts.items()}


QUOTES = [{"source_id": "G1.1", "amount": 620, "currency": "USD", "covers": "return", "words": "typically cost $620"},
          {"source_id": "G1.2", "amount": 580, "currency": "USD", "covers": "return", "words": "round-trip fare Chennai to Denpasar: USD 580"},
          {"source_id": "G1.3", "amount": 1.5, "currency": "USD", "covers": "unclear", "words": "from as low as US$1.50"},
          {"source_id": "G1.4", "amount": 250, "currency": "USD", "covers": "unclear", "words": "deals from $250"},
          {"source_id": "G1.2", "amount": 99, "currency": "USD", "covers": "return", "words": "not in the source"}]


def test_the_typical_fare_pools_real_prices_and_ignores_teasers():
    from travelex.agents.transport import typical_fare
    from travelex.schemas import FareQuote
    quotes = [FareQuote(**q) for q in QUOTES]
    assert typical_fare(quotes, fare_sources()) == (600, 580, 620, 2, ["G1.1", "G1.2"])  # banner, teaser, unquoted: out
    assert typical_fare(quotes[:1], fare_sources()) is None  # one source is not a pattern


def test_a_teaser_fare_takes_the_typical_fare(use_fake_llm):
    from travelex.agents.transport import RAISED, extract_routes
    request = trip(person("A"), stops=[Stop(place="Bali", country="ID")], month="2027-03", nights=6)
    use_fake_llm(transport_module, tool_call("RouteList", {"home_airports": ["MAA"], "fare_quotes": QUOTES, "options": [
        {"mode": "flight", "route": "MAA-SIN-DPS", "adult_fare": {"amount": 250, "currency": "USD"},
         "evidence": "Chennai to Bali deals from $250 with Scoot.", "source_ids": ["G1.4"]}]}))
    out = extract_routes({"group": request.groups[0], "request": request, "source_prefix": "G1.",
                          "sources": fare_sources()})
    [fare] = out["options"]
    assert (fare.adult_fare.amount, fare.fare_basis) == (600, "estimated")
    assert fare.warnings == [RAISED, "Typical return fare from 2 sources: 580 to 620 USD."]
    assert "typical return fare 600 USD from 2 sources" in out["events"][0]["detail"]


def test_fixed_fare_searches_run_first(use_fake_llm, fake_search, search_calls):
    from travelex.agents.transport import fare_queries, transport_agent
    fake_search.append(page("https://www.kayak.com/flight-routes/Chennai-MAA/Bali-DPS", "Chennai to Bali from $480"))
    request = trip(person("A"), stops=[Stop(place="Bali", country="ID")], month="2027-03", nights=6)
    llm = use_fake_llm(transport_module, AIMessage("Done."))
    out = transport_agent({"group": request.groups[0], "request": request, "source_prefix": "G1.", "sources": {},
                           "today": TODAY, "messages": []})
    assert search_calls == fare_queries(request, request.groups[0]) == [
        "Chennai to Bali round trip flight price March 2027",
        "cheap flights Chennai to Bali March 2027 kayak skyscanner google flights"]
    assert list(out["sources"]) == ["G1.1"]  # the page reading failed softly: no Tavily key in tests
    assert "Fare pages for the flight have already been fetched" in llm.calls[0][1].content


def test_three_friends_never_share_one_double_room(use_fake_llm, fake_search):
    fake_search.append(page("https://example.org/bali-hotels", "Bali guesthouses from $10 a night"))
    shared = {**BHUTAN_COSTS, "rooms": 1, "room_config": "1 double room for 3 adults (sharing)"}
    use_fake_llm(costs_module, tool_call("StopCosts", shared))
    friends = trip(person("A", origin="Dallas"), person("B", origin="Dubai"), person("C"), month="2027-03", nights=4)
    costs = destination_costs(stop_input(friends))["costs"]["Bhutan"]
    assert (costs.rooms, costs.room_config) == (2, "2 rooms (at most 2 adults each)")


def test_cost_evidence_must_be_in_the_source(use_fake_llm, fake_search):
    fake_search.append(page("https://example.org/bali-hotels", "Budget guesthouses in Ubud from $15 a night."))
    hotel = {"item": "budget hotel", "source_id": "C1.1", "quote": "Budget guesthouses in Ubud from $15 a night",
             "amount": 15, "currency": "USD", "unit": "per room-night"}
    claims = [hotel,
              {**hotel, "item": "food", "quote": "Meals cost $5 a day", "amount": 5, "unit": "per person-day"},  # not on the page
              {**hotel, "item": "local transport", "quote": "Budget guesthouses in Ubud", "amount": 3,
               "unit": "per person-day"}]  # no price
    rates = [{"style": "budget", "per_room_night": 15}, {"style": "mid-range", "per_room_night": 60}]
    use_fake_llm(costs_module, tool_call("StopCosts", {**BHUTAN_COSTS, "lodging_rates": rates, "evidence": claims}))
    bali = trip(person("A"), stops=[Stop(place="Bali", country="ID", cities=["Ubud"])], month="2027-03", nights=4)
    costs = destination_costs(stop_input(bali))["costs"]["Bali"]
    assert [(x.item, x.source_id) for x in costs.evidence] == [("budget hotel", "C1.1")]


def test_the_visa_summary_keeps_abbreviations_whole():
    from travelex.agents.visa import summarise
    from travelex.schemas import VisaRule
    rule = VisaRule(passport="US", country="ID", visa_required=True, document="visa_on_arrival", fee=35,
                    fee_currency="USD", notes="U.S. citizens can get a visa on arrival. Extensions cost extra.")
    assert summarise(rule, "USD").endswith(". U.S. citizens can get a visa on arrival.")


def iceland_rule():
    """What a Vienna trip found: a Schengen rule researched on Iceland's pages, saved as Austria's own."""
    from travelex.schemas import Citation, VisaRule
    pages = [("https://www.axa-schengen.com/en/visa/visit-schengen/india-iceland",
              "Iceland Visa Application Process for Indians", SCHENGEN_WIDE),
             ("https://www.government.is/topics/foreign-affairs/visa-to-iceland", "Apply for a visa to Iceland",
              "A uniform Schengen visa is valid for travel within the whole Schengen area for up to 90 days.")]
    return VisaRule(passport="IN", country="AT", zone="Schengen", visa_required=True, document="visa", confirmed=True,
                    cited_for="AT", citations=[Citation(url=u, title=t, quote=q, supports="requirement",
                                                         relevance="applies") for u, t, q in pages])


def test_another_schengen_countrys_pages_are_never_sources_here(use_fake_llm, fake_search):
    from travelex.agents.visa import other_member
    rule = iceland_rule()
    assert [other_member(c, "AT") for c in rule.citations] == ["Iceland", "Iceland"]
    assert other_member(rule.citations[0], "IS") is None  # still fine for Iceland itself
    visa_rules.remember(rule)
    fake_search.append(page("https://www.bmeia.gv.at/en/travel-stay/entry-and-residence-in-austria",
                            "Indian citizens need a Schengen visa to visit Austria.", title="Entry to Austria"))
    calls = []
    use_fake_llm(visa_module, responder=judge_by_text(calls))
    request = trip(person("A", origin="Dubai"), stops=[Stop(place="Vienna", country="AT")], month="2026-12")
    visa = visa_check(visa_state(request))["visas"][0]
    assert "Extract the entry ru" in calls  # Austria researched on its own pages
    assert all("iceland" not in c.url for c in visa.rule.citations)
