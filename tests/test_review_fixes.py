"""Regression tests for the code review's findings: each reproduces the reported failure."""
import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph

import travelex.agents.costs as costs_module
import travelex.agents.food as food_module
import travelex.agents.intake as intake_module
import travelex.tools.search as search_module
from builders import option, person, result, stop_costs, trip
from fakes import page, tool_call
from travelex.agents.alternatives import apply_changes, problems
from travelex.agents.common import register
from travelex.agents.costs import checked_evidence, confirm_fee, destination_costs
from travelex.agents.evidence import price_problem
from travelex.agents.feasibility import assess
from travelex.agents.food import (allergen_terms, allergies, breaks_diet, conflicts, food_agent,
                                  forbidden_terms)
from travelex.agents.gaps import apply, find_gaps
from travelex.agents.intake import tidy
from travelex.agents.itinerary import cruise_brief, food_sends
from travelex.agents.qa import MAX_QA_ROUNDS, check, route_after_qa
from travelex.agents.review import real_changes
from travelex.agents.transport import route_transport
from travelex.agents.visa import confirm_seed, summarise
from travelex.graph import open_checkpointer, route_research, run_config
from travelex.limits import LimitReached, RunLimits, limits_for
from travelex.report import render_markdown, to_double_check
from travelex.report_html import render_html
from travelex.retry import call_with_retry
from travelex.schemas import (Citation, CostEvidence, DayPlan, FeeRule, DestinationIdea, Dish, FoodGuide, Itinerary, LodgingRate, MeetingPlan, Place,
                              Source, Stop,
                              TripChanges, TripRequest, VisaRule)
from travelex.tools import fees, knowledge
from travelex.trip import allocate_nights, group_key, resolve_nights, stop_key

FRANCE_JAPAN = [Stop(place="France", country="FR", nights=3), Stop(place="Japan", country="JP", nights=3)]


# ---------- [P1] every journey between stops is priced, or said to be missing ----------

def france_japan(onward=None):
    request = trip(person("A", origin="London", passport="GB"), stops=FRANCE_JAPAN, nights=None,
                   budget=(100000, "USD"))
    costs = {"France": stop_costs("France", fee_country=None, onward=onward),
             "Japan": stop_costs("Japan", fee_country=None)}
    return request, costs


def test_a_missing_journey_between_stops_is_never_ok():
    request, costs = france_japan()
    f = assess(request, [result("G1", option("LHR-CDG", 200))], costs)
    assert not f.ok and not f.complete
    assert f.unpriced == ["France to Japan: no price found for this journey."]
    assert f.reasons == [] and f.status() == "fits so far, but 1 item not priced"


def test_a_missing_journey_is_not_a_budget_problem_for_the_alternatives():
    request, costs = france_japan()
    f = assess(request, [result("G1", option("LHR-CDG", 200))], costs)
    assert problems({"request": request, "feasibility": f, "season": {}}) == []


def test_a_priced_journey_between_stops_is_charged_and_complete():
    request, costs = france_japan(onward=option("CDG-NRT", 700, fare_covers="one_way"))
    f = assess(request, [result("G1", option("LHR-CDG", 200))], costs)
    assert f.ok and f.complete
    assert any(l.label.startswith("CDG-NRT") and l.amount == 700 for l in f.lines)


def test_the_gap_agent_is_asked_to_find_a_missing_journey_and_can_fill_it():
    request, costs = france_japan()
    state = {"request": request, "costs": costs, "group_results": {"G1": result("G1", option("LHR-CDG", 200))}}
    gap = next(g for g in find_gaps(state) if g.item == "onward")
    assert (gap.kind, gap.key) == ("route", "France")
    quote = "Paris to Tokyo one way from $650"
    sources = {"R.1": Source(id="R.1", kind="web", url="https://fares.example/cdg-nrt", title="Fares",
                             retrieved_at="2026-10-30T00:00:00+00:00", text=f"Cheap flights. {quote} in May.")}
    call = {"name": "record_route", "args": {"gap_id": gap.id, "route": "CDG-NRT", "mode": "flight", "amount": 650,
                                             "currency": "USD", "source_id": "R.1", "quote": quote}}
    gaps = {gap.id: gap}
    assert apply(call, gaps, costs, {}, sources, request).startswith("Accepted")
    assert costs["France"].onward.route == "CDG-NRT" and costs["France"].onward.fare_covers == "one_way"
    f = assess(request, [result("G1", option("LHR-CDG", 200))], costs)
    assert f.complete


# ---------- [P1] allergies are enforced, apart from diet ----------

def peanut_trip():
    return trip(person("A"), person("Kid", type="child", age=6), stops=[Stop(place="Thailand", country="TH")],
                nights=3, diet=["vegetarian", "peanut allergy"])


def test_allergies_are_read_apart_from_diet():
    assert allergies(["vegetarian", "peanut allergy"]) == ["peanut"]
    assert allergies(["nut allergy"]) == ["peanut", "tree nut"]
    assert allergies(["gluten-free", "shellfish allergy"]) == ["gluten", "shellfish"]
    assert allergies(["vegetarian", "halal"]) == []


def test_peanut_noodles_are_dropped_for_a_peanut_allergy(use_fake_llm, fake_search):
    fake_search.append(page("https://eat.example/bkk", "Try the street food on Yaowarat Road.", title="Bangkok food"))
    use_fake_llm(food_module, tool_call("FoodGuide", {"city": "Bangkok", "dishes": [
        {"name": "Peanut noodles", "description": "Rice noodles tossed in peanut sauce", "diet_tags": ["vegetarian"]},
        {"name": "Pad thai", "description": "Stir-fried noodles", "diet_tags": ["vegetarian", "nut-free"]},
        {"name": "Satay tofu (peanut-free version)", "description": "Grilled tofu"},  # satay is a peanut dish
        {"name": "Tofu skewers", "description": "Grilled tofu, made without peanuts"},
        {"name": "Veg green curry", "description": "Coconut curry with aubergine", "diet_tags": ["vegetarian"]}]}))
    out = food_agent({"city": "Bangkok", "request": peanut_trip(), "source_prefix": "F1."})
    kept = {d.name: d for d in out["food"]["Bangkok"].dishes}
    assert set(kept) == {"Tofu skewers", "Veg green curry"}  # a "nut-free" tag never clears pad thai
    assert all(d.allergens_unknown == ["peanut"] for d in kept.values())  # never called safe


def test_qa_flags_an_allergen_conflict_so_the_food_agent_runs_again():
    food = {"Bangkok": FoodGuide(city="Bangkok", dishes=[Dish(name="Peanut noodles", description="")],
                                 diet_notes="Say 'mai sai tua'")}
    plan = Itinerary(days=[DayPlan(day=i, city="Bangkok", title="Day") for i in range(1, 5)])
    report = check({"request": peanut_trip(), "itinerary": plan, "food": food})
    assert "allergen:Bangkok" in report.issues


def test_a_vegetarian_tag_does_not_clear_meat_in_the_description():
    veg = forbidden_terms(["vegetarian"])
    assert breaks_diet(Dish(name="Larb", description="Minced chicken salad", diet_tags=["vegetarian"]), veg) == "chicken"
    assert breaks_diet(Dish(name="Som tam", description="Papaya salad, ask for no fish sauce"), forbidden_terms(["vegan"])) is None
    assert conflicts(Dish(name="Mango sticky rice", description="Sticky rice", diet_tags=["contains nuts"]),
                     allergen_terms(["tree nut"])) == "tree nut"


# ---------- [P1] cost and fee evidence must state the amount recorded ----------

def bali_page(text):
    return Source(id="C1.1", kind="web", url="https://stay.example/bali", title="Bali hotel prices",
                  retrieved_at="2026-10-30T00:00:00+00:00", text=text)


def hotel_claim(amount, currency="USD", quote="Luxury villas in Ubud from $15 a night"):
    return CostEvidence(item="luxury hotel", source_id="C1.1", quote=quote, amount=amount, currency=currency,
                        unit="per room-night")


def bali_costs(*claims, luxury=900):
    return stop_costs("Bali", fee_country=None, lodging_rates=[LodgingRate(style="luxury", per_room_night=luxury)],
                      evidence=list(claims))


def test_a_15_dollar_quote_cannot_back_a_900_dollar_room():
    page_ = bali_page("Luxury villas in Ubud from $15 a night, breakfast extra.")
    assert checked_evidence(bali_costs(hotel_claim(15)), {"C1.1": page_}, ["Bali", "Ubud"]) == []


def test_a_quote_stating_the_recorded_rate_is_kept():
    page_ = bali_page("Luxury villas in Ubud from $900 a night, breakfast extra.")
    claim = hotel_claim(900, quote="Luxury villas in Ubud from $900 a night")
    assert checked_evidence(bali_costs(claim), {"C1.1": page_}, ["Bali", "Ubud"]) == [claim]


def test_a_quote_in_another_currency_or_unit_or_place_is_rejected():
    page_ = bali_page("Luxury villas in Ubud from €900 a night. Phuket villas from $900 per week.")
    euro = hotel_claim(900, quote="Luxury villas in Ubud from €900 a night")  # recorded as USD
    week = hotel_claim(900, quote="Phuket villas from $900 per week")  # wrong unit, and not Bali
    assert checked_evidence(bali_costs(euro, week), {"C1.1": page_}, ["Bali", "Ubud"]) == []


def test_an_unconfirmed_fee_is_used_for_the_trip_but_never_cached(use_fake_llm, fake_search):
    fake_search.append(page("https://example.org/bali-levy", "Visitors to Bali pay a levy on arrival."))
    levy = {"country": "ID", "place": "Bali", "name": "Bali tourist levy", "basis": "per_trip", "amount": 150000,
            "currency": "IDR", "source_id": "C1.1", "quote": "Visitors to Bali pay a levy on arrival"}  # no amount
    use_fake_llm(costs_module, tool_call("StopCosts", {"place": "Bali", "currency": "IDR", "lodging_per_room_night": 600000,
                                                       "rooms": 1, "room_config": "1 double",
                                                       "food_per_person_day": 200000, "fees": [levy]}))
    request = trip(person("A", passport="AU", origin="Sydney"), stops=[Stop(place="Bali", country="ID")],
                   month="2027-07", nights=4)
    out = destination_costs({"stop": request.stops[0], "next_stop": None, "source_prefix": "C1.",
                             "request": request, "today": None})
    assert [(f.name, f.confirmed) for f in out["costs"]["Bali"].fees] == [("Bali tourist levy", False)]
    assert fees.rules_for("ID", "Bali") is None  # researched again next time
    f = assess(request, [result("G1", option("SYD-DPS", 500))], out["costs"])
    assert any("unconfirmed" in l.label for l in f.lines if l.category == "destination_fee")


# ---------- [P2] a source ID keeps meaning the same page across revisions ----------

def test_a_retained_citation_still_points_at_its_own_page_after_a_revision(use_fake_llm, fake_search):
    """Version 1 had Paris as city 1 (prefix F1.). Version 2 puts Tokyo first: Tokyo's research must number after
    Paris's, so the Paris guide kept from version 1 still cites a Paris page."""
    paris = {sid: Source(id=sid, kind="web", url=f"https://paris.example/{n}", title="Paris food",
                         retrieved_at="2026-10-30T00:00:00+00:00", text="Le Comptoir in Saint-Germain.")
             for n, sid in enumerate(["F1.1", "F1.2"], 1)}
    kept_guide = FoodGuide(city="Paris", dishes=[Dish(name="Croque monsieur", description="Ham and cheese")],
                           places=[Place(name="Le Comptoir", source_ids=["F1.2"])])
    tokyo_first = Itinerary(days=[DayPlan(day=1, city="Tokyo", title="Arrive"), DayPlan(day=2, city="Paris", title="Fly")])
    state = {"request": trip(person("A"), stops=FRANCE_JAPAN, nights=1), "itinerary": tokyo_first,
             "food": {"Paris": kept_guide}, "sources": paris}
    (send,) = food_sends(state)
    assert (send.arg["city"], send.arg["source_prefix"]) == ("Tokyo", "F1.") and set(send.arg["taken"]) == {"F1.1", "F1.2"}

    fake_search.append(page("https://tokyo.example/ramen", "Ichiran in Shibuya serves tonkotsu ramen.", title="Tokyo"))
    use_fake_llm(food_module, tool_call("FoodGuide", {"city": "Tokyo", "dishes": [{"name": "Ramen", "description": "Noodle soup"}],
                                                      "places": [{"name": "Ichiran", "source_ids": ["F1.3"]}]}))
    out = food_agent(send.arg)
    merged = {**paris, **out["sources"]}  # what the state's merge_dicts reducer does
    assert set(out["sources"]) == {"F1.3"}  # numbered after Paris's IDs, never over them
    assert merged["F1.2"].url == "https://paris.example/2"  # the kept Paris citation still resolves to Paris
    assert out["food"]["Tokyo"].places[0].source_ids == ["F1.3"]


def test_every_research_branch_numbers_after_the_ids_already_given_out():
    old = {"G1.1": Source(id="G1.1", kind="web", url="https://a.example", title="a", text="a",
                          retrieved_at="2026-10-30T00:00:00+00:00")}
    new, found = register([{"url": "https://b.example", "title": "b", "text": "b", "kind": "web",
                            "retrieved_at": "2026-10-31T00:00:00+00:00"}], "G1.", {}, taken=old)
    assert list(new) == ["G1.2"] and found[0].id == "G1.2"
    request = trip(person("A"), stops=FRANCE_JAPAN, nights=None, month="2027-05")
    sends = route_research({"request": request, "today": None, "sources": old})
    group = next(s for s in sends if s.node == "group_research")
    assert set(group.arg["taken"]) == {"G1.1"}


# ---------- [P2] a different place in the same country is a change to the trip ----------

def changes(*stops):
    return TripChanges(stops=list(stops))


def test_bali_to_jakarta_runs_the_research_again():
    bali = trip(person("A"), stops=[Stop(place="Bali", country="ID")], nights=4)
    kept = real_changes(changes(Stop(place="Jakarta", country="ID")), bali, towns=["Ubud", "Seminyak"])
    assert kept.stops and kept.stops[0].place == "Jakarta"


def test_moving_nights_between_towns_already_planned_is_a_change_to_the_days():
    norway = trip(person("A"), stops=[Stop(place="Norway", country="NO", cities=["Oslo"])], nights=6)
    kept = real_changes(changes(Stop(place="Norway", country="NO", cities=["Oslo", "Bergen"])), norway,
                        towns=["Oslo", "Bergen"])
    assert kept.stops is None


def test_adding_a_new_town_in_the_same_country_runs_the_research_again():
    norway = trip(person("A"), stops=[Stop(place="Oslo", country="NO")], nights=6)
    kept = real_changes(changes(Stop(place="Oslo", country="NO"), Stop(place="Tromsø", country="NO")), norway,
                        towns=["Oslo"])
    assert [s.place for s in kept.stops] == ["Oslo", "Tromsø"]
    dropped = real_changes(changes(Stop(place="Bali", country="ID")),
                           trip(person("A"), stops=[Stop(place="Bali", country="ID"), Stop(place="Lombok", country="ID")]),
                           towns=["Ubud", "Gili"])
    assert dropped.stops is not None  # dropping Lombok is a change too


# ---------- [P2] stale evidence is fetched again, and a rule is as old as its evidence ----------

class CountingTavily:
    def __init__(self):
        self.calls = 0

    def search(self, **params):
        self.calls += 1
        return {"results": [{"url": "https://visa.example", "title": "Visa", "content": f"fetch {self.calls}"}]}


def cache_with(monkeypatch, tmp_path, age_days):
    tavily = CountingTavily()
    monkeypatch.setattr(search_module, "CACHE_DIR", tmp_path / "tavily")
    monkeypatch.setattr(search_module, "_tavily", lambda: tavily)
    search_module.search("japan visa")  # cached now
    path = next((tmp_path / "tavily").glob("*.json"))
    data = json.loads(path.read_text(encoding="utf-8"))
    data["retrieved_at"] = (datetime.now(timezone.utc) - timedelta(days=age_days)).isoformat()
    path.write_text(json.dumps(data), encoding="utf-8")
    return tavily


def test_a_cached_search_older_than_the_ttl_is_fetched_again(monkeypatch, tmp_path):
    tavily = cache_with(monkeypatch, tmp_path, age_days=10)
    assert search_module.search("japan visa")[0]["text"] == "fetch 2" and tavily.calls == 2


def test_a_fresh_cached_search_is_reused_unless_a_refresh_is_forced(monkeypatch, tmp_path):
    tavily = cache_with(monkeypatch, tmp_path, age_days=1)
    search_module.search("japan visa")
    assert tavily.calls == 1
    monkeypatch.setenv("TRAVELEX_REFRESH", "1")
    search_module.search("japan visa")
    assert tavily.calls == 2


def test_a_rule_built_today_from_old_evidence_has_already_expired():
    rule = VisaRule(passport="IN", country="JP", visa_required=True, document="visa")
    knowledge.remember("visa", "IN-JP", rule, VisaRule, evidence_at=datetime.now(timezone.utc) - timedelta(days=40))
    assert knowledge.recall("visa", "IN-JP", VisaRule) is None
    knowledge.remember("visa", "IN-JP", rule, VisaRule, evidence_at=datetime.now(timezone.utc) - timedelta(days=3))
    assert knowledge.recall("visa", "IN-JP", VisaRule) == rule


# ---------- [P2] a problem QA couldn't fix is shown, never dropped ----------

def test_an_unresolved_day_count_is_labelled_incomplete_in_the_report():
    request = trip(person("A"), stops=[Stop(place="Bhutan", country="BT", cities=["Paro"])], nights=5)
    short = Itinerary(days=[DayPlan(day=i, city="Paro", title="Explore") for i in range(1, 6)])  # 5 days, trip is 6
    food = {"Paro": FoodGuide(city="Paro", dishes=[Dish(name="Ema datshi", description="Chillies and cheese")])}
    state = {"request": request, "itinerary": short, "food": food, "qa_round": MAX_QA_ROUNDS + 1}
    state["qa_report"] = check(state)
    assert state["qa_report"].issues == ["day_count"]
    assert route_after_qa(state) == "save_report"  # out of repair rounds: saved anyway…
    md = render_markdown(state)
    assert "**This plan is incomplete:** The plan has 5 days, but the trip is 6 days." in md  # …but labelled
    assert to_double_check(state)[0] == "Still wrong in this plan: The plan has 5 days, but the trip is 6 days."
    assert "Incomplete plan" in render_html(state)


# ---------- reuse keys change whenever research would ----------

def test_a_new_passport_wheelchair_age_or_city_changes_the_group_key():
    base = trip(person("A", origin="Dubai"), person("K", type="child", age=7, origin="Dubai"),
                stops=[Stop(place="Japan", country="JP", cities=["Tokyo"])], nights=5, month="2027-04")
    key = lambda r: group_key(r, r.groups[0])
    swap = lambda label, **kw: base.model_copy(update={"travellers": [t.model_copy(update=kw) if t.label == label else t
                                                                      for t in base.travellers]})
    assert key(swap("A", passport="GB")) != key(base)
    assert key(swap("A", needs=["wheelchair"])) != key(base)
    assert key(swap("K", age=1, type="infant")) != key(base)
    other_city = base.model_copy(update={"stops": [Stop(place="Japan", country="JP", cities=["Osaka"])]})
    assert key(other_city) != key(base)
    assert key(base.model_copy(update={"nights": 3})) == key(base)  # a shorter trip still reuses the research


# ---------- a passing error is retried before a node gives up ----------

def test_a_model_that_times_out_once_still_writes_the_food_guide(use_fake_llm, fake_search):
    fake_search.append(page("https://eat.example/paro", "Try ema datshi at Sonam Trophel.", title="Paro food"))
    guide = tool_call("FoodGuide", {"city": "Paro", "dishes": [{"name": "Ema datshi", "description": "Chillies"}]})
    replies = iter([TimeoutError("read timed out"), guide])

    def flaky(messages):
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply
    use_fake_llm(food_module, responder=flaky)
    out = food_agent({"city": "Paro", "request": trip(person("A")), "source_prefix": "F1."})
    assert "errors" not in out and out["food"]["Paro"].dishes[0].name == "Ema datshi"


def test_a_bad_request_is_not_retried():
    calls = []

    def bad():
        calls.append(1)
        raise ValueError("schema mismatch")
    with pytest.raises(ValueError):
        call_with_retry(bad)
    assert len(calls) == 1


# ---------- nights spent waiting for the others are priced ----------

def test_an_early_group_pays_for_the_nights_it_waits_at_the_meeting_point():
    request = trip(person("A"), person("B", origin="Dubai"), person("C", origin="Dubai"), nights=4,
                   budget=(100000, "USD"))
    dubai = next(g for g in request.groups if g.origin == "Dubai")
    meeting = MeetingPlan(meeting_point="Paro", day1=None, arrivals={}, buffer_nights={dubai.id: 2}, options=[])
    results = [result(g.id, option("XXX-PBH", 300)) for g in request.groups]
    without = assess(request, results, {"Bhutan": stop_costs()})
    f = assess(request, results, {"Bhutan": stop_costs()}, meeting=meeting)
    waiting = [l for l in f.lines if l.label.startswith("Waiting for the others in Paro (Dubai group)")]
    assert [(l.category, l.amount) for l in waiting] == [("lodging", 60 * 1 * 2), ("food", 15 * 2 * 2)]
    assert f.total == round(without.total + 120 + 60, 2)


# ---------- one run shares a limit on searches, tokens and time ----------

def run_inside(fn, limits):
    """Call fn as a step of a run with these limits, the way graph nodes are called."""
    return RunnableLambda(lambda _: fn()).invoke(None, config={"configurable": {"limits": limits}})


def test_searches_stop_at_the_run_limit_but_cached_pages_are_free(monkeypatch, tmp_path):
    tavily = CountingTavily()
    monkeypatch.setattr(search_module, "CACHE_DIR", tmp_path / "tavily")
    monkeypatch.setattr(search_module, "_tavily", lambda: tavily)
    limits = RunLimits(max_searches=1)
    run_inside(lambda: search_module.search("japan visa"), limits)
    run_inside(lambda: search_module.search("japan visa"), limits)  # cached: costs nothing
    with pytest.raises(LimitReached):
        run_inside(lambda: search_module.search("japan hotels"), limits)
    assert (tavily.calls, limits.searches) == (1, 1)


def test_an_agent_stops_searching_when_the_run_limit_is_reached():
    wants_more = {"messages": [tool_call("web_search", {"query": "more flights"})]}
    assert run_inside(lambda: route_transport(wants_more), RunLimits()) == "transport_tools"
    spent = RunLimits(max_tokens=10)
    spent.add_tokens(10)
    assert run_inside(lambda: route_transport(wants_more), spent) == "extract_routes"


def test_a_resumed_run_keeps_counting_and_its_config_checkpoints(tmp_path):
    assert limits_for("thread-1") is limits_for("thread-1") and limits_for("thread-1") is not limits_for("thread-2")
    graph = StateGraph(dict)
    graph.add_node("step", lambda state: {"done": True})
    graph.add_edge(START, "step")
    graph.add_edge("step", END)
    app = graph.compile(checkpointer=open_checkpointer(tmp_path / "checkpoints.sqlite"))
    assert app.invoke({"done": False}, run_config("thread-1"))["done"]  # the limits object never reaches the database


# ---------- the reviewed golden dataset is frozen ----------

GOLDEN_V1_SHA256 = "f4ae6d45d8ed779c308aea72aaee44e85436e81dc6cbe0bbafab05acc9bf95ea"


def test_golden_v1_is_frozen():
    """v1 is the dataset experiments were compared on: changing a case would invalidate every delta. Put new or
    changed cases in a new version (v2) instead of editing v1."""
    text = (Path(__file__).parents[1] / "evals" / "golden" / "travelex_golden_v1.jsonl").read_text(encoding="utf-8")
    assert hashlib.sha256(text.replace("\r\n", "\n").encode()).hexdigest() == GOLDEN_V1_SHA256


# ---------- found in the live check: nights per stop, and "Budget GBP 9000" ----------

LIVE_QUERY = ("Two adults from London on British passports: 4 nights in Paris, then 5 nights in Tokyo, in May 2027. "
              "Budget GBP 9000 including flights. One of us has a peanut allergy.")


def live_request(**changes):
    """What the model returned for LIVE_QUERY: no nights on the stops, and the first length as the trip's."""
    fields = {"stops": [{"place": "Paris", "country": "FR", "cities": ["Paris"]},
                        {"place": "Tokyo", "country": "JP", "cities": ["Tokyo"]}],
              "travellers": [{"label": f"Adult {i}", "type": "adult", "passport": "GB", "origin": "London"}
                             for i in (1, 2)],
              "month": "2027-05", "nights": 4, "duration_text": "4 nights in Paris, then 5 nights in Tokyo",
              "budget": {"amount": 9000, "currency": "GBP"}, "raw_query": LIVE_QUERY}
    return TripRequest(**{**fields, **changes})


def test_nights_given_per_stop_add_up_to_the_trip():
    request = tidy(live_request(), date(2026, 11, 1))
    assert [s.nights for s in request.stops] == [4, 5]
    assert resolve_nights(request) == (9, None)
    assert [n for _, n in allocate_nights(request)] == [4, 5]


def test_a_whole_trip_length_said_besides_the_stops_is_kept():
    request = tidy(live_request(duration_text="two weeks: 4 nights in Paris, the rest in Tokyo", nights=None,
                                raw_query="Two weeks: 4 nights in Paris, the rest in Tokyo."), date(2026, 11, 1))
    assert resolve_nights(request) == (13, None)
    assert [n for _, n in allocate_nights(request)] == [4, 9]


def test_budget_before_an_amount_is_money_not_budget_hotels(use_fake_llm):
    model_says = live_request(lodging_style="budget").model_dump(mode="json")
    use_fake_llm(intake_module, tool_call("TripRequest", model_says))
    request = intake_module.intake({"raw_query": LIVE_QUERY, "today": date(2026, 11, 1)})["request"]
    assert request.lodging_style is None  # code picks the level that fits the budget
    on_a_budget = "Two adults from London to Paris in May 2027, on a budget, about GBP 2000."
    use_fake_llm(intake_module, tool_call("TripRequest", {**model_says, "raw_query": on_a_budget}))
    assert intake_module.intake({"raw_query": on_a_budget, "today": date(2026, 11, 1)})["request"].lodging_style == "budget"


# ---------- destination ideas are screened before they are offered ----------

def no_place(*ideas, month="2027-07", budget=(4000, "USD"), passport="US"):
    return TripRequest(travellers=[person(f"A{i}", passport=passport, origin="Dubai") for i in (1, 2, 3)],
                       month=month, nights=5, budget={"amount": budget[0], "currency": budget[1]},
                       destination_ideas=[DestinationIdea(**i) for i in ideas], raw_query="Somewhere in the mountains")


def test_an_idea_in_a_season_to_avoid_is_left_out_and_the_question_says_why():
    request = tidy(no_place({"place": "Bhutan", "country": "BT", "why": "monasteries"},
                            {"place": "Ladakh", "country": "IN", "why": "high passes, dry in July"}), date(2026, 11, 1))
    assert [i.place for i in request.destination_ideas] == ["Ladakh"]
    assert "Ladakh (high passes, dry in July)" in request.clarification_needed
    assert "I left out Bhutan: July is a season to avoid there" in request.clarification_needed


def test_an_idea_whose_fees_alone_break_the_budget_is_left_out():
    request = tidy(no_place({"place": "Bhutan", "country": "BT"}, {"place": "Nepal", "country": "NP"},
                            month="2027-10", budget=(800, "USD")), date(2026, 11, 1))
    assert [i.place for i in request.destination_ideas] == ["Nepal"]  # 3 × 5 nights × $100 SDF = $1,500 > $800
    assert "tourist fees alone (1,500 USD for 5 nights)" in request.clarification_needed


def test_ideas_that_need_care_are_kept_with_a_warning():
    request = tidy(no_place({"place": "Japan", "country": "JP"}, month="2027-06", passport="IN"), date(2026, 11, 1))
    (japan,) = request.destination_ideas
    assert japan.note.startswith("June:")  # rainy season: a caveat, not a reason to leave it out
    assert "visa in advance" not in japan.note  # India → Japan is an e-visa, not an embassy visa


def test_if_every_idea_fails_they_are_all_offered_with_what_is_wrong():
    request = tidy(no_place({"place": "Bhutan", "country": "BT"}), date(2026, 11, 1))
    (bhutan,) = request.destination_ideas
    assert bhutan.note.startswith("July is a season to avoid there")


# ---------- a cruise is out of scope: say so, and offer what TravelEx can do ----------

CRUISE_QUERY = "I want to plan a cruise trip from dubai to an european country in the month of march next year for three weeks"
DUBAI_ADULT = {"label": "Adult 1", "type": "adult", "passport": "IN", "origin": "Dubai"}


def test_a_cruise_request_is_not_planned_as_a_hotel_stay(use_fake_llm):
    """What happened: the model put "cruise" in interests and planned 20 hotel nights in Venice."""
    use_fake_llm(intake_module, tool_call("TripRequest", {
        "stops": [{"place": "Venice", "country": "IT"}], "interests": ["cruise"], "month": "2027-03",
        "duration_text": "three weeks", "travellers": [DUBAI_ADULT]}))
    request = intake_module.intake({"raw_query": CRUISE_QUERY, "today": date(2026, 11, 1)})["request"]
    assert (request.trip_type, request.stops) == ("cruise", [])
    assert request.missing[0] == "trip_type"
    assert request.clarification_needed.startswith("TravelEx plans land trips: it can't find, book or price cruises")


def test_with_nobody_to_ask_a_cruise_becomes_a_land_trip_to_the_same_region():
    request = TripRequest(trip_type="cruise", travellers=[DUBAI_ADULT], month="2027-03", raw_query=CRUISE_QUERY)
    out = intake_module.clarify({"request": request, "today": date(2026, 11, 1)}, {"configurable": {"auto_approve": True}})
    assert out["clarify_answer"].startswith("TravelEx can't plan cruises: plan a land trip")


def around_cruise_request(*stops, **changes):
    fields = {"trip_type": "around_cruise", "stops": list(stops), "travellers": [DUBAI_ADULT], "month": "2027-03",
              "duration_text": "three weeks", "cruise_nights": 14, "budget": {"amount": 5000, "currency": "USD"},
              "raw_query": CRUISE_QUERY}
    return tidy(TripRequest(**{**fields, **changes}), date(2026, 11, 1))


def test_around_a_cruise_only_the_nights_on_land_are_planned():
    request = around_cruise_request(Stop(place="Venice", country="IT"), Stop(place="Barcelona", country="ES"))
    assert [s.nights for s in request.stops] == [2, 1] and resolve_nights(request) == (3, None)  # not three weeks
    assert request.missing == [] and request.destination == "Venice → Barcelona (around your cruise)"


def test_the_cruise_between_two_ports_is_not_researched_or_priced():
    request = around_cruise_request(Stop(place="Venice", country="IT"), Stop(place="Barcelona", country="ES"))
    sends = route_research({"request": request, "today": None, "sources": {}})
    venice = next(s for s in sends if s.node == "destination_costs" and s.arg["stop"].place == "Venice")
    assert venice.arg["next_stop"] is None  # no "Venice to Barcelona train bus flight price" search
    costs = {"Venice": stop_costs("Venice", fee_country=None), "Barcelona": stop_costs("Barcelona", fee_country=None)}
    f = assess(request, [result("G1", option("DXB-VCE", 600))], costs)
    assert f.complete and not any("Venice to Barcelona" in u for u in f.unpriced)
    assert all(g.item != "onward" for g in find_gaps({"request": request, "costs": costs, "group_results": {}}))


def test_the_plan_says_the_cruise_is_not_in_it():
    request = around_cruise_request(Stop(place="Venice", country="IT"))
    assert resolve_nights(request) == (2, None)
    state = {"request": request, "itinerary": Itinerary(days=[DayPlan(day=i, city="Venice", title="Day") for i in (1, 2, 3)])}
    md = render_markdown(state)
    assert "> **Your cruise (14 nights) isn't in this plan:** TravelEx can't plan or price cruises." in md
    assert any(c.startswith("Your cruise is not priced") for c in to_double_check(state))
    assert "plan or price cruises" in render_html(state)  # the banner (its apostrophes are escaped in HTML)
    assert "you don't plan: no days at sea" in cruise_brief(request)


def test_a_cruise_that_returns_to_its_port_is_one_stop():
    venice = Stop(place="Venice", country="IT")
    request = around_cruise_request(venice, venice, destination_ideas=["Mediterranean"])
    assert [(s.place, s.nights) for s in request.stops] == [("Venice", 2)]
    assert request.destination_ideas == []  # a port is chosen: the ideas are no longer offered


# ========== second review ==========

# ---------- [P1] a seed visa rule is confirmed claim by claim ----------

def seed_bt(**changes):
    fields = {"passport": "US", "country": "BT", "visa_required": True, "document": "e_visa", "fee": 40,
              "fee_currency": "USD", "max_stay_days": 30, "provenance": "seed"}
    return VisaRule(**{**fields, **changes})


def cite(quote, supports):
    return Citation(url="https://www.visit.doi.gov.bt/visa", title="Visa", quote=quote, supports=supports,
                    relevance="applies")


def research_says(**rule):
    fields = {"passport": "US", "country": "BT", "visa_required": True, "document": "e_visa", "confirmed": True,
              "citations": [cite("All tourists need a visa, applied for online", "requirement"),
                            cite("The visa fee is USD 90", "fee"), cite("Visas are issued for up to 30 days", "stay")]}
    return VisaRule(**{**fields, **rule})


def test_a_seed_fee_contradicted_by_its_citation_is_not_confirmed():
    merged = confirm_seed(seed_bt(), "US", research_says(fee=90, fee_currency="USD"))
    assert merged.confirmed  # the requirement still matches
    assert merged.unconfirmed == ["fee"] and merged.fee == 40
    assert "fee as 90 USD" in merged.notes
    assert not any(c.supports == "fee" for c in merged.citations)  # "USD 90" is not shown as backing USD 40
    assert "(fee not confirmed)" in summarise(merged)


def test_a_changed_document_or_stay_is_flagged_too():
    on_arrival = confirm_seed(seed_bt(), "US", research_says(document="visa_on_arrival", fee=40))
    assert not on_arrival.confirmed and "visa on arrival" in on_arrival.notes
    assert not any(c.supports == "requirement" for c in on_arrival.citations)
    shorter = confirm_seed(seed_bt(max_stay_days=90), "US", research_says(fee=40, max_stay_days=30))
    assert "stay" in shorter.unconfirmed and "stay as 30 days" in shorter.notes


def test_a_seed_fee_is_confirmed_only_by_its_amount_and_currency():
    same = confirm_seed(seed_bt(), "US", research_says(
        fee=40, citations=[cite("All tourists need a visa", "requirement"), cite("The visa fee is USD 40", "fee")]))
    assert same.confirmed and same.unconfirmed == ["stay"]  # no quote states the 30 days
    rupees = confirm_seed(seed_bt(), "US", research_says(
        fee=None, citations=[cite("All tourists need a visa", "requirement"), cite("The visa fee is INR 40", "fee")]))
    assert "fee" in rupees.unconfirmed


# ---------- [P1] evidence must name the item it prices and its unit ----------

def bali_source(text):
    return Source(id="C1.1", kind="web", url="https://bali.example/costs", title="Bali costs",
                  retrieved_at="2026-10-30T00:00:00+00:00", text=text)


def test_a_hotel_price_never_confirms_a_tourist_levy():
    hotels = "Hotels in Bali cost USD 100 per room per night"
    levy = FeeRule(country="ID", place="Bali", name="Bali tourist levy", basis="per_trip", amount=100, currency="USD",
                   source_id="C1.1", quote=hotels)
    assert not confirm_fee(levy, {"C1.1": bali_source(hotels)}, ["Bali"]).confirmed
    real = "Foreign tourists pay a levy of IDR 150,000 per person on arrival in Bali"
    levy = levy.model_copy(update={"amount": 150000, "currency": "IDR", "quote": real})
    assert confirm_fee(levy, {"C1.1": bali_source(real)}, ["Bali"]).confirmed
    nightly = levy.model_copy(update={"quote": "A levy of IDR 150,000 per person per night applies in Bali"})
    assert not confirm_fee(nightly, {"C1.1": bali_source(nightly.quote)}, ["Bali"]).confirmed  # not per trip


def test_a_per_person_price_is_not_a_room_rate_and_a_price_must_say_what_it_is():
    per_person = "Bali hotels from USD 50 per person per night"
    assert price_problem(50, "USD", "per room-night", per_person, bali_source(per_person), ["Bali"], item="budget hotel")
    no_unit = "Bali hotels from USD 50"
    assert "not a price per room-night" in price_problem(50, "USD", "per room-night", no_unit, bali_source(no_unit),
                                                         ["Bali"], item="budget hotel")
    not_food = "Bali scooter hire USD 5 a day"
    assert "doesn't say it prices food" in price_problem(5, "USD", "per person-day", not_food, bali_source(not_food),
                                                         ["Bali"], item="food")
    ok = "Bali hotels from USD 50 a night per room"
    assert price_problem(50, "USD", "per room-night", ok, bali_source(ok), ["Bali"], item="budget hotel") == ""


# ---------- [P2] dropping a city changes the trip, its research and what QA expects ----------

def test_dropping_a_city_updates_the_request_research_and_qa():
    norway = trip(person("A"), stops=[Stop(place="Norway", country="NO", cities=["Oslo", "Bergen"])], nights=6)
    kept = real_changes(TripChanges(stops=[Stop(place="Norway", country="NO", cities=["Oslo"])]), norway,
                        towns=["Oslo", "Bergen"])
    assert kept.stops and kept.stops[0].cities == ["Oslo"]  # not discarded as "no trip change"
    new = apply_changes(norway, kept, date(2026, 11, 1))
    assert new.stops[0].cities == ["Oslo"]
    assert stop_key(new, new.stops[0], None) != stop_key(norway, norway.stops[0], None)  # costs researched again
    assert group_key(new, new.groups[0]) != group_key(norway, norway.groups[0])
    oslo_only = Itinerary(days=[DayPlan(day=i, city="Oslo", title="Day") for i in range(1, 8)])
    food = {"Oslo": FoodGuide(city="Oslo", dishes=[Dish(name="Fiskesuppe", description="Fish soup")])}
    assert not any(i.startswith("city_missing") for i in check({"request": new, "itinerary": oslo_only, "food": food}).issues)


# ---------- found by the live smoke test: Tavily's rate limit is retried ----------

def test_a_search_rate_limited_once_is_retried(monkeypatch, tmp_path):
    from tavily.errors import UsageLimitExceededError
    calls = []

    class Limited:
        def search(self, **params):
            calls.append(1)
            if len(calls) == 1:
                raise UsageLimitExceededError("Your request has been blocked due to excessive requests.")
            return {"results": [{"url": "https://eat.example", "title": "Paro food", "content": "Ema datshi"}]}
    monkeypatch.setattr(search_module, "_tavily", lambda: Limited())
    assert search_module.search("Paro local food")[0]["text"] == "Ema datshi" and len(calls) == 2


def test_out_of_credits_is_not_retried():
    from tavily.errors import ForbiddenError
    calls = []

    def no_credits():
        calls.append(1)
        raise ForbiddenError("This request exceeds your plan's set usage limit.")
    with pytest.raises(ForbiddenError):
        call_with_retry(no_credits)
    assert len(calls) == 1


# ---------- found by the baseline run (travelex-5552cfa8) ----------

from travelex.agents.food import checked_tags  # noqa: E402
from travelex.agents.intake import clarify, intake, route_after_clarify  # noqa: E402
from travelex.agents.itinerary import country_of  # noqa: E402

BASELINE_TODAY = date(2026, 10, 27)


def places(request):
    return [(s.place, s.country, s.cities) for s in request.stops]


def test_stops_in_one_country_get_their_own_names():
    """E4: "Thailand: Bangkok, Chiang Mai and Phuket" became three "Thailand" stops, which shared one set of rates
    and left two "Thailand to Thailand" journeys unpriced. E5 did the same with Marrakech and Fes."""
    thailand = trip(person("A"), stops=[Stop(place="Thailand", country="TH", cities=[c])
                                        for c in ("Bangkok", "Chiang Mai", "Phuket")], nights=9)
    assert places(tidy(thailand, BASELINE_TODAY)) == [("Bangkok", "TH", []), ("Chiang Mai", "TH", []),
                                                      ("Phuket", "TH", [])]
    morocco = trip(person("A"), stops=[Stop(place="Morocco", country="MA", cities=["Marrakech"]),
                                       Stop(place="Sahara", country="MA"),
                                       Stop(place="Morocco", country="MA", cities=["Fes"])], nights=5)
    request = tidy(morocco, BASELINE_TODAY)
    assert [s.place for s in request.stops] == ["Marrakech", "Sahara", "Fes"]
    costs = {s.place: stop_costs(s.place, fee_country=None) for s in request.stops}
    f = assess(request, [result("G1", option("CDG-RAK", 200))], costs)
    assert not any("Morocco to" in u for u in f.unpriced)  # each journey is named by its own town


def test_one_place_split_over_two_stops_in_a_row_is_one_stop():
    split = trip(person("A"), stops=[Stop(place="Bali", country="ID", nights=3), Stop(place="Bali", country="ID", nights=2)],
                 nights=None)
    assert [(s.place, s.nights) for s in tidy(split, BASELINE_TODAY).stops] == [("Bali", 5)]


def test_a_return_visit_is_researched_once():
    """Tokyo, Kyoto, back to Tokyo: researching the last Tokyo again (with nowhere onward) would overwrite the first
    visit's Tokyo → Kyoto journey."""
    tokyo, kyoto = Stop(place="Tokyo", country="JP"), Stop(place="Kyoto", country="JP")
    request = trip(person("A"), stops=[tokyo, kyoto, tokyo], nights=6, month="2027-04")
    assert places(tidy(request, BASELINE_TODAY)) == [("Tokyo", "JP", []), ("Kyoto", "JP", []), ("Tokyo", "JP", [])]
    sends = [s for s in route_research({"request": request, "today": None, "sources": {}}) if s.node == "destination_costs"]
    assert [(s.arg["stop"].place, s.arg["next_stop"].place) for s in sends] == [("Tokyo", "Kyoto"), ("Kyoto", "Tokyo")]


def test_no_destination_and_several_home_cities_is_sent_back_for_correction(use_fake_llm):
    """E3: "2 adults … from Kuala Lumpur, Seoul and Busan" read as three home cities and no destination."""
    who = lambda label, origin: {"label": label, "type": "adult", "passport": "MY", "origin": origin}
    misread = {"stops": [], "month": "2027-11", "duration_text": "7 days", "diet": ["halal", "nut allergy"],
               "travellers": [who("Adult 1", "Kuala Lumpur"), who("Adult 2", "Seoul")]}
    # what the live model sent back: the stops right, but the nut allergy and Adult 2's home city gone
    corrected = {**misread, "stops": [{"place": "South Korea", "country": "KR", "cities": ["Seoul", "Busan"]}],
                 "diet": ["halal"], "travellers": [who("Adult 1", "Kuala Lumpur"), who("Adult 2", None)]}
    llm = use_fake_llm(intake_module, tool_call("TripRequest", misread), tool_call("TripRequest", corrected))
    out = intake({"raw_query": "2 adults with Malaysian passports from Kuala Lumpur, Seoul and Busan for 7 days",
                  "today": BASELINE_TODAY})
    check = llm.calls[1][-1].content
    assert len(llm.calls) == 2 and "the trip has no destination" in check and '"origin":"Seoul"' in check
    assert 'Read "from Kuala Lumpur, Seoul and Busan" again' in check  # the live model kept its reading without it
    request = out["request"]
    assert places(request) == [("South Korea", "KR", ["Seoul", "Busan"])]
    assert request.diet == ["halal", "nut allergy"]  # kept from the first reading
    assert [t.origin for t in request.travellers] == ["Kuala Lumpur", "Kuala Lumpur"]  # not Seoul, now a stop


def test_a_place_the_corrected_reading_drops_is_named_back(use_fake_llm):
    """Live, the second reading of "from Kuala Lumpur, Seoul and Busan" kept Seoul as the only stop."""
    who = lambda label, origin: {"label": label, "type": "adult", "passport": "MY", "origin": origin}
    base = {"month": "2027-11", "duration_text": "7 days", "travellers": [who("Adult 1", "Kuala Lumpur"),
                                                                          who("Adult 2", "Kuala Lumpur")]}
    seoul_only = {**base, "stops": [{"place": "Seoul", "country": "KR"}]}
    both = {**base, "stops": [{"place": "Seoul", "country": "KR"}, {"place": "Busan", "country": "KR"}]}
    llm = use_fake_llm(intake_module, tool_call("TripRequest", seoul_only), tool_call("TripRequest", both))
    out = intake({"raw_query": "2 adults from Kuala Lumpur, Seoul and Busan for 7 days", "today": BASELINE_TODAY})
    assert len(llm.calls) == 2 and "names Busan, which is neither a home city nor a stop" in llm.calls[1][-1].content
    assert [s.place for s in out["request"].stops] == ["Seoul", "Busan"]


@pytest.mark.parametrize("raw, diet, kept", [
    ("We eat halal only and one of us has a nut allergy.", ["halal"], ["halal", "nut allergy"]),  # E3, live
    ("My son is allergic to peanuts", [], ["allergic to peanuts"]),
    ("I'm coeliac and lactose intolerant", ["gluten-free"], ["gluten-free", "lactose intolerant"]),
    ("Our 6-year-old has a peanut allergy", ["peanut allergy"], ["peanut allergy"]),  # already there
    ("Vegetarian, no allergies", ["vegetarian"], ["vegetarian"]),
])
def test_an_allergy_the_request_states_is_never_dropped(raw, diet, kept):
    request = trip(person("A"), diet=diet, raw_query=raw)
    assert tidy(request, BASELINE_TODAY).diet == kept


@pytest.mark.parametrize("raw, lists", [
    ("2 adults from Kuala Lumpur, Seoul and Busan for 7 days", [["Kuala Lumpur", "Seoul", "Busan"]]),
    ("Two adults from Dubai and Chennai to Bhutan", [["Dubai", "Chennai"]]),
    ("3 friends from London, around £3,000.", []),  # one place; the rest isn't a name
    ("one flying from Lagos on a Nigerian passport, one from Nairobi", []),
])
def test_from_lists_reads_only_place_names(raw, lists):
    from travelex.agents.intake import from_lists
    assert [names for _, names in from_lists(raw)] == lists


@pytest.mark.parametrize("text, nights", [
    ("from Kuala Lumpur, Seoul and Busan for 7 days in November", {}),  # E3: the whole trip, not Busan's share
    ("Seoul, Busan for 7 days", {}),
    ("Seoul for 3 nights, then Busan for 4 nights", {"Seoul": 3, "Busan": 4}),
    ("4 nights in Seoul and 3 nights in Busan", {"Seoul": 4, "Busan": 3}),
])
def test_a_length_after_a_list_of_stops_is_the_whole_trip(text, nights):
    from travelex.trip import stop_lengths
    stops = [Stop(place="Seoul", country="KR"), Stop(place="Busan", country="KR")]
    assert stop_lengths(text, stops)[0] == nights


def test_one_home_city_and_no_destination_is_not_rechecked(use_fake_llm):
    """Destination ideas are a normal answer to "somewhere warm": no extra model call for them."""
    llm = use_fake_llm(intake_module, tool_call("TripRequest", {"stops": [], "travellers": [
        {"label": "Adult 1", "type": "adult", "passport": "GB", "origin": "London"}]}))
    intake({"raw_query": "Somewhere warm from London", "today": BASELINE_TODAY})
    assert len(llm.calls) == 1


def test_auto_mode_with_no_destination_ends_and_says_why():
    """It used to return without counting a round, so route_after_clarify sent it back to clarify forever."""
    request = trip(person("A"), stops=[])
    out = clarify({"request": request, "clarify_round": 0}, {"configurable": {"auto_approve": True}})
    assert out["events"][0]["action"] == "no destination"
    assert route_after_clarify({"request": request, **out}) == END


def test_food_research_names_the_country_and_drops_places_from_elsewhere(use_fake_llm, search_calls, fake_search):
    """D4: Iceland's "South Coast Region" got South Coast Plaza in Costa Mesa, California."""
    fake_search += [page("https://ca.example/eats", "South Coast Plaza in Costa Mesa serves shrimp and grits.",
                         title="Costa Mesa eats"),
                    page("https://is.example/vik", "Halldórskaffi in Vík is a cosy cafe on the south coast of Iceland.",
                         title="Where to eat in south Iceland")]
    llm = use_fake_llm(food_module, tool_call("FoodGuide", {"city": "South Coast Region", "dishes": [
        {"name": "Kjötsúpa", "description": "Lamb soup"}], "places": [
        {"name": "South Coast Plaza", "source_ids": ["F1.1"]}, {"name": "Halldórskaffi", "source_ids": ["F1.2"]}]}))
    request = trip(person("A", passport="GB", origin="London"), stops=[Stop(place="Iceland", country="IS")])
    out = food_agent({"city": "South Coast Region", "country": country_of(request, "South Coast Region"),
                      "source_prefix": "F1.", "request": request})
    assert all("Iceland" in q for q in search_calls)
    assert "Country: Iceland" in llm.calls[0][-1].content
    assert [p.name for p in out["food"]["South Coast Region"].places] == ["Halldórskaffi"]


def test_a_food_city_gets_the_country_of_its_stop():
    request = trip(person("A"), stops=[Stop(place="Italy", country="IT", nights=4),
                                       Stop(place="Switzerland", country="CH", nights=3)], nights=None)
    plan = Itinerary(days=[DayPlan(day=d, city=c, title="") for d, c in
                           [(1, "Milan"), (3, "Lake Como"), (6, "Lugano"), (8, "Zurich")]])
    assert [country_of(request, c, plan) for c in ("Milan", "Lake Como", "Lugano", "Zurich")] == ["IT", "IT", "CH", "CH"]
    assert country_of(request, "Italy") == "IT"  # a stop's own name


@pytest.mark.parametrize("name, description, tags, kept", [
    ("Cuy chactado", "Fried guinea pig", ["contains pork"], []),             # guinea pig is not pork
    ("Suadero tacos", "Beef brisket tacos", ["contains pork", "spicy"], ["spicy"]),
    ("Baccalà mantecato", "Whipped salt cod on polenta", ["vegetarian"], []),  # cod is fish
    ("Nasi campur", "Rice with pork satay", ["halal"], ["contains pork"]),     # never "halal"; pork named
    ("Veg momos", "Like the chicken ones, but with cabbage", ["vegetarian"], ["vegetarian"]),
    ("Pastel de nata", "Custard tart", ["vegetarian"], ["vegetarian"]),
])
def test_diet_tags_are_checked_against_the_ingredients(name, description, tags, kept):
    assert checked_tags(Dish(name=name, description=description, diet_tags=tags)) == kept


# ---------- alternatives that really fix the problem (judge: 0.45 across the baseline) ----------

import travelex.agents.alternatives as alternatives_module  # noqa: E402
from travelex.agents.alternatives import levers, make_alternatives_agent, trade_offs  # noqa: E402
from travelex.schemas import CostLine, Feasibility  # noqa: E402


def bhutan_costs(budget: float) -> Feasibility:
    """3 adults, 4 nights: flights 900 whatever the length; the SDF (1,200) and hotels (240) by the night."""
    lines = [CostLine(category=c, label=c, amount=a, currency="USD", amount_report=a)
             for c, a in (("transport", 900), ("destination_fee", 1200), ("lodging", 240))]
    return Feasibility(ok=2340 <= budget, currency="USD", total=2340, budget=budget, gap=2340 - budget, lines=lines)


def test_the_model_is_told_how_the_costs_scale():
    request = trip(person("A"), person("B"), person("C"), nights=4)
    assert levers({"request": request, "feasibility": bhutan_costs(800)}) == [
        "Each night costs about 360 USD (hotels, food, guides, vehicles, daily fees); about 900 USD (flights, visas, "
        "one-off fees) is paid however long the trip is.",
        "The fixed costs alone are over the 800 USD budget: fewer nights or cheaper hotels cannot fit it.",
        "Hotels (mid-range) are 240 USD of the total."]
    assert levers({"request": request, "feasibility": bhutan_costs(2000)})[1] == \
        "Fewer nights alone fits at 3 nights or fewer (now 4)."


def test_trade_offs_are_worked_out_from_the_two_trips():
    old = trip(person("A"), nights=4, month="2027-10")
    new = old.model_copy(update={"stops": [Stop(place="Nepal", country="NP")], "month": "2027-11", "nights": 3,
                                 "lodging_style": "budget"})
    assert trade_offs(old, new) == ["goes to Nepal instead of Bhutan", "travels in November 2027 instead of October 2027",
                                    "1 fewer night (3 instead of 4)", "budget hotels instead of mid-range"]
    assert trade_offs(old, old) == []


class FakeResearch:
    """Costs a trip: Nepal fits the 1,000 USD budget, anything in Bhutan doesn't."""
    def __init__(self):
        self.trips = []

    def invoke(self, seed, config):
        self.trips.append(seed["request"])
        total = 700 if seed["request"].stops[0].place == "Nepal" else 1500
        return {"feasibility": Feasibility(ok=total <= 1000, currency="USD", total=total, budget=1000, gap=total - 1000)}


def test_no_fix_on_the_first_try_gets_a_second_with_what_fell_short(use_fake_llm):
    """C2, D5: every alternative still over budget, one of them "budget hotels" on a trip that already had them."""
    first = tool_call("AlternativeIdeas", {"ideas": [
        {"title": "Budget hotels", "why": "Cheaper rooms", "changes": {"lodging_style": "budget"}},
        {"title": "Three nights", "why": "Fewer SDF nights", "changes": {"duration_text": "3 nights"}}]})
    second = tool_call("AlternativeIdeas", {"ideas": [
        {"title": "Nepal instead", "why": "No daily fee", "changes": {"stops": [{"place": "Nepal", "country": "NP"}]}}]})
    llm = use_fake_llm(alternatives_module, first, second)
    research = FakeResearch()
    request = trip(person("A"), nights=4, month="2027-10", lodging_style="budget", budget=(1000, "USD"))
    state = {"request": request, "today": BASELINE_TODAY, "raw_query": "", "season": {},
             "feasibility": Feasibility(ok=False, currency="USD", total=1500, budget=1000, gap=500)}
    out = make_alternatives_agent(research)(state, {"configurable": {}})

    assert len(research.trips) == 2  # "budget hotels" changed nothing, so it was never costed
    retry = llm.calls[1][-1].content
    assert "Tried, and none of these fixes it:" in retry and "Three nights" in retry and "over budget by 500" in retry
    assert [a.title for a in out["alternatives"]] == ["Nepal instead", "Three nights"]  # the fix first
    nepal, shorter = out["alternatives"]
    assert nepal.trade_offs == ["goes to Nepal instead of Bhutan"]
    assert shorter.trade_offs == ["1 fewer night (3 instead of 4)", "still over budget by 500 USD"]
    assert any(e["action"] == "tried again" for e in out["events"])


def test_picking_for_the_user_keeps_a_destination_that_can_fit():
    """B3: "see Prague and Vienna" on €1,200 was swapped for Slovakia, the one alternative that fitted. Prague and
    Vienna could fit (a train from Berlin leaves room for nights there), so the closest one that keeps them wins.
    C2: 3 adults to Bhutan on $800 can't fit (flights alone break it), so Nepal is right."""
    from travelex.agents.alternatives import best, destination_can_fit, parse_choice
    from travelex.schemas import Alternative, TripChanges
    def option(title, total, trade_offs):
        return Alternative(title=title, why="", changes=TripChanges(), request=trip(person("A")),
                           feasibility=Feasibility(ok=total <= 1200, currency="EUR", total=total, budget=1200,
                                                   gap=total - 1200), trade_offs=trade_offs)
    alternatives = [option("Shorter", 1450, ["2 fewer nights (3 instead of 5)"]),
                    option("Slovakia", 1190, ["goes to Slovakia instead of Prague → Vienna"])]
    assert best(alternatives) == 1  # "cheapest": what fits, wherever it is
    assert best(alternatives, keep_destination=True) == 0
    assert parse_choice("pick for me", alternatives) == "auto" and parse_choice("you choose", alternatives) == "auto"

    # C2, live: "raise the budget to $3,645" was only 640 over its own new budget, so it looked closest
    bigger = option("Bigger budget", 1900, ["a budget of 1,800 EUR instead of 1,200 EUR"])
    bigger = bigger.model_copy(update={"feasibility": bigger.feasibility.model_copy(update={"budget": 1800, "gap": 100})})
    far = [option("Sri Lanka", 1900, ["goes to Sri Lanka instead of Prague → Vienna"]), bigger]
    assert best(far) == 1 and best(far, spend_more=False) == 0

    request = trip(person("A"), person("B"), person("C"), nights=4)
    assert not destination_can_fit({"request": request, "feasibility": bhutan_costs(800)})  # flights 900 > 800
    assert destination_can_fit({"request": request, "feasibility": bhutan_costs(2000)})  # room for 3 nights
    assert not destination_can_fit({"request": request, "feasibility": bhutan_costs(1500)})  # 1 night isn't a trip
    assert destination_can_fit({"request": request, "feasibility": None})  # a season problem: keep the place


def test_a_first_try_that_fixes_it_is_not_tried_again(use_fake_llm):
    llm = use_fake_llm(alternatives_module, tool_call("AlternativeIdeas", {"ideas": [
        {"title": "Nepal instead", "why": "No daily fee", "changes": {"stops": [{"place": "Nepal", "country": "NP"}]}}]}))
    state = {"request": trip(person("A"), nights=4, month="2027-10", budget=(1000, "USD")), "today": BASELINE_TODAY,
             "raw_query": "", "season": {}, "feasibility": Feasibility(ok=False, currency="USD", total=1500, budget=1000, gap=500)}
    out = make_alternatives_agent(FakeResearch())(state, {"configurable": {}})
    assert len(llm.calls) == 1 and [a.title for a in out["alternatives"]] == ["Nepal instead"]
