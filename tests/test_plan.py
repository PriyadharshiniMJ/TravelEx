"""Steps 4-5: alternatives, itinerary, food, QA and the report, and the whole graph end to end."""
import re
from datetime import date

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import travelex.agents.alternatives as alternatives_module
import travelex.agents.costs as costs_module
import travelex.agents.food as food_module
import travelex.agents.gaps as gaps_module
import travelex.agents.intake as intake_module
import travelex.agents.itinerary as itinerary_module
import travelex.agents.season as season_module
import travelex.agents.transport as transport_module
import travelex.agents.visa as visa_module
import travelex.report as report_module
from builders import person, trip
from fakes import page, tool_call
from travelex.agents.alternatives import apply_changes, best, parse_choice, problems
from travelex.agents.food import breaks_diet, forbidden_terms, grounded
from travelex.agents.itinerary import itinerary_planner
from travelex.agents.qa import check
from travelex.graph import build_graph, initial_state, run_config, serializer
from travelex.schemas import (
    Alternative, DayPlan, Dish, Feasibility, FoodGuide, Itinerary, Money, Place, SeasonVerdict, Source, Stop,
    TripChanges,
)

TODAY = date(2026, 10, 27)
NEPAL = {"place": "Nepal", "country": "NP", "cities": ["Kathmandu"]}


# ---------- alternatives: the pure parts ----------

def feas(total, budget=1000, ok=None):
    return Feasibility(ok=total <= budget if ok is None else ok, currency="USD", total=total, budget=budget,
                       gap=total - budget)


def alt(title, total, season="good"):
    return Alternative(title=title, why="", changes=TripChanges(), request=trip(person("A")), feasibility=feas(total),
                       season={"Bhutan": SeasonVerdict(place="Bhutan", month=10, verdict=season)})


def test_problems_name_the_budget_gap_and_bad_seasons():
    state = {"request": trip(person("A")), "feasibility": feas(1500),
             "season": {"Bhutan": SeasonVerdict(place="Bhutan", month=7, verdict="avoid", reasons=["monsoon"])}}
    assert problems(state) == ["Over budget by 500 USD (1,500 vs 1,000).", "Bhutan is best avoided then: monsoon."]
    assert problems({**state, "feasibility": feas(900), "season": {}}) == []


def test_apply_changes():
    base = trip(person("A"), start_date=date(2027, 10, 10), end_date=date(2027, 10, 16), exact_dates=True, nights=None)
    shorter = apply_changes(base, TripChanges(duration_text="4 days"), TODAY)
    assert (shorter.start_date, shorter.end_date) == (date(2027, 10, 10), date(2027, 10, 13))
    later = apply_changes(base, TripChanges(month="2027-11"), TODAY)
    assert (later.month, later.start_date, later.exact_dates) == ("2027-11", None, False)
    nepal = apply_changes(base, TripChanges(stops=[Stop(**NEPAL)], lodging_style="budget"), TODAY)
    assert (nepal.destination, nepal.lodging_style, len(nepal.groups)) == ("Nepal", "budget", 1)


@pytest.mark.parametrize("answer, expected", [
    ("option 2", 1), ("2", 1), (2, 1), ({"choice": 3}, 2), ("keep", "keep"), ("", "keep"), (0, "keep"),
    ("the cheapest one", 1), ("None of these, show me cheaper options", None), ("somewhere warmer", None),
    ("option 9", None),
])
def test_parse_choice(answer, expected):
    alternatives = [alt("Fewer nights", 1200), alt("Nepal", 900), alt("October", 950, season="avoid")]
    assert parse_choice(answer, alternatives) == expected


def test_best_prefers_the_cheapest_fix_then_the_smallest_gap():
    assert best([alt("A", 990), alt("B", 900), alt("C", 800, season="avoid")]) == 1
    assert best([alt("A", 1500), alt("B", 1200)]) == 1


# ---------- food and QA ----------

def test_diet_filter():
    veg = forbidden_terms(["vegetarian", "peanut allergy"])
    assert breaks_diet(Dish(name="Chicken momos", description="steamed"), veg) == "chicken"
    assert breaks_diet(Dish(name="Veg momos", description="unlike the chicken ones, filled with cabbage"), veg) is None
    assert breaks_diet(Dish(name="Ema datshi", description="chillies and cheese"), veg) is None
    assert breaks_diet(Dish(name="Ramen", description="pork broth"), forbidden_terms(["halal"])) == "pork"
    assert forbidden_terms(["peanut allergy"]) == set()


def test_places_must_be_named_by_their_source():
    sources = {"F1.1": Source(id="F1.1", url="https://a", title="Where to eat in Paro", kind="web",
                              retrieved_at="2026-10-27T00:00:00+00:00", text="Try Sonam Trophel for ema datshi.")}
    assert grounded(Place(name="Sonam Trophel", source_ids=["F1.1"]), sources)
    assert not grounded(Place(name="Made Up Café", source_ids=["F1.1"]), sources)
    assert not grounded(Place(name="Sonam Trophel", source_ids=[]), sources)


def days(n, city="Paro", strenuous_day=None, rest=False):
    from travelex.schemas import Activity
    return [DayPlan(day=i, city=city, title="Day", rest_day=rest and i == 3,
                    activities=[Activity(name="Hike", strenuous=True)] * (2 if i == strenuous_day else 1))
            for i in range(1, n + 1)]


def test_qa_checks():
    family = trip(person("A"), person("Kid", type="child", age=5), nights=5, diet=["vegetarian"])
    food = {"Paro": FoodGuide(city="Paro", dishes=[Dish(name="Pork curry", description="")])}
    report = check({"request": family, "itinerary": Itinerary(days=days(4, strenuous_day=2)), "food": food})
    assert report.issues == ["day_count", "pacing:day 2", "rest_day", "diet:Paro"]  # no guide wanted for home (Chennai)
    good = {"Paro": FoodGuide(city="Paro", dishes=[Dish(name="Ema datshi", description="")]),
            "Chennai": FoodGuide(city="Chennai", dishes=[Dish(name="Idli", description="")])}
    report = check({"request": family, "itinerary": Itinerary(days=days(6, rest=True)), "food": good})
    assert report.passed and "No sourced places to eat in Paro." in report.notes


def test_itinerary_days_are_numbered_and_dated(use_fake_llm):
    llm = use_fake_llm(itinerary_module, tool_call("Itinerary", {"days": [
        {"day": 7, "city": "Paro", "title": "Arrive"}, {"day": 7, "city": "Thimphu", "title": "Capital"}]}))
    request = trip(person("A"), start_date=date(2027, 10, 10), end_date=date(2027, 10, 11), nights=None)
    out = itinerary_planner({"request": request})
    assert [(d.day, d.date) for d in out["itinerary"].days] == [(1, date(2027, 10, 10)), (2, date(2027, 10, 11))]
    assert "Exactly 2 days" in llm.calls[0][0].content


# ---------- the whole graph ----------

def everything(calls):
    """One fake model for every agent, answering by its system prompt. `calls` counts calls per agent."""
    def respond(messages):
        system, task = messages[0].content, messages[-1].content
        agent = next(name for prefix, name in [
            ("You turn a traveller's", "intake"), ("List this group's", "routes"), ("You find how", "transport"),
            ("You estimate", "costs"), ("Extract the entry rule", "visa"), ("You help travellers", "alternatives"),
            ("You plan a realistic", "itinerary"), ("You write a short food guide", "food"),
            ("You write a month-by-month", "season"), ("You check whether quotes", "verify"),
            ("A traveller has commented", "review"), ("You fill gaps", "gaps")]
            if system.startswith(prefix))
        calls[agent] = calls.get(agent, 0) + 1
        if agent == "intake":
            return tool_call("TripRequest", {
                "stops": [{"place": "Bhutan", "country": "BT"}], "month": "2027-10", "duration_text": "4-5 days",
                "budget": {"amount": 1000, "currency": "USD"},
                "travellers": [{"label": f"Adult {i}", "type": "adult", "passport": "US", "origin": "Dubai"}
                               for i in (1, 2, 3)]})
        if agent == "routes":
            return tool_call("RouteList", {"home_airports": ["DXB"], "options": [
                {"mode": "flight", "route": "DXB-KTM-PBH" if "Bhutan" in task else "DXB-KTM",
                 "adult_fare": {"amount": 150, "currency": "USD"}, "max_layover_hours": 2}]})
        if agent == "costs":
            return tool_call("StopCosts", {"place": "x", "currency": "USD", "lodging_per_room_night": 30, "rooms": 2,
                                           "room_config": "2 doubles", "food_per_person_day": 10})
        if agent == "alternatives":
            return tool_call("AlternativeIdeas", {"ideas": [
                {"title": "Nepal instead", "why": "No daily fee", "changes": {"stops": [NEPAL]}},
                {"title": "Three days", "why": "Fewer SDF nights", "changes": {"duration_text": "3 days"}}]})
        if agent == "itinerary":
            n = int(re.search(r"Exactly (\d+) days", system).group(1))
            city = "Kathmandu" if "Nepal" in task else "Paro"
            return tool_call("Itinerary", {"days": [{"day": i, "city": city, "title": "Explore"} for i in range(1, n + 1)]})
        if agent == "visa":
            sid = re.search(r"^\[([^\]]+)\]", task, re.MULTILINE)  # IDs look like G1.1, or A1.1.G1.1 in an alternative
            country = re.search(r"Country visited: .+ \((\w\w)\)", task).group(1)
            evidence = [{"source_id": sid.group(1), "quote": "US citizens need a visa", "supports": "requirement"}] if sid else []
            document = "visa_on_arrival" if country == "NP" else "e_visa"  # what the stored rules say
            return tool_call("VisaResearch", {"rule": {"passport": "US", "country": country, "visa_required": True,
                                                       "document": document}, "evidence": evidence})
        if agent == "verify":
            shown = [int(i) for i in re.findall(r"^\[(\d+)\]", task, re.MULTILINE)]
            return tool_call("RelevanceCheck", {"verdicts": [{"index": i, "verdict": "applies", "reason": "t"}
                                                            for i in shown]})
        if agent == "review":  # "make it N days" changes the trip; anything else changes the days
            comment = re.search(r"Comment: (.+)", task).group(1)
            days = re.search(r"make it (\d+) days", comment)
            return tool_call("PlanComment", {"summary": comment, "trip_changes":
                                             {"duration_text": f"{days.group(1)} days"} if days else {}})
        if agent == "food":
            city = re.search(r"City: (.+)", task).group(1)
            return tool_call("FoodGuide", {"city": city, "dishes": [{"name": "Local thali", "description": "rice"}]})
        return AIMessage("Done.")  # transport loop: no searching
    return respond


@pytest.fixture
def fake_everything(use_fake_llm, fake_search, monkeypatch, tmp_path):
    monkeypatch.setattr(report_module, "RUNS_DIR", tmp_path / "runs")
    fake_search.append(page("https://www.visaguide.example/us", "US citizens need a visa and pay the SDF.", title="Visa rules"))
    calls: dict[str, int] = {}
    for module in (intake_module, transport_module, costs_module, visa_module, alternatives_module,
                   itinerary_module, food_module, season_module, gaps_module):
        use_fake_llm(module, responder=everything(calls))
    return calls


def run(graph, payload, config):
    for _ in graph.stream(payload, config):
        pass
    snapshot = graph.get_state(config)
    return snapshot, (snapshot.interrupts[0].value if snapshot.interrupts else None)


def test_over_budget_trip_offers_costed_alternatives_and_switches(fake_everything):
    graph = build_graph(InMemorySaver(serde=serializer()))
    config = run_config("alts")
    _, pause = run(graph, initial_state("3 US adults Dubai to Bhutan 4-5 days $800", TODAY), config)
    assert pause["kind"] == "clarify"
    _, pause = run(graph, Command(resume=pause["request"]), config)

    assert pause["kind"] == "alternatives"
    assert any("Over budget" in p for p in pause["problems"])  # 3 × 4 × $100 SDF alone is $1,200
    by_title = {a["title"]: a for a in pause["alternatives"]}
    assert by_title["Nepal instead"]["fixes"] and "fits the budget" in by_title["Nepal instead"]["summary"]
    assert not by_title["Three days"]["fixes"]  # 3 × 2 nights × $100 SDF plus flights is still over $1,000
    assert fake_everything["routes"] == 2  # the main run + Nepal; "Three days" reused the Bhutan flights

    snapshot, pause = run(graph, Command(resume="1"), config)
    values = snapshot.values
    assert pause["kind"] == "review" and values["chosen_alternative"] == "Nepal instead"
    assert values["request"].destination == "Nepal" and values["feasibility"].ok
    assert set(values["costs"]) == {"Nepal"} and set(values["season"]) == {"Nepal"}  # replaced, not merged
    assert {d.city for d in values["itinerary"].days} == {"Kathmandu"}
    assert set(values["food"]) == {"Kathmandu"}  # the destination only, never home
    assert values["qa_report"].passed
    report = (values["report_dir"] + "/report.md")
    text = open(report, encoding="utf-8").read()
    for heading in ("# Nepal", "## Day by day", "## Getting there", "## Visas and entry", "## Food guide",
                    "## Budget", "## Alternatives considered"):
        assert heading in text
    assert "**Nepal instead**" in text and "← chosen" in text
    html = open(values["report_dir"] + "/report.html", encoding="utf-8").read()
    for part in ('class="hero"', "<h1>Nepal</h1>", 'class="timeline"', 'class="option picked"', 'class="visa ',
                 'class="meter"', "🇳🇵", "🇺🇸", 'id="src-1"', 'class="fn" href="https://www.visaguide.example/us" target="_blank"', "Switched to: Nepal instead"):
        assert part in html, part
    assert 'href="#' not in html  # in the app's frame an in-page link loads the app itself
    assert "“US citizens need a visa”" in text  # the verified visa quote, with its footnote
    assert "[Visa rules](https://www.visaguide.example/us)" in text and "a travel site, about your passport" in text


def test_feedback_gets_a_second_round_then_the_cap_holds(fake_everything):
    graph = build_graph(InMemorySaver(serde=serializer()))
    config = run_config("feedback")
    _, pause = run(graph, initial_state("3 US adults Dubai to Bhutan", TODAY), config)
    _, pause = run(graph, Command(resume=pause["request"]), config)
    _, pause = run(graph, Command(resume="None of these, show me cheaper options"), config)
    assert pause["kind"] == "alternatives"  # round 2
    snapshot, pause = run(graph, Command(resume="something else entirely"), config)
    assert pause["kind"] == "review"  # the cap: no third round, plan the trip as it stands
    values = snapshot.values
    assert values["alt_round"] == 2 and values.get("chosen_alternative") is None
    assert "does not fit" in open(values["report_dir"] + "/report.md", encoding="utf-8").read()


def test_qa_sends_a_short_itinerary_back_once(fake_everything):
    first = {"done": False}

    def bad_first_draft(state):
        out = itinerary_planner(state)
        if not first["done"]:
            first["done"] = True
            plan = out["itinerary"]
            out["itinerary"] = plan.model_copy(update={"days": plan.days[:-1]})
        return out

    graph = build_graph(InMemorySaver(serde=serializer()), nodes={"itinerary_planner": bad_first_draft})
    config = run_config("qa", auto_approve=True)
    snapshot, _ = run(graph, initial_state("3 US adults Dubai to Bhutan", TODAY), config)
    values = snapshot.values
    qa_events = [e for e in values["events"] if e["agent"] == "qa"]
    assert [e["action"] for e in qa_events] == ["found problems", "passed"]
    assert "day_count" in qa_events[0]["detail"]
    assert values["qa_round"] == 2 and fake_everything["itinerary"] == 2


def test_tidy_plan_drops_what_nobody_asked_for():
    from travelex.agents.itinerary import tidy_plan
    from travelex.schemas import Activity
    plan = Itinerary(days=[DayPlan(day=1, city="Paro", title="Hike",
                                   activities=[Activity(name="Tiger's Nest", strenuous=True, kid_option="Not applicable")])],
                     trimmed_variant="Drop day 3", notes=["Ensure visa approval for Bhutan", "Pack layers",
                                                          "The SDF is included in your package"])
    adults = tidy_plan(plan, trip(person("A"), nights=4))
    assert adults.days[0].activities[0].kid_option == "" and adults.trimmed_variant == ""
    assert adults.notes == ["Pack layers"]
    family = tidy_plan(plan, trip(person("A"), person("Kid", type="child", age=6), nights=None, duration_range=[4, 5]))
    assert family.days[0].activities[0].kid_option == "Not applicable" and family.trimmed_variant == "Drop day 3"


# ---------- food research starts early for cities the user named ----------

def named_cities_llm(calls):
    """everything(), except the trip names Paro and Thimphu, and the planner writes "Paro Valley"."""
    base = everything(calls)

    def respond(messages):
        system, task = messages[0].content, messages[-1].content
        if system.startswith("You turn a traveller's"):
            calls["intake"] = calls.get("intake", 0) + 1
            return tool_call("TripRequest", {
                "stops": [{"place": "Bhutan", "country": "BT", "cities": ["Paro", "Thimphu"]}], "month": "2027-10",
                "duration_text": "5 days", "budget": {"amount": 100000, "currency": "USD"},
                "travellers": [{"label": "Adult 1", "type": "adult", "passport": "US", "origin": "Dubai"}]})
        if system.startswith("You plan a realistic"):
            calls["itinerary"] = calls.get("itinerary", 0) + 1
            n = int(re.search(r"Exactly (\d+) days", system).group(1))
            return tool_call("Itinerary", {"days": [{"day": i, "city": "Paro Valley" if i % 2 else "Thimphu",
                                                     "title": "Explore"} for i in range(1, n + 1)]})
        return base(messages)
    return respond


def test_named_cities_get_food_guides_during_research(use_fake_llm, fake_search, monkeypatch, tmp_path):
    monkeypatch.setattr(report_module, "RUNS_DIR", tmp_path / "runs")
    calls: dict[str, int] = {}
    for module in (intake_module, transport_module, costs_module, visa_module, alternatives_module,
                   itinerary_module, food_module, season_module):
        use_fake_llm(module, responder=named_cities_llm(calls))
    graph = build_graph(InMemorySaver(serde=serializer()))
    config = run_config("early-food", auto_approve=True)
    snapshot, _ = run(graph, initial_state("Paro and Thimphu, 5 days", TODAY), config)
    values = snapshot.values
    agents = [e["agent"] for e in values["events"]]
    assert agents.index("food_agent") < agents.index("meeting_plan") < agents.index("itinerary_planner")
    assert calls["food"] == 2  # Paro and Thimphu, once each: "Paro Valley" reused the early Paro guide
    assert values["qa_report"].passed
    html = open(values["report_dir"] + "/report.html", encoding="utf-8").read()
    assert html.count('class="food-card"') == 2 and "🍽" in html


def test_research_for_alternatives_fetches_no_food():
    from travelex.graph import build_research_graph
    assert "food_early" not in build_research_graph().get_graph().nodes


def test_switching_keeps_only_early_guides_for_cities_still_on_the_trip():
    from travelex.agents.alternatives import switch_to
    nepal = trip(person("A"), stops=[Stop(place="Nepal", country="NP", cities=["Kathmandu"])])
    food = {"Paro": FoodGuide(city="Paro", dishes=[]), "Kathmandu": FoodGuide(city="Kathmandu", dishes=[])}
    update = switch_to(Alternative(title="Nepal", why="", changes=TripChanges(), request=nepal), "picked", food)
    assert set(update["food"].value) == {"Kathmandu"}


# ---------- the trip at a glance, shown while the itinerary is still being written ----------

def previews(graph, payload, config) -> list[tuple]:
    """(step, itinerary so far, preview page) for every trip at a glance the app would show."""
    from travelex.graph import settled
    from travelex.report_html import render_preview_html
    shown = []
    for namespace, update in graph.stream(payload, config, stream_mode="updates", subgraphs=True):
        for node, output in update.items():
            if not namespace and (view := settled(node, graph.get_state(config).values, output)):
                shown.append((node, view.get("itinerary"), render_preview_html(view)))
    return shown


def test_preview_waits_until_the_trip_is_settled(fake_everything):
    graph = build_graph(InMemorySaver(serde=serializer()))
    shown = previews(graph, initial_state("3 US adults Dubai to Bhutan", TODAY), run_config("preview", auto_approve=True))
    assert [p[0] for p in shown] == ["choose_alternative"]  # over budget: no Bhutan preview, only the Nepal switch
    node, itinerary, html = shown[0]
    assert itinerary is None  # shown before the days are planned
    assert "<h1>Nepal</h1>" in html and 'class="option' in html and 'class="meter' in html
    assert 'class="timeline"' not in html and "being written now" in html


@pytest.fixture
def fits_budget(use_fake_llm, fake_search, monkeypatch, tmp_path):
    """everything(), for a trip with room in the budget: no alternatives, straight to the plan."""
    import travelex.agents.review as review_module
    monkeypatch.setattr(report_module, "RUNS_DIR", tmp_path / "runs")
    calls: dict[str, int] = {}
    base = everything(calls)
    prompts: list = []

    def roomy(messages):
        prompts.append(messages)
        reply = base(messages)
        if messages[0].content.startswith("You turn a traveller's"):
            reply.tool_calls[0]["args"]["budget"] = {"amount": 100000, "currency": "USD"}
        return reply
    for module in (intake_module, transport_module, costs_module, visa_module, alternatives_module,
                   itinerary_module, food_module, season_module, review_module):
        use_fake_llm(module, responder=roomy)
    gap_model = use_fake_llm(gaps_module, responder=roomy)
    return calls, prompts, gap_model


def test_preview_shows_straight_away_when_nothing_needs_fixing(fits_budget):
    graph = build_graph(InMemorySaver(serde=serializer()))
    shown = previews(graph, initial_state("3 US adults Dubai to Bhutan", TODAY), run_config("fits", auto_approve=True))
    assert [p[0] for p in shown] == ["feasibility"]
    assert "<h1>Bhutan</h1>" in shown[0][2]


def test_state_after_applies_overwrites_and_merges():
    from langgraph.types import Overwrite
    from travelex.graph import state_after
    saved = {"costs": {"Bhutan": 1}, "sources": {"S1": 1}, "events": [1], "request": "old"}
    view = state_after(saved, {"costs": Overwrite({"Nepal": 2}), "sources": {"G1.1": 2}, "events": [2],
                               "request": "new"})
    assert view == {"costs": {"Nepal": 2}, "sources": {"S1": 1, "G1.1": 2}, "events": [1], "request": "new"}


@pytest.mark.parametrize("travellers, kid_friendly", [((person("Solo", origin="Dallas"),), False),
                                                      ((person("A"), person("Kid", type="child", age=6)), True)])
def test_food_is_kid_friendly_only_when_children_travel(use_fake_llm, fake_search, travellers, kid_friendly):
    from travelex.agents.food import food_agent
    fake_search.append(page("https://example.com/oslo", "Try fiskesuppe at Fisketorget."))
    llm = use_fake_llm(food_module, tool_call("FoodGuide", {"city": "Oslo", "dishes": [
        {"name": "Fiskesuppe", "description": "Creamy fish soup", "kid_friendly": True}]}))
    out = food_agent({"city": "Oslo", "request": trip(*travellers), "source_prefix": "F1."})
    assert out["food"]["Oslo"].dishes[0].kid_friendly is kid_friendly
    assert ("never mention children" in llm.calls[0][0].content) is not kid_friendly


def test_alternatives_never_go_back_in_time_or_cut_the_budget():
    from travelex.agents.alternatives import checked
    request = trip(person("A"), budget=(5000, "USD"))  # TODAY is 2026-10-27
    june = checked(TripChanges(month="2026-06", budget=Money(amount=1500, currency="USD")), request, TODAY)
    assert (june.month, june.budget) == ("2027-06", None)
    later = checked(TripChanges(month="2026-12", budget=Money(amount=6000, currency="USD")), request, TODAY)
    assert (later.month, later.budget.amount) == ("2026-12", 6000)


# ---------- comments on the finished plan ----------

def finished_plan(config):
    graph = build_graph(InMemorySaver(serde=serializer()))
    _, pause = run(graph, initial_state("3 US adults Dubai to Bhutan", TODAY), config)
    assert pause["kind"] == "clarify"
    snapshot, pause = run(graph, Command(resume=pause["request"]), config)
    if pause["kind"] == "handoff":  # the fake visa research leaves a rule unconfirmed: carry on
        snapshot, pause = run(graph, Command(resume=""), config)
    assert pause["kind"] == "review" and pause["revision"] == 0
    return graph, snapshot


def test_a_comment_on_the_days_reworks_only_the_itinerary(fits_budget):
    calls, prompts, _ = fits_budget
    config = run_config("comment-days")
    graph, first = finished_plan(config)
    research = {k: calls[k] for k in ("costs", "routes", "season") if k in calls}
    snapshot, pause = run(graph, Command(resume="make day 2 a rest day"), config)
    values = snapshot.values
    assert pause["kind"] == "review" and pause["revision"] == 1
    assert {k: calls[k] for k in research} == research  # no new research
    assert calls["itinerary"] == 2 and values["revisions"] == ["make day 2 a rest day"]
    planner = [m for m in prompts if m[0].content.startswith("You plan a realistic")][-1][-1].content
    assert "comment on the previous version: make day 2 a rest day" in planner and "- Day 1 · Paro" in planner
    assert values["report_dir"] != first.values["report_dir"] and values["report_dir"].endswith("_v2")
    html = open(values["report_dir"] + "/report.html", encoding="utf-8").read()
    assert "Version 2, updated after your comments: make day 2 a rest day." in html

    snapshot, pause = run(graph, Command(resume="looks good, thanks"), config)
    assert pause is None and snapshot.next == ()


def test_a_comment_that_changes_the_trip_reruns_its_research(fits_budget):
    calls, _, _ = fits_budget
    config = run_config("comment-trip")
    graph, first = finished_plan(config)
    assert len(first.values["itinerary"].days) == 5  # "4-5 days"
    snapshot, pause = run(graph, Command(resume="make it 3 days"), config)
    values = snapshot.values
    assert pause["kind"] == "review"
    assert len(values["itinerary"].days) == 3 and values["request"].duration_text == "3 days"
    assert values["feasibility"].total < first.values["feasibility"].total  # fewer nights, re-costed
    assert any(e["action"] == "changing the trip" for e in values["events"])


def test_comments_stop_after_the_last_allowed_round():
    from travelex.agents.review import MAX_REVISIONS, review_plan
    state = {"revisions": ["a"] * MAX_REVISIONS}
    assert review_plan(state, {"configurable": {}}) == {"plan_comment": None}
    assert review_plan({"revisions": []}, {"configurable": {"auto_approve": True}}) == {"plan_comment": None}


@pytest.mark.parametrize("comment, done", [("", True), ("looks good, thanks", True), ("No changes needed.", True),
                                           ("perfect", True), ("make day 2 easier", False),
                                           ("no hiking on day 3", False)])
def test_finished_or_a_real_comment(comment, done):
    from travelex.agents.review import finished
    assert finished(comment) is done


def test_changes_that_change_nothing_are_dropped():
    from travelex.agents.review import finished, real_changes
    norway = trip(person("A"), stops=[Stop(place="Norway", country="NO")], nights=None, month="2026-12",
                  duration_text="7 days", duration_range=[7, 7])
    moved = real_changes(TripChanges(stops=[Stop(place="Bergen", country="NO", nights=2),
                                            Stop(place="Oslo", country="NO", nights=4)],
                                     duration_text="6 nights", month="2026-12"), norway, towns=["Oslo", "Bergen"])
    assert moved == TripChanges()  # same country, same length, same month: a change to the days
    iceland = real_changes(TripChanges(stops=[Stop(place="Iceland", country="IS")], month="2026-12"), norway)
    assert [s.country for s in iceland.stops] == ["IS"] and iceland.month is None
    assert not finished("change the plan") and finished("nothing else to change, thanks")


# ---------- the gap-filling agent ----------

def test_gaps_are_the_prices_no_source_states():
    from travelex.agents.gaps import find_gaps
    from travelex.schemas import CostEvidence, GroupResult, RouteOption, StopCosts
    request = trip(person("A"), person("B", origin="Dubai"), stops=[Stop(place="Bali", country="ID")], nights=4,
                   lodging_style="mid-range")
    costs = StopCosts(place="Bali", currency="USD", lodging_per_room_night=43, rooms=1, room_config="1 double",
                      food_per_person_day=20, local_transport_per_person_day=10,
                      evidence=[CostEvidence(item="mid-range hotel", source_id="C1.1", quote="$43 a night", amount=43,
                                             currency="USD", unit="per room-night")])
    one_source = RouteOption(mode="flight", route="MAA-DPS", adult_fare=Money(amount=300, currency="USD"),
                             source_ids=["G1.1"])
    gaps = find_gaps({"request": request, "costs": {"Bali": costs},
                      "group_results": {"G1": GroupResult(group_id="G1", options=[one_source]),
                                        "G2": GroupResult(group_id="G2")}})
    assert [(g.id, g.kind, g.item) for g in gaps] == [
        ("gap1", "fare", "fare"), ("gap2", "route", "fare"), ("gap3", "cost", "food"), ("gap4", "cost", "local transport")]
    assert gaps[1].text == "No route found from Dubai to Bali."


def records_state(*calls):
    from datetime import datetime
    from travelex.schemas import Gap, StopCosts
    page_text = "In Ubud a warung meal is about IDR 50,000, so food runs $8 to $12 a day."
    gap = Gap(id="gap1", kind="cost", key="Bali", item="food", text="Bali: food per person per day")
    return {"request": trip(person("A"), stops=[Stop(place="Bali", country="ID")], nights=4),
            "costs": {"Bali": StopCosts(place="Bali", currency="IDR", lodging_per_room_night=600000, rooms=1,
                                        room_config="1 double", food_per_person_day=300000)},
            "sources": {"R.1": Source(id="R.1", url="https://unfoldbali.com/food", title="Food in Bali", kind="web",
                                      retrieved_at=datetime(2026, 10, 27), text=page_text)},
            "gaps": [gap], "group_results": {}, "gap_messages": [AIMessage("", tool_calls=list(calls))]}


FOOD = {"gap_id": "gap1", "amount": 10, "currency": "USD", "source_id": "R.1", "quote": "food runs $8 to $12 a day"}


def test_a_recorded_price_must_be_stated_by_its_source():
    from travelex.agents.gaps import gap_tools
    from travelex.tools import fx
    out = gap_tools(records_state({"name": "record_price", "id": "c1", "args": FOOD},
                                  {"name": "record_price", "id": "c2", "args": {**FOOD, "amount": 9}}))
    first, second = [m.content for m in out["gap_messages"]]
    assert first.startswith("Accepted: gap1") and second.startswith("Rejected: gap1 is not an open gap")
    food = out["costs"]["Bali"]
    assert food.food_per_person_day == round(fx.convert(10, "USD", "IDR"), 2)  # kept in the stop's currency
    assert [(x.item, x.source_id) for x in food.evidence] == [("food", "R.1")]
    assert out["gaps"][0].status == "filled" and "unfoldbali.com" in out["gaps"][0].note


@pytest.mark.parametrize("args, why", [
    ({"quote": "meals are cheap"}, "the quote is not in R.1"),
    ({"amount": 30}, "30 is not stated in the quote"),
    ({"source_id": "R.9"}, "there is no source R.9"),
    ({"quote": "IDR 50,000", "amount": 50000}, "the quote is not in USD"),
    ({"quote": "IDR 50,000", "amount": 50000, "currency": "INR"}, "the quote is not in INR")])
def test_records_that_fail_the_checks_are_rejected(args, why):
    from travelex.agents.gaps import gap_tools
    out = gap_tools(records_state({"name": "record_price", "id": "c1", "args": {**FOOD, **args}}))
    assert why in out["gap_messages"][0].content
    assert out["gaps"][0].status == "open" and out["costs"] == {}


def test_a_gap_the_agent_gives_up_on_is_left_for_the_traveller():
    from travelex.agents.gaps import gap_tools
    out = gap_tools(records_state({"name": "give_up", "id": "g", "args": {
        "gap_id": "gap1", "reason": "no page states daily food costs for Ubud"}}))
    assert (out["gaps"][0].status, out["gaps"][0].note) == ("gave_up", "no page states daily food costs for Ubud")


def test_the_agent_searches_records_and_the_budget_cites_it(fits_budget, fake_search):
    calls, _, gap_model = fits_budget
    fake_search.append(page("https://paro.example/food", "Meals in Paro cost about $12 a day for one person.",
                            title="Bhutan on a budget"))

    def gap_filler(messages):
        food = re.search(r"- (gap\d+): Bhutan: food", messages[1].content).group(1)
        results = [m for m in messages if isinstance(m, ToolMessage)]
        if not results:
            return tool_call("web_search", {"query": "Paro meal prices per day"})
        if len(results) == 1:
            sid = re.search(r"\[(R\.\d+)\][^\[]*Meals in Paro", results[0].content).group(1)
            return tool_call("record_price", {"gap_id": food, "amount": 12, "currency": "USD", "source_id": sid,
                                              "quote": "Meals in Paro cost about $12 a day"})
        return AIMessage("Filled the food price.")
    gap_model.responder = gap_filler
    graph = build_graph(InMemorySaver(serde=serializer()))
    snapshot, _ = run(graph, initial_state("3 US adults Dubai to Bhutan", TODAY), run_config("gaps", auto_approve=True))
    values = snapshot.values
    assert values["costs"]["Bhutan"].food_per_person_day == 12
    food = next(l for l in values["feasibility"].lines if l.category == "food")
    assert food.source_ids and food.quote == "Meals in Paro cost about $12 a day"
    assert [g.status for g in values["gaps"] if g.item == "food"] == ["filled"]
    assert [e["action"] for e in values["events"] if e["agent"] == "fill_gaps"][-1] == "finished"



@pytest.mark.parametrize("quote, amount, ok", [
    ('[R.1] Food in Ubud: "food runs $8 to $12 a day"', 10, True),  # label and lead-in stripped, words exact
    ("food runs $8 to $12 a day", 12, True), ("food runs $8 to $12 a day", 13, False),
    ("IDR 50,000, so food runs $8 to $12 a day", 50000, True),
    ("Sun 12/6 8:12 am-1:10 pm 2 stops $699", 400, False)])  # between the numbers is not stated
def test_quotes_are_trimmed_to_the_sources_words_and_must_state_the_amount(quote, amount, ok):
    from travelex.agents.gaps import stated, verbatim
    text = "In Ubud a warung meal is about IDR 50,000, so food runs $8 to $12 a day. Sun 12/6 8:12 am-1:10 pm 2 stops $699"
    words = verbatim(quote, text)
    assert bool(words) and stated(amount, words) == ok


def test_gaps_still_open_at_the_limit_go_to_the_traveller(use_fake_llm):
    from travelex.agents.gaps import MAX_ROUNDS, NOT_FOUND_IN_TIME, gap_agent
    use_fake_llm(gaps_module, AIMessage("Searched enough."))
    from travelex.schemas import Gap
    from langchain_core.messages import HumanMessage
    busy = [HumanMessage("gaps")] + [tool_call("web_search", {"query": "q"}) for _ in range(MAX_ROUNDS)]
    open_gap = Gap(id="gap1", kind="cost", key="Bali", item="food", text="Bali: food")
    out = gap_agent({"gap_messages": busy, "gaps": [open_gap]})
    assert out["gaps"][0].status == "gave_up" and out["gaps"][0].note == NOT_FOUND_IN_TIME
    assert out["events"][0]["detail"] == "0 of 1 gaps filled; left for the traveller: gap1"


def test_a_failing_model_ends_gap_filling_not_the_trip():
    from travelex.agents.gaps import gap_agent
    from travelex.schemas import Gap
    from langchain_core.messages import HumanMessage
    out = gap_agent({"gap_messages": [HumanMessage("gaps")], "gaps": [Gap(id="gap1", kind="cost", key="Bali",
                                                                            item="food", text="Bali: food")]})
    assert out["events"][0]["action"] == "stopped" and out["gaps"][0].status == "gave_up" and out["errors"]



def test_a_sourced_price_far_from_the_estimate_is_another_item():
    from travelex.agents.gaps import gap_tools
    state = records_state({"name": "record_price", "id": "c1", "args": FOOD})
    state["gaps"][0] = state["gaps"][0].model_copy(update={"estimate_usd": 60})  # $10 a day is under a quarter
    out = gap_tools(state)
    assert "far from the estimate of 60.00 USD" in out["gap_messages"][0].content and out["costs"] == {}



# ---------- handing off to the traveller ----------

def stuck_state(**extra):
    from travelex.schemas import Gap, GroupResult, VisaCheck
    request = trip(person("A", passport="IN", origin="Chennai"), stops=[Stop(place="Bali", country="ID")], nights=4)
    unconfirmed = VisaCheck(traveller="A", passport="IN", country="ID", rule=None, summary="Not known: check yourself")
    return {"request": request, "group_results": {"G1": GroupResult(group_id="G1", visas=[unconfirmed])},
            "gaps": [Gap(id="gap1", kind="route", key="G1", item="fare", text="No route found from Chennai to Bali.",
                         status="gave_up", note="No source found within the search limit."),
                     Gap(id="gap2", kind="cost", key="Bali", item="food", text="Bali: food", status="gave_up")],
            "handoff_round": 0, "traveller_notes": [], **extra}


def test_what_can_stop_the_trip_goes_to_the_traveller():
    from travelex.agents.handoff import open_issues
    issues = open_issues(stuck_state())
    assert issues == ["No route found from Chennai to Bali. (No source found within the search limit.)",
                      "A (IN) entering ID: Not known: check yourself"]  # an estimated food price doesn't stop anyone


def test_a_hint_sends_the_agent_back_and_continue_goes_on(monkeypatch):
    import travelex.agents.handoff as handoff
    from travelex.agents.handoff import ask_traveller, route_after_handoff
    monkeypatch.setattr(handoff, "interrupt", lambda payload: "try flights via Singapore")
    out = ask_traveller(stuck_state(), {"configurable": {}})
    assert out["traveller_notes"] == ["try flights via Singapore"]
    assert [g.status for g in out["gaps"]] == ["open", "gave_up"]  # the route reopens, the food estimate stays
    assert "The traveller says: try flights via Singapore" in out["gap_messages"][0].content
    assert route_after_handoff({**stuck_state(), **out}) == "gap_agent"

    monkeypatch.setattr(handoff, "interrupt", lambda payload: "continue")
    out = ask_traveller(stuck_state(), {"configurable": {}})
    assert out["handoff_round"] == 1 and "gaps" not in out
    assert route_after_handoff({**stuck_state(), **out}) == "feasibility"
    assert ask_traveller(stuck_state(handoff_round=1), {"configurable": {}}) == {}  # asked once per trip
    assert ask_traveller(stuck_state(), {"configurable": {"auto_approve": True}}) == {}


def test_the_plan_lists_what_is_still_open_and_what_the_traveller_said():
    from travelex.report import to_double_check
    checks = to_double_check(stuck_state(traveller_notes=["I already hold a valid US visa"]))
    assert checks[0].startswith("Not settled by our research: No route found from Chennai to Bali.")
    assert "You told us: I already hold a valid US visa" in checks


def test_the_graph_pauses_for_the_traveller_and_the_hint_reaches_the_agent(fits_budget):
    _, _, gap_model = fits_budget
    gap_model.responder = lambda messages: AIMessage("Done.")  # finds nothing: what's open goes to the traveller
    graph = build_graph(InMemorySaver(serde=serializer()))
    config = run_config("handoff")
    _, pause = run(graph, initial_state("3 US adults Dubai to Bhutan", TODAY), config)
    _, pause = run(graph, Command(resume=pause["request"]), config)
    assert pause["kind"] == "handoff" and pause["issues"]
    snapshot, pause = run(graph, Command(resume="Bhutan visas for US citizens are arranged by the tour operator"), config)
    assert pause["kind"] == "review"  # asked once: the plan went on
    assert snapshot.values["traveller_notes"] == ["Bhutan visas for US citizens are arranged by the tour operator"]
    html = open(snapshot.values["report_dir"] + "/report.html", encoding="utf-8").read()
    assert "You told us: Bhutan visas for US citizens are arranged by the tour operator" in html


def test_alternatives_research_shares_the_runs_limits_and_concurrency(fake_everything, monkeypatch):
    """Second review: an alternative's research ran with no limits and no concurrency setting."""
    import travelex.graph as graph_module
    import travelex.tools.search as search_module
    from langgraph.config import get_config
    from travelex.limits import current, research_stopped
    monkeypatch.setattr(graph_module, "MAX_CONCURRENCY", 1)
    seen, inner = [], search_module.search

    def recorder(query, *args, **kwargs):  # what each search inside the run can see of its limits
        seen.append((query, current(), get_config().get("max_concurrency"), research_stopped()))
        return inner(query, *args, **kwargs)
    monkeypatch.setattr(search_module, "search", recorder)
    graph = build_graph(InMemorySaver(serde=serializer()))
    config = run_config("alts-limits")
    _, pause = run(graph, initial_state("3 US adults Dubai to Bhutan 4-5 days $800", TODAY), config)
    limits = config["configurable"]["limits"]
    limits.max_searches = 0  # the trip has used its allowance before the alternatives are costed
    _, pause = run(graph, Command(resume=pause["request"]), config)
    assert pause["kind"] == "alternatives"
    nepal = [s for s in seen if "Nepal" in s[0]]  # the "Nepal instead" alternative's research
    assert nepal and all(lim is limits and conc == 1 and stopped for _, lim, conc, stopped in nepal)
