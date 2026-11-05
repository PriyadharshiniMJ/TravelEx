from datetime import date

import pytest
from langgraph.checkpoint.memory import InMemorySaver

import travelex.graph as graph_module
from langgraph.types import Command

import travelex.agents.intake as intake_module
from builders import person
from fakes import tool_call
from travelex.agents.intake import tidy
from travelex.graph import initial_state, run_config, serializer
from travelex.schemas import Stop, TripRequest

TODAY = date(2026, 10, 27)
BHUTAN = {"place": "Bhutan", "country": "BT"}
QUERY = ("I want to plan a trip to Bhutan for two adults, one travelling from Dubai and another from Chennai. "
         "Budget is $1500. Give an itinerary for 4-5 days.")


def trip_args(passport=None, **extra) -> dict:
    """What the LLM returns for the A1 golden query, before and after the user names their passports."""
    return {"stops": [BHUTAN], "duration_range": [4, 5], "budget": {"amount": 1500, "currency": "USD"},
            "travellers": [{"label": "Adult 1", "type": "adult", "passport": passport, "origin": "Dubai"},
                           {"label": "Adult 2", "type": "adult", "passport": passport, "origin": "Chennai"}],
            **extra}


# ---------- tidy: code-side checks ----------

def test_tidy_lists_what_is_missing_and_writes_a_question():
    request = tidy(TripRequest(**trip_args()), TODAY)
    assert request.missing == ["passport_country", "travel_month"]
    assert "passport" in request.clarification_needed
    assert [g.origin for g in request.groups] == ["Dubai", "Chennai"]


def test_tidy_complete_request_has_no_question():
    request = tidy(TripRequest(**trip_args("IN", month="2027-10", clarification_needed="Anything else?")), TODAY)
    assert request.missing == [] and request.clarification_needed is None


def test_tidy_flags_past_dates():
    request = tidy(TripRequest(**trip_args("BR", month="2026-03")), TODAY)
    assert request.flags == ["date_in_past"]
    assert request.missing == ["travel_dates"]
    assert "past" in request.clarification_needed


def test_tidy_drops_dates_the_user_never_gave():
    invented = tidy(TripRequest(**trip_args("NG", start_date="2027-04-01", end_date="2027-04-06")), TODAY)
    assert (invented.start_date, invented.month) == (None, "2027-04")
    typed = tidy(TripRequest(**trip_args("CA", start_date="2027-04-10", end_date="2027-04-20", exact_dates=True)), TODAY)
    assert (typed.start_date.day, typed.end_date.day) == (10, 20)


def test_tidy_fixes_travellers():
    travellers = [person("Adult", age=72), person("Adult"), person("Baby", type="child", age=1),
                  person("Kid", type="child", origin=None)]
    request = tidy(TripRequest(stops=[Stop(**BHUTAN)], travellers=travellers, month="2027-10", nights=5), TODAY)
    assert [t.label for t in request.travellers] == ["Adult", "Adult 2", "Baby", "Kid"]
    assert request.travellers[0].needs == ["senior"]
    assert request.travellers[2].type == "infant"
    assert request.missing == ["origin", "child_ages"]


def test_tidy_no_destination():
    request = tidy(TripRequest(travellers=[person("A", passport="GB", origin="London")], month="2026-12",
                               duration_range=(7, 7), destination_ideas=["Sri Lanka", "Oman"]), TODAY)
    assert request.missing == ["destination"]
    assert request.destination == "destination to be chosen"


# ---------- the graph: ask, answer, confirm, season ----------

@pytest.fixture(autouse=True)
def stop_after_season(monkeypatch):
    """These tests are about intake and clarify: skip the research branches (tested in test_research.py)
    and stop before the itinerary (test_plan.py)."""
    monkeypatch.setattr(graph_module, "route_research", lambda state, **options: [graph_module.season_send(state)])


def build_graph(checkpointer):
    return graph_module.build_graph(checkpointer, interrupt_before=["itinerary_planner", "alternatives_agent"])


def run_until_pause(graph, payload, config):
    for _ in graph.stream(payload, config):
        pass
    snapshot = graph.get_state(config)
    return snapshot, (snapshot.interrupts[0].value if snapshot.interrupts else None)


def test_golden_a1_asks_for_passports_and_month_then_checks_the_season(use_fake_llm):
    llm = use_fake_llm(intake_module, tool_call("TripRequest", trip_args()),
                       tool_call("TripRequest", trip_args("IN", month="2027-10")))
    graph = build_graph(InMemorySaver(serde=serializer()))
    config = run_config("t1")

    snapshot, pause = run_until_pause(graph, initial_state(QUERY, TODAY), config)
    assert pause["missing"] == ["passport_country", "travel_month"]
    assert "duration" not in pause["missing"]  # "4-5 days" is a range, not a gap

    snapshot, pause = run_until_pause(graph, Command(resume="Both have Indian passports. October 2027."), config)
    assert "Indian passports" in llm.calls[1][-1].content  # the answer reaches intake with the old request
    assert pause["question"] is None and pause["missing"] == []  # just the confirmation card

    snapshot, pause = run_until_pause(graph, Command(resume=pause["request"]), config)
    assert pause is None
    assert snapshot.values["season"]["Bhutan"].verdict == "good"
    assert len(llm.calls) == 2  # the season came from the seed profile, no LLM call


def test_auto_mode_never_pauses(use_fake_llm):
    use_fake_llm(intake_module, tool_call("TripRequest", trip_args()))
    graph = build_graph(InMemorySaver(serde=serializer()))
    snapshot, pause = run_until_pause(graph, initial_state(QUERY, TODAY), run_config("t2", auto_approve=True))
    assert pause is None
    assert snapshot.values["season"]["Bhutan"].verdict == "unknown"  # no month yet
    assert any("proceeding without" in e["detail"] for e in snapshot.values["events"])


NO_PLACE = {"travellers": [{"label": "Friend", "type": "adult", "passport": "GB", "origin": "London"}],
            "month": "2026-12", "duration_range": [7, 7], "destination_ideas": ["Sri Lanka", "Oman"],
            "clarification_needed": "Which dates in December?"}


def test_no_destination_asks_again_instead_of_ending(use_fake_llm):
    from travelex.agents.intake import MAX_CLARIFY_ROUNDS
    use_fake_llm(intake_module, tool_call("TripRequest", NO_PLACE))
    graph = build_graph(InMemorySaver(serde=serializer()))
    config = run_config("t3")
    _, pause = run_until_pause(graph, initial_state("Somewhere warm for a week in December", TODAY), config)
    assert pause["ideas"] == ["Sri Lanka", "Oman"]
    assert pause["question"] == "Where would you like to go? A few ideas that fit: Sri Lanka; Oman."  # not the dates
    for _ in range(MAX_CLARIFY_ROUNDS - 1):  # Continue with no place: the card comes back
        snapshot, pause = run_until_pause(graph, Command(resume=pause["request"]), config)
        assert pause["kind"] == "clarify"
    snapshot, pause = run_until_pause(graph, Command(resume=pause["request"]), config)
    assert pause is None and not snapshot.values.get("season")  # the question rounds ran out: nothing to plan


def test_auto_mode_picks_the_first_idea(use_fake_llm, fake_search):
    picked = {**NO_PLACE, "stops": [{"place": "Sri Lanka", "country": "LK"}], "destination_ideas": [],
              "clarification_needed": None}
    llm = use_fake_llm(intake_module, tool_call("TripRequest", NO_PLACE), tool_call("TripRequest", picked))
    graph = build_graph(InMemorySaver(serde=serializer()))
    snapshot, pause = run_until_pause(graph, initial_state("Somewhere warm for a week in December", TODAY),
                                      run_config("auto-idea", auto_approve=True))
    assert "The user answered: Let's go to Sri Lanka" in llm.calls[1][-1].content
    assert snapshot.values["request"].destination == "Sri Lanka"


def test_edited_card_is_accepted_without_another_llm_call(use_fake_llm):
    llm = use_fake_llm(intake_module, tool_call("TripRequest", trip_args("IN", month="2027-07")))
    graph = build_graph(InMemorySaver(serde=serializer()))
    config = run_config("t4")
    _, pause = run_until_pause(graph, initial_state(QUERY, TODAY), config)
    edited = {**pause["request"], "month": "2027-10"}
    snapshot, _ = run_until_pause(graph, Command(resume=edited), config)
    assert snapshot.values["request"].month == "2027-10"
    assert snapshot.values["season"]["Bhutan"].verdict == "good"
    assert len(llm.calls) == 1


# ---------- answers never lose what was known (the Sri Lanka run) ----------

SRI_LANKA = {"stops": [{"place": "Sri Lanka", "country": "LK"}], "budget": {"amount": 200000, "currency": "INR"},
             "start_date": "2026-12-01", "end_date": "2026-12-07", "exact_dates": True}


def sri_lanka(passport, age=None, **extra) -> dict:
    return {**SRI_LANKA, **extra, "travellers": [
        {"label": "Adult 1", "type": "adult", "passport": passport, "origin": "Dubai"},
        {"label": "Adult 2", "type": "adult", "age": age, "passport": passport, "origin": "Chennai"},
        {"label": "Child (7)", "type": "child", "age": 7, "passport": passport, "origin": "Chennai"}]}


def test_keep_known_restores_what_a_reparse_dropped():
    from travelex.agents.intake import keep_known
    known = TripRequest(**sri_lanka("IN", age=35))
    forgetful = TripRequest(**{**sri_lanka(None), "start_date": None, "end_date": None, "exact_dates": False,
                               "budget": None})
    merged = keep_known(known, forgetful)
    assert [t.passport for t in merged.travellers] == ["IN", "IN", "IN"]
    assert merged.travellers[1].age == 35
    assert (merged.start_date.day, merged.end_date.day, merged.budget.amount) == (1, 7, 200000)


def test_keep_known_lets_the_user_change_things():
    from travelex.agents.intake import keep_known
    known = TripRequest(**sri_lanka("IN"))
    changed = TripRequest(**{**sri_lanka("GB"), "start_date": None, "end_date": None, "exact_dates": False,
                             "month": "2027-01"})
    merged = keep_known(known, changed)
    assert [t.passport for t in merged.travellers] == ["GB", "GB", "GB"]
    assert (merged.month, merged.start_date) == ("2027-01", None)  # new month replaces the old dates


@pytest.mark.parametrize("reply", ["yes", "Yes!", "ok", "looks good", "correct", "go ahead"])
def test_yes_accepts_the_card_without_another_parse(use_fake_llm, fake_search, reply):
    llm = use_fake_llm(intake_module, tool_call("TripRequest", sri_lanka("IN", age=35)),
                       responder=None)
    graph = build_graph(InMemorySaver(serde=serializer()))
    config = run_config(f"yes-{reply}")
    _, pause = run_until_pause(graph, initial_state("Sri Lanka, Dubai and Chennai, Indian passports", TODAY), config)
    assert pause["question"] is None  # nothing missing: the card only
    snapshot, pause = run_until_pause(graph, Command(resume=reply), config)
    assert pause is None and len(llm.calls) == 1
    assert [t.passport for t in snapshot.values["request"].travellers] == ["IN", "IN", "IN"]


def guessed_family() -> dict:
    """"My family flies from Chennai with our seven-year-old": the second parent is a guess."""
    args = sri_lanka("IN", age=35)
    args["travellers"][1]["assumed"] = True
    return args


def test_tidy_asks_about_guessed_travellers():
    from travelex.agents.intake import describe
    request = tidy(TripRequest(**guessed_family()), TODAY)
    assert request.missing == ["party_size"]
    assert request.clarification_needed.startswith("I guessed who's coming (Adult 2)")
    assert "Adult 2 35 (assumed)" in describe(request)


def test_confirming_the_card_clears_the_guess(use_fake_llm, fake_search):
    llm = use_fake_llm(intake_module, tool_call("TripRequest", guessed_family()))
    graph = build_graph(InMemorySaver(serde=serializer()))
    config = run_config("guessed")
    _, pause = run_until_pause(graph, initial_state("My family flies from Chennai with our 7-year-old", TODAY), config)
    assert pause["missing"] == ["party_size"] and "Adult 2" in pause["question"]
    snapshot, pause = run_until_pause(graph, Command(resume="yes"), config)
    assert pause is None and len(llm.calls) == 1
    request = snapshot.values["request"]
    assert not any(t.assumed for t in request.travellers) and request.missing == []


def test_the_sri_lanka_answers_keep_the_passports(use_fake_llm):
    """Replays the real run: passports given in an answer, then the model forgets them on a later re-parse."""
    use_fake_llm(intake_module, tool_call("TripRequest", {**sri_lanka(None), "start_date": None, "end_date": None,
                                                          "exact_dates": False, "month": "2026-12"}),
                 tool_call("TripRequest", sri_lanka("IN", age=35)),
                 tool_call("TripRequest", {**sri_lanka(None), "duration_text": "6 days"}))  # forgets on the 3rd parse
    graph = build_graph(InMemorySaver(serde=serializer()))
    config = run_config("sri-lanka")
    _, pause = run_until_pause(graph, initial_state("Sri Lanka from Dubai and Chennai", TODAY), config)
    assert "passport_country" in pause["missing"]
    _, pause = run_until_pause(graph, Command(resume="indian passport, end date 7th dec"), config)
    _, pause = run_until_pause(graph, Command(resume="also we like beaches"), config)
    assert [t["passport"] for t in pause["request"]["travellers"]] == ["IN", "IN", "IN"]


def test_a_traveller_starting_from_a_stop_is_sent_back_for_correction(use_fake_llm):
    """Live misreading: "2 adults from Chennai, Lisbon and Porto in Portugal" became travellers from Lisbon and Porto."""
    misread = {"stops": [{"place": "Lisbon", "country": "PT", "cities": ["Lisbon"]}], "month": "2027-05",
               "duration_text": "7 days", "travellers": [
                   {"label": "Adult 1", "type": "adult", "passport": "IN", "origin": "Chennai"},
                   {"label": "Adult 2", "type": "adult", "passport": "IN", "origin": "Lisbon"},
                   {"label": "Adult 3", "type": "adult", "passport": "IN", "origin": "Porto"}]}
    corrected = {**misread, "stops": [{"place": "Portugal", "country": "PT", "cities": ["Lisbon", "Porto"]}],
                 "travellers": misread["travellers"][:1] + [{**misread["travellers"][1], "origin": "Chennai"}]}
    llm = use_fake_llm(intake_module, tool_call("TripRequest", misread), tool_call("TripRequest", corrected))
    from travelex.agents.intake import intake
    out = intake({"raw_query": "2 adults with Indian passports from Chennai, Lisbon and Porto in Portugal",
                  "today": TODAY})
    assert len(llm.calls) == 2 and "Adult 2 starts from Lisbon" in llm.calls[1][-1].content
    request = out["request"]
    assert [t.origin for t in request.travellers] == ["Chennai", "Chennai"]
    assert request.stops[0].cities == ["Lisbon", "Porto"]


@pytest.mark.parametrize("query, style", [("High-end hotels in Sri Lanka", "luxury"),
                                          ("Sri Lanka, cheapest possible", "budget"), ("Sri Lanka", None)])
def test_hotel_words_set_the_level_once_at_intake(use_fake_llm, query, style):
    use_fake_llm(intake_module, tool_call("TripRequest", sri_lanka("IN")))
    request = intake_module.intake({"raw_query": query, "today": TODAY})["request"]
    assert request.lodging_style == style


def test_nights_on_a_stop_must_fit_the_trip():
    norway = {"stops": [{"place": "Norway", "country": "NO", "nights": 7}], "travellers": [
        {"label": "Adult 1", "type": "adult", "passport": "IN", "origin": "Dallas"}],
        "start_date": "2026-12-01", "end_date": "2026-12-07", "exact_dates": True, "duration_text": "7 days"}
    request = tidy(TripRequest(**norway), TODAY)
    assert request.stops[0].nights is None
    from travelex.trip import allocate_nights
    assert [n for _, n in allocate_nights(request)] == [6]
    split = {**norway, "stops": [{"place": "Oslo", "country": "NO", "nights": 2}, {"place": "Bergen", "country": "NO"}]}
    assert [s.nights for s in tidy(TripRequest(**split), TODAY).stops] == [2, None]  # the user's split is kept
