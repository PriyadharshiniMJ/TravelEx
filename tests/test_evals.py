"""The eval harness: the target runs a golden-style case end to end (fake model and search), and every
evaluator scores it without crashing. Live runs: python evals/run_evals.py."""
import json
import sys
from pathlib import Path

import pytest

EVALS = Path(__file__).resolve().parents[1] / "evals"
sys.path.insert(0, str(EVALS))

from evaluators import CODE_EVALUATORS, route_matches, score  # noqa: E402
from target import run_case  # noqa: E402
from test_plan import fake_everything  # noqa: E402,F401  (fixture)

CASES = {json.loads(line)["metadata"]["id"]: json.loads(line)
         for line in (EVALS / "golden" / "travelex_golden_v1.jsonl").read_text(encoding="utf-8").splitlines()}


def by_key(results):
    return {r["key"]: r for r in results}


def test_route_matching():
    assert route_matches("DXB-DEL-PBH", "DXB-DEL-PBH")
    assert route_matches("DXB-DEL-PBH", "dxb-del-ixb-pbh")
    assert not route_matches("DXB-DEL-PBH", "DXB-CCU-PBH")
    assert score("x", [])["score"] is None
    assert score("x", [("a", True), ("b", False)]) == {"key": "x", "score": 0.5, "comment": "failed: b"}


def test_c2_case_runs_and_every_evaluator_scores(fake_everything):
    case = CASES["C2"]  # 3 US adults, Bhutan, 4-5 days, $800: the fake intake matches it closely
    outputs = run_case(case["inputs"])
    results = by_key(e(case["inputs"], outputs, case["outputs"]) for e in CODE_EVALUATORS)

    assert results["itinerary"]["score"] is None  # C2 has no itinerary expectations
    assert results["route"]["score"] == 1.0  # over budget, so alternatives ran
    assert results["fees"]["score"] == 1.0, results["fees"]["comment"]  # SDF $1,200 on the trip as asked
    feasibility = results["feasibility"]
    assert "ok = False" in feasibility["comment"] or feasibility["score"] == 1.0
    assert outputs["first"]["request"]["stops"][0]["place"] == "Bhutan"
    assert outputs["request"]["stops"][0]["place"] == "Nepal"  # auto-picked the alternative that fits


def test_e5_faults_are_injected_and_caught(fake_everything):
    case = CASES["E5"]
    outputs = run_case(case["inputs"])
    qa = by_key(e(case["inputs"], outputs, case["outputs"]) for e in CODE_EVALUATORS)["qa"]
    assert [e["action"] for e in outputs["qa_events"]][:1] == ["found problems"]
    assert "day_count" in outputs["qa_events"][0]["detail"]
    assert qa["score"] is not None and qa["score"] >= 0.6, qa["comment"]


# ---------- evaluators never pass a bad or missing output (second review) ----------

from evaluators import completion, critical, food_checks, transport_checks  # noqa: E402

GROUP = {"request": {"groups": [{"id": "G1", "origin": "Dubai", "travellers": ["A"]}], "travellers": []}}


def guide(*dishes, places=(), notes=""):
    return {"dishes": [{"name": n, "description": d} for n, d in dishes], "places": list(places), "diet_notes": notes}


def test_a_peanut_dish_fails_a_peanut_allergy_case_and_is_critical():
    out = {**GROUP, "completed": True, "food": {"Bangkok": guide(("Peanut noodles", "noodles in peanut sauce"))}}
    exp = {"food": {"diet": ["peanut allergy"]}}
    result = food_checks({}, out, exp)
    assert result["score"] < 1 and "Peanut noodles (peanut)" in result["comment"]
    assert critical({}, out, exp)["score"] == 0


def test_one_ordinary_cafe_is_not_three_halal_places():
    cafe = {"name": "Cafe Onion", "area": "Seongsu", "why": "great coffee", "source_ids": ["F1.1"]}
    out = {**GROUP, "food": {"Seoul": guide(("Bibimbap", "rice"), places=[cafe])}, "source_ids": ["F1.1"]}
    result = food_checks({}, out, {"food": {"halal_places_min_per_city": 3}})
    assert result["score"] < 1 and "≥3 halal place(s) per city" in result["comment"]


def test_an_invented_source_id_fails_sources_present():
    place = {"name": "Ichiran", "area": "Shibuya", "why": "ramen", "source_ids": ["F1.9"]}  # never fetched
    out = {**GROUP, "completed": True, "food": {"Tokyo": guide(("Ramen", "soup"), places=[place])}, "source_ids": ["F1.1"]}
    exp = {"food": {"sources_present": True}}
    assert "places cite real sources" in food_checks({}, out, exp)["comment"]
    assert critical({}, out, exp)["score"] == 0


def test_no_routes_fails_the_layover_rule_and_a_report_mention_is_not_a_route():
    out = {**GROUP, "group_results": {}, "costs": {}, "report_md": "Mexico City, then Cancún"}
    result = transport_checks({}, out, {"transport": {"rules": {"max_layover_hours": 4}, "intercity_any_route": ["MEX-CUN"]}})
    assert result["score"] == 0, result["comment"]


def test_a_run_stopped_at_the_pause_limit_is_unfinished(fake_everything, monkeypatch):
    import target
    monkeypatch.setattr(target, "MAX_PAUSES", 1)  # C2 pauses at the trip card, then again at the alternatives
    outputs = run_case(CASES["C2"]["inputs"])
    assert (outputs["completed"], outputs["termination_reason"]) == (False, "pause_limit")
    assert outputs["pending_interrupt"] == "alternatives"
    assert completion({}, outputs, {})["score"] == 0 and critical({}, outputs, {})["score"] == 0


def test_a_finished_run_says_so(fake_everything):
    outputs = run_case(CASES["C2"]["inputs"])
    assert (outputs["completed"], outputs["termination_reason"]) == (True, "finished")


# ---------- the judges never run on the agent's own model ----------

def test_the_judges_use_their_own_model(monkeypatch):
    import evaluators
    from fakes import FakeLLM, tool_call
    roles = []

    def get_llm(role, **kwargs):
        roles.append(role)
        return FakeLLM(responses=[tool_call("Grade", {"score": 4, "reason": "sensible pacing"})])
    monkeypatch.setattr(evaluators, "get_llm", get_llm)
    result = evaluators.JUDGES[0]({"query": "Bhutan, 5 days"}, {"itinerary": {"days": [{"day": 1}]}}, {})
    assert roles == ["judge"] and result["score"] == 0.8


def test_a_judge_on_the_agents_model_is_refused(monkeypatch):
    from evaluators import judge_model_problem
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4.1-mini")
    monkeypatch.delenv("LLM_MODEL_JUDGE", raising=False)
    assert "LLM_MODEL_JUDGE is not set" in judge_model_problem()
    monkeypatch.setenv("LLM_MODEL_JUDGE", "openai:gpt-4.1-mini")
    assert "the agent's own model" in judge_model_problem()
    monkeypatch.setenv("LLM_MODEL_JUDGE", "openai:gpt-5.2")
    assert judge_model_problem() is None


# ---------- success: the task-completion metric ----------

from evaluators import success  # noqa: E402

DONE = {**GROUP, "completed": True, "termination_reason": "finished", "qa_report": {"passed": True, "issues": []},
        "feasibility": {"ok": False, "gap": 300, "unpriced": []}}  # over budget is still a success


def test_a_finished_checked_fully_priced_plan_is_done_well():
    assert success({}, DONE, {}) == {"key": "success", "score": 1, "comment": "done well"}


def test_each_way_a_plan_falls_short_is_named():
    cases = {"not finished (pause_limit)": {"completed": False, "termination_reason": "pause_limit"},
             "QA did not pass (day_count)": {"qa_report": {"passed": False, "issues": ["day_count"]}},
             "not fully priced (France to Japan: no price found for this journey.)":
                 {"feasibility": {"unpriced": ["France to Japan: no price found for this journey."]}},
             "not costed": {"feasibility": None}}
    for why, change in cases.items():
        result = success({}, {**DONE, **change}, {})
        assert result["score"] == 0 and why in result["comment"], (why, result)
    peanut = {**DONE, "food": {"Bangkok": guide(("Peanut noodles", "peanut sauce"))}}
    assert "critical: food: no dish with an allergen" in success({}, peanut, {"food": {"diet": ["peanut allergy"]}})["comment"]


def test_a_case_that_expects_no_plan_succeeds_by_finishing_cleanly():
    declined = {**GROUP, "completed": True, "termination_reason": "finished"}
    assert success({}, declined, {"expect_plan": False})["score"] == 1


def test_success_scores_a_real_run(fake_everything):
    outputs = run_case(CASES["C2"]["inputs"])
    result = success({}, outputs, CASES["C2"]["outputs"])
    assert result["score"] == 1, result["comment"]  # over budget, switched to Nepal, finished, checked, priced


# ---------- found by the live smoke test: crashes and a stale LangSmith copy are reported ----------

def test_a_crashed_run_says_why(monkeypatch):
    import target

    def expired(inputs):
        raise RuntimeError("Error code: 401 - Your API key has expired.")
    monkeypatch.setattr(target, "run_case", expired)
    outputs = target.run_case_safely({"query": "Bhutan"})
    assert outputs["completed"] is False and outputs["termination_reason"].startswith("error: RuntimeError: Error code: 401")
    assert "error: RuntimeError" in completion({}, outputs, {})["comment"]


def test_a_country_with_cities_is_met_by_a_stop_per_city():
    """A2 expects "Japan: Tokyo and Kyoto"; intake now files that as a Tokyo stop and a Kyoto stop."""
    from evaluators import request_checks
    exp = {"request": {"stops": [{"place": "Japan", "country": "JP", "cities": ["Tokyo", "Kyoto"]}]}}
    by_city = {"request": {"stops": [{"place": "Tokyo", "country": "JP", "cities": []},
                                     {"place": "Kyoto", "country": "JP", "cities": []}]}}
    assert request_checks({}, by_city, exp)["score"] == 1
    only_tokyo = {"request": {"stops": by_city["request"]["stops"][:1]}}
    assert request_checks({}, only_tokyo, exp)["score"] == 0


def test_a_run_that_ends_without_a_destination_says_so():
    """E3 reached the end with no stops and was reported as "finished"."""
    from types import SimpleNamespace
    from builders import person, trip
    import target
    snapshot = SimpleNamespace(interrupts=[], next=(), values={"request": trip(person("A"), stops=[])})
    run = target.status(snapshot)
    assert run == {"completed": True, "pending_interrupt": None, "termination_reason": "ended without a destination"}
    assert completion({}, run, {})["comment"] == "ended without a destination"
    assert "no plan (ended without a destination)" in success({}, run, {})["comment"]
    assert success({}, run, {"expect_plan": False})["score"] == 1  # a request TravelEx must decline


def test_a_stale_langsmith_copy_is_caught():
    from types import SimpleNamespace
    from upload_dataset import local_cases, stale_ids
    local = local_cases()
    copy = [SimpleNamespace(metadata=c["metadata"], inputs=c["inputs"], outputs=c["outputs"]) for c in local.values()]
    assert stale_ids(copy) == []
    old_e2 = json.loads(json.dumps(local["E2"]["outputs"]))
    old_e2["food"]["cities"].append("Toronto")  # what LangSmith still had
    copy = [SimpleNamespace(metadata=c["metadata"], inputs=c["inputs"], outputs=old_e2 if i == "E2" else c["outputs"])
            for i, c in local.items()]
    assert stale_ids(copy) == ["E2"]
