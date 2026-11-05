"""QA: code checks on the finished plan. Problems the planner or food agent can fix send the run back (max 2);
everything else is reported as a note."""
from langgraph.types import Send

from travelex.agents.common import event
from travelex.agents.food import allergen_terms, allergies, breaks_diet, conflicts, forbidden_terms
from travelex.agents.itinerary import cities, food_sends, guide_for
from travelex.schemas import QAReport
from travelex.state import TripState
from travelex.trip import has_vulnerable, resolve_nights

MAX_QA_ROUNDS = 2
PLAN_ISSUES = ("day_count", "pacing", "rest_day", "city_missing")
FOOD_ISSUES = ("food_missing", "diet", "allergen")
ALLERGY_WORDS = ("allerg", "nut", "peanut", "gluten", "shellfish", "dairy", "lactose")


def check(state: TripState) -> QAReport:
    request, plan, food = state["request"], state.get("itinerary"), state.get("food") or {}
    issues, notes = [], []
    nights, _ = resolve_nights(request)
    if plan is None:
        return QAReport(passed=False, issues=["day_count"])
    if nights and len(plan.days) != nights + 1:
        issues.append("day_count")
    day_cities = " | ".join(d.city.casefold() for d in plan.days)
    issues += [f"city_missing:{c}" for s in request.stops for c in s.cities if c.casefold() not in day_cities]
    if has_vulnerable(request):
        issues += [f"pacing:day {d.day}" for d in plan.days if sum(a.strenuous for a in d.activities) > 1]
        if nights and nights >= 5 and not any(d.rest_day for d in plan.days):
            issues.append("rest_day")

    terms, allergens = forbidden_terms(request.diet), allergen_terms(allergies(request.diet))
    allergy = any(any(w in d.lower() for w in ALLERGY_WORDS) for d in request.diet)
    for city in cities(plan):
        guide = guide_for(city, food)
        if guide is None:
            issues.append(f"food_missing:{city}")
            continue
        if any(breaks_diet(dish, terms) for dish in guide.dishes):
            issues.append(f"diet:{city}")
        if any(conflicts(dish, allergens) for dish in guide.dishes):  # separate from the advice note below
            issues.append(f"allergen:{city}")
        if allergy and not guide.diet_notes:
            notes.append(f"No allergy advice for {city}.")
        if not guide.places:
            notes.append(f"No sourced places to eat in {city}.")

    for gid, result in (state.get("group_results") or {}).items():
        if not result.options:
            notes.append(f"No transport found for group {gid}.")
    notes += [f"Visa for {v.traveller} → {v.country} is unknown." for r in (state.get("group_results") or {}).values()
              for v in r.visas if v.rule is None]
    if (f := state.get("feasibility")) and not f.ok:
        notes.append("The plan does not fit the budget: " + " ".join(f.reasons))
    return QAReport(passed=not issues, issues=issues, notes=notes)


def qa(state: TripState) -> dict:
    report = check(state)
    detail = "passed" if report.passed else "; ".join(report.issues)
    return {"qa_report": report, "qa_round": state.get("qa_round", 0) + 1,
            "events": [event("qa", "passed" if report.passed else "found problems", detail)]}


def route_after_qa(state: TripState) -> str | list[Send]:
    report, rounds = state["qa_report"], state.get("qa_round", 0)
    if report.passed or rounds > MAX_QA_ROUNDS:
        return "save_report"
    if any(i.startswith(PLAN_ISSUES) for i in report.issues):
        return "itinerary_planner"
    redo = {i.split(":", 1)[1] for i in report.issues if i.startswith(FOOD_ISSUES)}
    return food_sends(state, only=redo) or "save_report"
