"""Evaluators for the golden dataset. Each reads one section of reference_outputs and scores the share of its
checks that pass; a section the case doesn't specify scores None (not applicable). A check passes only on evidence:
an output that is missing (no routes, no food guides, no sources) fails it, never passes it by default.
`critical` reports the failures that make a plan unsafe or untrue, apart from the mean; `completion` whether the
run finished. Three LLM judges grade quality where code can't."""
import os
import re

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from travelex.agents.food import allergen_terms, allergies, breaks_diet, conflicts, forbidden_terms
from travelex.llm import get_llm, structured
from travelex.schemas import Dish

Check = tuple[str, bool]


def low(text) -> str:
    return str(text or "").casefold()


def codes(route: str) -> list[str]:
    return [c.strip().casefold() for c in route.split("-")]


def route_matches(expected: str, actual: str) -> bool:
    """Expected stops appear in order in the actual route ("DXB-DEL-PBH" matches "DXB-DEL-PBH" and
    "DXB-DEL-IXB-PBH")."""
    it = iter(codes(actual))
    return all(code in it for code in codes(expected))


def score(key: str, checks: list[Check]) -> dict:
    if not checks:
        return {"key": key, "score": None, "comment": "not applicable"}
    failed = [name for name, ok in checks if not ok]
    return {"key": key, "score": round(1 - len(failed) / len(checks), 3),
            "comment": "all passed" if not failed else "failed: " + "; ".join(failed)}


def as_first_asked(out: dict) -> dict:
    """The outputs with the original trip's research in place of an alternative the run switched to."""
    return out | {k: v for k, v in (out.get("first") or {}).items() if v is not None}


def section(name, original: bool = False):
    """Decorator: run the check function only when the case specifies `name`, and score its checks.
    original=True: judge the trip as first asked, not an alternative the run may have switched to."""
    def wrap(fn):
        def evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
            expected = reference_outputs.get(name)
            out = as_first_asked(outputs) if original else outputs
            return score(name, fn(expected, out) if expected is not None else [])
        evaluator.__name__ = name
        evaluator.checks = lambda expected, outputs: fn(expected, as_first_asked(outputs) if original else outputs)
        return evaluator
    return wrap


# ---------- intake & clarify ----------

def traveller_matches(expected: dict, actual: dict) -> bool:
    for field in ("type", "age", "age_months", "passport"):
        if field in expected and actual.get(field) != expected[field]:
            return False
    if "origin" in expected and low(expected["origin"]) not in low(actual.get("origin")):
        return False
    return all(n in actual.get("needs", []) for n in expected.get("needs", []))


@section("request", original=True)
def request_checks(exp: dict, out: dict) -> list[Check]:
    req, checks = out.get("request") or {}, []
    if "stops" in exp:
        got = [(low(s["place"]), s["country"]) for s in req.get("stops", [])]
        named = lambda city, country: any(c == country and (low(city) in p or p in low(city)) for p, c in
                                          [(low(x), s["country"]) for s in req.get("stops", []) for x in [s["place"], *s.get("cities", [])]])
        for s in exp["stops"]:
            # "Japan: Tokyo and Kyoto" is met by one Japan stop, or by a Tokyo stop and a Kyoto stop in Japan
            checks.append((f"stop {s['place']} ({s['country']})",
                           any(s["country"] == c and (low(s["place"]) in p or p in low(s["place"])) for p, c in got)
                           or bool(s.get("cities")) and all(named(city, s["country"]) for city in s["cities"])))
    if "travellers" in exp:
        pool = list(req.get("travellers", []))
        for t in exp["travellers"]:
            match = next((a for a in pool if traveller_matches(t, a)), None)
            checks.append((f"traveller {t}", match is not None))
            if match:
                pool.remove(match)
        checks.append(("traveller count", len(req.get("travellers", [])) == len(exp["travellers"])))
    simple = {"group_count": len(req.get("groups", [])), "nights": out.get("nights"),
              "trimmed_nights": out.get("trimmed_nights"), "month": req.get("month"),
              "start_date": req.get("start_date"), "end_date": req.get("end_date"),
              "duration_range": req.get("duration_range")}
    checks += [(f"{k} = {exp[k]} (got {simple[k]})", simple[k] == exp[k]) for k in simple if k in exp]
    if "budget" in exp:
        b = req.get("budget") or {}
        checks.append(("budget", b.get("currency") == exp["budget"]["currency"]
                       and abs((b.get("amount") or 0) - exp["budget"]["amount"]) < 1))
    for d in exp.get("diet", []):
        checks.append((f"diet {d}", any(low(d) in low(x) or low(x) in low(d) for x in req.get("diet", []))))
    for n in exp.get("needs_any", []):
        checks.append((f"need {n}", any(n in t.get("needs", []) for t in req.get("travellers", []))))
    return checks


@section("clarify")
def clarify_checks(exp: dict, out: dict) -> list[Check]:
    first = next((p for p in out.get("pauses", []) if p["kind"] == "clarify"), None)
    asked = bool(first and first.get("question"))
    checks = [(f"asked = {exp['asked']}", asked == exp["asked"])]
    missing = (first or {}).get("missing", [])
    checks += [(f"asked for {f}", f in missing) for f in exp.get("fields_required", [])]
    checks += [(f"did not ask for {f}", f not in missing) for f in exp.get("fields_forbidden", [])]
    flags = ((first or {}).get("request") or {}).get("flags", [])
    checks += [(f"flag {f}", f in flags) for f in exp.get("flags", [])]
    if exp.get("suggests_destinations"):
        checks.append(("suggested destinations", bool((first or {}).get("ideas"))))
    return checks


# ---------- season & alternatives ----------

@section("season")
def season_checks(exp: dict, out: dict) -> list[Check]:
    """Judge the trip as first asked (the season_check events), not the alternative it may have switched to."""
    events = out.get("season_events", [])
    seen = {e["action"] for e in events}
    reasons = low(" ".join(e["detail"] for e in events))
    checks = [(f"verdict {exp['verdict']} (got {sorted(seen)})", exp["verdict"] in seen)]
    if exp.get("reasons_any"):
        checks.append((f"reason mentions one of {exp['reasons_any']}", any(low(w) in reasons for w in exp["reasons_any"])))
    if "stricter_for_vulnerable" in exp:
        checks.append(("stricter because of children/seniors", any(w in reasons for w in ("child", "senior", "kids"))))
    return checks


@section("route")
def route_checks(exp: dict, out: dict) -> list[Check]:
    ran = "alternatives" in out.get("agents_run", [])
    return [(f"alternatives ran = {exp['alternatives']}", ran == exp["alternatives"])]


def idea_matches(want: dict, alt: dict) -> bool:
    req = alt.get("request") or {}
    places = " ".join(low(s["place"]) + " " + " ".join(low(c) for c in s.get("cities", [])) for s in req.get("stops", []))
    text = low(alt.get("title")) + " " + low(alt.get("why")) + " " + places
    if "place" in want and low(want["place"]) not in text:
        return False
    if "months" in want:
        month = req.get("month") or (req.get("start_date") or "")[:7]
        if not month or int(month[5:7]) not in want["months"]:
            return False
    if "mentions_any" in want and not any(low(w) in text for w in want["mentions_any"]):
        return False
    if want.get("lower_altitude") and not any(w in text for w in ("lower", "altitude", "warmer", "goa", "kerala")):
        return False
    return True


@section("alternatives")
def alternatives_checks(exp: dict, out: dict) -> list[Check]:
    alts, checks = out.get("alternatives") or [], []
    if "count_min" in exp:
        checks.append((f"at least {exp['count_min']} alternatives", len(alts) >= exp["count_min"]))
    if "count_max" in exp:
        checks.append((f"at most {exp['count_max']} alternatives", len(alts) <= exp["count_max"]))
    if exp.get("each_has_feasibility"):
        checks.append(("every alternative costed", bool(alts) and all(a.get("feasibility") for a in alts)))
    if exp.get("at_least_one_feasible"):
        checks.append(("one alternative fits", any((a.get("feasibility") or {}).get("ok") for a in alts)))
    if "include_any" in exp:
        checks.append((f"includes one of {exp['include_any']}", any(idea_matches(w, a) for w in exp["include_any"] for a in alts)))
    if "destinations_any" in exp:
        checks.append(("destination idea", any(idea_matches({"place": d}, a) for d in exp["destinations_any"] for a in alts)))
    if "max_alt_round" in exp:
        checks.append((f"alt_round ≤ {exp['max_alt_round']}", (out.get("alt_round") or 0) <= exp["max_alt_round"]))
    return checks


@section("final")
def final_checks(exp: dict, out: dict) -> list[Check]:
    f = out.get("feasibility") or {}
    report = low(out.get("report_md"))
    return [("feasible, or the report says it doesn't fit", bool(f.get("ok")) or "does not fit" in report)]


# ---------- transport, meeting, visas, costs ----------

def group_for(out: dict, origin: str) -> tuple[dict, dict]:
    groups = (out.get("request") or {}).get("groups", [])
    g = next((g for g in groups if low(origin) in low(g["origin"]) or low(g["origin"]) in low(origin)), None)
    return (g or {}), ((out.get("group_results") or {}).get(g["id"]) if g else None) or {}


@section("transport", original=True)
def transport_checks(exp: dict, out: dict) -> list[Check]:
    checks = []
    for want in exp.get("groups", []):
        group, result = group_for(out, want["origin"])
        options = result.get("options", [])
        name = want["origin"]
        if "travellers" in want:
            checks.append((f"{name}: {want['travellers']} travellers", len(group.get("travellers", [])) == want["travellers"]))
        if "min_options" in want:
            checks.append((f"{name}: ≥{want['min_options']} options", len(options) >= want["min_options"]))
        if "must_include_any_route" in want:
            checks.append((f"{name}: a route like {want['must_include_any_route']}",
                           any(route_matches(r, o["route"]) for r in want["must_include_any_route"] for o in options)))
        for mode in want.get("must_include_modes", []):
            checks.append((f"{name}: a {mode} option", any(o["mode"] == mode for o in options)))
        if "cheapest_mode" in want and options:
            chosen = options[result.get("chosen", 0)]
            checks.append((f"{name}: cheapest is {want['cheapest_mode']}", chosen["mode"] == want["cheapest_mode"]))
        if "overland_via_any" in want:
            text = " ".join(low(o["route"]) + " " + " ".join(low(l["destination"]) for l in o.get("legs", [])) for o in options)
            checks.append((f"{name}: overland via {want['overland_via_any']}", any(low(v) in text for v in want["overland_via_any"])))
    rules = exp.get("rules", {})
    chosen = [r["options"][r.get("chosen", 0)] for r in (out.get("group_results") or {}).values() if r.get("options")]
    groups = len((out.get("request") or {}).get("groups", []))
    if rules:
        checks.append(("a route for every group", bool(chosen) and len(chosen) == groups))

    def layover(o):  # a direct route has none; a connecting route with no figure can't be checked, so it fails
        return 0 if len(o.get("legs", [])) <= 1 else o.get("max_layover_hours")
    if "max_layover_hours" in rules:
        checks.append(("short layovers", bool(chosen) and all(layover(o) is not None and layover(o) <= rules["max_layover_hours"]
                                                               for o in chosen)))
    if "no_overnight_bus_for" in rules:
        checks.append(("no overnight bus", bool(chosen) and not any(l.get("overnight") and l["mode"] == "bus"
                                                                    for o in chosen for l in o.get("legs", []))))
    if rules.get("assistance_noted"):
        checks.append(("assistance noted", any(o.get("accessibility_note") for o in chosen)))
    legs = [c.get("onward") for c in (out.get("costs") or {}).values() if c.get("onward")]
    if "intercity_modes_any" in exp:
        checks.append(("between-stop transport", any(l["mode"] in ("surface", "flight+overland") or
                                                     any(x["mode"] in exp["intercity_modes_any"] for x in l.get("legs", [])) for l in legs)))
    if "intercity_any_route" in exp:  # a priced journey between the stops; naming both cities in the report isn't one
        checks.append(("between-stop route", any(route_matches(r, l["route"]) for r in exp["intercity_any_route"] for l in legs)))
    return checks


@section("meeting", original=True)
def meeting_checks(exp: dict, out: dict) -> list[Check]:
    meeting, checks = out.get("meeting") or {}, []
    arrivals = [d for d in (meeting.get("arrivals") or {}).values() if d]
    if exp.get("day1") == "latest_arrival":  # no meeting plan at all is a failure; unknown arrival days are not
        checks.append(("day 1 is the latest arrival", bool(meeting) and (not arrivals or meeting.get("day1") == max(arrivals))))
    kinds = [o["kind"] for o in meeting.get("options", [])]
    checks += [(f"offers {k}", k in kinds) for k in exp.get("options_include", [])]
    if exp.get("hub_offered_if_shared"):
        routes = [r["options"][r.get("chosen", 0)]["route"] for r in (out.get("group_results") or {}).values() if r.get("options")]
        shared = set(codes(routes[0])[:-1]).intersection(*(set(codes(r)[:-1]) for r in routes[1:])) if len(routes) > 1 else set()
        checks.append(("hub offered when groups share one", not shared or "meet_at_hub_fly_together" in kinds))
    if exp.get("buffer_night_if_arrivals_differ") and len(set(arrivals)) > 1:
        checks.append(("buffer night", any(meeting.get("buffer_nights", {}).values())))
    return checks


@section("visa", original=True)
def visa_checks(exp: list, out: dict) -> list[Check]:
    visas = [v for r in (out.get("group_results") or {}).values() for v in r.get("visas", [])]
    checks = []
    for want in exp:
        found = [v for v in visas if v.get("passport") == want["passport"] and v["country"] == want["country"]]
        name = f"{want['passport']}→{want['country']}"
        if not found or found[0].get("rule") is None:
            checks.append((f"{name}: known", False))
            continue
        rule = found[0]["rule"]
        checks.append((f"{name}: visa_required={want['visa_required']}", rule["visa_required"] == want["visa_required"]))
        by_law = rule.get("notes") in ("Domestic travel.", "Free movement inside the Schengen area.")
        relevant = [c for c in rule.get("citations", []) if c.get("supports") == "requirement"
                    and c.get("relevance") in ("applies", "all_visitors")]
        checks.append((f"{name}: requirement cited by a relevant, verbatim quote", by_law or bool(rule.get("confirmed") and relevant)))
        for field in ("document", "zone", "fee_currency"):
            if field in want:
                checks.append((f"{name}: {field}={want[field]} (got {rule.get(field)})", rule.get(field) == want[field]))
        if "fee" in want:
            checks.append((f"{name}: fee {want['fee']} (got {rule.get('fee')})", rule.get("fee") == want["fee"]))
        for doc in want.get("documents_any", []):
            checks.append((f"{name}: accepts {doc}", doc in rule.get("accepted_ids", [])))
    return checks


@section("fees", original=True)
def fee_checks(exp: list, out: dict) -> list[Check]:
    f = out.get("feasibility") or {}
    lines = [l for l in f.get("lines", []) if l["category"] == "destination_fee"]
    passports = {t["label"]: t.get("passport") for t in (out.get("request") or {}).get("travellers", [])}
    checks = []
    for want in exp:
        named = [l for l in lines if any(low(n) in low(l["label"]) for n in want.get("name_any", [""]))]
        if "passport" in want:
            named = [l for l in named if passports.get(l["traveller"]) == want["passport"]]
        label = f"fee {want.get('name_any')} {want.get('passport', '')}".strip()
        checks.append((f"{label}: charged", bool(named)))
        if named and "total" in want:
            checks.append((f"{label}: {want['total']} {want['currency']}", any(abs(l["amount"] - want["total"]) < 1 and
                                                                             l["currency"] == want["currency"] for l in named)))
        if named and "total_all" in want:
            total = sum(l["amount"] for l in named if l["currency"] == want["currency"])
            checks.append((f"{label}: all travellers {want['total_all']} {want['currency']} (got {total:g})",
                           abs(total - want["total_all"]) < 1))
    return checks


@section("costs", original=True)
def cost_checks(exp: dict, out: dict) -> list[Check]:
    costs, checks = list((out.get("costs") or {}).values()), []
    f = out.get("feasibility") or {}
    if exp.get("child_fares_applied"):
        kids = {t["label"] for t in (out.get("request") or {}).get("travellers", []) if t["type"] != "adult"}
        fares = {}
        for l in (l for l in f.get("lines", []) if l["category"] == "transport"):
            fares[l["traveller"]] = fares.get(l["traveller"], 0) + l["amount_report"]
        checks.append(("children pay a child fare", bool(kids) and all(fares.get(k, 0) < max(fares.values() or [0]) for k in kids)))
    if "vehicle_capacity_min" in exp:
        checks.append(("vehicle seats everyone", any((c.get("vehicle_capacity") or 0) >= exp["vehicle_capacity_min"] for c in costs)))
    if exp.get("vehicle_accessible"):
        checks.append(("accessible vehicle", any(c.get("vehicle_accessible") for c in costs)))
    return checks


@section("feasibility", original=True)
def feasibility_checks(exp: dict, out: dict) -> list[Check]:
    f, checks = out.get("feasibility") or {}, []
    if not f:
        return [("feasibility computed", False)]
    first = (out.get("feasibility_events") or [{}])[0]  # the trip as first asked, before any alternative
    if "ok" in exp:
        checks.append((f"ok = {exp['ok']} (first check: {first.get('action')})", (first.get("action") == "ok") == exp["ok"]))
    if "gap_gt" in exp:
        checks.append(("over budget", "over budget" in low(first.get("detail"))))
    for word in exp.get("reasons_include", []):
        checks.append((f"reason names {word}", low(word) in low(first.get("detail"))))
    if exp.get("math_consistent"):
        lines = sum(l["amount_report"] for l in f.get("lines", []))
        people = sum(f.get("per_traveller", {}).values()) + f.get("shared", 0)
        checks.append(("totals add up", abs(lines - f["total"]) < 1 and abs(people - f["total"]) < 1))
    if "report_currency" in exp:
        checks.append((f"currency {exp['report_currency']}", f["currency"] == exp["report_currency"]))
    if exp.get("fx_rate_date_present"):
        checks.append(("rate date shown", bool(f.get("fx_as_of"))))
    checks += [(f"converts {c}", c in f.get("conversions", [])) for c in exp.get("conversions_include", [])]
    if "visa_fees_per_traveller" in exp:
        per = {}
        for l in f.get("lines", []):
            if l["category"] == "visa":
                per[l["traveller"]] = per.get(l["traveller"], 0) + 1
        checks.append(("one visa fee each", bool(per) and set(per.values()) == {exp["visa_fees_per_traveller"]}))
    return checks


# ---------- itinerary, food, QA, report ----------

@section("itinerary")
def itinerary_checks(exp: dict, out: dict) -> list[Check]:
    plan = out.get("itinerary") or {}
    days = plan.get("days", [])
    checks = [("itinerary present", bool(days))]
    if not days:
        return checks
    cities = [low(d["city"]) for d in days]
    if "days" in exp:
        checks.append((f"{exp['days']} days (got {len(days)})", len(days) == exp["days"]))
    checks += [(f"visits {c}", any(low(c) in x for x in cities)) for c in exp.get("cities_include", [])]
    if "start_city" in exp:
        checks.append((f"starts in {exp['start_city']}", low(exp["start_city"]) in cities[0]))
    if "end_city" in exp:
        checks.append((f"ends in {exp['end_city']}", low(exp["end_city"]) in cities[-1]))
    if (rule := exp.get("activity_not_on_day")):
        day = days[rule["day"] - 1]
        checks.append((f"{rule['activity']} not on day {rule['day']}",
                       not any(low(rule["activity"])[:8] in low(a["name"]) for a in day.get("activities", []))))
    if exp.get("rest_day"):
        checks.append(("a rest day", any(d.get("rest_day") for d in days)))
    if "max_strenuous_per_day" in exp:
        checks.append(("pacing", all(sum(a.get("strenuous", False) for a in d.get("activities", [])) <= exp["max_strenuous_per_day"] for d in days)))
    text = " ".join(low(d.get("transfer")) + " " + " ".join(low(a["name"]) for a in d.get("activities", [])) for d in days)
    if "intercity_mentions_any" in exp:
        checks.append(("transfer named", any(low(w) in text for w in exp["intercity_mentions_any"])))
    if "intercity_legs_min" in exp:
        checks.append((f"≥{exp['intercity_legs_min']} transfers", sum(1 for d in days if d.get("transfer")) >= exp["intercity_legs_min"]))
    return checks


@section("food")
def food_checks(exp: dict, out: dict) -> list[Check]:
    food, checks = out.get("food") or {}, []
    have = {low(c) for c in food}
    visited = {low(d["city"]) for d in (out.get("itinerary") or {}).get("days", [])}
    homes = {low(g["origin"]) for g in (out.get("request") or {}).get("groups", [])} - visited
    checks.append(("no food guide for a home city", bool(food) and not any(h in c or c in h for h in homes for c in have)))
    for c in exp.get("cities", []) + exp.get("cities_include", []):
        checks.append((f"guide for {c}", any(low(c) in h or h in low(c) for h in have)))
    text = lambda g: " ".join(low(d["name"]) + " " + low(d["description"]) for d in g.get("dishes", []))
    everything = " ".join(text(g) for g in food.values())
    if "must_mention_any" in exp:
        checks.append((f"mentions one of {exp['must_mention_any']}", any(low(w) in everything for w in exp["must_mention_any"])))
    for city, words in exp.get("must_mention_any_by_city", {}).items():
        guide = next((g for c, g in food.items() if low(city) in low(c)), {})
        places = " ".join(low(p["name"]) + " " + low(p.get("area")) for p in guide.get("places", []))
        checks.append((f"{city}: one of {words}", any(low(w) in text(guide) + " " + places for w in words)))
    terms, allergens = forbidden_terms(exp.get("diet", [])), allergen_terms(allergies(exp.get("diet", [])))
    dishes = [Dish.model_validate(d) for g in food.values() for d in g.get("dishes", [])]
    if terms:
        checks.append(("no off-diet dishes", bool(dishes) and not any(breaks_diet(d, terms) for d in dishes)))
    if allergens:
        found = [f"{d.name} ({a})" for d in dishes if (a := conflicts(d, allergens))]
        checks.append((f"no dish with an allergen{': ' + ', '.join(found) if found else ''}", bool(dishes) and not found))
    for word in exp.get("forbidden_terms", []):
        checks.append((f"never recommends {word}", not any(re.search(rf"\b{re.escape(word)}\b", d.name, re.I) for d in dishes)))
    guides = list(food.values())  # every "in every city" check needs at least one guide to be true
    if exp.get("kid_friendly_pick_per_city"):
        checks.append(("kid-friendly pick in every city", bool(guides) and all(any(d.get("kid_friendly") for d in g.get("dishes", []))
                                                                               for g in guides)))
    if "places_min_per_city" in exp:
        checks.append((f"≥{exp['places_min_per_city']} places per city", bool(guides) and all(
            len(g.get("places", [])) >= exp["places_min_per_city"] for g in guides)))
    if "halal_places_min_per_city" in exp:  # places that say they are halal, as many as asked for
        halal = lambda p: re.search(r"halal|muslim[- ]friendly", " ".join(low(p.get(k)) for k in ("name", "area", "why")))
        n = exp["halal_places_min_per_city"]
        checks.append((f"≥{n} halal place(s) per city", bool(guides) and all(
            sum(bool(halal(p)) for p in g.get("places", [])) >= n for g in guides)))
    if exp.get("sources_present"):  # every place cites a source this run really fetched
        known = set(out.get("source_ids") or [])
        places = [p for g in guides for p in g.get("places", [])]
        checks.append(("places cite real sources", bool(places) and all(
            p.get("source_ids") and set(p["source_ids"]) <= known for p in places)))
    if exp.get("diet_notes_every_city") or exp.get("allergy_warnings_present"):
        checks.append(("diet notes everywhere", bool(guides) and all(g.get("diet_notes") for g in guides)))
    if "warns_about_any" in exp:
        notes = " ".join(low(g.get("diet_notes")) for g in food.values()) + " " + everything
        checks.append((f"warns about {exp['warns_about_any']}", any(low(w) in notes for w in exp["warns_about_any"])))
    return checks


@section("qa")
def qa_checks(exp: dict, out: dict) -> list[Check]:
    events, checks = out.get("qa_events", []), []
    final = out.get("qa_report") or {}
    if "pass" in exp:
        checks.append(("QA passed", bool(final.get("passed")) == exp["pass"]))
    if "first_pass" in exp:
        checks.append(("first QA verdict", bool(events) and (events[0]["action"] == "passed") == exp["first_pass"]))
    for issue in exp.get("issues_include", []):
        checks.append((f"QA caught {issue}", bool(events) and issue.split(":")[0] in events[0]["detail"]))
    if "loops" in exp:
        checks.append((f"{exp['loops']} QA loop(s)", len(events) - 1 == exp["loops"]))
    return checks


@section("report")
def report_checks(exp: dict, out: dict) -> list[Check]:
    report = low(out.get("report_md"))
    return [(f"report mentions one of {exp['mentions_any']}", any(low(w) in report for w in exp["mentions_any"]))]


CODE_EVALUATORS = [request_checks, clarify_checks, season_checks, route_checks, alternatives_checks, final_checks,
                   transport_checks, meeting_checks, visa_checks, fee_checks, cost_checks, feasibility_checks,
                   itinerary_checks, food_checks, qa_checks, report_checks]


def completion(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """The run reached the end of the workflow: not stopped at a pause nobody answered."""
    done = bool(outputs.get("completed"))
    why = (outputs.get("termination_reason") or "finished") if done else \
        f"{outputs.get('termination_reason')}, waiting at {outputs.get('pending_interrupt')}"
    return {"key": "completion", "score": 1 if done else 0, "comment": why}


# Failures that make a plan unsafe or untrue. They are reported apart from the mean, and fail the case whatever its
# mean: a wrong visa rule or an allergen is not made up for by good food elsewhere.
CRITICAL = {"visa": ("visa_required", ": known"), "feasibility": ("totals add up",),
            "food": ("no off-diet dishes", "no dish with an allergen", "never recommends", "real sources")}


def critical(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    failed = [] if outputs.get("completed") else [f"run not completed ({outputs.get('termination_reason')})"]
    for evaluator in CODE_EVALUATORS:
        words, expected = CRITICAL.get(evaluator.__name__), reference_outputs.get(evaluator.__name__)
        if words and expected is not None:
            failed += [f"{evaluator.__name__}: {name}" for name, ok in evaluator.checks(expected, outputs)
                       if not ok and any(w in name for w in words)]
    return {"key": "critical", "score": 0 if failed else 1, "comment": "; ".join(failed) or "none"}


def success(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Done well: the task-completion metric. The run finished, the plan passed QA, everything the trip needs is
    priced, and nothing critical failed. Fitting the budget and the season are not part of it: a correct "this
    doesn't fit" plan is a success too. A case that expects no plan (expect_plan false, e.g. a request TravelEx
    must decline) succeeds by finishing with no critical failure."""
    missing = [] if outputs.get("completed") else [f"not finished ({outputs.get('termination_reason')})"]
    if reference_outputs.get("expect_plan", True):
        qa, f = outputs.get("qa_report") or {}, outputs.get("feasibility") or {}
        if outputs.get("termination_reason") == "ended without a destination":
            missing.append("no plan (ended without a destination)")
        if not qa.get("passed"):
            missing.append("QA did not pass" + (f" ({', '.join(qa['issues'])})" if qa.get("issues") else " (no QA report)"))
        if not f:
            missing.append("not costed")
        elif f.get("unpriced"):
            missing.append(f"not fully priced ({'; '.join(f['unpriced'])})")
    if (severe := critical(inputs, outputs, reference_outputs))["score"] == 0:
        missing.append(f"critical: {severe['comment']}")
    return {"key": "success", "score": 0 if missing else 1, "comment": "; ".join(missing) or "done well"}


# ---------- LLM judges ----------

class Grade(BaseModel):
    score: int = Field(ge=1, le=5, description="1 poor, 3 acceptable, 5 excellent")
    reason: str


JUDGE = """You grade one part of a travel plan for the travellers described. Be strict and specific.
{rubric}
Answer with a score from 1 to 5 and one or two sentences of reason."""
RUBRICS = {
    "itinerary_quality": "Is the day-by-day plan realistic: sensible order and travel times, pacing that suits "
                         "the children or seniors, time to acclimatise at altitude, and the travellers' interests?",
    "food_relevance": "Are the dishes and places really local to each city, and do they respect the diet and "
                      "allergies? Is there something for children when children travel?",
    "alternatives_quality": "Do the alternatives fix the actual problem (budget or season), differ from each "
                            "other, and keep what the travellers wanted?",
}


def judge_model_problem() -> str | None:
    """Why the judges can't run: a judge on the agent's own model grades its own work, and tends to be generous."""
    from travelex.llm import model_name_for
    judge, agent = os.getenv("LLM_MODEL_JUDGE", "").strip(), model_name_for("intake")
    if not judge:
        return "LLM_MODEL_JUDGE is not set. Set it to a different model from the agent's, e.g. openai:gpt-5.2."
    if judge == agent:
        return f"LLM_MODEL_JUDGE is {judge}, the agent's own model. Use a different model to judge."
    return None


def judge(key: str, part: str):
    def evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        content = {"itinerary_quality": outputs.get("itinerary"), "food_relevance": outputs.get("food"),
                   "alternatives_quality": [{k: a.get(k) for k in ("title", "why", "trade_offs", "feasibility")}
                                            for a in outputs.get("alternatives") or []]}[key]
        if not content:
            return {"key": key, "score": None, "comment": "not applicable"}
        context = f"Travellers' request: {inputs['query']}"
        if chosen := outputs.get("chosen_alternative"):  # else the judge marks the plan down for "ignoring" the request
            reasons = "; ".join((outputs.get("first") or {}).get("feasibility", {}).get("reasons", []))
            context += f"\nThe original plan didn't work ({reasons}), so the travellers switched to: {chosen}."
        messages = [SystemMessage(JUDGE.format(rubric=RUBRICS[key])),
                    HumanMessage(f"{context}\n\n{part}: {str(content)[:12000]}")]
        grade = structured(get_llm("judge"), Grade, messages)  # LLM_MODEL_JUDGE: not the agent's own model
        return {"key": key, "score": grade.score / 5, "comment": grade.reason}
    evaluator.__name__ = key
    return evaluator


JUDGES = [judge("itinerary_quality", "Itinerary"), judge("food_relevance", "Food guides"),
          judge("alternatives_quality", "Alternatives")]
