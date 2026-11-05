"""Runs one golden case through the real graph and returns what the evaluators read."""
from datetime import date
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from travelex.agents.food import food_agent
from travelex.agents.itinerary import itinerary_planner
from travelex.graph import build_graph, initial_state, run_config, serializer
from travelex.report import render_markdown, to_jsonable
from travelex.trip import resolve_nights

MAX_PAUSES = 12  # clarify, handoff, alternatives and comment rounds together; more means the run is stuck


def fault_nodes(fixtures: dict) -> dict:
    """Swap in nodes that fail once, so a case can check that QA catches it (golden E5)."""
    nodes = {}
    if fixtures.get("itinerary_planner_first_call") == "five_days_for_six_day_trip":
        first = {"done": False}

        def short_first_draft(state):
            out = itinerary_planner(state)
            if not first["done"]:
                first["done"] = True
                plan = out["itinerary"]
                out["itinerary"] = plan.model_copy(update={"days": plan.days[:-1]})
            return out
        nodes["itinerary_planner"] = short_first_draft
    if city := fixtures.get("food_agent_skip_city_first_pass"):
        skipped = {"done": False}

        def skip_once(state):
            if state["city"].casefold() == city.casefold() and not skipped["done"]:
                skipped["done"] = True
                return {}
            return food_agent(state)
        nodes["food_agent"] = skip_once
    return nodes


def default_answer(pause: dict):
    """When the case's answers run out: accept the trip card, let the agent pick an alternative as it does with
    nobody to ask (keeping the destination when it can fit), finish without comments."""
    return {"clarify": pause.get("request"), "alternatives": "pick for me", "review": "", "handoff": ""}[pause["kind"]]


def run_case(inputs: dict, config_extra: dict | None = None) -> dict:
    graph = build_graph(InMemorySaver(serde=serializer()), nodes=fault_nodes(inputs.get("fixtures") or {}))
    thread = uuid4().hex
    answers = list(inputs.get("resume") or [])
    config = run_config(thread, auto_approve=False, eval_case=True) | (config_extra or {})
    payload, pauses = initial_state(inputs["query"], date.fromisoformat(inputs["today"])), []
    for _ in range(MAX_PAUSES + 1):
        for _ in graph.stream(payload, config):
            pass
        snapshot = graph.get_state(config)
        if not snapshot.interrupts or len(pauses) == MAX_PAUSES:
            break
        pause = snapshot.interrupts[0].value
        pauses.append(pause)
        payload = Command(resume=answers.pop(0) if answers else default_answer(pause))
    return summarise(snapshot.values, pauses, first_research(graph, config), status(snapshot))


def run_case_safely(inputs: dict) -> dict:
    """run_case, but a run that crashes (an expired key, a bug) comes back as an unfinished run saying why, so the
    completion and success evaluators report the error instead of scoring an empty output."""
    try:
        return run_case(inputs)
    except Exception as e:
        return {"completed": False, "pending_interrupt": None,
                "termination_reason": f"error: {type(e).__name__}: {str(e)[:240]}"}


def status(snapshot) -> dict:
    """Whether the workflow reached its end. A run still waiting at a pause is unfinished, and fails completion."""
    if snapshot.interrupts:
        return {"completed": False, "pending_interrupt": snapshot.interrupts[0].value.get("kind"),
                "termination_reason": "pause_limit"}
    if snapshot.next:
        return {"completed": False, "pending_interrupt": None, "termination_reason": f"stopped before {snapshot.next[0]}"}
    request = snapshot.values.get("request")
    planned = bool(request and request.stops)  # the workflow can end at the question with nothing to plan
    return {"completed": True, "pending_interrupt": None,
            "termination_reason": "finished" if planned else "ended without a destination"}


FIRST_KEYS = ("request", "season", "group_results", "costs", "meeting", "feasibility")


def first_research(graph, config) -> dict:
    """The research for the trip as first asked: the oldest checkpoint with a budget check. Choosing an
    alternative replaces these in the final state, but the golden expectations are about the original."""
    first = {}
    for snapshot in graph.get_state_history(config):  # newest first, so the last match is the oldest
        if snapshot.values.get("feasibility") is not None:
            first = {k: snapshot.values.get(k) for k in FIRST_KEYS}
    return first


def summarise(values: dict, pauses: list[dict], first: dict | None = None, run_status: dict | None = None) -> dict:
    """Everything an evaluator needs, as plain JSON."""
    request = values.get("request")
    nights, trimmed = resolve_nights(request) if request else (None, None)
    events = values.get("events") or []
    out = to_jsonable({k: v for k, v in values.items() if k not in ("events", "sources")})
    out |= {"nights": nights, "trimmed_nights": trimmed, "pauses": pauses,
            "agents_run": list(dict.fromkeys(e["agent"] for e in events)),
            "qa_events": [e for e in events if e["agent"] == "qa"],
            # the trip as first asked: an alternative replaces season and feasibility in the final state
            "season_events": [e for e in events if e["agent"] == "season_check"],
            "feasibility_events": [e for e in events if e["agent"] == "feasibility"],
            "report_md": render_markdown(values) if request and request.stops else "",
            "source_count": len(values.get("sources") or {}),
            "source_ids": sorted(values.get("sources") or {}),
            **(run_status or {"completed": None, "pending_interrupt": None, "termination_reason": "unknown"}),
            "first": first_json(first or {})}
    return out


def first_json(first: dict) -> dict:
    out = to_jsonable(first)
    if first.get("request"):
        out["nights"], out["trimmed_nights"] = resolve_nights(first["request"])
    return out
