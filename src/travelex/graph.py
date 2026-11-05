"""Wires the agents into the main graph, the research graph and the per-group subgraph.

intake → clarify → [season_check ‖ group_research × groups ‖ destination_costs × stops] → meeting_plan
→ feasibility → (alternatives_agent → choose_alternative)* → itinerary_planner → [food_agent × cities]
→ qa → (itinerary_planner | food_agent)* → save_report
"""
import os
import sqlite3
from datetime import date
from functools import partial
from pathlib import Path

import pydantic
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.types import Overwrite, RetryPolicy, Send

import travelex.schemas as schemas
from travelex.agents.alternatives import (
    choose_alternative, make_alternatives_agent, route_after_choice, route_after_feasibility,
)
from travelex.agents.common import event, taken_ids
from travelex.agents.costs import destination_costs
from travelex.agents.feasibility import feasibility_node
from travelex.agents.food import food_agent
from travelex.agents.handoff import ask_traveller, route_after_handoff
from travelex.agents.gaps import gap_agent, gap_tools, plan_gap_filling, route_after_planning, route_gaps
from travelex.agents.intake import clarify, intake, route_after_clarify
from travelex.agents.itinerary import country_of, guide_for, itinerary_planner, named_cities, route_after_itinerary
from travelex.agents.meeting import meeting_node
from travelex.agents.review import review_plan, route_after_review
from travelex.agents.qa import qa, route_after_qa
from travelex.agents.season import season_check
from travelex.agents.transport import extract_routes, route_transport, transport_agent, transport_tools
from travelex.agents.visa import assemble_group, visa_check
from travelex.report import save_report
from travelex.schemas import GroupResult
from travelex.limits import TokenCounter, limits_for
from travelex.retry import is_transient
from travelex.state import GroupInput, GroupState, TripState
from travelex.trip import group_key, next_stop, stop_key

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DB = PROJECT_ROOT / "runs" / "checkpoints.sqlite"
LLM_RETRY = RetryPolicy(max_attempts=3, retry_on=is_transient)
MAX_CONCURRENCY = int(os.getenv("TRAVELEX_MAX_CONCURRENCY") or 8)  # parallel steps (research branches) at once
RECURSION_LIMIT = 300  # ReAct loops, clarify rounds, alternatives and QA loops need far more than the default 25


def run_config(thread_id: str, auto_approve: bool = False, **metadata) -> dict:
    """The config every run uses: checkpoint thread, auto-approve switch, step limit, the run-wide limits on
    searches, tokens and time (shared by every branch), how many steps run at once, and LangSmith labels."""
    limits = limits_for(thread_id)
    return {"configurable": {"thread_id": thread_id, "auto_approve": auto_approve, "limits": limits},
            "callbacks": [TokenCounter(limits)], "max_concurrency": MAX_CONCURRENCY,
            "recursion_limit": RECURSION_LIMIT, "run_name": "travelex",
            "tags": ["travelex", "auto" if auto_approve else "interactive"],
            "metadata": {"thread_id": thread_id, **metadata}}


def initial_state(query: str, today: date | None = None) -> dict:
    """today is passed in, not read inside nodes, so evals can pin it."""
    return {"raw_query": query, "today": today or date.today(), "namespace": "", "clarify_round": 0,
            "alt_round": 0, "qa_round": 0, "revisions": [], "handoff_round": 0, "traveller_notes": []}


def state_after(saved: dict, output: dict) -> dict:
    """The state as it will be once a step's output is saved, for showing results while the run continues.
    While a step's update streams, the checkpoint may still hold the state before it. Overwrite values replace
    and dict channels merge, as the graph's reducers do; the trace channels are left out."""
    view = dict(saved)
    for key, value in output.items():
        if isinstance(value, Overwrite):
            view[key] = value.value
        elif key in ("events", "errors"):
            continue
        elif isinstance(value, dict) and isinstance(saved.get(key), dict):
            view[key] = {**saved[key], **value}
        else:
            view[key] = value
    return view


PREVIEW_AFTER = {"feasibility": route_after_feasibility, "choose_alternative": route_after_choice}


def settled(node: str, saved: dict, output) -> dict | None:
    """The state after a top-level step whose research is final (the itinerary is written next), for the trip
    at a glance. None while alternatives may still change the trip, so no preview shows a plan about to change."""
    if node not in PREVIEW_AFTER or not isinstance(output, dict):
        return None
    view = state_after(saved, output)
    return view if PREVIEW_AFTER[node](view) == "itinerary_planner" else None


def serializer() -> JsonPlusSerializer:
    """Let our Pydantic models back out of checkpoints. Any model left off this list comes back as a dict."""
    models = [(schemas.__name__, name) for name, obj in vars(schemas).items()
              if isinstance(obj, type) and issubclass(obj, pydantic.BaseModel) and obj.__module__ == schemas.__name__]
    return JsonPlusSerializer(allowed_msgpack_modules=models)


def open_checkpointer(path: Path = CHECKPOINT_DB) -> SqliteSaver:
    """check_same_thread=False: LangGraph and Streamlit both run nodes on worker threads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False), serde=serializer())


# ---------- research: transport and visas per group, costs per stop ----------

def build_group_subgraph():
    graph = StateGraph(GroupState)
    graph.add_node("transport_agent", transport_agent, retry_policy=LLM_RETRY)
    graph.add_node("transport_tools", transport_tools)
    graph.add_node("extract_routes", extract_routes, retry_policy=LLM_RETRY)
    graph.add_node("visa_check", visa_check)
    graph.add_node("assemble_group", assemble_group)
    graph.add_edge(START, "transport_agent")  # the flight search and the visa check start together:
    graph.add_edge(START, "visa_check")       # neither needs the other
    graph.add_conditional_edges("transport_agent", route_transport, ["transport_tools", "extract_routes"])
    graph.add_edge("transport_tools", "transport_agent")
    graph.add_edge(["extract_routes", "visa_check"], "assemble_group")  # waits for both
    graph.add_edge("assemble_group", END)
    return graph.compile()


def make_group_research(subgraph):
    def group_research(state: GroupInput, config: RunnableConfig) -> dict:
        """Run one travel group's subgraph. Fail-soft: one broken group must not sink the trip."""
        group = state["group"]
        try:
            out = subgraph.invoke(state, config)  # parent config, so stream(subgraphs=True) shows inner steps
        except GraphBubbleUp:
            raise
        except Exception as e:
            return {"group_results": {group.id: GroupResult(group_id=group.id, status="failed")},
                    "errors": [f"group_research ({group.origin}): {type(e).__name__}: {e}"],
                    "events": [event("group_research", "failed", f"{group.origin}: {e}")]}
        return {"group_results": {group.id: out["result"]}, "sources": out.get("sources") or {},
                "events": out.get("events") or [], "errors": out.get("errors") or []}
    return group_research


def season_send(state: TripState) -> Send:
    ns = state.get("namespace", "")
    return Send("season_check", {"request": state["request"], "today": state["today"], "namespace": ns,
                                 "taken": taken_ids(state, f"{ns}S")})


def plan_research(state: TripState) -> dict:
    """Nothing to compute: its outgoing edge launches every research branch at once."""
    return {}


def route_research(state: TripState, early_food: bool = False) -> list[Send]:
    """The season, one branch per travel group and one per stop, all in parallel. A result whose key still
    matches (same inputs, e.g. an alternative that only shortens the trip) is reused instead of researched.
    early_food: also start food guides for the cities the user named, which need no itinerary."""
    request, today, ns = state["request"], state["today"], state.get("namespace", "")
    have_groups, have_costs = state.get("group_results") or {}, state.get("costs") or {}
    sends = [season_send(state)] + [Send("group_research", {"group": g, "source_prefix": f"{ns}G{i}.", "request": request,
                                                            "today": today, "taken": taken_ids(state, f"{ns}G{i}.")})
             for i, g in enumerate(request.groups, 1)
             if getattr(have_groups.get(g.id), "key", None) != group_key(request, g)]
    stops = request.stops
    for i, stop in enumerate(stops):
        nxt = next_stop(request, i)
        if nxt is None and any(s.place.casefold() == stop.place.casefold() for s in stops[:i]):
            continue  # back where the trip began ("Tokyo, Kyoto, Tokyo"): the first visit's rates and onward leg stand
        if getattr(have_costs.get(stop.place), "key", None) != stop_key(request, stop, nxt):
            sends.append(Send("destination_costs", {"stop": stop, "next_stop": nxt, "source_prefix": f"{ns}C{i + 1}.",
                                                    "request": request, "today": today,
                                                    "taken": taken_ids(state, f"{ns}C{i + 1}.")}))
    if early_food:
        have = state.get("food") or {}
        sends += [Send("food_early", {"city": city, "country": country_of(request, city),
                                      "source_prefix": f"{ns}E{i}.", "request": request,
                                      "taken": taken_ids(state, f"{ns}E{i}.")})
                  for i, city in enumerate(named_cities(request), 1) if guide_for(city, have) is None]
    return sends


def add_research(graph: StateGraph, early_food=None, fill_gaps: bool = False) -> None:
    """plan_research → [season_check ‖ group_research × groups ‖ destination_costs × stops
    (‖ food_early × named city)] → meeting_plan (→ fill_gaps → gap_agent ⇄ gap_tools → ask_traveller) → feasibility.
    early_food: the food node to run during research. Only the main trip passes one: costing an alternative
    needs no food guides. It is a separate node because food_agent leads on to the final check.
    fill_gaps: let an agent chase the prices and routes the research couldn't source (the main trip only)."""
    graph.add_node("plan_research", plan_research)
    graph.add_node("season_check", season_check, retry_policy=LLM_RETRY)
    graph.add_node("group_research", make_group_research(build_group_subgraph()))
    graph.add_node("destination_costs", destination_costs, retry_policy=LLM_RETRY)
    graph.add_node("meeting_plan", meeting_node)
    graph.add_node("feasibility", feasibility_node)
    branches = ["season_check", "group_research", "destination_costs"]
    if early_food is not None:
        graph.add_node("food_early", early_food, retry_policy=LLM_RETRY)
        graph.add_edge("food_early", "meeting_plan")
        branches.append("food_early")
    graph.add_conditional_edges("plan_research", partial(route_research, early_food=early_food is not None), branches)
    graph.add_edge("season_check", "meeting_plan")
    graph.add_edge("group_research", "meeting_plan")
    graph.add_edge("destination_costs", "meeting_plan")
    if fill_gaps:
        graph.add_node("fill_gaps", plan_gap_filling)
        graph.add_node("gap_agent", gap_agent)
        graph.add_node("gap_tools", gap_tools)
        graph.add_node("ask_traveller", ask_traveller)
        graph.add_edge("meeting_plan", "fill_gaps")
        graph.add_conditional_edges("fill_gaps", route_after_planning, ["gap_agent", "ask_traveller"])
        graph.add_conditional_edges("gap_agent", route_gaps, ["gap_tools", "ask_traveller"])
        graph.add_edge("gap_tools", "gap_agent")
        graph.add_conditional_edges("ask_traveller", route_after_handoff, ["gap_agent", "feasibility"])
    else:
        graph.add_edge("meeting_plan", "feasibility")


def build_research_graph():
    """The research on its own, for costing alternatives. checkpointer=False: never inherit the parent's,
    so several alternatives can be researched at once from one node."""
    graph = StateGraph(TripState)
    add_research(graph)
    graph.add_edge(START, "plan_research")
    graph.add_edge("feasibility", END)
    return graph.compile(checkpointer=False)


# ---------- the whole trip ----------

def build_graph(checkpointer=None, nodes: dict | None = None, interrupt_before: list[str] | None = None):
    """nodes: replacement node functions by name, for evals that inject faults (e.g. a bad first itinerary).
    interrupt_before: stop before these nodes (tests that only exercise part of the graph)."""
    swap = {"itinerary_planner": itinerary_planner, "food_agent": food_agent} | (nodes or {})
    graph = StateGraph(TripState)
    graph.add_node("intake", intake, retry_policy=LLM_RETRY)
    graph.add_node("clarify", clarify)
    add_research(graph, early_food=swap["food_agent"], fill_gaps=True)
    graph.add_node("alternatives_agent", make_alternatives_agent(build_research_graph()))
    graph.add_node("choose_alternative", choose_alternative)
    graph.add_node("itinerary_planner", swap["itinerary_planner"], retry_policy=LLM_RETRY)
    graph.add_node("food_agent", swap["food_agent"], retry_policy=LLM_RETRY)
    graph.add_node("qa", qa)
    graph.add_node("save_report", save_report)
    graph.add_node("review_plan", review_plan, retry_policy=LLM_RETRY)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "clarify")
    graph.add_conditional_edges("clarify", route_after_clarify, ["intake", "clarify", "plan_research", END])
    graph.add_conditional_edges("feasibility", route_after_feasibility, ["alternatives_agent", "itinerary_planner"])
    graph.add_edge("alternatives_agent", "choose_alternative")
    graph.add_conditional_edges("choose_alternative", route_after_choice, ["alternatives_agent", "itinerary_planner"])
    graph.add_conditional_edges("itinerary_planner", route_after_itinerary, ["food_agent", "qa"])
    graph.add_edge("food_agent", "qa")
    graph.add_conditional_edges("qa", route_after_qa, ["itinerary_planner", "food_agent", "save_report"])
    graph.add_edge("save_report", "review_plan")
    graph.add_conditional_edges("review_plan", route_after_review, ["itinerary_planner", "plan_research", END])
    return graph.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)
