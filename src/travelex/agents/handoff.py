"""Handoff: when the agents are stuck on something that can stop the trip (no way to get there, an entry rule no
source confirms), pause and ask the traveller (interrupt). A hint goes back to the gap-filling agent for another try;
"continue" goes on with the plan as it is, and what's unresolved is listed in the plan to double-check."""
import re
import time

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from travelex.agents.common import event
from travelex.state import TripState
from travelex.tools import visa_rules

MAX_HANDOFFS = 1  # one hint per trip: after that, the plan goes on and lists what's still open
CONTINUE = re.compile(r"(continue|skip|go ahead|proceed|carry on|ok|okay|fine|no|none|nothing|yes)[.! ]*", re.IGNORECASE)


def open_issues(state: TripState) -> list[str]:
    """What the agents couldn't settle that could stop the trip: journeys with no route, entry rules no source
    confirms. Estimated prices are not here: they are labelled in the budget and don't stop anyone travelling."""
    issues = [g.text + (f" ({g.note})" if g.note else "") for g in state.get("gaps") or []
              if g.kind == "route" and g.status == "gave_up"]
    for result in (state.get("group_results") or {}).values():
        for v in result.visas:
            if v.rule is None or not (v.rule.confirmed or visa_rules.by_law(v.rule)):
                issues.append(f"{v.traveller} ({v.passport or 'passport unknown'}) entering {v.country}: "
                              f"{v.summary or 'no entry rule found'}")
    return issues


def ask_traveller(state: TripState, config: RunnableConfig) -> dict:
    """Pause with the open issues. Resume with "" (or "continue") to go on, or a hint for the agent to try again."""
    issues, rounds = open_issues(state), state.get("handoff_round", 0)
    if not issues or rounds >= MAX_HANDOFFS or config.get("configurable", {}).get("auto_approve"):
        return {}
    answer = interrupt({"kind": "handoff", "issues": issues})
    hint = answer.strip() if isinstance(answer, str) else ""
    if not hint or CONTINUE.fullmatch(hint):
        return {"handoff_round": rounds + 1,
                "events": [event("ask_traveller", "continued", f"{len(issues)} open: " + "; ".join(issues))]}
    started = time.perf_counter()
    gaps = [g.model_copy(update={"status": "open", "note": ""}) if g.status == "gave_up" and g.kind == "route"
            else g for g in state.get("gaps") or []]  # only what the traveller was shown: a visa hint finds no fares
    update = {"handoff_round": rounds + 1, "traveller_notes": [*(state.get("traveller_notes") or []), hint],
              "gaps": gaps, "events": [event("ask_traveller", "traveller answered", hint, started=started)]}
    if any(g.status == "open" for g in gaps):  # the agent tries again with the hint, with a fresh round budget
        update["gap_messages"] = [HumanMessage(f"The traveller says: {hint}\nUse this to try the open gaps again.")]
    return update


def route_after_handoff(state: TripState) -> str:
    return "gap_agent" if any(g.status == "open" for g in state.get("gaps") or []) else "feasibility"
