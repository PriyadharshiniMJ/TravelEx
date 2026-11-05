"""Graph state: the main TripState, the per-group GroupState and the per-city FoodInput."""
from datetime import date
from operator import add
from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph.message import add_messages

from travelex.schemas import (
    Alternative, Feasibility, FoodGuide, Gap, GroupResult, Itinerary, MeetingPlan, QAReport,
    SeasonVerdict, Source, Stop, StopCosts, TravelGroup, TripRequest,
)


def merge_dicts(left: dict | None, right: dict | None) -> dict:
    """Merge updates key by key. Returning None from a node clears the dict, e.g. before an alternative re-runs."""
    return {} if right is None else {**(left or {}), **right}


def tool_rounds(messages: list[AnyMessage]) -> int:
    """AI tool-call turns since the last HumanMessage; a new instruction resets the step cap."""
    rounds = 0
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, AIMessage) and message.tool_calls:
            rounds += 1
    return rounds


class TripState(TypedDict):
    raw_query: str
    today: date  # passed in, never read from the clock, so runs and evals are reproducible
    namespace: str  # source-ID prefix: "" for the main run, "A1." for alternative 1's research
    request: TripRequest | None
    clarify_answer: str | None
    clarify_round: int
    season: Annotated[dict[str, SeasonVerdict], merge_dicts]  # stop place → verdict
    group_results: Annotated[dict[str, GroupResult], merge_dicts]  # group id → result
    costs: Annotated[dict[str, StopCosts], merge_dicts]  # stop place → researched rates and fees
    meeting: MeetingPlan | None
    gaps: list[Gap]  # what the research couldn't source, and what the gap-filling agent did about each
    gap_messages: Annotated[list[AnyMessage], add_messages]  # the gap-filling agent's conversation
    handoff_round: int  # times the plan paused to ask the traveller about something the agents couldn't settle
    traveller_notes: list[str]  # what the traveller told us then, e.g. "I already hold a valid US visa"
    feasibility: Feasibility | None
    alternatives: list[Alternative]
    alt_round: int
    alt_feedback: str | None  # the user's words when none of the alternatives suit
    chosen_alternative: str | None  # title of the alternative the trip switched to
    itinerary: Itinerary | None
    food: Annotated[dict[str, FoodGuide], merge_dicts]  # city → guide
    qa_report: QAReport | None
    qa_round: int
    plan_comment: str | None  # the traveller's latest comment on the finished plan, for the itinerary planner
    replan: bool  # that comment changes the trip itself, so its research runs again
    revisions: list[str]  # what each round of comments changed, oldest first
    sources: Annotated[dict[str, Source], merge_dicts]
    report_dir: str | None
    events: Annotated[list[dict], add]
    errors: Annotated[list[str], add]


class GroupInput(TypedDict):
    """The Send() payload for one travel group."""
    group: TravelGroup
    source_prefix: str  # "G1.", "G2."… so parallel branches never clash on source IDs
    request: TripRequest
    today: date
    taken: NotRequired[dict[str, Source]]  # IDs earlier research gave out under this prefix: number after them


class GroupState(GroupInput):
    messages: Annotated[list[AnyMessage], add_messages]
    options: list  # RouteOptions from the transport branch
    visas: list  # VisaChecks from the visa branch, which runs at the same time
    result: GroupResult | None
    sources: Annotated[dict[str, Source], merge_dicts]
    errors: Annotated[list[str], add]
    events: Annotated[list[dict], add]


class StopInput(TypedDict):
    """The Send() payload for one stop's cost research."""
    stop: Stop
    next_stop: Stop | None
    source_prefix: str
    request: TripRequest
    today: date
    taken: NotRequired[dict[str, Source]]


class FoodInput(TypedDict):
    """The Send() payload for one city's food guide."""
    city: str
    source_prefix: str
    request: TripRequest
    country: NotRequired[str | None]  # ISO code of the city's country: searches and checks name it
    taken: NotRequired[dict[str, Source]]
