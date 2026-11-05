"""Where and when the travel groups meet: pure Python over each group's chosen route."""
from datetime import date, datetime

from langsmith import traceable

from travelex.agents.common import event
from travelex.schemas import GroupResult, MeetingOption, MeetingPlan, TripRequest


def hubs(route: str) -> list[str]:
    """Every place on a route like "MAA-CCU-PBH" except the destination. An origin can be a hub too:
    someone flying LOS-NBO-ZNZ can meet a traveller who starts in Nairobi."""
    return route.split("-")[:-1]


ARRIVAL_SLACK_DAYS = 3


def plausible(arrival: datetime | None, request: TripRequest) -> date | None:
    """An arrival date we can trust: only with exact trip dates, and within a few days of the start."""
    if arrival is None or request.start_date is None:
        return None
    return arrival.date() if abs((arrival.date() - request.start_date).days) <= ARRIVAL_SLACK_DAYS else None


@traceable(name="meeting.plan_meeting")
def plan_meeting(request: TripRequest, results: list[GroupResult]) -> MeetingPlan:
    """Day 1 is the latest group's arrival; earlier groups get buffer nights or can meet at a shared hub."""
    by_group = {r.group_id: r for r in results}
    chosen = {g.id: by_group[g.id].chosen_option() if g.id in by_group else None for g in request.groups}

    arrivals = {gid: plausible(o.arrival() if o else None, request) for gid, o in chosen.items()}
    known = [d for d in arrivals.values() if d]
    day1 = max(known) if known else request.start_date
    buffer_nights = {gid: (day1 - d).days if d and day1 else 0 for gid, d in arrivals.items()}

    ends = [o.legs[-1].destination if o.legs else o.route.split("-")[-1] for o in chosen.values() if o]
    meeting_point = ends[0] if ends else (request.stops[0].place if request.stops else "destination")

    waiting = [g.origin for g in request.groups if buffer_nights.get(g.id)]
    if len(request.groups) == 1:
        note = "Everyone travels together."
    elif waiting:
        note = f"Travellers from {', '.join(waiting)} arrive early and wait {max(buffer_nights.values())} night(s)."
    elif len(known) == len(request.groups):
        note = "All groups arrive the same day."
    else:
        note = f"Each group travels on its own; book arrivals at {meeting_point} for the same day."
    options = [MeetingOption(kind="meet_at_destination", place=meeting_point, note=note)]

    routes = [o.route for o in chosen.values() if o]
    if len(request.groups) > 1 and len(routes) == len(request.groups):
        shared = set(hubs(routes[0])).intersection(*(hubs(r) for r in routes[1:]))
        options += [MeetingOption(kind="meet_at_hub_fly_together", place=hub,
                                  note=f"Every group passes through {hub}: meet there and travel in together.")
                    for hub in sorted(shared)]

    return MeetingPlan(meeting_point=meeting_point, day1=day1, arrivals=arrivals,
                       buffer_nights=buffer_nights, options=options)


def meeting_node(state: dict) -> dict:
    plan = plan_meeting(state["request"], list((state.get("group_results") or {}).values()))
    options = ", ".join(f"{o.kind.replace('_', ' ')} ({o.place})" for o in plan.options)
    return {"meeting": plan, "events": [event("meeting_plan", "planned", f"Day 1 {plan.day1 or 'TBC'}; {options}")]}
