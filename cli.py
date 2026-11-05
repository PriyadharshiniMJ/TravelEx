"""Run TravelEx from the terminal: python cli.py "<your trip, in your own words>" [--auto] [--today YYYY-MM-DD]"""
import argparse
import sys
from datetime import date
from pathlib import Path
from uuid import uuid4

from langchain_core.tracers.langchain import wait_for_all_tracers
from langgraph.types import Command

from travelex.agents.intake import describe
from travelex.agents.review import finished
from travelex.report import hotel_note, plan_currency, shown, who_says
from travelex.graph import build_graph, initial_state, open_checkpointer, run_config
from travelex.schemas import TripRequest

ICONS = {"good": "✓", "ok_with_caveats": "!", "avoid": "✗", "unknown": "?"}


def ask(payload: dict) -> str | dict:
    if payload["kind"] == "alternatives":
        return ask_alternatives(payload)
    if payload["kind"] == "handoff":
        return ask_handoff(payload)
    return ask_clarify(payload)


def ask_handoff(payload: dict) -> str:
    """What the agents couldn't settle. A hint lets them try again; Enter goes on with the plan as it is."""
    print("\n  Our research couldn't settle these:")
    for issue in payload["issues"]:
        print(f"  ! {issue}")
    return input("\n  Anything that helps (e.g. 'try flights via Singapore'), or Enter to continue: ").strip()


def ask_alternatives(payload: dict) -> str:
    """Show what's wrong and the costed alternatives. A number picks one; Enter keeps the plan; text is feedback."""
    print(f"\n  Your plan: {payload['current']}")
    for problem in payload["problems"]:
        print(f"  ! {problem}")
    print("\n  Alternatives:")
    for a in payload["alternatives"]:
        print(f"    {a['n']}. {'✓' if a['fixes'] else ' '} {a['summary']}\n         {a['why']}")
    return input("\n  Pick a number, Enter to keep your plan, or say what you'd prefer: ").strip()


def ask_clarify(payload: dict) -> str | dict:
    """Show the trip card and question. Enter accepts the card; anything else is an answer."""
    request = TripRequest.model_validate(payload["request"])
    print(f"\n  Trip: {describe(request)}")
    if payload.get("ideas"):
        labels = payload.get("idea_labels") or {}
        print("  Ideas:\n" + "\n".join(f"    - {labels.get(p, p)}" for p in payload["ideas"]))
    if payload.get("question"):
        print(f"\n  {payload['question']}")
    answer = input("\n  Your answer (Enter to accept the trip as shown): ").strip()
    return answer or payload["request"]


def run(query: str, auto: bool, today: date) -> tuple[dict, bool]:
    """Returns the final state, and whether the finished plan was already shown (at the last comment prompt)."""
    graph = build_graph(open_checkpointer())
    thread = uuid4().hex
    config = run_config(thread, auto_approve=auto)
    payload, plan_shown = initial_state(query, today), False
    while True:
        for update in graph.stream(payload, config, stream_mode="updates"):
            for node, change in update.items():
                for e in (change or {}).get("events", []) if isinstance(change, dict) else []:
                    print(f"  [{e['agent']}] {e['action']}: {e['detail']}")
        snapshot = graph.get_state(config)
        if not snapshot.interrupts:
            return snapshot.values, plan_shown
        pause = snapshot.interrupts[0].value
        if pause["kind"] == "review":  # the plan is saved: show it, then take comments
            show_plan(snapshot.values)
            answer = input("\n  Anything to change? Describe it, or press Enter to finish: ").strip()
            plan_shown = finished(answer)  # a comment means a new version still to show
        else:
            answer = ask(pause)
        payload = Command(resume=answer)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--auto", action="store_true", help="never pause for questions")
    parser.add_argument("--today", type=date.fromisoformat, default=date.today(), help="pin today's date")
    args = parser.parse_args()
    try:
        state, plan_shown = run(args.query, args.auto, args.today)
    finally:
        wait_for_all_tracers()  # short-lived process: let LangSmith finish uploading

    if not state.get("request") or not state["request"].stops:
        print("\nNo destination chosen, so there is nothing to plan yet.")
        return 1
    if not plan_shown:
        show_plan(state)
    return 0


def show_plan(state: dict) -> None:
    print_summary(state)
    if plan := state.get("itinerary"):
        print("\nDay by day")
        for d in plan.days:
            when = f" {d.date:%d %b}" if d.date else ""
            print(f"  Day {d.day}{when} · {d.city}: {d.title}" + (" (rest)" if d.rest_day else ""))
    if state.get("report_dir"):
        print(f"\nFull plan: {Path(state['report_dir']) / 'report.md'} (also report.html and report.json)")


def print_summary(state: dict) -> None:
    request = state["request"]
    print(f"\nTrip: {describe(request)}")
    print("\nSeason")
    for place, verdict in (state.get("season") or {}).items():
        why = f" — {'; '.join(verdict.reasons)}" if verdict.reasons else ""
        print(f"  {ICONS[verdict.verdict]} {place}: {verdict.verdict}{why}")

    cur = plan_currency(state)  # every amount in the budget's currency
    groups = {g.id: g for g in request.groups}
    for gid, result in (state.get("group_results") or {}).items():
        print(f"\nFrom {groups[gid].origin} ({', '.join(groups[gid].travellers)})")
        for i, o in enumerate(result.options):
            mark = "→" if i == result.chosen else " "
            print(f"  {mark} {o.route} [{o.mode}] {shown(o.adult_fare.amount, o.adult_fare.currency, cur)} "
                  f"return per adult ({o.fare_basis}){f' — {o.kid_note}' if o.kid_note else ''}")
            for warning in o.warnings:
                print(f"      ! {warning}")
        for v in result.visas:
            print(f"    visa {v.traveller} [{v.passport or '?'}] → {v.country}: {v.summary}")
            for c in (v.rule.citations if v.rule else [])[:2]:
                print(f"      “{c.quote}” — {c.url} ({who_says(c)})")

    f = state.get("feasibility")
    for place, c in (state.get("costs") or {}).items():
        guide = f", guide {shown(c.guide_per_day, c.currency, cur)}/day" if c.guide_per_day else ""
        rate = c.room_rate(f.lodging_style if f else None)
        print(f"\nIn {place}: {shown(rate, c.currency, cur)}/night × {c.rooms} ({c.room_config}), "
              f"food {shown(c.food_per_person_day, c.currency, cur)}/person/day{guide}")
        if c.onward:
            print(f"  then {c.onward.route} [{c.onward.mode}] "
                  f"{shown(c.onward.adult_fare.amount, c.onward.adult_fare.currency, cur)} one way per adult")
        for line in (f.lines if f else []):
            if line.category == "destination_fee" and f"({place})" in line.label:
                print(f"  fee: {line.traveller} — {line.label.split(f' ({place})')[0]}: "
                      f"{line.amount_report:,.0f} {f.currency}")

    if (meeting := state.get("meeting")) and len(state["request"].groups) > 1:
        print(f"\nMeeting: Day 1 {meeting.day1 or 'TBC'} — " + "; ".join(f"{o.place}: {o.note}" for o in meeting.options))
    if f:
        status = f.status()
        budget = f" of {f.budget:,.0f}" if f.budget is not None else ""
        print(f"\nBudget: {f.counted:,.0f} {f.currency}{budget} — {status}")
        if f.lodging_style:
            print(f"  {hotel_note(f)}")
        if f.excluded:
            print(f"  flights to and from home ({f.excluded:,.0f} {f.currency}) are outside the budget; "
                  f"whole trip {f.total:,.0f} {f.currency}")
        for who, amount in f.per_traveller.items():
            print(f"  {who}: {amount:,.0f} {f.currency}")
        if f.shared:
            print(f"  shared (rooms, food, guide, vehicle): {f.shared:,.0f} {f.currency}")
        for reason in f.reasons + f.unpriced:
            print(f"  ! {reason}")
        if f.fx_source:
            print(f"  rates: {f.fx_source}, {f.fx_as_of}")
    for error in state.get("errors") or []:
        print(f"  error: {error}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
