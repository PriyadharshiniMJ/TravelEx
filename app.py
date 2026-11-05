"""TravelEx web UI: streamlit run app.py"""
from datetime import date
from pathlib import Path
from uuid import uuid4

import streamlit as st
import streamlit.components.v1 as components
from langgraph.types import Command

from travelex.agents.intake import AFFIRMATIVE, describe
from travelex.graph import PREVIEW_AFTER, build_graph, initial_state, open_checkpointer, run_config, settled
from travelex.report_html import render_preview_html
from travelex.schemas import TripRequest

st.set_page_config(page_title="TravelEx", page_icon="🧭", layout="wide")

LABELS = {
    "intake": "Understanding your trip", "clarify": "Trip card", "season_check": "Checking the season",
    "transport": "Finding routes", "visa_check": "Checking visas", "group_research": "Finding routes",
    "destination_costs": "Pricing each stop", "meeting_plan": "Planning where you meet",
    "feasibility": "Checking the budget", "alternatives": "Looking for alternatives",
    "choose_alternative": "Your choice", "itinerary_planner": "Planning the days", "food_agent": "Finding food",
    "qa": "Final check", "save_report": "Saving the plan", "review_plan": "Your changes",
    "fill_gaps": "Filling research gaps", "ask_traveller": "Your input",
}
PLAN_HEIGHT = 1400  # the plan scrolls inside its own frame
PREVIEW_HEIGHT = 900
HOTELS = {"Let TravelEx choose": None, "Budget": "budget", "Mid-range": "mid-range", "Luxury": "luxury"}
DOWNLOADS = [("report.md", "text/markdown"), ("report.html", "text/html"), ("report.json", "application/json")]
PLACEHOLDER = "Where to, who's going, from where, when, and your budget"


def md_safe(text) -> str:
    """Streamlit renders $...$ as LaTeX, and prices are full of dollar signs."""
    return str(text).replace("$", "\\$")


@st.cache_resource
def get_graph():
    """One compiled graph and SQLite connection per server process, shared by every rerun."""
    return build_graph(open_checkpointer())


def reset() -> None:
    st.session_state.pop("hotels", None)
    st.session_state.update(stage="input", thread_id=None, pause=None, events=[], report_dir=None, error=None,
                            versions=[],
                            pause_id=st.session_state.get("pause_id", 0) + 1)


def fresh(name: str) -> str:
    """A widget key that is new for every pause, so each question opens with an empty answer box
    (Streamlit keeps a widget's value for as long as its key stays the same)."""
    return f"{name}-{st.session_state.get('pause_id', 0)}"


def event_line(e: dict) -> str:
    seconds = f" · {e['ms'] / 1000:.1f}s" if e.get("ms") else ""
    queries = "".join(f"\n- `{q}`" for q in e.get("queries") or [])
    return md_safe(f"**{e['action']}** {e.get('detail', '')}{seconds}{queries}")


def advance(payload) -> None:
    """Stream the graph until it pauses or finishes, showing each agent's steps live, then rerun."""
    graph = get_graph()
    config = run_config(st.session_state.thread_id, st.session_state.auto_approve)
    boxes, pause = {}, None
    preview = st.empty()  # above the progress boxes: the trip at a glance, once the research is done
    try:
        for namespace, update in graph.stream(payload, config, stream_mode="updates", subgraphs=True):
            for node, output in update.items():
                if node == "__interrupt__":
                    pause = output[0].value
                elif isinstance(output, dict):
                    if not namespace and node in PREVIEW_AFTER:
                        if view := settled(node, graph.get_state(config).values, output):  # never a plan about to change
                            show_preview(preview, view)
                    nested = not namespace and node == "group_research"  # its events were streamed already
                    for e in output.get("events") or []:
                        if nested and e["agent"] != "group_research":
                            continue
                        st.session_state.events.append(e)
                        label = LABELS.get(e["agent"], e["agent"])
                        if label not in boxes:
                            boxes[label] = st.status(label)
                        boxes[label].markdown(event_line(e))
    except Exception as e:  # show the failure, keep the trace so far
        st.session_state.update(stage="error", error=f"{type(e).__name__}: {e}")
    else:
        for box in boxes.values():
            box.update(state="complete")
        values = graph.get_state(config).values
        if pause:
            st.session_state.update(stage=pause["kind"], pause=pause, pause_id=st.session_state.get("pause_id", 0) + 1)
            if pause["kind"] == "review":  # the plan is saved and open for comments
                st.session_state.report_dir = values["report_dir"]
                if values["report_dir"] not in st.session_state.versions:
                    st.session_state.versions.append(values["report_dir"])
        elif values.get("report_dir"):
            st.session_state.update(stage="done", report_dir=values["report_dir"])
        else:
            st.session_state.update(stage="error", error="No destination was chosen, so there is nothing to plan.")
    st.rerun()


def show_preview(slot, state: dict) -> None:
    if not (state.get("request") and state.get("feasibility")):
        return
    with slot.container():
        st.success("Here's your trip at a glance. The day-by-day plan and food guide are on their way.")
        components.html(render_preview_html(state), height=PREVIEW_HEIGHT, scrolling=True)


def input_stage() -> None:
    st.title("TravelEx")
    st.caption("Plan a trip for anyone, from anywhere, to anywhere: routes, visas, costs, a day-by-day plan "
               "and food to try, with the sources behind them.")
    with st.form(fresh("query")):
        query = st.text_area("Describe your trip in your own words", placeholder=PLACEHOLDER, height=120,
                             key=fresh("query-text"))
        auto = st.toggle("Don't ask me anything: accept the trip card and pick the best alternative myself")
        submitted = st.form_submit_button("Plan my trip", type="primary")
    if submitted and query.strip():
        st.session_state.update(thread_id=uuid4().hex, auto_approve=auto, events=[])
        advance(initial_state(query.strip(), date.today()))


def clarify_stage(pause: dict) -> None:
    request = TripRequest.model_validate(pause["request"])
    st.subheader("Check your trip")
    st.markdown(md_safe(f"**{describe(request)}**"))
    labels = pause.get("idea_labels") or {}
    if pause.get("ideas") and request.stops:
        st.markdown(md_safe("Ideas: " + ", ".join(pause["ideas"])))
    if pause.get("question"):
        st.info(md_safe(pause["question"]))
    wanted = st.session_state.get("hotels", request.lodging_style)  # a pick survives answering a question
    with st.form(fresh("answer"), clear_on_submit=True):
        place = None
        if pause.get("ideas") and not request.stops:  # nothing to plan without a place: offer the ideas to click
            place = st.radio("Pick a destination", pause["ideas"], index=None, key=fresh("idea"),
                             format_func=lambda p: md_safe(labels.get(p, p)), help="Or type somewhere else below.")
        answer = st.text_input("Answer, or correct anything above (e.g. \"we all have Indian passports\")",
                               key=fresh("answer-text"),
                               help="Leave it empty, or type yes, to plan the trip as shown.")
        hotels = st.selectbox("Hotels", list(HOTELS), index=list(HOTELS.values()).index(wanted), key=fresh("hotels"),
                              help="TravelEx picks the most comfortable hotels that leave room in your budget "
                                   "for activities. The plan shows what the other levels would cost.")
        if st.form_submit_button("Continue", type="primary"):  # one button: nothing typed is ever lost
            if HOTELS[hotels] != wanted:
                st.session_state.hotels = HOTELS[hotels]
            typed = " ".join(filter(None, [f"Let's go to {place}." if place else "", answer.strip()]))
            if typed and not AFFIRMATIVE.fullmatch(typed):
                advance(Command(resume=typed))  # back to intake; the card, and this pick, come back
            else:
                advance(Command(resume={**pause["request"], "lodging_style": HOTELS[hotels]}))


def alternatives_stage(pause: dict) -> None:
    st.subheader("Your plan doesn't quite work")
    st.markdown(md_safe(f"**{pause['current']}**"))
    current = get_graph().get_state(run_config(st.session_state.thread_id)).values
    if current.get("feasibility"):
        with st.expander("Your current plan at a glance"):
            components.html(render_preview_html(current), height=PREVIEW_HEIGHT, scrolling=True)
    for problem in pause["problems"]:
        st.warning(md_safe(problem))
    options = {a["n"]: a for a in pause["alternatives"]}
    with st.form(fresh("alternatives"), clear_on_submit=True):
        n = st.radio("Alternatives, each researched and costed", list(options),
                     format_func=lambda i: f"{'✓ ' if options[i]['fixes'] else ''}{options[i]['summary']}",
                     captions=[md_safe(a["why"]) for a in options.values()], key=fresh("alternatives-choice"))
        feedback = st.text_input("Or tell us what you'd prefer instead", key=fresh("feedback-text"))
        pick, keep, other = st.columns(3)
        if pick.form_submit_button("Switch to this one", type="primary"):
            advance(Command(resume=n))
        if keep.form_submit_button("Keep my plan"):
            advance(Command(resume="keep"))
        if other.form_submit_button("Show me others") and feedback.strip():
            advance(Command(resume=feedback.strip()))


def done_stage() -> None:
    folder = Path(st.session_state.report_dir)
    report = (folder / "report.md").read_text(encoding="utf-8")
    columns = st.columns(len(DOWNLOADS) + 1)
    for column, (name, mime) in zip(columns, DOWNLOADS):
        column.download_button(f"Download {name}", (folder / name).read_bytes(), file_name=name, mime=mime)
    if columns[-1].button("Plan another trip"):
        reset()
        st.rerun()
    if st.session_state.stage == "review":
        comment_form(st.session_state.pause)
    earlier = st.session_state.get("versions", [])[:-1]
    if earlier:
        with st.expander(f"Earlier versions ({len(earlier)})"):
            for n, old in enumerate(earlier, 1):
                st.download_button(f"Version {n} (report.html)", (Path(old) / "report.html").read_bytes(),
                                   file_name=f"plan-v{n}.html", mime="text/html", key=f"v{n}-{old}")
    plan, text = st.tabs(["Your plan", "Plain text"])
    with plan:
        components.html((folder / "report.html").read_text(encoding="utf-8"), height=PLAN_HEIGHT, scrolling=True)
    with text:
        st.markdown(md_safe(report), unsafe_allow_html=True)  # the report folds its cost lines into <details>


def handoff_stage(pause: dict) -> None:
    """The agents are stuck on something that could stop the trip: let the traveller help, or go on."""
    st.subheader("We need your help with this")
    st.caption("Our research couldn't settle these. A hint lets the agents try again; otherwise we'll plan the "
               "trip as it is and list them under \"Things to double-check\".")
    for issue in pause["issues"]:
        st.warning(md_safe(issue))
    with st.form(fresh("handoff"), clear_on_submit=True):
        hint = st.text_input("Anything that helps? (optional)", key=fresh("handoff-text"),
                             placeholder="e.g. try flights via Singapore, or: I already hold a valid US visa")
        if st.form_submit_button("Continue", type="primary"):
            advance(Command(resume=hint.strip()))


def comment_form(pause: dict) -> None:
    """Rework the finished plan from a comment: the days, or the trip itself (dates, places, hotels, budget)."""
    left = pause["max_revisions"] - pause["revision"]
    with st.form(fresh("comment"), clear_on_submit=True):
        comment = st.text_area("Want anything changed? Tell us and we'll rework the plan", height=80,
                               key=fresh("comment-text"),
                               placeholder="e.g. make day 3 more relaxed, add a northern lights tour, "
                                           "skip Bergen, or make it 5 days")
        if st.form_submit_button("Rework my plan", type="primary") and comment.strip():
            advance(Command(resume=comment.strip()))
    st.caption(f"You can ask for changes {left} more time{'s' if left != 1 else ''}.")


def sidebar() -> None:
    with st.sidebar:
        st.header("What the agents did")
        for e in st.session_state.events:
            st.markdown(f"**{LABELS.get(e['agent'], e['agent'])}** · {event_line(e)}")
        if st.session_state.stage != "input" and st.button("Start over"):
            reset()
            st.rerun()


def main() -> None:
    if "stage" not in st.session_state:
        reset()
    sidebar()
    stage = st.session_state.stage
    if stage == "input":
        input_stage()
    elif stage == "clarify":
        clarify_stage(st.session_state.pause)
    elif stage == "alternatives":
        alternatives_stage(st.session_state.pause)
    elif stage == "handoff":
        handoff_stage(st.session_state.pause)
    elif stage in ("done", "review"):
        done_stage()
    else:
        st.error(st.session_state.error)
        if st.button("Start over"):
            reset()
            st.rerun()


main()
