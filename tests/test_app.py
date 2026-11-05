from pathlib import Path

from streamlit.testing.v1 import AppTest

from builders import person, trip

APP = str(Path(__file__).parents[1] / "app.py")


def test_app_starts_on_the_trip_form():
    at = AppTest.from_file(APP).run(timeout=30)
    assert not at.exception
    assert at.title[0].value == "TravelEx"
    assert at.button[0].label == "Plan my trip"


def test_clarify_and_alternatives_pauses_render():
    request = trip(person("Adult 1", passport=None, origin="Dubai"), month="2027-10")
    at = AppTest.from_file(APP)
    at.session_state["stage"] = "clarify"
    at.session_state["pause"] = {"kind": "clarify", "question": "Which passports do you hold?", "missing": [],
                                 "ideas": [], "request": request.model_dump(mode="json")}
    for key, value in {"thread_id": "t", "events": [], "report_dir": None, "error": None, "auto_approve": False}.items():
        at.session_state[key] = value
    at.run(timeout=30)
    assert not at.exception
    assert "Which passports" in at.info[0].value

    at.text_input[0].input("Indian passports")  # typed but not sent: the next question must not inherit it
    at.session_state["pause_id"] = at.session_state["pause_id"] + 1 if "pause_id" in at.session_state else 1
    at.session_state["pause"] = {**at.session_state["pause"], "question": "Which month?"}
    at.run(timeout=30)
    assert not at.exception
    assert at.text_input[0].value == ""

    at.session_state["stage"] = "alternatives"
    at.session_state["pause"] = {"kind": "alternatives", "current": "Bhutan", "problems": ["Over budget by $500"],
                                 "alternatives": [{"n": 1, "title": "Nepal", "why": "No SDF", "fixes": True,
                                                   "summary": "Nepal [go to Nepal]: 900 USD, fits the budget"}]}
    at.run(timeout=30)
    assert not at.exception
    assert at.radio[0].options == ["✓ Nepal [go to Nepal]: 900 USD, fits the budget"]


def test_the_trip_card_offers_a_hotel_level():
    for asked, shown in [(None, "Let TravelEx choose"), ("luxury", "Luxury")]:
        request = trip(person("Adult 1", origin="Dubai"), month="2027-10", lodging_style=asked)
        at = AppTest.from_file(APP)
        at.session_state["stage"] = "clarify"
        at.session_state["pause"] = {"kind": "clarify", "question": None, "missing": [], "ideas": [],
                                     "request": request.model_dump(mode="json")}
        for key, value in {"thread_id": "t", "events": [], "report_dir": None, "error": None,
                           "auto_approve": False}.items():
            at.session_state[key] = value
        at.run(timeout=30)
        assert not at.exception
        assert at.selectbox[0].label == "Hotels" and at.selectbox[0].value == shown
        assert at.selectbox[0].options == ["Let TravelEx choose", "Budget", "Mid-range", "Luxury"]


def test_a_finished_plan_takes_comments(tmp_path):
    for name in ("report.md", "report.html", "report.json"):
        (tmp_path / name).write_text("<p>plan</p>", encoding="utf-8")
    at = AppTest.from_file(APP)
    at.session_state["stage"] = "review"
    at.session_state["pause"] = {"kind": "review", "revision": 1, "max_revisions": 3, "report_dir": str(tmp_path)}
    for key, value in {"thread_id": "t", "events": [], "report_dir": str(tmp_path), "error": None,
                       "auto_approve": False, "versions": [str(tmp_path / "v1"), str(tmp_path)]}.items():
        at.session_state[key] = value
    (tmp_path / "v1").mkdir()
    (tmp_path / "v1" / "report.html").write_text("<p>v1</p>", encoding="utf-8")
    at.run(timeout=30)
    assert not at.exception
    assert at.text_area[0].label.startswith("Want anything changed?")
    assert any(b.label == "Rework my plan" for b in at.button)
    assert "2 more times" in at.caption[0].value
    assert at.expander[0].label == "Earlier versions (1)"



def test_the_handoff_pause_renders():
    at = AppTest.from_file(APP)
    at.session_state["stage"] = "handoff"
    at.session_state["pause"] = {"kind": "handoff", "issues": ["No route found from Chennai to Bali."]}
    for key, value in {"thread_id": "t", "events": [], "report_dir": None, "error": None,
                       "auto_approve": False}.items():
        at.session_state[key] = value
    at.run(timeout=30)
    assert not at.exception
    assert at.warning[0].value == "No route found from Chennai to Bali."
    assert at.text_input[0].label == "Anything that helps? (optional)"


def test_the_trip_card_offers_the_ideas_to_pick():
    from travelex.schemas import TripRequest
    request = TripRequest(travellers=[person("A")], month="2027-03", destination_ideas=["Bali", "Phuket"])
    at = AppTest.from_file(APP)
    at.session_state["stage"] = "clarify"
    at.session_state["pause"] = {"kind": "clarify", "question": "Where would you like to go?", "missing": [],
                                 "ideas": ["Bali", "Phuket"], "request": request.model_dump(mode="json")}
    for key, value in {"thread_id": "t", "events": [], "report_dir": None, "error": None,
                       "auto_approve": False}.items():
        at.session_state[key] = value
    at.run(timeout=30)
    assert not at.exception
    assert at.radio[0].label == "Pick a destination" and at.radio[0].options == ["Bali", "Phuket"]
