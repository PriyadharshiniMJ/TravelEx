from datetime import date, datetime

from builders import option, person, result, trip
from travelex.agents.meeting import plan_meeting
from travelex.schemas import Stop

ZANZIBAR = Stop(place="Zanzibar", country="TZ")


def test_day1_is_the_latest_arrival_and_early_groups_wait():
    request = trip(person("Sydney 1", passport="AU", origin="Sydney"),
                   person("Auckland friend", passport="NZ", origin="Auckland"), stops=[Stop(place="Bali", country="ID")],
                   start_date=date(2027, 7, 9), end_date=date(2027, 7, 16), exact_dates=True)
    results = [result("G1", option("SYD-DPS", 700, arrive=datetime(2027, 7, 9, 15))),
               result("G2", option("AKL-BNE-DPS", 900, arrive=datetime(2027, 7, 10, 9)))]
    plan = plan_meeting(request, results)
    assert plan.day1 == date(2027, 7, 10)
    assert plan.buffer_nights == {"G1": 1, "G2": 0}
    assert plan.meeting_point == "DPS"
    assert "Sydney" in plan.options[0].note


def test_shared_hub_offers_meet_and_travel_together():
    request = trip(person("Family"), person("Grandma", age=68, origin="Dubai"), start_date=date(2027, 10, 5),
                   end_date=date(2027, 10, 10), exact_dates=True)
    results = [result("G1", option("MAA-CCU-PBH", 300, arrive=datetime(2027, 10, 5, 12))),
               result("G2", option("DXB-CCU-PBH", 400, arrive=datetime(2027, 10, 5, 12)))]
    plan = plan_meeting(request, results)
    assert [(o.kind, o.place) for o in plan.options] == [
        ("meet_at_destination", "PBH"), ("meet_at_hub_fly_together", "CCU")]
    assert plan.options[0].note == "All groups arrive the same day."


def test_one_groups_origin_can_be_the_hub():
    request = trip(person("Lagos", passport="NG", origin="Lagos"), person("Nairobi", passport="KE", origin="Nairobi"),
                   stops=[ZANZIBAR])
    results = [result("G1", option("LOS-NBO-ZNZ", 650)), result("G2", option("NBO-ZNZ", 200))]
    assert [o.place for o in plan_meeting(request, results).options] == ["ZNZ", "NBO"]


def test_single_group_travels_together():
    plan = plan_meeting(trip(person("A"), person("B")), [result("G1", option("MAA-CCU-PBH", 300))])
    assert [o.kind for o in plan.options] == ["meet_at_destination"]
    assert plan.options[0].note == "Everyone travels together."


def test_reports_show_the_meeting_note_only_when_groups_must_meet():
    from travelex.report import section_getting_there
    from travelex.report_html import getting_there
    for travellers, shown in [((person("A"),), False), ((person("A"), person("B", origin="Dubai")), True)]:
        request = trip(*travellers)
        results = {g.id: result(g.id, option("MAA-CCU-PBH", 300)) for g in request.groups}
        state = {"request": request, "group_results": results, "meeting": plan_meeting(request, list(results.values()))}
        assert ("🤝" in getting_there(state)) is shown
        assert any(line.startswith("**Meeting:**") for line in section_getting_there(state)) is shown


def test_unknown_arrivals_fall_back_to_the_start_date_and_first_stop():
    request = trip(person("A"), stops=[ZANZIBAR], start_date=date(2027, 8, 10), end_date=date(2027, 8, 16))
    plan = plan_meeting(request, [])
    assert plan.day1 == date(2027, 8, 10)
    assert plan.arrivals == {"G1": None}
    assert plan.meeting_point == "Zanzibar"


def test_invented_or_month_only_arrivals_are_ignored():
    results = [result("G1", option("MAA-CCU-PBH", 300, arrive=datetime(1201, 11, 25))),
               result("G2", option("DXB-CCU-PBH", 400, arrive=datetime(2027, 10, 5)))]
    exact = trip(person("A"), person("B", origin="Dubai"), start_date=date(2027, 10, 5), end_date=date(2027, 10, 9))
    assert plan_meeting(exact, results).arrivals == {"G1": None, "G2": date(2027, 10, 5)}
    month_only = trip(person("A"), person("B", origin="Dubai"), month="2027-10")
    plan = plan_meeting(month_only, results)
    assert plan.arrivals == {"G1": None, "G2": None}
    assert plan.options[0].note == "Each group travels on its own; book arrivals at PBH for the same day."
