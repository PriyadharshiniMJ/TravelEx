import pytest

from builders import option, person, result, stop_costs, trip, visa
from travelex.agents.feasibility import assess
from travelex.schemas import AgeBand, FeeRule, Money, Stop, VisaCheck, VisaRule
from travelex.tools import fx


def assert_adds_up(f):
    assert f.total == pytest.approx(sum(l.amount_report for l in f.lines), abs=0.01)
    assert f.total == pytest.approx(sum(f.per_traveller.values()) + f.shared, abs=0.05)


def fares(f) -> dict[str, float]:
    """Each traveller's transport: the way out and the way home are separate lines."""
    total: dict[str, float] = {}
    for l in f.lines:
        if l.category == "transport":
            total[l.traveller] = total.get(l.traveller, 0) + l.amount
    return total


def test_golden_c2_over_budget_because_of_sdf():
    travellers = [person(f"A{i}", passport="US", origin="Dubai") for i in range(1, 4)]
    request = trip(*travellers, nights=None, duration_range=(4, 5), budget=(800, "USD"))
    f = assess(request, [result("G1", option("DXB-DEL-PBH", 450))], {"Bhutan": stop_costs()})
    assert sum(l.amount_report for l in f.lines if l.category == "destination_fee") == 1200
    assert not f.ok and f.gap > 0
    assert any("SDF" in r for r in f.reasons)
    assert_adds_up(f)


def test_golden_c1_mixed_passports_cost_per_traveller():
    ae, ind = person("AE adult", passport="AE", origin="Dubai"), person("IN adult", passport="IN")
    request = trip(ae, ind, nights=4, budget=(1500, "USD"))
    results = [result("G1", option("DXB-DEL-PBH", 420), visas=[visa(ae)]),
               result("G2", option("MAA-CCU-PBH", 310), visas=[visa(ind)])]
    f = assess(request, results, {"Bhutan": stop_costs()})

    by = {(l.category, l.traveller): l for l in f.lines}
    assert by[("visa", "AE adult")].amount == 40
    assert ("visa", "IN adult") not in by  # the entry permit is free
    assert by[("destination_fee", "AE adult")].amount == 400
    sdf_in = by[("destination_fee", "IN adult")]
    assert (sdf_in.amount, sdf_in.currency) == (4800, "INR")
    assert sdf_in.amount_report == pytest.approx(fx.convert(4800, "INR", "USD"), abs=0.01)
    assert f.per_traveller["AE adult"] == pytest.approx(420 + 40 + 400)
    assert_adds_up(f)


def test_golden_c3_galapagos_park_fee_per_trip_with_child_band():
    park = FeeRule(country="EC", place="Galápagos", name="Galápagos National Park entry fee", basis="per_trip",
                   amount=200, currency="USD",
                   age_bands=[AgeBand(min_age=0, max_age=11, share=0.5, label="under 12 (50%)")])
    family = [person("A1", passport="DE", origin="Frankfurt"), person("A2", passport="DE", origin="Frankfurt"),
              person("K4", type="child", age=4, passport="DE", origin="Frankfurt"),
              person("K9", type="child", age=9, passport="DE", origin="Frankfurt")]
    stops = [Stop(place="Ecuador mainland", country="EC", nights=5), Stop(place="Galápagos", country="EC", nights=4)]
    request = trip(*family, stops=stops, nights=None, budget=(12000, "EUR"))
    costs = {"Ecuador mainland": stop_costs("Ecuador mainland", fee_country=None),
             "Galápagos": stop_costs("Galápagos", fee_country=None, fees=[park])}
    f = assess(request, [result("G1", option("FRA-MAD-UIO", 1100))], costs)
    fee_usd = sum(l.amount for l in f.lines if l.category == "destination_fee")
    assert fee_usd == 600  # 2 × 200 + 2 × 100, charged once per trip, not per night
    assert f.currency == "EUR" and "USD->EUR" in f.conversions
    assert_adds_up(f)


def test_golden_c4_one_schengen_visa_covers_three_countries():
    a, b = person("A", origin="Delhi"), person("B", origin="Delhi")
    schengen = {c: VisaRule(passport="IN", country=c, zone="Schengen", visa_required=True, document="visa",
                            fee=90, fee_currency="EUR") for c in ("FR", "IT", "CH")}
    visas = [VisaCheck(traveller=t.label, passport="IN", country=c, rule=r) for t in (a, b) for c, r in schengen.items()]
    stops = [Stop(place="France", country="FR"), Stop(place="Italy", country="IT"),
             Stop(place="Switzerland", country="CH")]
    request = trip(a, b, stops=stops, nights=9, budget=(400000, "INR"))
    costs = {s.place: stop_costs(s.place, currency="EUR", fee_country=None) for s in stops}
    f = assess(request, [result("G1", option("DEL-CDG", 700), visas=visas)], costs)
    visa_fees = [l for l in f.lines if l.category == "visa"]
    assert [(l.traveller, l.amount, l.currency) for l in visa_fees] == [("A", 90, "EUR"), ("B", 90, "EUR")]
    lodging_nights = sorted(l.label for l in f.lines if l.category == "lodging")
    assert lodging_nights == ["France: mid-range hotel, 1 double, 3 nights", "Italy: mid-range hotel, 1 double, 3 nights",
                              "Switzerland: mid-range hotel, 1 double, 3 nights"]
    assert_adds_up(f)


def test_golden_c5_budget_in_aed_converts_every_line():
    a, b = person("A", origin="Dubai"), person("B", origin="Dubai")
    levy = FeeRule(country="ID", place="Bali", name="Bali tourist levy", basis="per_trip", amount=150000,
                   currency="IDR")
    request = trip(a, b, stops=[Stop(place="Bali", country="ID")], nights=5, budget=(8000, "AED"))
    costs = {"Bali": stop_costs("Bali", currency="IDR", fee_country=None, fees=[levy],
                                lodging_per_room_night=900000, food_per_person_day=300000)}
    f = assess(request, [result("G1", option("DXB-DPS", 2200, currency="AED"))], costs)
    assert f.currency == "AED"
    assert "IDR->AED" in f.conversions
    assert (f.fx_as_of, f.fx_source) == (fx.SEED_AS_OF, fx.SEED_SOURCE)
    assert_adds_up(f)


def test_within_budget_is_ok_with_no_reasons():
    f = assess(trip(person("A")), [result("G1", option("MAA-CCU-PBH", 300))], {"Bhutan": stop_costs()})
    assert f.ok and f.reasons == [] and f.gap < 0
    assert f.conversions == ["INR->USD"]  # the Indian SDF is priced in rupees


def test_missing_costs_transport_or_destination_is_never_ok():
    request = trip(person("A"), person("B", origin="Dubai"), budget=(100000, "USD"))
    f = assess(request, [result("G1", option("MAA-CCU-PBH", 300))], {})
    assert not f.ok and not f.complete and f.reasons == []  # missing prices are not a budget problem
    assert "No transport found from Dubai." in f.unpriced
    assert "Costs for Bhutan are missing." in f.unpriced
    f = assess(trip(person("A"), stops=[]), [result("G1", option("MAA-CCU-PBH", 300))], {})
    assert "No destination has been chosen yet." in f.reasons


def test_child_and_infant_fares_and_food():
    kid, baby = person("Kid", type="child", age=8), person("Baby", type="infant")
    request = trip(person("Mum"), kid, baby, nights=2, budget=None)
    quoted = option("MAA-CCU-PBH", 400)
    f = assess(request, [result("G1", quoted)], {"Bhutan": stop_costs(food_per_person_day=10)})
    assert fares(f) == {"Mum": 400, "Kid": 300, "Baby": 40}
    food = next(l for l in f.lines if l.category == "food")
    assert food.amount == 10 * 2 * 3  # 2 eaters (no infant) × 3 days

    with_child_fare = quoted.model_copy(update={"child_fare": Money(amount=250, currency="USD")})
    f = assess(request, [result("G1", with_child_fare)], {"Bhutan": stop_costs()})
    assert fares(f)["Kid"] == 250


def test_a_solo_traveller_pays_for_everything_and_nothing_is_shared():
    from travelex.report import section_budget
    from travelex.report_html import Notes, budget
    request = trip(person("Solo", passport="US", origin="Dubai"))
    f = assess(request, [result("G1", option("DXB-DEL-PBH", 450))], {"Bhutan": stop_costs()})
    assert f.shared == 0 and all(l.traveller == "Solo" for l in f.lines)
    assert f.per_traveller["Solo"] == pytest.approx(f.total, abs=0.05)
    state = {"request": request, "feasibility": f}
    assert "Shared" not in "\n".join(section_budget(state)) and "Shared" not in budget(state, Notes({}))


def test_the_way_out_and_the_way_home_are_separate_cost_lines():
    from travelex.agents.transport import ONE_WAY_DOUBLED
    request = trip(person("A"))
    f = assess(request, [result("G1", option("MAA-CCU-PBH", 500))], {"Bhutan": stop_costs()})
    assert [(l.label, l.amount) for l in f.lines if l.category == "transport"] == [
        ("Getting there: MAA-CCU-PBH (flight, half the return fare)", 250),
        ("Getting home: PBH-CCU-MAA (flight, half the return fare)", 250)]

    two_stops = trip(person("A"), stops=[Stop(place="Lisbon", country="PT"), Stop(place="Porto", country="PT")])
    doubled = option("MAA-LIS", 800, warnings=[ONE_WAY_DOUBLED])
    f = assess(two_stops, [result("G1", doubled)], {"Lisbon": stop_costs("Lisbon", fee_country=None),
                                                    "Porto": stop_costs("Porto", fee_country=None)})
    assert [l.label for l in f.lines if l.category == "transport"] == [
        "Getting there: MAA-LIS (flight, one-way fare)", "Getting home: Porto to Chennai (flight, one-way fare)"]


def test_a_budget_excluding_airfare_leaves_the_flights_out():
    from travelex.report import section_budget
    request = trip(person("Solo", passport="US", origin="Dallas"), budget=(1000, "USD"), flights_in_budget=False)
    f = assess(request, [result("G1", option("DFW-PBH", 900))], {"Bhutan": stop_costs()})
    assert f.excluded == 900 and f.counted == pytest.approx(f.total - 900)
    assert f.gap == pytest.approx(f.counted - 1000)
    with_flights = assess(request.model_copy(update={"flights_in_budget": True}),
                          [result("G1", option("DFW-PBH", 900))], {"Bhutan": stop_costs()})
    assert with_flights.excluded == 0 and with_flights.gap == pytest.approx(f.gap + 900)
    assert "outside your budget" in "\n".join(section_budget({"request": request, "feasibility": f}))


RATES = [{"style": "budget", "per_room_night": 50}, {"style": "mid-range", "per_room_night": 120},
         {"style": "luxury", "per_room_night": 400}]


@pytest.mark.parametrize("budget, style", [(5000, "luxury"), (2000, "mid-range"), (1300, "budget"), (900, "budget")])
def test_hotels_follow_the_budget(budget, style):
    from travelex.agents.feasibility import fit_to_budget
    request = trip(person("A"), person("B"), stops=[Stop(place="Japan", country="JP")], nights=4, budget=(budget, "USD"))
    costs = {"Japan": stop_costs("Japan", fee_country=None, lodging_rates=RATES)}
    f = fit_to_budget(request, [result("G1", option("MAA-NRT", 300))], costs)
    assert (f.lodging_style, f.lodging_chosen) == (style, True)
    assert any(l.label.startswith(f"Japan: {style} hotel") for l in f.lines)


def test_a_hotel_level_the_traveller_asks_for_is_kept_even_over_budget():
    from travelex.agents.feasibility import fit_to_budget
    from travelex.report import hotel_note
    request = trip(person("A"), stops=[Stop(place="Japan", country="JP")], nights=4, budget=(900, "USD"),
                   lodging_style="luxury")
    costs = {"Japan": stop_costs("Japan", fee_country=None, lodging_rates=RATES)}
    f = fit_to_budget(request, [result("G1", option("MAA-NRT", 300))], costs)
    assert (f.lodging_style, f.lodging_chosen, f.ok) == ("luxury", False, False)
    assert [o.style for o in f.lodging_others] == ["mid-range", "budget"]
    assert hotel_note(f).startswith("Hotels: luxury, as you asked. Instead, mid-range hotels would save 1,120 USD")


def test_the_planner_and_the_report_get_the_hotel_level_and_spending_money():
    from travelex.agents.feasibility import fit_to_budget
    from travelex.agents.itinerary import research_brief
    from travelex.report import section_budget
    request = trip(person("A"), person("B"), stops=[Stop(place="Japan", country="JP")], nights=4, budget=(5000, "USD"))
    costs = {"Japan": stop_costs("Japan", fee_country=None, lodging_rates=RATES)}
    f = fit_to_budget(request, [result("G1", option("MAA-NRT", 300))], costs)
    state = {"request": request, "feasibility": f, "costs": costs}
    brief = research_brief(state)
    assert "Hotels: luxury (chosen to fit the budget)" in brief
    assert f"about {f.spare:,.0f} USD" in brief and "per person per day" in brief
    assert f"Hotels: luxury, the most comfortable that fits your budget. {f.spare:,.0f} USD is left" in "\n".join(section_budget(state))


def test_a_car_is_charged_only_when_needed_otherwise_local_transport():
    request = trip(person("Solo"), stops=[Stop(place="Norway", country="NO")], nights=6, budget=(5000, "USD"))
    def transport(**kwargs):
        costs = {"Norway": stop_costs("Norway", fee_country=None, vehicle_per_day=128,
                                      local_transport_per_person_day=20, **kwargs)}
        f = assess(request, [result("G1", option("DFW-OSL", 900))], costs)
        return [(l.category, l.label, l.amount) for l in f.lines if l.category in ("vehicle", "local_transport")]
    assert transport() == [("local_transport", "Norway: local transport, 1 person × 7 days", 140)]
    assert transport(vehicle_needed=True) == [("vehicle", "Norway: car with driver × 7 days", 896)]


@pytest.mark.parametrize("party, rooms", [
    (("adult", "adult", "adult"), 2), (("adult", "adult"), 1), (("adult",), 1),
    (("adult", "adult", "child", "child"), 1), (("adult", "adult", "child", "child", "child"), 2),
    (("adult", "adult", "infant"), 1), (("adult", "adult", "adult", "adult"), 2)])
def test_rooms_needed(party, rooms):
    from travelex.agents.costs import rooms_needed
    assert rooms_needed(trip(*[person(f"P{i}", type=t, age=8 if t == "child" else None)
                               for i, t in enumerate(party)])) == rooms


def test_every_cost_line_says_where_its_amount_comes_from():
    from datetime import datetime
    from travelex.report import section_budget
    from travelex.report_html import Notes, budget
    from travelex.schemas import CostEvidence, Source
    request = trip(person("A", passport="US", origin="Dubai"), nights=4, lodging_style="budget")
    costs = {"Bhutan": stop_costs(lodging_rates=RATES, evidence=[CostEvidence(
        item="budget hotel", source_id="C1.1", quote="Paro guesthouses from $50 a night", amount=50, currency="USD",
        unit="per room-night")])}
    flight = option("DXB-DEL-PBH", 450, source_ids=["G1.2"], evidence="Dubai to Paro return USD 450")
    f = assess(request, [result("G1", flight)], costs)
    cited = {l.category: (l.source_ids, l.urls, l.quote) for l in f.lines}
    assert cited["lodging"] == (["C1.1"], [], "Paro guesthouses from $50 a night")
    assert cited["transport"] == (["G1.2"], [], "Dubai to Paro return USD 450")
    assert cited["food"] == ([], [], "")  # no source states it: an estimate
    sdf = cited["destination_fee"]
    assert sdf[1] and sdf[1][0].startswith("http") and sdf[2] == ""  # the rule's page; no words to quote

    sources = {sid: Source(id=sid, url=f"https://example.org/{sid}", title=sid, kind="web",
                           retrieved_at=datetime(2026, 10, 27), text="x") for sid in ("C1.1", "G1.2")}
    state = {"request": request, "feasibility": f, "sources": sources}
    html = budget(state, Notes(sources))
    assert "<th>Source</th>" in html and 'href="https://example.org/C1.1"' in html and ">estimate<" in html
    assert 'title="“Paro guesthouses from $50 a night” · C1.1">example.org ↗</a>' in html  # a site chip, words on hover
    from travelex.report import Footnotes
    md = "\n".join(section_budget(state, Footnotes(sources)))
    assert "| Amount | Source |" in md and "| estimate |" in md and "[example.org](https://example.org/C1.1)" in md
    assert "Bhutan: budget hotel, 1 double, 4 nights: “Paro guesthouses from $50 a night”" in md


def test_a_pooled_fare_shows_one_site_and_how_many_more():
    from datetime import datetime
    from travelex.report_html import Notes
    from travelex.schemas import CostLine, Source
    sources = {sid: Source(id=sid, url=f"https://{site}/fares", title=site, kind="web",
                           retrieved_at=datetime(2026, 10, 27), text="x")
               for sid, site in (("G1.1", "momondo.com"), ("G1.2", "skyscanner.com"), ("G1.3", "kayak.com"))}
    line = CostLine(category="transport", label="Getting there", amount=300, currency="USD", amount_report=300,
                    source_ids=["G1.1", "G1.2", "G1.3"], quote="Typical return fare from 3 sources")
    chip = Notes(sources).cite_cost(line)
    assert ">momondo.com ↗</a>" in chip and 'title="skyscanner.com, kayak.com">+2</span>' in chip
