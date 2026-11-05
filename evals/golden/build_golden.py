"""Source of the golden dataset. Edit the cases here, then run: python evals/golden/build_golden.py"""
import json
from pathlib import Path

OUT = Path(__file__).with_name("travelex_golden_v1.jsonl")
TODAY = "2026-10-27"


def adult(passport, origin, **kw):
    return {"type": "adult", "passport": passport, "origin": origin, **kw}


def child(age, passport, origin):
    return {"type": "child", "age": age, "passport": passport, "origin": origin}


def stop(place, country, **kw):
    return {"place": place, "country": country, **kw}


def case(id, focus, nodes, query, outputs, resume=(), review="", **inputs):
    return {"inputs": {"query": query, "today": TODAY, "resume": list(resume), **inputs},
            "outputs": outputs,
            "metadata": {"id": id, "focus": focus, "nodes": nodes, "needs_review": bool(review),
                         "review_notes": review}}


A = ("intake_clarify", ["intake", "clarify"])
B = ("transport_meeting", ["group_research.transport_agent", "meeting_plan"])
C = ("visa_costs_feasibility", ["visa_check", "destination_costs", "feasibility"])
D = ("season_alternatives", ["season_check", "alternatives_agent", "choose_alternative"])
E = ("itinerary_food_qa", ["itinerary_planner", "food_agent", "qa"])

BHUTAN_FAMILY = ("Family of 4 (2 adults, kids aged 5 and 11) flying from Chennai, and grandma (68) joining from Dubai. "
                 "All Indian passports. Bhutan, 6 days in October 2027, budget $4000, vegetarian.")
US_TRIO_BHUTAN = "3 adults with US passports flying from Dubai to Bhutan, 4-5 days in October 2027, budget $800."
JAPAN_FAMILY = ("Family of 4 from Toronto: 2 adults and kids aged 6 and 10, all Canadian passports. Japan, Tokyo and "
                "Kyoto, 10 to 20 April 2027. Budget CAD 12,000. Our 6-year-old has a peanut allergy.")

CASES = [
    # ---------- A. intake + clarify ----------
    case("A1", *A, "I want to plan a trip to Bhutan for two adults, one travelling from Dubai and another from "
                   "Chennai. Budget is $1500. Give an itinerary for 4-5 days.",
         {"clarify": {"asked": True, "fields_required": ["passport_country", "travel_month"],
                      "fields_forbidden": ["duration"]},
          "request": {"stops": [stop("Bhutan", "BT")],
                      "travellers": [adult("IN", "Dubai"), adult("IN", "Chennai")], "group_count": 2,
                      "duration_range": [4, 5], "nights": 4, "trimmed_nights": 3, "month": "2027-10",
                      "budget": {"amount": 1500, "currency": "USD"}}},
         resume=["Both have Indian passports. We want to go in October 2027."]),
    case("A2", *A, JAPAN_FAMILY,
         {"clarify": {"asked": False},
          "request": {"stops": [stop("Japan", "JP", cities=["Tokyo", "Kyoto"])],
                      "travellers": [adult("CA", "Toronto"), adult("CA", "Toronto"), child(6, "CA", "Toronto"),
                                     child(10, "CA", "Toronto")],
                      "group_count": 1, "start_date": "2027-04-10", "end_date": "2027-04-20", "nights": 10,
                      "budget": {"amount": 12000, "currency": "CAD"}, "diet": ["peanut allergy"]}}),
    case("A3", *A, "My parents (70 and 72) will fly from Mumbai on Indian passports. My partner and I will come "
                   "from Berlin on German passports. Two weeks in Italy and Switzerland in September 2027, "
                   "total budget €9,000.",
         {"clarify": {"asked": False},
          "request": {"stops": [stop("Italy", "IT"), stop("Switzerland", "CH")],
                      "travellers": [adult("IN", "Mumbai", age=70, needs=["senior"]),
                                     adult("IN", "Mumbai", age=72, needs=["senior"]),
                                     adult("DE", "Berlin"), adult("DE", "Berlin")],
                      "group_count": 2, "month": "2027-09", "nights": 13,
                      "budget": {"amount": 9000, "currency": "EUR"}}}),
    case("A4", *A, "Somewhere warm with beaches for a week in December. 3 friends from London, around £3,000.",
         {"clarify": {"asked": True, "fields_required": ["destination", "passport_country"],
                      "suggests_destinations": True},
          "request": {"stops": [stop("Sri Lanka", "LK")], "travellers": [adult("GB", "London")] * 3,
                      "group_count": 1, "month": "2026-12", "nights": 6,
                      "budget": {"amount": 3000, "currency": "GBP"}}},
         resume=["British passports. Let's do Sri Lanka."]),
    case("A5", *A, "Peru for 2 adults and our 8-month-old baby from São Paulo in March 2026, budget US$4,000.",
         {"clarify": {"asked": True, "fields_required": ["passport_country", "travel_dates"],
                      "flags": ["date_in_past"]},
          "request": {"stops": [stop("Peru", "PE")],
                      "travellers": [adult("BR", "São Paulo"), adult("BR", "São Paulo"),
                                     {"type": "infant", "age_months": 8, "passport": "BR", "origin": "São Paulo"}],
                      "group_count": 1, "month": "2027-03", "nights": 8,
                      "budget": {"amount": 4000, "currency": "USD"}}},
         resume=["Brazilian passports. Make it March 2027, 9 days."]),

    # ---------- B. transport + meeting_plan ----------
    case("B1", *B, "Two adults: one flying from Lagos on a Nigerian passport, one from Nairobi on a Kenyan passport. "
                   "Zanzibar for 6 days in August 2027, budget US$3,000.",
         {"request": {"stops": [stop("Zanzibar", "TZ")], "group_count": 2, "nights": 5},
          "transport": {"groups": [
              {"origin": "Lagos", "min_options": 2,
               "must_include_any_route": ["LOS-NBO-ZNZ", "LOS-ADD-ZNZ", "LOS-DAR-ZNZ", "LOS-DXB-ZNZ"]},
              {"origin": "Nairobi", "min_options": 1, "must_include_any_route": ["NBO-ZNZ"]}]},
          "meeting": {"day1": "latest_arrival", "hub_offered_if_shared": True}},
         review="Confirm current Lagos-Zanzibar connections and NBO-ZNZ direct service."),
    case("B2", *B, BHUTAN_FAMILY,
         {"request": {"stops": [stop("Bhutan", "BT")], "group_count": 2, "nights": 5},
          "transport": {"groups": [{"origin": "Chennai", "travellers": 4, "min_options": 2},
                                   {"origin": "Dubai", "travellers": 1, "min_options": 2}],
                        "rules": {"no_overnight_bus_for": ["child", "senior"], "max_layover_hours": 4}},
          "meeting": {"day1": "latest_arrival", "options_include": ["meet_at_destination"],
                      "hub_offered_if_shared": True},
          "costs": {"child_fares_applied": True}}),
    case("B3", *B, "4 students from Berlin with German passports, cheapest possible way to see Prague and Vienna, "
                   "6 days in May 2027, €1,200 total for all of us.",
         {"request": {"stops": [stop("Prague", "CZ"), stop("Vienna", "AT")], "group_count": 1, "nights": 5},
          "transport": {"groups": [{"origin": "Berlin", "must_include_modes": ["surface"], "cheapest_mode": "surface"}],
                        "intercity_modes_any": ["train", "bus"]},
          "visa": [{"passport": "DE", "country": "CZ", "visa_required": False},
                   {"passport": "DE", "country": "AT", "visa_required": False}]}),
    case("B4", *B, "A couple from Sydney with Australian passports and a friend from Auckland with a New Zealand "
                   "passport, meeting in Bali for 7 days in July 2027, budget AUD 7,000.",
         {"request": {"stops": [stop("Bali", "ID")], "group_count": 2, "nights": 6},
          "transport": {"groups": [
              {"origin": "Sydney", "must_include_any_route": ["SYD-DPS"]},
              {"origin": "Auckland", "must_include_any_route": ["AKL-DPS", "AKL-SYD-DPS", "AKL-BNE-DPS", "AKL-MEL-DPS"]}]},
          "meeting": {"day1": "latest_arrival", "buffer_night_if_arrivals_differ": True}},
         review="Confirm whether Auckland-Denpasar is flown direct in July."),
    case("B5", *B, "Family of 5 from Chicago, one of us uses a wheelchair. US passports. Mexico City and Cancún, "
                   "8 days in March 2027, budget $9,000.",
         {"request": {"stops": [stop("Mexico", "MX", cities=["Mexico City", "Cancún"])], "group_count": 1,
                      "nights": 7, "needs_any": ["wheelchair"]},
          "transport": {"groups": [{"origin": "Chicago", "min_options": 2}],
                        "rules": {"prefer_fewest_transfers": True, "assistance_noted": True},
                        "intercity_any_route": ["MEX-CUN"]},
          "costs": {"vehicle_capacity_min": 5, "vehicle_accessible": True}}),

    # ---------- C. visa_check + destination_costs + feasibility ----------
    case("C1", *C, "Two adults to Bhutan: one from Dubai with a UAE passport, one from Chennai with an Indian "
                   "passport. 5 days in October 2027, budget $1500.",
         {"request": {"nights": 4},
          "visa": [{"passport": "AE", "country": "BT", "visa_required": True, "fee": 40, "fee_currency": "USD"},
                   {"passport": "IN", "country": "BT", "visa_required": False, "document": "entry_permit"}],
          "fees": [{"passport": "AE", "name_any": ["SDF", "Sustainable Development Fee"], "total": 400, "currency": "USD"},
                   {"passport": "IN", "name_any": ["SDF", "Sustainable Development Fee"], "total": 4800, "currency": "INR"}],
          "feasibility": {"math_consistent": True}},
         review="Verify the Bhutan visa fee ($40), SDF $100/night, Indian SDF ₹1200/night and the entry-permit rule."),
    case("C2", *C, US_TRIO_BHUTAN,
         {"request": {"nights": 4, "duration_range": [4, 5]},
          "visa": [{"passport": "US", "country": "BT", "visa_required": True, "fee": 40, "fee_currency": "USD"}],
          "fees": [{"name_any": ["SDF", "Sustainable Development Fee"], "total_all": 1200, "currency": "USD"}],
          "feasibility": {"ok": False, "gap_gt": 0, "reasons_include": ["SDF"], "math_consistent": True},
          "route": {"alternatives": True}},
         review="Verify SDF $100/night and the visa fee."),
    case("C3", *C, "2 adults and kids aged 4 and 9, German passports, from Frankfurt. Ecuador for 10 days in "
                   "August 2027, including 4 nights in the Galápagos. Budget €12,000.",
         {"request": {"stops": [stop("Ecuador", "EC"), stop("Galápagos", "EC", nights=4)], "nights": 9},
          "visa": [{"passport": "DE", "country": "EC", "visa_required": False}],
          "fees": [{"name_any": ["Galápagos", "national park"], "basis": "per_trip", "total_all": 600, "currency": "USD"}],
          "feasibility": {"report_currency": "EUR", "conversions_include": ["USD->EUR"], "math_consistent": True}},
         review="Confirm the Galápagos park fee ($200 adult, $100 under 12) and whether the $20 transit card applies."),
    case("C4", *C, "2 adults with Indian passports from Delhi: France, Italy and Switzerland for 10 days in June "
                   "2027, budget ₹4 lakh.",
         {"request": {"stops": [stop("France", "FR"), stop("Italy", "IT"), stop("Switzerland", "CH")], "nights": 9},
          "visa": [{"passport": "IN", "country": c, "visa_required": True, "zone": "Schengen",
                    "fee": 90, "fee_currency": "EUR"} for c in ("FR", "IT", "CH")],
          "feasibility": {"visa_fees_per_traveller": 1, "math_consistent": True}},
         review="Confirm the Schengen short-stay visa fee (€90) for adults."),
    case("C5", *C, "2 adults with Indian passports from Dubai, Bali for 6 days in September 2027, budget AED 8,000.",
         {"request": {"stops": [stop("Bali", "ID")], "nights": 5, "budget": {"amount": 8000, "currency": "AED"}},
          "visa": [{"passport": "IN", "country": "ID", "visa_required": True, "document": "visa_on_arrival"}],
          "fees": [{"name_any": ["Bali", "levy"], "basis": "per_trip"}],
          "feasibility": {"report_currency": "AED", "fx_rate_date_present": True,
                          "conversions_include": ["IDR->AED"], "math_consistent": True}},
         review="Confirm Indonesia's visa on arrival for Indian passports and the Bali tourist levy (IDR 150,000)."),

    # ---------- D. season_check + alternatives_agent + choose_alternative ----------
    case("D1", *D, "2 adults with Indian passports from Delhi, Bhutan for 6 days in July 2027, budget $1500.",
         {"season": {"verdict": "avoid", "reasons_any": ["monsoon", "landslide", "flight"]},
          "route": {"alternatives": True},
          "alternatives": {"count_min": 2, "count_max": 3, "each_has_feasibility": True,
                           "include_any": [{"place": "Bhutan", "months": [10, 11]}, {"months": [7]}]},
          "itinerary": {"present": True}},
         resume=["option 1"]),
    case("D2", *D, "2 adults with British passports from Manchester want to hike the Inca Trail to Machu Picchu in "
                   "February 2027, 10 days, budget £5,000.",
         {"season": {"verdict": "avoid", "reasons_any": ["closed", "February"]},
          "route": {"alternatives": True},
          "alternatives": {"count_min": 2, "each_has_feasibility": True,
                           "include_any": [{"place": "Inca Trail", "months": [4, 5, 6, 7, 8, 9, 10]},
                                           {"months": [2], "mentions_any": ["train", "Salkantay", "Lares"]}]},
          "itinerary": {"present": True}},
         resume=["option 1"], review="Confirm the Inca Trail still closes every February."),
    case("D3", *D, "2 adults and kids aged 3 and 6, Indian passports, from Bengaluru, Ladakh for 6 days in January "
                   "2027, budget ₹2 lakh.",
         {"season": {"verdict": "avoid", "reasons_any": ["cold", "altitude"], "stricter_for_vulnerable": True},
          "route": {"alternatives": True},
          "alternatives": {"count_min": 2, "each_has_feasibility": True,
                           "include_any": [{"place": "Ladakh", "months": [6, 7, 8, 9]}, {"lower_altitude": True}]}},
         resume=["option 2"]),
    case("D4", *D, "Family of 4 with kids aged 8 and 12, British passports, from London. Iceland for 6 days in "
                   "December 2027, budget £8,000.",
         {"season": {"verdict": "ok_with_caveats", "reasons_any": ["daylight", "icy", "ice"]},
          "route": {"alternatives": False},
          "report": {"mentions_any": ["daylight", "northern lights"]}, "itinerary": {"present": True}}),
    case("D5", *D, "2 adults with US passports from New York, an overwater villa in the Maldives for 14 nights in "
                   "January 2027, budget $2,500.",
         {"route": {"alternatives": True},
          "alternatives": {"max_alt_round": 2, "each_has_feasibility": True},
          "final": {"feasible_or_explicitly_infeasible": True}},
         resume=["None of these, show me cheaper options", "option 1"]),

    # ---------- E. itinerary_planner + food_agent + qa ----------
    case("E1", *E, "A couple with Indian passports from Mumbai, Bhutan for 6 days in October 2027, budget $2500. "
                   "We love hiking and photography.",
         {"itinerary": {"days": 6, "cities_include": ["Paro", "Thimphu", "Punakha"], "start_city": "Paro",
                        "end_city": "Paro", "activity_not_on_day": {"activity": "Tiger's Nest", "day": 1}},
          "food": {"cities": ["Paro", "Thimphu", "Punakha"], "must_mention_any": ["ema datshi", "red rice", "momo"],
                   "places_min_per_city": 2, "sources_present": True},
          "qa": {"pass": True}}),
    case("E2", *E, JAPAN_FAMILY + " My partner is vegetarian.",
         {"itinerary": {"days": 11, "cities_include": ["Tokyo", "Kyoto"], "rest_day": True, "max_strenuous_per_day": 1,
                        "intercity_mentions_any": ["shinkansen", "train"]},
          "food": {"cities": ["Tokyo", "Kyoto"], "diet": ["vegetarian", "peanut allergy"],
                   "must_mention_any_by_city": {"Tokyo": ["ramen", "sushi", "tonkatsu", "monjayaki"],
                                                "Kyoto": ["yudofu", "kaiseki", "shojin ryori", "matcha"]},
                   "warns_about_any": ["dashi", "fish stock"], "allergy_warnings_present": True,
                   "kid_friendly_pick_per_city": True},
          "qa": {"pass": True}}),
    case("E3", *E, "2 adults with Malaysian passports from Kuala Lumpur, Seoul and Busan for 7 days in November "
                   "2027, budget US$3,000. We eat halal only and one of us has a nut allergy.",
         {"itinerary": {"days": 7, "cities_include": ["Seoul", "Busan"]},
          "food": {"cities": ["Seoul", "Busan"], "diet": ["halal", "nut allergy"],
                   "diet_notes_every_city": True, "halal_places_min_per_city": 1,
                   "must_mention_any_by_city": {"Seoul": ["Itaewon"]},
                   "forbidden_terms": ["pork", "samgyeopsal"], "allergy_warnings_present": True},
          "qa": {"pass": True}}),
    case("E4", *E, "2 adults and our 8-year-old, South African passports, from Johannesburg. Thailand for 10 days in "
                   "December 2027: Bangkok, Chiang Mai and Phuket. Budget ZAR 90,000.",
         {"itinerary": {"days": 10, "cities_include": ["Bangkok", "Chiang Mai", "Phuket"], "intercity_legs_min": 2},
          "food": {"cities": ["Bangkok", "Chiang Mai", "Phuket"],
                   "must_mention_any_by_city": {"Bangkok": ["pad thai", "boat noodles", "mango sticky rice"],
                                                "Chiang Mai": ["khao soi"],
                                                "Phuket": ["southern curry", "gaeng som", "moo hong", "tom yum"]},
                   "kid_friendly_pick_per_city": True},
          "qa": {"pass": True}}),
    case("E5", *E, "A couple with French passports from Paris: Morocco for 6 days in October 2027, Marrakech, a "
                   "night in the Sahara, then Fes. Budget €3,000.",
         {"qa": {"first_pass": False, "issues_include": ["day_count", "food_missing:Fes"], "loops": 1, "pass": True},
          "itinerary": {"days": 6, "cities_include": ["Marrakech", "Fes"]},
          "food": {"cities_include": ["Marrakech", "Fes"]}},
         fixtures={"itinerary_planner_first_call": "five_days_for_six_day_trip",
                   "food_agent_skip_city_first_pass": "Fes"}),
]

if __name__ == "__main__":
    OUT.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in CASES), encoding="utf-8")
    print(f"wrote {len(CASES)} cases to {OUT}")
