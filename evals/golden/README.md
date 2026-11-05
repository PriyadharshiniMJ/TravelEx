# Golden dataset: travelex_golden_v1

There are 25 end-to-end cases, 5 for each focus area. They cover destinations worldwide and travellers with many different passports.

The cases are written in **`build_golden.py`**. Edit them there, then run `python evals/golden/build_golden.py` to regenerate `travelex_golden_v1.jsonl`. A test fails if the two files differ.

| Focus | Nodes with strict expectations | Cases |
|---|---|---|
| `intake_clarify` | `intake`, `clarify` | A1 Bhutan (4–5 days, asks for passports and month) · A2 Japan family · A3 Italy + Switzerland, two groups · A4 "somewhere warm" → Sri Lanka · A5 Peru, date in the past |
| `transport_meeting` | `group_research.transport_agent`, `meeting_plan` | B1 Lagos + Nairobi → Zanzibar · B2 Bhutan family + grandma · B3 Berlin → Prague/Vienna by surface · B4 Sydney + Auckland → Bali · B5 Mexico with a wheelchair user |
| `visa_costs_feasibility` | `visa_check`, `destination_costs`, `feasibility` | C1 Bhutan with UAE + Indian passports · C2 Bhutan, fees exceed budget · C3 Galápagos park fee · C4 one Schengen visa for 3 countries · C5 Bali levy, budget in AED |
| `season_alternatives` | `season_check`, `alternatives_agent`, `choose_alternative` | D1 Bhutan monsoon · D2 Inca Trail closed in February · D3 Ladakh winter with small kids · D4 Iceland in December (caveats only, no alternatives) · D5 Maldives over budget, 2-round cap |
| `itinerary_food_qa` | `itinerary_planner`, `food_agent`, `qa` | E1 Bhutan hiking · E2 Japan, vegetarian + peanut allergy · E3 Korea, halal + nut allergy · E4 Thailand, 3 cities · E5 Morocco with a broken first draft |

Each line is a LangSmith example made of `inputs`, `outputs` and `metadata`.

- `inputs.today` fixes "today" for resolving dates. Never read the clock in tests.
- `inputs.resume` holds the answers given to each `interrupt()`, in order. When the list runs out, the run continues with `auto_approve`.
- `inputs.fixtures` (E5 only) swaps in canned node outputs to exercise the QA loop.
- `outputs` is **partial**. An evaluator checks only the keys that are present and skips the rest.
  - Countries are ISO 3166 alpha-2 codes.
  - Routes are IATA codes joined by `-`.
  - `*_any` means at least one listed item must match.
- `metadata.needs_review` marks facts to check against official sources before v1 is frozen: visa rules, fees, routes and closures. `review_notes` says what to check.

Visa expectations are checked in two ways:
- **Known pairs** (those in the curated seed rules) are checked by `tests/test_golden_dataset.py` against `visa_rules.py`.
- **Researched pairs** are checked against what the visa agent found during the eval run.

Conventions:
- "N days" means N−1 nights.
- A range such as "4–5 days" is planned at the upper end, with a trimmed variant.
