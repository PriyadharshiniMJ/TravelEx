# TravelEx

A travel-planning agent for anyone, from anywhere, to anywhere. Describe a trip in your own words, for example
*"two adults to Bhutan, one from Dubai and one from Chennai, Indian passports, 4–5 days in October, $1500"*, and
it returns:

- routes for each traveller group;
- visa rules for each passport;
- tourist fees and costs for each stop;
- a budget check, with costed alternatives when the plan doesn't fit or the season is wrong;
- a day-by-day itinerary;
- food to try in every city;
- a report with sources.

## Setup

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt -r requirements-openai.txt -e .
copy .env.example .env      # then fill in LLM_MODEL, the provider key, TAVILY_API_KEY and LangSmith
```

LangSmith: set `LANGSMITH_TRACING=true`. Accounts outside the US also need `LANGSMITH_ENDPOINT`: for example `https://apac.api.smith.langchain.com` for APAC, or `https://eu.api.smith.langchain.com` for the EU.

## Use it

```
.venv\Scripts\python cli.py "your trip in your own words"            # answers questions in the terminal
.venv\Scripts\python cli.py --auto "your trip"                       # never pauses
.venv\Scripts\streamlit run app.py                                   # web UI
```

Each plan is saved to `runs/<time>_<destination>/` as `report.md`, `report.html` and `report.json`.

## How it works

LangGraph runs a fixed workflow: the order of steps is set in code, not chosen by an LLM.

```
intake → clarify (asks for what's missing)
  → [season ‖ transport + visas per traveller group ‖ costs + fees per stop ‖ food per named city]   (in parallel)
  → meeting_plan → fill gaps (an agent chases weak prices and one-source fares; at most 4 rounds)
  → ask the traveller                    (only when something could stop the trip: no route, an unconfirmed entry rule)
  → feasibility (pure Python)
  → alternatives → you choose            (only when over budget or the season is wrong; at most 2 rounds)
  → itinerary → [food per city] → qa     (QA sends fixable problems back, at most 2 rounds)
  → save_report → review                 (comment on the plan: day changes re-plan the days, trip changes
                                          re-run the research they affect; at most 3 rounds)
```

With `--auto` the run never pauses. Clarify plans with what it has, and takes the first destination idea when no
place is named. Ask the traveller and review are skipped, and "pick for me" chooses the alternative.

- **LLMs read, research and write; Python does the arithmetic.** That covers dates, nights, currency, fees, totals, choosing a route and checking the plan.
- **Agents look up rules from a seed set first, then a cache, then research.** Visa rules, tourist fees and seasons come from a small hand-written seed set (`src/travelex/tools/`), then from rules learned earlier and cached in `.cache/rules/`, and otherwise from new research with sources.
- **Code checks what the model says.**
  - A non-stop flight needs a word-for-word quote from a source.
  - Fares are capped, and outliers are dropped.
  - A place to eat must be named by a source.
  - Dishes that break the diet or contain a stated allergen are removed, and diet tags are checked against the ingredients.
  - Food places must be in the city's own country.
  - A quoted price must state that amount, in that currency, about that place.
  - The plan's notes can't state visa, fee or price facts.
- **No destination yet?** The intake suggests ideas, and code screens them for free against the stored season, fees and visa rules for the travel month and passports.
- **Alternatives are costed, and their trade-offs are spelled out.** Each one lists what it changes (for example "1 fewer night") and what is still wrong. If none fixes the problem, the agent tries once more. "Pick for me" keeps the destination when it can still fit and never raises the budget.
- **Alternatives re-research only what changed.** Results are reused when their inputs match: a shorter stay re-prices nothing, and a new destination re-researches that stop and the flights.

## Limits

One trip shares a budget across every parallel branch, so a looping agent can't run up an unbounded bill. When a
limit is reached, research stops and the plan is still written from what was found. Set them in `.env`:

| Setting | Default | |
|---|---|---|
| `TRAVELEX_MAX_SEARCHES` | 150 | searches and page fetches per trip |
| `TRAVELEX_MAX_TOKENS` | 3000000 | model tokens per trip |
| `TRAVELEX_MAX_SECONDS` | 1200 | time per trip |
| `TRAVELEX_MAX_CONCURRENCY` | 8 | searches at once (use 4 on Tavily's free plan) |
| `TRAVELEX_SEARCH_TTL_DAYS` | 7 | cached pages older than this are fetched again; `TRAVELEX_REFRESH=1` ignores the cache |
| `LANGSMITH_HIDE_INPUTS` | off | `true` redacts passports, ages and diets from traces |

## Tests and evals

```
.venv\Scripts\python -m pytest                            # offline: fake model, fake search
.venv\Scripts\python evals/upload_dataset.py              # golden dataset → LangSmith (once)
.venv\Scripts\python evals/run_evals.py --ids A1 C4       # a few live cases
.venv\Scripts\python evals/run_evals.py                   # all 25 cases (slow, real API costs)
.venv\Scripts\python evals/run_evals.py --no-judges       # code checks only, no LLM judges
```

The golden dataset has 25 cases, 5 per focus area, across 19 destination countries. The cases are written in `evals/golden/build_golden.py`. Visa, fee and route facts marked `needs_review` must be checked against official sources.

Each case is scored by:
- **code checks for each step:** request, clarify, season, routes, transport, meeting, visas, fees, costs, feasibility, alternatives, itinerary, food, QA and the report;
- **`completion`:** the run reached the end;
- **`critical`:** fails the case on a wrong visa rule, totals that don't add up, an off-diet or allergen dish, or a made-up source, whatever the mean;
- **`success` (done well):** finished, passed QA, fully priced and nothing critical. A correct "this doesn't fit" plan counts as a success;
- **LLM judges for the itinerary, food and alternatives:** run on `LLM_MODEL_JUDGE`, a different model from the agent's, so it never grades its own work.

On Tavily's free plan, run evals one case at a time: `--concurrency 1` (the default) with `TRAVELEX_MAX_CONCURRENCY=4`.


