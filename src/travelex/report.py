"""Saves a finished plan under runs/<timestamp>_<slug>/ as report.md, report.html and report.json."""
import json
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from travelex.agents.common import event, official
from travelex.agents.intake import describe
from travelex.agents.itinerary import cities, guide_for, same_city
from travelex.schemas import Citation, Source
from travelex.state import TripState
from travelex.tools import fx, visa_rules
from travelex.trip import resolve_nights

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "runs"
VERDICT = {"good": "✓ good", "ok_with_caveats": "! fine, with caveats", "avoid": "✗ best avoided", "unknown": "? unknown"}
def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "trip"


def esc(text: str) -> str:
    """Make text safe inside a Markdown table cell."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def domain(url: str) -> str:
    return urlparse(url).netloc.removeprefix("www.") or url


def shown_food(state: TripState) -> dict:
    """Guides for cities on the itinerary only: a city named up front but left out of the plan gets none."""
    food, plan = state.get("food") or {}, state.get("itinerary")
    if plan is None:
        return food
    visited = cities(plan)
    return {city: g for city, g in food.items() if any(same_city(city, v) for v in visited)}


def plan_currency(state: TripState) -> str:
    """The one currency the plan is shown in: the budget's (₹ budget → every amount in INR)."""
    f, request = state.get("feasibility"), state.get("request")
    if f:
        return f.currency
    return request.budget.currency if request and request.budget else "USD"


def shown(amount: float, currency: str, target: str) -> str:
    """An amount converted into the plan's currency at today's rates; as quoted if the rate is unknown."""
    try:
        return money(fx.convert(amount, currency, target), target)
    except fx.FxError:
        return money(amount, currency)


def money(amount: float, currency: str) -> str:
    return f"{amount:,.0f} {currency}"


QA_WORDS = {"pacing": "Day {} has more than one strenuous activity.",
            "city_missing": "{} is on your trip but not in the day-by-day plan.",
            "food_missing": "There is no food guide for {}.",
            "diet": "A dish suggested in {} may not fit your diet.",
            "allergen": "A dish suggested in {} may contain something you're allergic to."}


def qa_problems(state: TripState) -> list[str]:
    """The final check's unresolved problems, in plain words. After its repair rounds the plan is saved anyway,
    so these must be shown, never dropped."""
    qa = state.get("qa_report")
    if qa is None or qa.passed:
        return []
    plan, out = state.get("itinerary"), []
    for issue in qa.issues:
        code, _, what = issue.partition(":")
        if code == "day_count":
            nights, _ = resolve_nights(state["request"])
            have = len(plan.days) if plan else 0
            out.append(f"The plan has {have} day{'s' * (have != 1)}, but the trip is {nights + 1} days."
                       if nights else "The day-by-day plan is missing.")
        elif code == "rest_day":
            out.append("There is no rest day, though the trip is long and has children or seniors.")
        else:
            out.append(QA_WORDS.get(code, f"{code}: {{}}").format(what.removeprefix("day ")))
    return out


def cruise_note(request) -> str:
    """Said at the top of a plan around the travellers' own cruise."""
    if request.trip_type != "around_cruise" or not request.stops:
        return ""
    length = f" ({request.cruise_nights} nights)" if request.cruise_nights else ""
    after = f" and after it in {request.stops[1].place}" if len(request.stops) == 2 else ""
    return (f"Your cruise{length} isn't in this plan: TravelEx can't plan or price cruises. This plan covers the "
            f"time around it: flights, nights in {request.stops[0].place} before boarding{after}, visas and costs. "
            f"Book the cruise with the cruise line, and check the entry rules for every port of call.")


def section_problems(state: TripState) -> list[str]:
    out = []
    if note := cruise_note(state["request"]):
        out.append(f"> **{note.split(':')[0]}:**{note.split(':', 1)[1]}")
    if problems := qa_problems(state):
        out.append("> **This plan is incomplete:** " + " ".join(problems))
    f = state.get("feasibility")
    if state.get("chosen_alternative"):
        out.append(f"> Switched to the alternative **{state['chosen_alternative']}** to fix the original plan.")
    if f and f.reasons:
        out += ["> **This plan does not fit:** " + " ".join(f.reasons)]
    if f and f.unpriced:
        out += ["> **Not everything is priced yet:** " + " ".join(f.unpriced)
                + " The total leaves these out, so the trip will cost more."]
    for v in (state.get("season") or {}).values():
        if v.verdict == "avoid":
            out.append(f"> **Season:** {v.place} is best avoided then: {'; '.join(v.reasons)}.")
    return out + ([""] if out else [])


def section_getting_there(state: TripState) -> list[str]:
    request = state["request"]
    groups = {g.id: g for g in request.groups}
    cur = plan_currency(state)
    out = ["## Getting there", ""]
    for gid, result in (state.get("group_results") or {}).items():
        group = groups.get(gid)
        if group is None:
            continue
        out.append(f"**From {group.origin}** ({', '.join(group.travellers)})")
        out.append("")
        if not result.options:
            out += ["- No route found. Search flights yourself.", ""]
        for i, o in enumerate(result.options):
            pick = " ← suggested" if i == result.chosen else ""
            notes = "; ".join(filter(None, [o.kid_note, o.accessibility_note, *o.warnings]))
            out.append(f"- `{o.route}` {o.mode}, {shown(o.adult_fare.amount, o.adult_fare.currency, cur)} return per adult "
                       f"({o.fare_basis}){pick}" + (f". {notes}" if notes else ""))
        out.append("")
    if (meeting := state.get("meeting")) and len(request.groups) > 1:  # one group has no one to meet
        out += [f"**Meeting:** {'; '.join(f'{o.place}: {o.note}' for o in meeting.options)}", ""]
    return out


KIND = {"destination_government": "the destination's government", "home_government": "your own government",
        "other_government": "another country's government", "third_party": "a travel site"}
APPLIES = {"applies": "about your passport", "all_visitors": "applies to every visitor",
           "unchecked": "relevance not checked"}


def who_says(c: Citation) -> str:
    """Who published the quote, and whom it applies to."""
    return f"{KIND.get(c.source_kind, c.source_kind)}, {APPLIES.get(c.relevance, c.relevance)}"


def section_visas(state: TripState, notes: "Footnotes") -> list[str]:
    """The visa table with a footnote for every source quoted, then the quotes themselves."""
    checks = [v for r in (state.get("group_results") or {}).values() for v in r.visas]
    if not checks:
        return []
    out = ["## Visas and entry", "", "| Traveller | Passport | Country | What you need | Sources |", "|---|---|---|---|---|"]
    for v in checks:
        rule = v.rule
        if rule is None:
            cited = "none found"
        elif visa_rules.by_law(rule):
            cited = "not needed: " + rule.notes.rstrip(".").lower()
        else:
            cited = notes.cite(rule.citations) or "⚠ no quote found"
        out.append(f"| {esc(v.traveller)} | {esc(v.passport or '?')} | {esc(v.country)} | {esc(v.summary)} | {esc(cited)} |")
    out.append("")
    seen = set()
    evidence = []
    for v in checks:
        if v.rule is None or (v.passport, v.country) in seen or not v.rule.citations:
            continue
        seen.add((v.passport, v.country))
        any_official = any(official(c.url) for c in v.rule.citations)
        evidence.append(f"**{v.passport} passport → {v.country}**" + ("" if any_official else
                        " ⚠ no government source found: confirm with the embassy"))
        evidence += [f"- _{c.supports.replace('_', ' ')}:_ “{c.quote}” {notes.cite([c])} · {who_says(c)}"
                     for c in v.rule.citations]
    if evidence:
        out += ["<details><summary>What the sources say</summary>", "", *evidence, "", "</details>", ""]
    return out + ["_Every quote above was found word for word on the page it links to, and checked to be about your "
                  "passport and a tourist visit. Rules change: check the official source before you apply._", ""]


def section_itinerary(state: TripState) -> list[str]:
    plan, food = state.get("itinerary"), state.get("food") or {}
    if plan is None:
        return []
    out = ["## Day by day", ""]
    seen_in_city: dict[str, int] = {}
    for d in plan.days:
        when = f" ({d.date:%a %d %b})" if d.date else ""
        out.append(f"### Day {d.day}{when}: {d.title} — {d.city}" + (" · rest day" if d.rest_day else ""))
        if d.transfer:
            out.append(f"_Transfer: {d.transfer}_")
        for a in d.activities:
            hard = " (strenuous)" if a.strenuous else ""
            kids = f" — with kids or seniors: {a.kid_option}" if a.kid_option else ""
            out.append(f"- {a.name}{hard}{kids}")
        if (guide := guide_for(d.city, food)) and guide.dishes:
            n = seen_in_city[d.city] = seen_in_city.get(d.city, -1) + 1  # rotate: a new dish each day here
            picks = [guide.dishes[(n * 2 + k) % len(guide.dishes)].name for k in range(min(2, len(guide.dishes)))]
            out.append(f"- 🍽 Try: {', '.join(dict.fromkeys(picks))}")
        out.append("")
    if plan.trimmed_variant:
        out += [f"**Shorter version:** {plan.trimmed_variant}", ""]
    out += [f"- {n}" for n in plan.notes] + ([""] if plan.notes else [])
    return out


def section_food(state: TripState, notes: "Footnotes") -> list[str]:
    food = state.get("food") or {}
    if not food:
        return []
    out = ["## Food guide", ""]
    for city, g in shown_food(state).items():
        out.append(f"### {city}")
        for dish in g.dishes:
            tags = ", ".join(dish.diet_tags + (["kid-friendly"] if dish.kid_friendly else [])
                             + [f"not confirmed {a}-free: ask" for a in dish.allergens_unknown])
            out.append(f"- **{dish.name}**: {dish.description}" + (f" _({tags})_" if tags else ""))
        for p in g.places:
            out.append(f"- 📍 {p.name}" + (f", {p.area}" if p.area else "") + (f": {p.why}" if p.why else "")
                       + notes.cite_ids(p.source_ids))
        if g.diet_notes:
            out.append(f"- _Diet: {g.diet_notes}_")
        out.append("")
    return out


def hotel_note(f) -> str:
    """Which hotel level the costs use, and why; and what is left to spend."""
    why = "the most comfortable that fits your budget" if f.lodging_chosen else "as you asked"
    left = f" {money(f.spare, f.currency)} is left for activities and extras." if f.spare else ""
    others = []
    for o in f.lodging_others:
        change = o.counted - f.counted
        over = " (over budget)" if f.budget is not None and o.counted > f.budget else ""
        others.append(f"{o.style} hotels would {'add' if change > 0 else 'save'} {money(abs(change), f.currency)}{over}")
    instead = f" Instead, {'; '.join(others)}." if others else ""
    return f"Hotels: {f.lodging_style}, {why}.{left}{instead}"


def section_budget(state: TripState, notes: "Footnotes | None" = None) -> list[str]:
    f = state.get("feasibility")
    if f is None:
        return []
    status = f.status() + (" ✓" if f.ok else " ✗" if f.reasons else " ⚠")
    budget = f" of {money(f.budget, f.currency)}" if f.budget is not None else ""
    out = ["## Budget", "", f"**{money(f.counted, f.currency)}{budget}: {status}**", "",
           *([f"_Flights to and from home ({money(f.excluded, f.currency)}) are outside your budget, as you asked. "
              f"The whole trip costs {money(f.total, f.currency)}._", ""] if f.excluded else []),
           *([hotel_note(f), ""] if f.lodging_style else []),
           "| Who | Cost |", "|---|---|", *[f"| {esc(w)} | {money(a, f.currency)} |" for w, a in f.per_traveller.items()],
           *([f"| Shared (rooms, food, guide, transport on the ground) | {money(f.shared, f.currency)} |"]
             if f.shared else []), "",
           "<details><summary>Every line</summary>", "", "| What | Who | Amount | Source |", "|---|---|---|---|",
           *[f"| {esc(l.label)} | {esc(l.traveller or 'shared')} | {money(l.amount_report, f.currency)} | "
             f"{esc(notes.cite_cost(l)) if notes else ''} |" for l in f.lines], "",
           *([f"- {esc(l.label)}: “{l.quote}”" for l in dict((l.label, l) for l in f.lines if l.quote).values()]
             if notes else []), "", "</details>", ""]
    if f.fx_source:
        out += [f"_Exchange rates: {f.fx_source}, {f.fx_as_of}. Fares and prices are estimates from web search._", ""]
    return out


def section_alternatives(state: TripState) -> list[str]:
    alts = state.get("alternatives") or []
    if not alts:
        return []
    out = ["## Alternatives considered", ""]
    for a in alts:
        f = a.feasibility
        cost = f"{money(f.counted, f.currency)}, {f.status()}" if f else "not costed"
        chosen = " ← chosen" if a.title == state.get("chosen_alternative") else ""
        out.append(f"- **{a.title}** ({cost}){chosen}: {a.why}"
                   + (f" _Trade-offs: {'; '.join(a.trade_offs)}._" if a.trade_offs else ""))
    return out + [""]


class Footnotes:
    """Numbers sources by URL in the order the report first cites them, so footnotes read [1], [2], [3]…
    Works for this run's Sources (by ID) and for cached Citations, which only keep the URL."""

    def __init__(self, sources: dict[str, Source]):
        self.sources, self.numbers, self.titles = sources, {}, {}

    def _mark(self, url: str, title: str) -> str:
        self.titles.setdefault(url, title or domain(url))
        return f"[{self.numbers.setdefault(url, len(self.numbers) + 1)}]"

    def cite(self, citations: list[Citation]) -> str:
        return "".join(dict.fromkeys(self._mark(c.url, c.title) for c in citations))

    def cite_ids(self, ids: list[str]) -> str:
        marks = "".join(dict.fromkeys(self._mark(self.sources[i].url, self.sources[i].title)
                                      for i in ids if i in self.sources))
        return f" {marks}" if marks else ""

    def cite_cost(self, line) -> str:
        """The source's site as a link ("+2" when a fare is pooled from more pages); "estimate" when no source
        states the amount."""
        urls = list(dict.fromkeys([self.sources[i].url for i in line.source_ids if i in self.sources] + line.urls))
        if not urls:
            return "estimate"
        return f"[{domain(urls[0])}]({urls[0]})" + (f" +{len(urls) - 1}" if len(urls) > 1 else "")

    def render(self) -> list[str]:
        if not self.numbers:
            return []
        return ["## Sources", "", *[f"{n}. [{self.titles[url]}]({url}) — {domain(url)}" + (" (official)" if official(url) else "")
                                    for url, n in sorted(self.numbers.items(), key=lambda kv: kv[1])]]


def to_double_check(state: TripState) -> list[str]:
    """Everything left open: the final check's unresolved problems and notes, prices we couldn't find or confirm,
    what the agents couldn't settle, and what the traveller told us about it."""
    from travelex.agents.handoff import open_issues  # here: the agents import from this module
    checks = [f"Still wrong in this plan: {p}" for p in qa_problems(state)]
    if state["request"].trip_type == "around_cruise":
        checks.append("Your cruise is not priced or checked here: its fare, port fees and each port's entry rules.")
    f = state.get("feasibility")
    checks += [f"Not priced, so not in the total: {u}" for u in (f.unpriced if f else [])]
    checks += [f"Fee not confirmed by a source: {label}" for label in dict.fromkeys(
        l.label.split(" (")[0] for l in (f.lines if f else []) if l.category == "destination_fee"
        and l.label.endswith("unconfirmed"))]
    checks += [f"Not settled by our research: {issue}" for issue in open_issues(state)]
    checks += [f"You told us: {note}" for note in state.get("traveller_notes") or []]
    qa = state.get("qa_report")
    return checks + (list(qa.notes) if qa else [])


def version_note(state: TripState) -> str:
    """"Version 3, updated after your comments: …" once the traveller has asked for changes."""
    revisions = state.get("revisions") or []
    return (f"Version {len(revisions) + 1}, updated after your comments: {'; '.join(revisions)}."
            if revisions else "")


def render_markdown(state: TripState) -> str:
    request = state["request"]
    notes = Footnotes(state.get("sources") or {})
    lines = [f"# {request.destination}", "", f"_{describe(request)}_", ""]
    if changed := version_note(state):
        lines += [f"> {changed}", ""]
    lines += section_problems(state)
    lines += ["## Season", "", *[f"- **{v.place}**: {VERDICT[v.verdict]}" + (f" — {'; '.join(v.reasons)}"
                                                                          if v.reasons else "")
                                  for v in (state.get("season") or {}).values()], ""]
    lines += section_itinerary(state) + section_getting_there(state) + section_visas(state, notes)
    lines += section_food(state, notes) + section_budget(state, notes) + section_alternatives(state)
    if checks := to_double_check(state):
        lines += ["## Things to double-check", "", *[f"- {n}" for n in checks], ""]
    lines += notes.render()
    return "\n".join(lines).rstrip() + "\n"


def to_jsonable(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    return value if isinstance(value, (str, int, float, bool, type(None))) else str(value)


def save_report(state: TripState) -> dict:
    request = state["request"]
    version = len(state.get("revisions") or []) + 1
    folder = RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{slug(request.destination)}{f'_v{version}' if version > 1 else ''}"
    folder.mkdir(parents=True, exist_ok=True)
    md = render_markdown(state)
    (folder / "report.md").write_text(md, encoding="utf-8")
    from travelex.report_html import render_html  # here, not at the top: report_html imports from this module
    (folder / "report.html").write_text(render_html(state), encoding="utf-8")
    keep = {k: v for k, v in state.items() if k not in ("events",)}
    (folder / "report.json").write_text(json.dumps(to_jsonable(keep), ensure_ascii=False, indent=1), encoding="utf-8")
    return {"report_dir": str(folder), "events": [event("save_report", "saved", str(folder))]}
