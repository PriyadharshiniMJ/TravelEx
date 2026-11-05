"""The plan as a bright, self-contained travel page (report.html, and what the app shows).
Built straight from the state, so each part gets its own treatment: a timeline, route chains, visa cards."""
from html import escape

from travelex.agents.intake import describe
from travelex.agents.itinerary import guide_for, same_city
from travelex.report import (APPLIES, KIND, domain, hotel_note, money, plan_currency, shown, shown_food,
                             cruise_note, qa_problems, to_double_check, version_note)
from travelex.schemas import Citation, Source
from travelex.state import TripState
from travelex.tools import visa_rules
from travelex.trip import resolve_nights

CITY_COLORS = ["#0aa5a5", "#ff6b57", "#f2a100", "#3cb371", "#7a5cfa", "#e8478b", "#1e88e5"]
LEG_ICON = {"flight": "✈", "train": "🚆", "bus": "🚌", "road": "🚗"}
MODE_ICON = {"flight": "✈", "flight+overland": "✈ 🚗", "surface": "🚆"}
SEASON = {"good": ("good", "Great time to go"), "ok_with_caveats": ("warn", "Good, with caveats"),
          "avoid": ("bad", "Best avoided"), "unknown": ("muted", "Season unknown")}
TRAVELLER_ICON = {"adult": "🧑", "child": "🧒", "infant": "👶"}


def e(text) -> str:
    return escape(str(text or ""))


def flag(code: str | None) -> str:
    """ISO country code → flag emoji ("IN" → 🇮🇳)."""
    if not code or len(code) != 2 or not code.isalpha():
        return "🏳"
    return "".join(chr(0x1F1E6 + ord(c) - ord("A")) for c in code.upper())


class Notes:
    """Footnotes by URL, shared by every section, rendered as linked superscripts."""

    def __init__(self, sources: dict[str, Source]):
        self.sources, self.numbers, self.titles = sources, {}, {}

    def _mark(self, url: str, title: str) -> str:
        """The footnote opens the source itself in a new tab. Not "#src-n": the app shows this page in a frame,
        where an in-page "#" link resolves against the app's address and loads the app inside the plan."""
        n = self.numbers.setdefault(url, len(self.numbers) + 1)
        self.titles.setdefault(url, title or domain(url))
        return (f'<a class="fn" href="{e(url)}" target="_blank" rel="noopener" '
                f'title="{e(self.titles[url])} ({e(domain(url))})">{n}</a>')

    def cite(self, citations: list[Citation]) -> str:
        return "".join(dict.fromkeys(self._mark(c.url, c.title) for c in citations))

    def cite_ids(self, ids: list[str]) -> str:
        return "".join(dict.fromkeys(self._mark(self.sources[i].url, self.sources[i].title)
                                     for i in ids if i in self.sources))

    def cite_cost(self, line) -> str:
        """A link chip naming the source's site, with the words behind the price on hover ("+2" when a fare is
        pooled from more pages); a quiet "estimate" pill when no source states the price."""
        pages = {self.sources[i].url: self.sources[i].title for i in line.source_ids if i in self.sources}
        pages |= {url: "" for url in line.urls if url not in pages}
        if not pages:
            return '<span class="tag est">estimate</span>'
        (url, title), *others = pages.items()
        hover = " · ".join(filter(None, [f"“{line.quote}”" if line.quote else "", title or domain(url)]))
        more = (f'<span class="more" title="{e(", ".join(domain(u) for u, _ in others))}">+{len(others)}</span>'
                if others else "")
        return (f'<a class="src" href="{e(url)}" target="_blank" rel="noopener" title="{e(hover)}">'
                f'{e(domain(url))} ↗</a>{more}')

    def render(self) -> str:
        if not self.numbers:
            return ""
        items = "".join(f'<li id="src-{n}"><a href="{e(url)}" target="_blank" rel="noopener">{e(self.titles[url])}</a>'
                        f' <span class="muted">{e(domain(url))}</span></li>'
                        for url, n in sorted(self.numbers.items(), key=lambda kv: kv[1]))
        return f'<section><h2>Sources</h2><ol class="sources">{items}</ol></section>'


def hero(state: TripState) -> str:
    request = state["request"]
    nights, _ = resolve_nights(request)
    when = (f"{request.start_date:%d %b} – {request.end_date:%d %b %Y}" if request.start_date and request.end_date
            else request.month or "Dates to be set")
    budget = money(request.budget.amount, request.budget.currency) if request.budget else "No budget set"
    people = "".join(f'<span class="person">{TRAVELLER_ICON.get(t.type, "🧑")} {e(t.label)}'
                     f'{f" · {t.age}" if t.age is not None else ""} <span title="passport">{flag(t.passport)}</span>'
                     f'<span class="muted"> from {e(t.origin or "?")}</span></span>' for t in request.travellers)
    stops = " → ".join(f"{flag(s.country)} {e(s.place)}" for s in request.stops)
    return f"""<header class="hero">
  <div class="hero-inner">
    <div class="eyebrow">Your trip plan</div>
    <h1>{e(request.destination)}</h1>
    <div class="facts"><span>📅 {e(when)}</span><span>🌙 {nights or "?"} nights</span><span>💰 {e(budget)}</span>
      <span>🗺 {stops}</span></div>
    <div class="people">{people}</div>
  </div>
</header>"""


def status(state: TripState) -> str:
    f, chips = state.get("feasibility"), []
    if problems := qa_problems(state):
        chips.append(f'<span class="badge bad" title="{e(" ".join(problems))}">⚠ Incomplete plan</span>')
    if f:
        if f.ok:
            chips.append('<span class="badge good">✓ Fits the budget</span>')
        else:
            chips.append(f'<span class="badge bad">Over budget by {e(money(f.gap, f.currency))}</span>'
                         if f.gap > 0 else f'<span class="badge warn" title="{e(" ".join(f.unpriced))}">'
                                           f'{e(f.status().capitalize())}</span>')
    for v in (state.get("season") or {}).values():
        tone, label = SEASON[v.verdict]
        chips.append(f'<span class="badge {tone}" title="{e("; ".join(v.reasons))}">{e(v.place)}: {label}</span>')
    if state.get("chosen_alternative"):
        chips.append(f'<span class="badge info">Switched to: {e(state["chosen_alternative"])}</span>')
    notes = "".join(f"<li>{e(r)}</li>" for v in (state.get("season") or {}).values() for r in v.reasons)
    incomplete = (f'<p class="callout bad"><b>This plan is incomplete:</b> {e(" ".join(problems))}</p>'
                  if problems else "")
    if note := cruise_note(state["request"]):
        incomplete = f'<p class="callout">🚢 {e(note)}</p>' + incomplete
    return (incomplete + f'<div class="status">{"".join(chips)}</div>'
            + (f'<ul class="season-notes">{notes}</ul>' if notes else ""))


def city_colors(state: TripState) -> dict[str, str]:
    plan = state.get("itinerary")
    cities = list(dict.fromkeys(d.city for d in plan.days)) if plan else []
    return {c: CITY_COLORS[i % len(CITY_COLORS)] for i, c in enumerate(cities)}


def transfer_icon(text: str) -> str:
    lower = text.lower()
    for word, icon in (("fly", "✈"), ("flight", "✈"), ("train", "🚆"), ("shinkansen", "🚆"), ("bus", "🚌"),
                       ("ferry", "⛴"), ("drive", "🚗"), ("car", "🚗"), ("taxi", "🚕")):
        if word in lower:
            return icon
    return "➜"


def timeline(state: TripState, colors: dict[str, str]) -> str:
    plan, food = state.get("itinerary"), state.get("food") or {}
    if plan is None:
        return ""
    seen: dict[str, int] = {}
    days = []
    for d in plan.days:
        color = colors.get(d.city, CITY_COLORS[0])
        when = f"{d.date:%a %d %b}" if d.date else ""
        acts = "".join(
            f'<li>{e(a.name)}{" <span class=\"tag hard\">strenuous</span>" if a.strenuous else ""}'
            f'{f"<div class=\"kid\">🧒 {e(a.kid_option)}</div>" if a.kid_option else ""}</li>' for a in d.activities)
        transfer = (f'<div class="transfer">{transfer_icon(d.transfer)} {e(d.transfer)}</div>' if d.transfer else "")
        dishes = ""
        if (guide := guide_for(d.city, food)) and guide.dishes:
            n = seen[d.city] = seen.get(d.city, -1) + 1
            picks = list(dict.fromkeys(guide.dishes[(n * 2 + k) % len(guide.dishes)].name
                                       for k in range(min(2, len(guide.dishes)))))
            dishes = '<div class="try">🍽 ' + "".join(f'<span class="chip food">{e(x)}</span>' for x in picks) + "</div>"
        rest = '<span class="tag rest">rest day</span>' if d.rest_day else ""
        days.append(f"""<li class="day" style="--city:{color}">
  <div class="dot">{d.day}</div>
  <div class="card">
    <div class="day-head"><span class="city">{e(d.city)}</span><span class="muted">{e(when)}</span>{rest}</div>
    <h3>{e(d.title)}</h3>{transfer}<ul class="acts">{acts}</ul>{dishes}
  </div>
</li>""")
    extra = ""
    if plan.trimmed_variant:
        extra += f'<p class="callout">✂ <b>Shorter version:</b> {e(plan.trimmed_variant)}</p>'
    if plan.notes:
        extra += '<ul class="tips">' + "".join(f"<li>💡 {e(n)}</li>" for n in plan.notes) + "</ul>"
    return f'<section><h2>Day by day</h2><ol class="timeline">{"".join(days)}</ol>{extra}</section>'


def route_chain(route: str) -> str:
    return '<span class="arrow">→</span>'.join(f'<span class="stop">{e(s)}</span>' for s in route.split("-"))


def getting_there(state: TripState) -> str:
    request = state["request"]
    groups, cards = {g.id: g for g in request.groups}, []
    cur = plan_currency(state)
    for gid, result in (state.get("group_results") or {}).items():
        group = groups.get(gid)
        if group is None:
            continue
        options = []
        for i, o in enumerate(result.options):
            pick = i == result.chosen
            notes = "".join(f'<div class="note">⚠ {e(w)}</div>' for w in o.warnings)
            notes += "".join(f'<div class="note">{e(x)}</div>' for x in (o.kid_note, o.accessibility_note) if x)
            options.append(f"""<div class="option{" picked" if pick else ""}">
  <div class="route">{MODE_ICON.get(o.mode, "✈")} {route_chain(o.route)}</div>
  <div class="fare"><b>{e(shown(o.adult_fare.amount, o.adult_fare.currency, cur))}</b> return per adult
    <span class="tag {"quoted" if o.fare_basis == "quoted" else "est"}">{e(o.fare_basis)}</span>
    {'<span class="tag pick">suggested</span>' if pick else ""}</div>{notes}
</div>""")
        body = "".join(options) or '<div class="note">No route found yet: search flights yourself.</div>'
        cards.append(f'<div class="group"><h3>From {e(group.origin)}</h3>'
                     f'<div class="muted">{e(", ".join(group.travellers))}</div>{body}</div>')
    meeting = ""
    if (m := state.get("meeting")) and len(request.groups) > 1:  # one group has no one to meet
        meeting = '<p class="callout">🤝 ' + " ".join(f"<b>{e(o.place)}</b>: {e(o.note)}" for o in m.options) + "</p>"
    return f'<section><h2>Getting there</h2><div class="groups">{"".join(cards)}</div>{meeting}</section>' if cards else ""


def visa_tone(rule) -> tuple[str, str]:
    if rule is None:
        return "muted", "Unknown"
    if visa_rules.by_law(rule) or (not rule.visa_required and rule.document == "none"):
        return "good", "No visa needed"
    return {"entry_permit": ("info", "Permit on arrival"), "visa_on_arrival": ("info", "Visa on arrival"),
            "eta": ("warn", "eTA before you fly"), "e_visa": ("warn", "e-Visa before you fly"),
            "visa": ("bad", "Visa before you fly")}.get(rule.document, ("warn", rule.document))


def visas(state: TripState, notes: Notes) -> str:
    checks = [v for r in (state.get("group_results") or {}).values() for v in r.visas]
    cards, seen = [], set()
    for v in checks:
        key = (v.passport, v.country)
        travellers = [c.traveller for c in checks if (c.passport, c.country) == key]
        if key in seen:
            continue
        seen.add(key)
        tone, label = visa_tone(v.rule)
        rule = v.rule
        cited = ""
        if rule and not visa_rules.by_law(rule):
            cited = ('<span class="tag good">✓ cited</span>' if rule.confirmed
                     else '<span class="tag bad">⚠ not confirmed</span>')
        quotes = ""
        if rule and rule.citations:
            items = "".join(f'<li>“{e(c.quote)}” {notes.cite([c])}<div class="who">{e(c.supports.replace("_", " "))} · '
                            f'{e(KIND.get(c.source_kind, c.source_kind))} · {e(APPLIES.get(c.relevance, c.relevance))}'
                            f'</div></li>' for c in rule.citations)
            quotes = f'<details><summary>What the sources say</summary><ul class="quotes">{items}</ul></details>'
        cards.append(f"""<div class="visa {tone}">
  <div class="visa-head"><span class="flags">{flag(v.passport)} → {flag(v.country)}</span>
    <span class="badge {tone}">{e(label)}</span>{cited}</div>
  <div class="muted">{e(", ".join(travellers))}</div>
  <p>{e(v.summary)}</p>{quotes}
</div>""")
    if not cards:
        return ""
    return (f'<section><h2>Visas and entry</h2><div class="visas">{"".join(cards)}</div>'
            '<p class="fine">Every quote was found word for word on the linked page and checked to be about your '
            'passport and a tourist visit. Rules change: check the official source before you apply.</p></section>')


def food_section(state: TripState, notes: Notes, colors: dict[str, str]) -> str:
    food = state.get("food") or {}
    cards = []
    for city, g in shown_food(state).items():
        color = next((c for name, c in colors.items() if same_city(name, city)), "#8a94a6")
        dishes = "".join(
            f'<li><b>{e(d.name)}</b> <span class="muted">{e(d.description)}</span>'
            + "".join(f'<span class="tag diet">{e(t)}</span>' for t in d.diet_tags)
            + ('<span class="tag kidtag">🧒 kid-friendly</span>' if d.kid_friendly else "")
            + "".join(f'<span class="tag warn" title="No source confirms it is free of {e(a)}: ask before ordering">'
                      f'⚠ not confirmed {e(a)}-free</span>' for a in d.allergens_unknown) + "</li>" for d in g.dishes)
        places = "".join(f'<li>📍 <b>{e(p.name)}</b>{f", {e(p.area)}" if p.area else ""}'
                         f'{f" — {e(p.why)}" if p.why else ""} {notes.cite_ids(p.source_ids)}</li>' for p in g.places)
        diet = f'<p class="diet-note">🥗 {e(g.diet_notes)}</p>' if g.diet_notes else ""
        cards.append(f'<div class="food-card" style="--city:{color}"><h3>{e(city)}</h3>'
                     f'<ul class="dishes">{dishes}</ul>{f"<ul class=\"places\">{places}</ul>" if places else ""}{diet}</div>')
    return f'<section><h2>Food to try</h2><div class="foods">{"".join(cards)}</div></section>' if cards else ""


def budget(state: TripState, notes: Notes) -> str:
    f = state.get("feasibility")
    if f is None:
        return ""
    meter = ""
    if f.budget:
        used = min(f.counted / f.budget, 1) * 100
        over = f.counted > f.budget
        meter = (f'<div class="meter{" over" if over else ""}"><div class="fill" style="width:{used:.1f}%"></div></div>'
                 f'<div class="meter-label"><b>{e(money(f.counted, f.currency))}</b> of {e(money(f.budget, f.currency))}'
                 f'{" — over by " + e(money(f.gap, f.currency)) if over else " — " + e(money(-f.gap, f.currency)) + " to spare"}'
                 f'{" so far" if f.unpriced and not over else ""}</div>')
    if f.lodging_style:
        meter += f'<p class="fine">🏨 {e(hotel_note(f))}</p>'
    if f.excluded:
        meter += (f'<p class="fine">Flights to and from home ({e(money(f.excluded, f.currency))}) are outside your '
                  f'budget, as you asked. The whole trip costs {e(money(f.total, f.currency))}.</p>')
    rows = list(f.per_traveller.items()) + ([("Shared: rooms, food, guide, local transport", f.shared)] if f.shared else [])
    top = max((a for _, a in rows), default=1) or 1
    bars = "".join(f'<div class="bar-row"><span>{e(who)}</span><div class="bar"><div style="width:{a / top * 100:.1f}%">'
                   f'</div></div><b class="num">{e(money(a, f.currency))}</b></div>' for who, a in rows)
    lines = "".join(f'<tr><td>{e(l.label)}</td><td>{e(l.traveller or "shared")}</td>'
                    f'<td class="num">{e(money(l.amount_report, f.currency))}</td><td>{notes.cite_cost(l)}</td></tr>'
                    for l in f.lines)
    reasons = "".join(f"<li>{e(r)}</li>" for r in f.reasons) + "".join(
        f"<li>Not priced, so not in the total: {e(r)}</li>" for r in f.unpriced)
    rates = (f'<p class="fine">Exchange rates: {e(f.fx_source)}, {e(f.fx_as_of)}. Fares and prices are estimates '
             f'from web search.</p>' if f.fx_source else "")
    return f"""<section><h2>Budget</h2>{meter}
  {f'<ul class="reasons">{reasons}</ul>' if reasons else ""}
  <div class="bars">{bars}</div>
  <details><summary>Every cost line</summary><div class="tablewrap"><table>
    <thead><tr><th>What</th><th>Who</th><th>Amount ({e(f.currency)})</th><th>Source</th></tr></thead><tbody>{lines}</tbody>
  </table></div></details>{rates}</section>"""


def alternatives(state: TripState) -> str:
    alts = state.get("alternatives") or []
    if not alts:
        return ""
    items = []
    for a in alts:
        f = a.feasibility
        cost = (f'{e(money(f.counted, f.currency))} · {e(f.status())}' if f else "not costed")
        chosen = '<span class="tag pick">chosen</span>' if a.title == state.get("chosen_alternative") else ""
        trade = f'<div class="muted">Trade-offs: {e("; ".join(a.trade_offs))}</div>' if a.trade_offs else ""
        items.append(f'<li><b>{e(a.title)}</b> {chosen}<div class="muted">{cost}</div><div>{e(a.why)}</div>'
                     f'{trade}</li>')
    return f'<section><h2>Alternatives considered</h2><ul class="alts">{"".join(items)}</ul></section>'


def double_check(state: TripState) -> str:
    checks = to_double_check(state)
    if not checks:
        return ""
    return ('<section><h2>Things to double-check</h2><ul class="checks">'
            + "".join(f"<li>{e(n)}</li>" for n in checks) + "</ul></section>")


def render_html(state: TripState) -> str:
    request = state["request"]
    notes = Notes(state.get("sources") or {})
    colors = city_colors(state)
    changed = f'<p class="callout">✏️ {e(version_note(state))}</p>' if version_note(state) else ""
    body = (hero(state) + '<main>' + changed + status(state) + timeline(state, colors) + getting_there(state)
            + visas(state, notes) + food_section(state, notes, colors) + budget(state, notes) + alternatives(state)
            + double_check(state) + notes.render()
            + f'<footer class="fine">Planned by TravelEx · {e(describe(request))}</footer></main>')
    return PAGE.replace("{title}", e(f"{request.destination} trip plan")).replace("{body}", body)


def render_preview_html(state: TripState) -> str:
    """The trip at a glance, shown as soon as the research is done: season, routes, visas and budget.
    The day-by-day plan and food follow; the full plan replaces this page when they are ready."""
    request = state["request"]
    notes = Notes(state.get("sources") or {})
    coming = ('<p class="callout">🗓 Your day-by-day plan and food guide are being written now. '
              'The full plan replaces this page when they are ready.</p>')
    body = (hero(state) + "<main>" + status(state) + coming + getting_there(state) + visas(state, notes)
            + budget(state, notes) + notes.render() + "</main>")
    return PAGE.replace("{title}", e(f"{request.destination} at a glance")).replace("{body}", body)


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"><title>{title}</title>
<base target="_blank">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=Nunito+Sans:wght@400;600;700&display=swap">
<style>
/* Layout: a bright sunrise header, then one column of cards; the timeline runs down the left edge. */
:root {
  --bg: #f3fbfb; --surface: #ffffff; --fg: #18323a; --muted: #5e7780; --line: #d9ecee;
  --sea: #0aa5a5; --sun: #ffb400; --coral: #ff6b57; --leaf: #2fae66; --sky: #3d8bfd;
  --good-bg: #e2f7ea; --good-fg: #17733f; --warn-bg: #fff3d6; --warn-fg: #8a5a00;
  --bad-bg: #ffe5e0; --bad-fg: #b2321f; --info-bg: #e3f0ff; --info-fg: #1f5fbf; --muted-bg: #eef2f3;
  --display: "Sora", "Segoe UI", system-ui, sans-serif;
  --body: "Nunito Sans", "Segoe UI", system-ui, -apple-system, sans-serif;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0f1d21; --surface: #16292e; --fg: #e6f3f4; --muted: #9db6bd; --line: #24414a;
  --good-bg: #133a26; --good-fg: #7fe0a6; --warn-bg: #3d2f0c; --warn-fg: #ffd277;
  --bad-bg: #44201a; --bad-fg: #ffa293; --info-bg: #142c4d; --info-fg: #9cc4ff; --muted-bg: #1d3238;
  color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #0f1d21; --surface: #16292e; --fg: #e6f3f4; --muted: #9db6bd; --line: #24414a;
  --good-bg: #133a26; --good-fg: #7fe0a6; --warn-bg: #3d2f0c; --warn-fg: #ffd277;
  --bad-bg: #44201a; --bad-fg: #ffa293; --info-bg: #142c4d; --info-fg: #9cc4ff; --muted-bg: #1d3238;
  color-scheme: dark }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font-family: var(--body); font-size: 15.5px; line-height: 1.55; }
a { color: var(--sea); }
.muted { color: var(--muted); }
.fine { color: var(--muted); font-size: 13px; }
h1, h2, h3 { font-family: var(--display); text-wrap: balance; margin: 0; }
h2 { font-size: 24px; display: flex; align-items: center; gap: 10px; }
h2::before { content: ""; width: 10px; height: 26px; border-radius: 4px;
  background: linear-gradient(var(--sun), var(--coral)); }
h3 { font-size: 17px; }

.hero { background: linear-gradient(120deg, #0aa5a5 0%, #19b6c9 38%, #ffb400 78%, #ff6b57 100%); color: #fff;
  padding-block: 36px 30px; padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px)); }
.hero-inner { max-width: 980px; margin: 0 auto; display: grid; gap: 12px; }
.eyebrow { text-transform: uppercase; letter-spacing: .14em; font-size: 12px; font-weight: 700; opacity: .9; }
.hero h1 { font-size: clamp(32px, 6vw, 56px); line-height: 1.02; text-shadow: 0 2px 12px rgba(0,0,0,.18); }
.facts { display: flex; flex-wrap: wrap; gap: 8px 18px; font-weight: 700; }
.people { display: flex; flex-wrap: wrap; gap: 8px; }
.person { background: rgba(255,255,255,.92); color: #18323a; border-radius: 999px; padding: 4px 12px; font-size: 14px; }

main { max-width: 980px; margin: 0 auto; display: grid; gap: 34px; padding-block: 26px 48px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
section { display: grid; gap: 14px; min-width: 0; }

.status { display: flex; flex-wrap: wrap; gap: 8px; }
.badge { display: inline-flex; align-items: center; gap: 6px; border-radius: 999px; padding: 4px 12px;
  font-weight: 700; font-size: 13.5px; }
.badge.good, .tag.good { background: var(--good-bg); color: var(--good-fg); }
.badge.warn, .tag.warn { background: var(--warn-bg); color: var(--warn-fg); }
.badge.bad, .tag.bad { background: var(--bad-bg); color: var(--bad-fg); }
.badge.info { background: var(--info-bg); color: var(--info-fg); }
.badge.muted { background: var(--muted-bg); color: var(--muted); }
.season-notes { margin: 0; padding-left: 20px; color: var(--muted); font-size: 14px; }

.tag { display: inline-block; border-radius: 6px; padding: 1px 7px; font-size: 12px; font-weight: 700;
  background: var(--muted-bg); color: var(--muted); margin-left: 6px; vertical-align: 1px; }
.tag.hard { background: var(--bad-bg); color: var(--bad-fg); }
.tag.rest { background: var(--good-bg); color: var(--good-fg); }
.tag.pick { background: var(--sun); color: #3b2a00; }
.tag.quoted { background: var(--good-bg); color: var(--good-fg); }
.tag.diet { background: var(--info-bg); color: var(--info-fg); }
.tag.kidtag { background: var(--warn-bg); color: var(--warn-fg); }
.tag.est { margin-left: 0; font-weight: 600; }
a.src { display: inline-block; max-width: 14em; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  vertical-align: middle; border-radius: 999px; padding: 1px 9px; font-size: 12px; font-weight: 600;
  text-decoration: none; background: var(--info-bg); color: var(--info-fg); }
a.src:hover { text-decoration: underline; }
.more { margin-left: 4px; font-size: 12px; font-weight: 600; color: var(--muted); cursor: help; }
.chip { display: inline-block; border-radius: 999px; padding: 2px 10px; font-size: 13px; font-weight: 700; margin: 2px 4px 2px 0; }
.chip.food { background: var(--warn-bg); color: var(--warn-fg); }

.timeline { list-style: none; margin: 0; padding: 0; display: grid; gap: 14px; position: relative; }
.timeline::before { content: ""; position: absolute; left: 19px; top: 8px; bottom: 8px; width: 3px;
  background: linear-gradient(var(--sea), var(--sun), var(--coral)); border-radius: 3px; }
.day { display: grid; grid-template-columns: 42px minmax(0, 1fr); gap: 12px; position: relative; }
.dot { width: 40px; height: 40px; border-radius: 50%; background: var(--city); color: #fff; font-family: var(--display);
  font-weight: 800; display: grid; place-items: center; box-shadow: 0 0 0 4px var(--bg); z-index: 1; }
.card { background: var(--surface); border: 1px solid var(--line); border-left: 5px solid var(--city);
  border-radius: 12px; padding: 12px 16px; display: grid; gap: 6px; min-width: 0; }
.day-head { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; font-size: 13.5px; }
.city { background: var(--city); color: #fff; border-radius: 999px; padding: 1px 10px; font-weight: 700; }
.transfer { font-size: 14px; color: var(--muted); }
.acts { margin: 0; padding-left: 18px; }
.kid { font-size: 13.5px; color: var(--warn-fg); }
.callout { background: var(--info-bg); color: var(--info-fg); border-radius: 10px; padding: 10px 14px; margin: 0; }
.callout.bad { background: var(--bad-bg); color: var(--bad-fg); margin-bottom: 10px; }
.tips { margin: 0; padding: 0; list-style: none; display: grid; gap: 4px; }

.groups, .visas, .foods { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; }
.group, .visa, .food-card { background: var(--surface); border: 1px solid var(--line); border-radius: 12px;
  padding: 14px 16px; display: grid; gap: 8px; align-content: start; min-width: 0; }
.option { border: 1px dashed var(--line); border-radius: 10px; padding: 8px 10px; display: grid; gap: 4px; }
.option.picked { border: 2px solid var(--sun); background: color-mix(in srgb, var(--sun) 8%, var(--surface)); }
.route { display: flex; flex-wrap: wrap; align-items: center; gap: 4px; font-weight: 700; }
.stop { background: var(--info-bg); color: var(--info-fg); border-radius: 6px; padding: 1px 7px; font-size: 13px; }
.arrow { color: var(--muted); }
.note { font-size: 13px; color: var(--muted); }

.visa { border-top: 5px solid var(--line); }
.visa.good { border-top-color: var(--leaf); } .visa.info { border-top-color: var(--sky); }
.visa.warn { border-top-color: var(--sun); } .visa.bad { border-top-color: var(--coral); }
.visa-head { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.flags { font-size: 22px; }
.visa p { margin: 0; }
details summary { cursor: pointer; color: var(--sea); font-weight: 700; }
.quotes { padding-left: 18px; margin: 8px 0 0; display: grid; gap: 8px; font-size: 14px; }
.who { color: var(--muted); font-size: 12.5px; }
a.fn { font-size: 11px; font-weight: 700; text-decoration: none; background: var(--info-bg); color: var(--info-fg);
  border-radius: 4px; padding: 0 4px; margin-left: 2px; vertical-align: 2px; }

.food-card { border-top: 5px solid var(--city); }
.dishes, .places { margin: 0; padding-left: 18px; display: grid; gap: 4px; font-size: 14.5px; }
.places { list-style: none; padding-left: 0; }
.diet-note { margin: 0; background: var(--good-bg); color: var(--good-fg); border-radius: 8px; padding: 6px 10px; font-size: 14px; }

.meter { height: 16px; border-radius: 999px; background: var(--muted-bg); overflow: hidden; }
.meter .fill { height: 100%; background: linear-gradient(90deg, var(--leaf), var(--sea)); border-radius: 999px; }
.meter.over .fill { background: linear-gradient(90deg, var(--sun), var(--coral)); }
.meter-label { font-size: 15px; }
.reasons { margin: 0; padding-left: 20px; color: var(--bad-fg); }
.bars { display: grid; gap: 8px; }
.bar-row { display: grid; grid-template-columns: minmax(90px, 220px) minmax(0, 1fr) auto; gap: 10px; align-items: center; font-size: 14px; }
.bar { height: 10px; background: var(--muted-bg); border-radius: 999px; overflow: hidden; }
.bar div { height: 100%; background: linear-gradient(90deg, var(--sea), var(--sky)); border-radius: 999px; }
.num { font-variant-numeric: tabular-nums; }
.tablewrap { overflow-x: auto; margin-top: 8px; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; background: var(--surface); }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .05em; }

.alts, .checks { margin: 0; padding-left: 20px; display: grid; gap: 8px; }
.checks li::marker { content: "⚠ "; }
.sources { margin: 0; padding-left: 24px; font-size: 14px; display: grid; gap: 2px; }
footer { border-top: 1px solid var(--line); padding-top: 12px; }
@media (max-width: 560px) { .bar-row { grid-template-columns: 1fr auto; } .bar { grid-column: 1 / -1; } }
@media print { .hero { -webkit-print-color-adjust: exact; print-color-adjust: exact; } details { display: block; } }
</style></head><body>{body}</body></html>"""
