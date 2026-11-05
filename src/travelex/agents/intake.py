"""Intake: free text -> TripRequest. Clarify: ask for what's missing, or confirm the trip card (interrupt)."""
import re
import time
from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import interrupt

from travelex.agents.common import event
from travelex.agents.food import allergies
from travelex.llm import get_llm, structured
from travelex.schemas import Traveller, TripRequest
from travelex.state import TripState
from travelex.agents.ideas import screen_ideas
from travelex.trip import (SENIOR_AGE, budget_means_money, group_by_origin, hinted_style, parse_duration,
                           resolve_nights, stop_lengths)

MAX_CLARIFY_ROUNDS = 3
RECHECKS = 2  # times the model is asked to correct a misread "from" list
QUESTIONS = {
    "travellers": "who is travelling (adults, and children with their ages)",
    "destination": "where you'd like to go",
    "passport_country": "the passport each traveller holds (visa rules depend on it)",
    "origin": "the city each traveller starts from",
    "travel_month": "when you'd like to travel (dates, or at least the month)",
    "travel_dates": "new travel dates, because the ones given are in the past",
    "duration": "how many days the trip should be",
    "child_ages": "each child's age (fares, fees and pacing depend on it)",
}

INTAKE_PROMPT = """You turn a traveller's free-text request into a TripRequest. Travellers can live anywhere and
go anywhere in the world. Today is {today}.

- travellers: one entry per person. "2 adults" is two entries; "my wife and I" is two adults.
  - Never add people silently. When the user gives no count ("my family", "we", "us"), list everyone they
    mention, add your best guess for the rest, and set assumed true on each guessed person.
  - label: short and unique, e.g. "Adult 1", "Child (9)", "Grandma".
  - type: "adult", "child" (2-17) or "infant" (under 2, set age_months if given).
  - passport: ISO 3166 alpha-2 code, ONLY when the user states a nationality or passport. Never guess it
    from the city someone lives in or flies from: a person flying from Dubai may hold any passport.
  - origin: the city that person starts from. When "from" is followed by more places than there are
    travellers, the first is home and the rest are stops: "2 adults from Delhi, Rome and Florence for 8 days"
    is two adults from Delhi visiting Rome and Florence. Every place the request names is either where someone
    starts or a stop; never drop one.
  - needs: "senior" for anyone {senior}+ or described as elderly; "wheelchair" or other access needs.
- stops: the countries or regions to visit, in order. place as the user said it; country as an ISO
  alpha-2 code (Bali → ID, Galápagos → EC, Ladakh → IN, Zanzibar → TZ). Put named cities in cities.
  Set nights on a stop only when the user gives that stop's length: "4 nights in Paris, then 5 in Tokyo" is
  Paris 4 and Tokyo 5 (9 nights in all), not a 4-night trip.
- trip_type: "land" for an ordinary trip, including one with a boat tour or a day cruise as an activity.
  "cruise" when the user wants a cruise holiday (days aboard a cruise ship): TravelEx can't plan cruises, so
  leave stops empty. "around_cruise" only when the user asks to plan the time around their own cruise: stops are
  the port where they board (nights = nights before boarding, 2 if not said) and, if the cruise ends at another
  port, that port (nights = nights after, 1 if not said); cruise_nights is the cruise's length; leave
  duration_text, duration_range and nights empty. "land" again when they choose a land trip instead.
- A region or vague place ("a European country", "somewhere in Asia", "the Caribbean") is not a destination:
  leave stops empty and suggest destination_ideas.
- If the user has not picked a destination, leave stops empty and suggest 2-3 destination_ideas that fit
  everything they said: what they want to do and see; a good season in their travel month (not monsoon, extreme
  heat or cold, or closures); easy to reach from every traveller's home city; entry their passports can manage;
  and realistic for the budget per person per day, flights included. Make them different from each other. Give
  place, ISO country code, and why in at most 6 words.
- Dates: exact dates → start_date, end_date (end = last day) and exact_dates true. Only a month or season → month "YYYY-MM"
  and leave start_date and end_date empty: never invent days the user didn't give. Use the next such
  month on or after today unless the user gives a year. Keep a year the user gives, even if it is past.
- Length: copy the user's own words into duration_text, e.g. "4-5 days", "a week", "5 nights".
  Also: "N days" → duration_range [N, N]; "4-5 days" → [4, 5]; "a week" → [7, 7]; "N nights" → nights N.
  Leave all three empty if the user didn't say.
- budget: amount and ISO currency. "$" is USD unless the user's context says otherwise; "₹3 lakh" is
  300000 INR; "₹1,50,000" is 150000 INR. Leave empty if not said. flights_in_budget false only when the user
  says the budget leaves out flights ("excluding airfare", "plus flights").
- diet: e.g. "vegetarian", "halal", "peanut allergy". interests: e.g. "hiking", "photography".
- clarification_needed: one friendly question asking for everything still missing (passports, travel
  month, children's ages, destination…). When there is no destination, offer the ideas in it. Else null.
- raw_query: the user's text, copied exactly."""


def unique_labels(travellers: list[Traveller]) -> list[Traveller]:
    seen: dict[str, int] = {}
    out = []
    for t in travellers:
        n = seen[t.label] = seen.get(t.label, 0) + 1
        out.append(t if n == 1 else t.model_copy(update={"label": f"{t.label} {n}"}))
    return out


def normalise(t: Traveller) -> Traveller:
    update = {}
    if t.type == "child" and t.age is not None and t.age < 2:
        update["type"] = "infant"
    if t.type == "adult" and t.age is not None and t.age >= SENIOR_AGE and "senior" not in t.needs:
        update["needs"] = [*t.needs, "senior"]
    return t.model_copy(update=update) if update else t


CRUISE_QUESTION = ("TravelEx plans land trips: it can't find, book or price cruises yet. I can plan a land trip to "
                   "the same region instead, or plan the time around your cruise: flights to the port where you "
                   "board, a night or two there before (and after), visas and costs. Which would you like? For the "
                   "second, tell me the port where the cruise starts and ends, and how many nights it lasts.")
# Plain cruise-holiday wording, a backstop for the model ("a dinner cruise in Paris" is an activity, not this)
CRUISE_TRIP = re.compile(r"\bcruise\s+(trip|holiday|vacation|ship|line|package)s?\b|\bon\s+a\s+cruise\b|"
                         r"\bgo(ing)?\s+cruising\b|\b\d+[-\s]?(night|day)s?\s+cruise\b|"
                         r"\bcruise\s+(from|to|around|across|through)\b", re.IGNORECASE)


def missing_fields(request: TripRequest, past: bool) -> list[str]:
    checks = {
        "trip_type": request.trip_type == "cruise",
        "travellers": not request.travellers,
        "party_size": any(t.assumed for t in request.travellers),
        "destination": not request.stops,
        "passport_country": any(t.passport is None for t in request.travellers),
        "origin": any(not t.origin for t in request.travellers),
        "travel_dates": past,
        "travel_month": not past and not (request.start_date or request.month),
        "duration": resolve_nights(request)[0] is None,
        "child_ages": any(t.type == "child" and t.age is None for t in request.travellers),
    }
    return [name for name, missing in checks.items() if missing]


DATE_FIELDS = ("start_date", "end_date", "exact_dates", "month")
LENGTH_FIELDS = ("duration_text", "duration_range", "nights")
SIMPLE_FIELDS = ("stops", "budget", "diet", "interests", "lodging_style")
TRAVELLER_FIELDS = ("passport", "age", "age_months", "origin", "needs")
AFFIRMATIVE = re.compile(r"(y|yes|yeah|yep|ok|okay|sure|correct|confirm(ed)?|right|fine|good|perfect|"
                         r"looks (good|right|fine)|that'?s (right|correct)|go ahead|proceed|plan it)[.! ]*", re.IGNORECASE)


FROM_PHRASE = re.compile(r"\bfrom\s+[^.;:]*?(?=\s+(?:for|in|on|with|during|budget|next|to)\b|[.;:]|$)", re.IGNORECASE)


def from_lists(raw: str) -> list[tuple[str, list[str]]]:
    """Each "from A, B and C" in the request with the place names it lists (capitalised words only, so
    "from London, around £3,000" lists just London)."""
    found = []
    for m in FROM_PHRASE.finditer(raw or ""):
        items = [i.strip() for i in re.split(r",\s*|\s+(?:and|&)\s+", m.group(0)[5:])]
        names = [i for i in items if i and all(w[0].isupper() for w in i.split())]
        if len(names) > 1:
            found.append((m.group(0), names))
    return found


def origin_conflicts(request: TripRequest, raw: str = "") -> list[str]:
    """Travellers who 'start' from a place the trip visits, or no destination while several home cities are
    read from one "from" list: misreadings, never a real plan."""
    visiting = {p.casefold() for s in request.stops for p in [s.place, *s.cities]}
    conflicts = [f"{t.label} starts from {t.origin}, which is on the trip" for t in request.travellers
                 if t.origin and t.origin.casefold() in visiting]
    homes = list(dict.fromkeys(t.origin for t in request.travellers if t.origin))
    if not request.stops and len(homes) > 1 and request.trip_type != "cruise":
        # "2 adults from Kuala Lumpur, Seoul and Busan": the places after the home city are the trip
        said = next((m.group(0) for m in FROM_PHRASE.finditer(raw) if sum(h.casefold() in m.group(0).casefold()
                                                                           for h in homes) > 1), "")
        quote = f' Read "{said}" again: when it names more places than there are travellers, the first is home ' \
                f'and the rest are stops; a place it names that is neither is missing.' if said else ""
        conflicts.append(f"the trip has no destination, yet the travellers start from {', '.join(homes)}. If the "
                         f"request lists places to visit after a home city, they are stops.{quote}")
    placed = visiting | {h.casefold() for h in homes}
    for said, names in from_lists(raw):  # live, the corrected reading of that list then dropped Busan
        if lost := [n for n in names if not any(n.casefold() in p or p in n.casefold() for p in placed)]:
            conflicts.append(f'"{said}" names {", ".join(lost)}, which is neither a home city nor a stop')
    return conflicts


def empty(value) -> bool:
    return value is None or value == [] or value is False or value == ""


def keep_known(previous: TripRequest, parsed: TripRequest) -> TripRequest:
    """A re-parse after an answer may add or change details, never forget them: the model sometimes rewrites
    the trip and drops passports or ages it was given. Dates and length are restored only as a whole, and only
    when the new parse has none, so "make it January" still replaces the old dates."""
    update = {f: getattr(previous, f) for f in SIMPLE_FIELDS if empty(getattr(parsed, f)) and not empty(getattr(previous, f))}
    for group in (DATE_FIELDS, LENGTH_FIELDS):
        if all(empty(getattr(parsed, f)) for f in group):
            update |= {f: getattr(previous, f) for f in group}
    old = previous.travellers
    if not parsed.travellers:
        update["travellers"] = old
    else:
        by_label = {t.label: t for t in old}
        same_size = len(old) == len(parsed.travellers)
        travellers = []
        for i, t in enumerate(parsed.travellers):
            before = by_label.get(t.label) or (old[i] if same_size else None)
            fill = {f: getattr(before, f) for f in TRAVELLER_FIELDS
                    if before and empty(getattr(t, f)) and not empty(getattr(before, f))}
            travellers.append(t.model_copy(update=fill) if fill else t)
        update["travellers"] = travellers
    return parsed.model_copy(update=update)


ALLERGEN_WORD = r"(?:(?:pea|ground|tree\s)?nuts?|sesame|shellfish|seafood|fish|eggs?|dairy|lactose|milk|gluten|wheat|soy)"
STATED_ALLERGY = re.compile(rf"\b{ALLERGEN_WORD}[\s-]+(?:allerg(?:y|ies|ic)|intoleran(?:ce|t))\b|"
                            rf"\ballergic\s+to\s+{ALLERGEN_WORD}\b|\b(?:coeliac|celiac)\b", re.IGNORECASE)


def stated_allergies(request: TripRequest) -> TripRequest:
    """Allergies the request states, added to diet when the model left them out: live, "one of us has a nut
    allergy" was dropped, so nothing filtered the food guides for nuts."""
    have = set(allergies(request.diet))
    missing = [m.group(0).lower() for m in STATED_ALLERGY.finditer(request.raw_query or "")
               if not set(allergies([m.group(0)])) <= have]
    return request.model_copy(update={"diet": request.diet + list(dict.fromkeys(missing))}) if missing else request


def keep_after_recheck(first: TripRequest, corrected: TripRequest) -> TripRequest:
    """The corrected reading of a misread "from" list keeps everything else the first reading had: live, it
    dropped a nut allergy and a traveller's home city. A home city that turned out to be a stop becomes the home
    city that is left ("from Kuala Lumpur, Seoul and Busan": Kuala Lumpur)."""
    visiting = {p.casefold() for s in corrected.stops for p in [s.place, *s.cities]}
    home = next((t.origin for t in first.travellers if t.origin and t.origin.casefold() not in visiting), None)
    travellers = [t.model_copy(update={"origin": home}) if t.origin and t.origin.casefold() in visiting else t
                  for t in first.travellers]
    merged = keep_known(first.model_copy(update={"travellers": travellers, "stops": []}), corrected)
    lists = {f: getattr(merged, f) + [x for x in getattr(first, f) if x.casefold() not in
                                      {y.casefold() for y in getattr(merged, f)}] for f in ("diet", "interests")}
    return merged.model_copy(update=lists)


def consistent_stops(request: TripRequest) -> TripRequest:
    """Nights on the stops must fit the trip: "7 days, 1st week of December" is 6 nights, even if the model also
    wrote 7 nights on the only stop. When they don't fit the dates or length, the stops' nights are dropped."""
    if not (request.start_date and request.end_date or request.duration_range or request.nights):
        return request  # the stops' nights are the only length there is
    total, _ = resolve_nights(request)
    fixed = [s.nights for s in request.stops if s.nights]
    if not fixed or (sum(fixed) < total and len(fixed) < len(request.stops)) or sum(fixed) == total:
        return request
    return request.model_copy(update={"stops": [s.model_copy(update={"nights": None}) for s in request.stops]})


def distinct_stops(request: TripRequest) -> TripRequest:
    """Every stop has its own place name: rates, journeys and checks are filed by it. "Thailand: Bangkok, Chiang
    Mai and Phuket" read as three "Thailand" stops becomes Bangkok, Chiang Mai and Phuket; one place split over
    two stops in a row is one stop. A place visited again later ("Tokyo, Kyoto, back to Tokyo") stays as said."""
    counts: dict[str, int] = {}
    for s in request.stops:
        counts[s.place.casefold()] = counts.get(s.place.casefold(), 0) + 1
    if all(n == 1 for n in counts.values()):
        return request
    named = [s.model_copy(update={"place": s.cities[0], "cities": s.cities[1:]})
             if counts[s.place.casefold()] > 1 and s.cities else s for s in request.stops]
    stops = []
    for s in named:
        last = stops[-1] if stops else None
        if last and last.place.casefold() == s.place.casefold() and last.country == s.country:
            nights = last.nights + s.nights if last.nights and s.nights else None
            cities = last.cities + [c for c in s.cities if c.casefold() not in {x.casefold() for x in last.cities}]
            stops[-1] = last.model_copy(update={"cities": cities, "nights": nights})
        else:
            stops.append(s)
    return request.model_copy(update={"stops": stops})


def around_cruise(request: TripRequest) -> TripRequest:
    """Plan only the time on land: nights before boarding at the first port (2 unless said) and after leaving the
    ship at a different last port (1 unless said). The trip's length is those nights, not the whole holiday.
    A cruise that ends where it started ("Venice → Venice") is one stop: the nights before boarding."""
    ports = request.stops[:2]
    if len(ports) == 2 and ports[0].place.casefold() == ports[1].place.casefold():
        ports = ports[:1]
    defaults = [2, 1]
    stops = [s.model_copy(update={"nights": s.nights or defaults[min(i, 1)]}) for i, s in enumerate(ports)]
    month = request.month or (f"{request.start_date:%Y-%m}" if request.start_date else None)
    return request.model_copy(update={"stops": stops, "duration_text": None, "duration_range": None, "nights": None,
                                      "start_date": None, "end_date": None, "exact_dates": False, "month": month})


def tidy(request: TripRequest, today: date) -> TripRequest:
    """Code-side fixes and checks: labels, infants, seniors, groups, past dates and what's still missing."""
    travellers = unique_labels([normalise(t) for t in request.travellers])
    request = request.model_copy(update={"travellers": travellers, "groups": group_by_origin(travellers)})
    if request.trip_type == "around_cruise":
        request = around_cruise(request)
    request = stated_allergies(distinct_stops(request))
    if request.start_date and not request.exact_dates:  # the model invented days from "6 days in April"
        request = request.model_copy(update={"month": request.month or f"{request.start_date:%Y-%m}",
                                             "start_date": None, "end_date": None})
    per_stop, rest = ({}, "") if request.trip_type == "around_cruise" else \
        stop_lengths(f"{request.duration_text or ''} | {request.raw_query or ''}", request.stops)
    if per_stop:  # "4 nights in Paris, then 5 in Tokyo": each stop's own length; the trip is their sum
        stops = [s.model_copy(update={"nights": per_stop.get(s.place, s.nights)}) for s in request.stops]
        request = request.model_copy(update={"stops": stops, "nights": None, "duration_range": None})
        days, nights = parse_duration(rest.split(" | ")[0])  # a whole-trip length said besides ("two weeks: …")
    elif request.trip_type == "around_cruise":
        days, nights = None, None  # "three weeks" is the whole holiday; we plan only the nights on land
    else:
        days, nights = parse_duration(request.duration_text)
    if days or nights:  # the user's own words beat the model's arithmetic ("6 days" is 5 nights, not 6)
        request = request.model_copy(update={"duration_range": days, "nights": nights})
    request = consistent_stops(request)
    past = bool((request.start_date and request.start_date < today)
                or (request.month and request.month < f"{today:%Y-%m}"))
    missing = missing_fields(request, past)
    flags = [f for f in request.flags if f != "date_in_past"] + (["date_in_past"] if past else [])
    question = request.clarification_needed
    asks = [QUESTIONS[m] for m in missing if m in QUESTIONS]
    if asks and not question:
        question = "To plan this I still need " + "; ".join(asks) + "."
    if past and "past" not in (question or ""):
        question = f"The travel date you gave is in the past (today is {today}). {question}"
    ideas, left_out = screen_ideas(request) if "destination" in missing else ([], [])  # a place is chosen: no ideas
    request = request.model_copy(update={"destination_ideas": ideas})
    screened = left_out or any(i.note for i in ideas)
    if "destination" in missing and ideas and (screened or not any(i.place.lower() in (question or "").lower()
                                                                     for i in ideas)):
        # the model asked about something else (exact dates…), never offered its places, or code screened them
        others = [QUESTIONS[m] for m in missing if m in QUESTIONS and m != "destination"]
        question = (f"Where would you like to go? A few ideas that fit: {'; '.join(i.label() for i in ideas)}."
                    + (f" I left out {'; '.join(left_out)}." if left_out else "")
                    + (f" I also need {'; '.join(others)}." if others else ""))
    if request.trip_type == "cruise":  # say plainly what TravelEx can't do, and what it can
        others = [QUESTIONS[m] for m in missing if m in QUESTIONS and m != "destination"]
        question = CRUISE_QUESTION + (f" I also need {'; '.join(others)}." if others else "")
    if guessed :=[t.label for t in travellers if t.assumed]:  # never plan for a guessed party without saying so
        question = f"I guessed who's coming ({', '.join(guessed)}): is that right? {question or ''}".strip()
    return request.model_copy(update={"missing": missing, "flags": flags,
                                      "clarification_needed": question if missing else None})


def describe(request: TripRequest) -> str:
    """One line for the trace and the trip card."""
    people = ", ".join(f"{t.label}{f' {t.age}' if t.age is not None else ''}{' (assumed)' if t.assumed else ''} "
                       f"[{t.passport or '?'}] "
                       f"from {t.origin or '?'}" for t in request.travellers)
    when = (f"{request.start_date} → {request.end_date}" if request.start_date
            else request.month or "dates unknown")
    nights, trimmed = resolve_nights(request)
    length = f"{nights} nights" + (f" (or {trimmed})" if trimmed else "") if nights else "length unknown"
    budget = f"{request.budget.amount:,.0f} {request.budget.currency}" if request.budget else "no budget"
    if request.budget and not request.flights_in_budget:
        budget += " (flights not included)"
    return f"{request.destination} · {when} · {length} · {budget} · {people or 'no travellers'}"


def intake(state: TripState) -> dict:
    started = time.perf_counter()
    raw, today = state["raw_query"], state["today"]
    text = f"Request: {raw}"
    previous = state.get("request")
    if answer := state.get("clarify_answer"):
        current = previous.model_dump_json(exclude={"raw_query", "groups", "missing", "flags"}) if previous else "{}"
        text += (f"\n\nWhat you understood so far: {current}"
                 f"\nYou asked: {previous.clarification_needed if previous else 'Can you clarify?'}"
                 f"\nThe user answered: {answer}\nUpdate the request with the answer and keep everything else.")
    prompt = INTAKE_PROMPT.format(today=today.isoformat(), senior=SENIOR_AGE)
    messages = [SystemMessage(prompt), HumanMessage(text)]
    parsed = structured(get_llm("intake"), TripRequest, messages)
    for _ in range(RECHECKS):  # "from Chennai, Lisbon and Porto" read as three home cities
        if not (conflicts := origin_conflicts(parsed, raw)):
            break
        answer_was = parsed.model_dump_json(include={"stops", "travellers"})
        messages.append(HumanMessage(f"Your answer had {answer_was}\nCheck it: {'; '.join(conflicts)}. "
                                     f"A place the travellers visit is a "
                                     f"stop, not where someone starts; keep a home city only where the request says "
                                     f"someone lives or flies from there. Return the corrected TripRequest."))
        parsed = keep_after_recheck(parsed, structured(get_llm("intake"), TripRequest, messages))
    if previous is not None and state.get("clarify_answer"):
        parsed = keep_known(previous, parsed)
    elif parsed.trip_type == "land" and CRUISE_TRIP.search(raw):  # the model planned a cruise as a land trip
        parsed = parsed.model_copy(update={"trip_type": "cruise", "stops": []})
    request = tidy(parsed.model_copy(update={"raw_query": raw}), today)
    if request.lodging_style == "budget" and budget_means_money(request):  # "Budget GBP 9000" is money, not hotels
        request = request.model_copy(update={"lodging_style": None})
    if request.lodging_style is None and (hint := hinted_style(request)):  # backs up the model: "cheapest possible"
        request = request.model_copy(update={"lodging_style": hint})
    return {"request": request, "clarify_answer": None,
            "events": [event("intake", "parsed request", describe(request), started=started)]}


def clarify(state: TripState, config: RunnableConfig) -> dict:
    """Pause with the trip card and, if anything is missing, a question.
    Resume with a str (an answer, sent back to intake) or a dict (the edited request, accepted as is)."""
    request, rounds = state["request"], state.get("clarify_round", 0)
    question = request.clarification_needed if rounds < MAX_CLARIFY_ROUNDS else None
    if config.get("configurable", {}).get("auto_approve"):
        if request.trip_type == "cruise" and rounds < MAX_CLARIFY_ROUNDS:  # nobody to ask: plan what we can
            pick = "TravelEx can't plan cruises: plan a land trip to the same region instead."
            return {"clarify_answer": pick, "clarify_round": rounds + 1,
                    "events": [event("clarify", "cruise not supported", pick)]}
        if not request.stops and request.destination_ideas and rounds < MAX_CLARIFY_ROUNDS:
            pick = f"Let's go to {request.destination_ideas[0].place}"  # nothing to plan without a place: take the first
            return {"clarify_answer": pick, "clarify_round": rounds + 1,
                    "events": [event("clarify", "auto-picked a destination", pick)]}
        if not request.stops:  # nobody to ask and no place to plan: end, and say why
            return {"clarify_round": MAX_CLARIFY_ROUNDS,
                    "events": [event("clarify", "no destination", "nothing to plan: the request names no place")]}
        detail = f"proceeding without: {', '.join(request.missing)}" if request.missing else describe(request)
        return {"events": [event("clarify", "auto-approved", detail)]}

    answer = interrupt({"kind": "clarify", "question": question, "missing": request.missing,
                        "ideas": [i.place for i in request.destination_ideas],
                        "idea_labels": {i.place: i.label() for i in request.destination_ideas},
                        "request": request.model_dump(mode="json")})

    if isinstance(answer, str) and AFFIRMATIVE.fullmatch(answer.strip()):
        answer = request.model_dump(mode="json")  # "yes" means accept the card, not new information
    if isinstance(answer, str) and answer.strip():
        return {"clarify_answer": answer, "clarify_round": rounds + 1,
                "events": [event("clarify", "user answered", answer)]}
    edited = request if isinstance(answer, str) else TripRequest.model_validate({**answer, "raw_query": request.raw_query})
    seen = [t.model_copy(update={"assumed": False}) for t in edited.travellers]  # the user saw the card and agreed
    confirmed = tidy(edited.model_copy(update={"travellers": seen}), state["today"])
    if not confirmed.stops:  # accepted without a place: ask again rather than end with nothing planned
        return {"request": confirmed, "clarify_round": rounds + 1,
                "events": [event("clarify", "needs a destination", "the card was accepted without one")]}
    return {"request": confirmed, "clarify_round": rounds + 1,
            "events": [event("clarify", "user confirmed trip", describe(confirmed))]}


def route_after_clarify(state: TripState) -> str:
    if state.get("clarify_answer"):
        return "intake"
    if state["request"].stops:
        return "plan_research"
    # no destination yet: ask again, until the question rounds run out
    return "clarify" if state.get("clarify_round", 0) < MAX_CLARIFY_ROUNDS else END
