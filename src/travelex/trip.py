"""Pure helpers over a TripRequest: grouping travellers, splitting nights across stops, spotting kids and seniors."""
import math
import re

from travelex.schemas import Stop, TravelGroup, Traveller, TripRequest

SENIOR_AGE = 65


def group_by_origin(travellers: list[Traveller]) -> list[TravelGroup]:
    """One group per departure city, in the order the cities first appear. Matching ignores case."""
    groups: dict[str, tuple[str, list[str]]] = {}
    for t in travellers:
        origin = (t.origin or "unknown").strip()
        groups.setdefault(origin.casefold(), (origin, []))[1].append(t.label)
    return [TravelGroup(id=f"G{i}", origin=origin, travellers=labels)
            for i, (origin, labels) in enumerate(groups.values(), 1)]


NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fourteen": 14, "fifteen": 15,
                "twenty": 20, "thirty": 30}
UNIT_DAYS = {"day": 1, "night": 1, "week": 7, "fortnight": 14}
NUMBER = r"(\d+|" + "|".join(NUMBER_WORDS) + r")"
LENGTH = re.compile(rf"{NUMBER}(?:\s*(?:-|–|to|or)\s*{NUMBER})?\s*(day|night|week|fortnight)s?\b", re.IGNORECASE)


def parse_duration(text: str | None) -> tuple[list[int] | None, int | None]:
    """The user's words → (duration_range in days, nights). "4-5 days" → ([4, 5], None); "5 nights" → (None, 5);
    "two weeks" → ([14, 14], None); "a fortnight" → ([14, 14], None). Unparseable → (None, None)."""
    match = LENGTH.search(text or "")
    if not match:
        return None, None
    low, high, unit = match.group(1), match.group(2) or match.group(1), match.group(3).lower()
    low, high = (int(n) if n.isdigit() else NUMBER_WORDS[n.lower()] for n in (low, high))
    if unit == "night":
        return None, high
    return [low * UNIT_DAYS[unit], high * UNIT_DAYS[unit]], None


def resolve_nights(request: TripRequest) -> tuple[int | None, int | None]:
    """(nights, trimmed_nights). "N days" is N-1 nights; a range plans the upper end, trimmed to the lower."""
    if request.start_date and request.end_date:
        return (request.end_date - request.start_date).days, None
    if request.duration_range:
        low, high = sorted(request.duration_range)
        return max(high - 1, 1), (max(low - 1, 1) if low < high else None)
    if request.nights:
        return request.nights, None
    if request.stops and all(s.nights for s in request.stops):
        return sum(s.nights for s in request.stops), None
    return None, None


def cruise_leg(request: TripRequest, i: int) -> bool:
    """The way from stop i to the next is the travellers' own cruise (boarding at the first stop, leaving the ship
    at the second): nothing for TravelEx to research or price."""
    return request.trip_type == "around_cruise" and i == 0 and len(request.stops) == 2


def next_stop(request: TripRequest, i: int) -> Stop | None:
    """The stop after stop i that the travellers travel on to themselves: none after the last stop, or when the
    cruise takes them there."""
    return request.stops[i + 1] if i + 1 < len(request.stops) and not cruise_leg(request, i) else None


def allocate_nights(request: TripRequest) -> list[tuple[Stop, int]]:
    """Nights per stop: the user's own numbers, then the rest split evenly (earlier stops get the remainder)."""
    total, _ = resolve_nights(request)
    fixed = sum(s.nights or 0 for s in request.stops)
    free = [s for s in request.stops if not s.nights]
    share, extra = divmod(max((total or fixed) - fixed, 0), len(free)) if free else (0, 0)
    plan = []
    for stop in request.stops:
        if stop.nights:
            plan.append((stop, stop.nights))
        else:
            plan.append((stop, share + (1 if extra > 0 else 0)))
            extra -= 1
    return plan


def trip_months(request: TripRequest) -> list[int]:
    """Every calendar month the trip touches, in order."""
    if request.start_date:
        end = request.end_date or request.start_date
        months, y, m = [], request.start_date.year, request.start_date.month
        while (y, m) <= (end.year, end.month):
            months.append(m)
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        return months
    if request.month:
        return [int(request.month[5:7])]
    return []


def is_senior(t: Traveller) -> bool:
    return "senior" in t.needs or (t.age is not None and t.type == "adult" and t.age >= SENIOR_AGE)


def is_vulnerable(t: Traveller) -> bool:
    """Children, infants and seniors: they change the season verdict, pacing and transport rules."""
    return t.type in ("child", "infant") or is_senior(t)


def has_vulnerable(request: TripRequest) -> bool:
    return any(is_vulnerable(t) for t in request.travellers)


# Words that ask for a hotel level. Never the bare word "budget": "Budget GBP 9000" is the money, not the hotels.
STYLE_HINTS = {"budget": ("cheap", "on a budget", "budget hotel", "budget accommodation", "budget stay",
                          "budget travel", "budget trip", "budget option", "tight budget", "low budget", "shoestring",
                          "backpack", "student", "hostel", "guesthouse"),
               "luxury": ("luxury", "villa", "5-star", "five-star", "resort", "high-end", "high end", "upscale",
                          "premium", "boutique")}
MONEY_BUDGET = re.compile(r"\bbudget\b\s*(?:of|is|:|=|around|about|up to|under)?\s*(?:[A-Z]{3}|[$€£¥₹]|rs\.?)?\s*\d",
                          re.IGNORECASE)


def budget_means_money(request: TripRequest) -> bool:
    """The user's "budget" is an amount ("Budget GBP 9000") and nothing asks for budget hotels: a "budget" hotel
    level read from it is a misreading."""
    return bool(MONEY_BUDGET.search(request.raw_query or "")) and hinted_style(request) != "budget"


# "4 nights in Paris", "Tokyo for 5 nights", "then 5 in Tokyo" (the unit said once for all)
WORD_NUMBERS = "|".join(NUMBER_WORDS)
STOP_LENGTH = (r"\b(?P<n>\d+|{words})\s*(?P<unit>nights?|days?)?\s+(?:in|at)\s+(?:the\s+)?{place}\b"
               r"|\b{place}\s+(?:for\s+)?(?P<n2>\d+|{words})\s*(?P<unit2>nights?|days?)\b")


LIST_BEFORE = re.compile(r"(?:,|(?:^|\s)(?:and|&))\s*$", re.IGNORECASE)


def count_of(word: str) -> int:
    return int(word) if word.isdigit() else NUMBER_WORDS[word.lower()]


def stop_lengths(text: str, stops: list[Stop]) -> tuple[dict[str, int], str]:
    """Nights per stop the user gave in their own words, by stop place, and the text with those phrases removed
    (what's left may still give the whole trip's length: "two weeks: 4 nights in Paris…"). "N days in X" on a
    trip with several stops counts as N nights there, except the last stop, where the trip ends: N - 1."""
    found: dict[str, tuple[int, str | None]] = {}
    rest = text or ""
    for stop in stops:
        for name in dict.fromkeys([stop.place, *stop.cities]):
            pattern = STOP_LENGTH.format(words=WORD_NUMBERS, place=re.escape(name))
            # "Seoul and Busan for 7 days" is the whole list's length, not Busan's
            in_list = lambda m: m.group("n2") and LIST_BEFORE.search(rest[max(0, m.start() - 6):m.start()])
            if match := next((m for m in re.finditer(pattern, rest, re.IGNORECASE) if not in_list(m)), None):
                unit = match.group("unit") or match.group("unit2")
                found[stop.place] = (count_of(match.group("n") or match.group("n2")),
                                     unit.lower().rstrip("s") if unit else None)
                rest = rest[:match.start()] + rest[match.end():]
                break
    units = {u for _, u in found.values() if u}
    if not units or len(units) > 1:
        return {}, text or ""  # no unit anywhere ("2 in Tokyo" could be people), or nights and days mixed
    days = units == {"day"}
    last = stops[-1].place
    nights = {place: max(count - (1 if days and place == last else 0), 1) for place, (count, _) in found.items()}
    return nights, rest


def hinted_style(request: TripRequest) -> str | None:
    """A hotel level the user's own words ask for ("cheapest possible", "high-end hotels"), if any."""
    words = " ".join([request.raw_query or "", *request.interests]).lower()
    return next((style for style, hints in STYLE_HINTS.items() if any(h in words for h in hints)), None)


def style_for(request: TripRequest) -> str:
    """Hotel level for cost research: what the user chose, else mid-range (the budget check prices every level)."""
    return request.lodging_style or "mid-range"


def rooms_needed(request_or_travellers) -> int:
    """At least one room per 2 adults, and per 4 people with children; infants share a bed."""
    travellers = getattr(request_or_travellers, "travellers", request_or_travellers)
    adults = sum(t.type == "adult" for t in travellers)
    children = sum(t.type == "child" for t in travellers)
    return max(math.ceil(adults / 2), math.ceil((adults + children) / 4), 1)


def party(request: TripRequest) -> str:
    adults = sum(t.type == "adult" for t in request.travellers)
    kids = [t.age for t in request.travellers if t.type == "child"]
    infants = sum(t.type == "infant" for t in request.travellers)
    needs = sorted({n for t in request.travellers for n in t.needs})
    return (f"{adults} adults" + (f", children aged {', '.join(str(a) for a in kids)}" if kids else "")
            + (f", {infants} infant(s)" if infants else "") + (f"; needs: {', '.join(needs)}" if needs else ""))


def _when(request: TripRequest) -> str:
    return f"{request.start_date}..{request.end_date}" if request.start_date else request.month or "?"


def group_key(request: TripRequest, group: TravelGroup) -> str:
    """Transport and visas depend on who travels (passport, age, needs such as a wheelchair) from where, every stop
    and city on the way, and when. Not on the number of nights, so a shorter trip reuses them."""
    by_label = {t.label: t for t in request.travellers}
    who = [f"{t.label}:{t.passport}:{t.type}:{t.age}:{','.join(sorted(t.needs))}"
           for t in (by_label[label] for label in group.travellers if label in by_label)]
    stops = [f"{s.place}({','.join(s.cities)}){s.country}" for s in request.stops] or ["?"]
    return "|".join([group.origin.casefold(), ";".join(who), ";".join(stops), _when(request)])


def stop_key(request: TripRequest, stop: Stop, next_stop: Stop | None) -> str:
    """A stop's rates depend on the place, where you go next, when, the hotel level and the party."""
    return "|".join([stop.place, ",".join(stop.cities), next_stop.place if next_stop else "-", _when(request),
                     style_for(request), party(request)])
