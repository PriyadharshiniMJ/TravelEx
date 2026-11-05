"""Checks that a quoted price backs the number it is cited for: the words are the source's own, they state that
amount, in that currency and unit, about this place. Used by the costs agent, the fee research and gap filling."""
import re

from travelex.agents.common import context_of, squash
from travelex.schemas import Source
from travelex.tools import fx

NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
RANGE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(?:-|–|to)\s*\D{0,4}?(\d[\d,]*(?:\.\d+)?)")
NEAR_RATE = 0.25  # a quoted amount more than 25% from the recorded rate is about something else

# Currency symbols and the codes each can stand for ("$" is many dollars; "¥" is yen or yuan)
SYMBOLS = {"$": {"USD", "AUD", "CAD", "SGD", "NZD", "HKD", "MXN", "TWD", "BSD", "FJD"}, "€": {"EUR"}, "£": {"GBP"},
           "¥": {"JPY", "CNY"}, "₹": {"INR"}, "฿": {"THB"}, "₩": {"KRW"}, "₱": {"PHP"}, "₫": {"VND"}, "₺": {"TRY"},
           "₪": {"ILS"}, "₦": {"NGN"}, "rp": {"IDR"}, "rm": {"MYR"}, "rs": {"INR", "LKR", "NPR", "PKR"},
           "nu": {"BTN"}, "dh": {"AED", "MAD"}, "aed": {"AED"}, "r$": {"BRL"}, "chf": {"CHF"}, "kr": {"NOK", "SEK", "DKK", "ISK"}}
WORD_SYMBOLS = re.compile(r"(?<![a-z])(rp|rm|rs|nu|dh|kr|chf|r\$)\.?\s*\d", re.IGNORECASE)
CODE = re.compile(r"\b([A-Z]{3})\b")

# A quote must say positively what it prices and per what: a number in the right currency is not enough
# ("Hotels in Bali cost USD 100 per room per night" is no evidence for a USD 100 tourist levy).
NIGHT = re.compile(r"\bnight(s|ly)?\b|/\s*night|\bpn\b", re.I)
DAY = re.compile(r"\b(a|per|each|every)\s+day\b|\bdaily\b|/\s*day\b|\bper\s+day\b|\bday\b", re.I)
UNIT_SAID = {"per room-night": NIGHT, "per person-day": DAY, "per day": DAY, "per trip": None}
ITEM_WORDS = {
    "hotel": re.compile(r"hotel|\broom|accommodation|guest\s?house|hostel|\bstay|villa|resort|lodg|\binns?\b|"
                        r"homestay|b&b|bed and breakfast|apartment|airbnb|ryokan|riad|motel|camp", re.I),
    "food": re.compile(r"food|meal|\beat|restaurant|dining|lunch|dinner|breakfast|caf[eé]|snack|warung|dish", re.I),
    "local transport": re.compile(r"\bbus|metro|train|taxi|tuk|transport|\bride|grab|uber|subway|tram|ferry|transit|"
                                  r"scooter|bike|travel ?card|\bpass\b", re.I),
    "vehicle": re.compile(r"\bcar\b|driver|vehicle|\bvan\b|jeep|hire|rental|chauffeur|4x4|suv", re.I),
    "guide": re.compile(r"guide", re.I),
    "other": re.compile(r"ticket|\bentry|entrance|admission|permit|\bfees?\b|\bpass\b", re.I),
}
FEE_WORDS = re.compile(r"\b(levy|levies|tax|taxes|fees?|charges?|surcharge|sdf|duty|permit|tariff)\b", re.I)
PER_TRIP = re.compile(r"\bper\s+(person|visitor|traveller|traveler|tourist|entry|trip|visit|stay|head|arrival)\b|"
                      r"\bone[-\s]time\b|\b(on|upon)\s+arrival\b|\beach\s+(person|visitor|traveller|traveler|tourist)\b|"
                      r"\bonce\b|\bper\s+(foreign\s+)?(visitor|tourist|national)s?\b", re.I)

# Words that show a price is for another unit than the one recorded
WRONG_UNIT = {
    "per room-night": re.compile(r"\bper (week|month|year)\b|\bweekly\b|\bmonthly\b|\bfor \d+ nights\b|\bpackage\b|"
                                 r"\bper\s+(person|guest|head|adult|bed|pax)\b|\bpp\b|\bdorm", re.I),
    "per person-day": re.compile(r"\bper (week|month|year|couple|family)\b|\bweekly\b|\bmonthly\b|\bfor two\b", re.I),
    "per day": re.compile(r"\bper (week|month|year|hour|km)\b|\bhourly\b|\bweekly\b|\bmonthly\b", re.I),
    "per trip": None,
}
UNIT_OF = {"budget hotel": "per room-night", "mid-range hotel": "per room-night", "luxury hotel": "per room-night",
           "food": "per person-day", "local transport": "per person-day", "vehicle": "per day", "guide": "per day",
           "other": "per trip"}


def verbatim(quote: str, text: str) -> str:
    """The part of a quote the source really contains. Models wrap a source's words in a label ("[R.3] …") or a
    lead-in of their own with the real words in quotation marks; the words themselves still have to be exact."""
    candidates = [quote, re.sub(r"^\s*\[[^\]]*\]\s*", "", quote),
                  *re.findall(r'"([^"]{8,})"', quote), *re.findall(r"“([^”]{8,})”", quote)]
    return next((c.strip() for c in candidates if c.strip() and squash(c) in squash(text)), "")


def number(n: str) -> float:
    return float(n.replace(",", ""))


def stated(amount: float, quote: str) -> bool:
    """The amount is one of the quote's numbers, or inside a range it gives ("$8 to $12 a day")."""
    if any(abs(number(n) - amount) <= 0.01 * max(amount, 1) for n in NUMBER.findall(quote)):
        return True
    return any(number(a) <= amount <= number(b) for a, b in RANGE.findall(quote))


def currencies_in(quote: str) -> set[str]:
    """The currencies a quote names, by ISO code or symbol; empty when it names none."""
    found = {c for c in CODE.findall(quote) if fx.known(c)}
    found |= {code for sym, codes in SYMBOLS.items() if len(sym) == 1 and sym in quote for code in codes}
    found |= {code for m in WORD_SYMBOLS.finditer(quote) for code in SYMBOLS[m.group(1).lower()]}
    return found


def currency_agrees(currency: str, quote: str) -> bool:
    """A known code, and the quote doesn't name a different currency. A quote with no currency can't disagree."""
    if not fx.known(currency):
        return False
    named = currencies_in(quote)
    return not named or currency.upper() in named


def near(amount: float, rate: float, quote: str) -> bool:
    """The quoted amount is the rate recorded (within NEAR_RATE), or the quote gives a range the rate falls in."""
    if rate <= 0:
        return False
    if abs(amount - rate) <= NEAR_RATE * rate:
        return True
    return any(number(a) <= rate <= number(b) for a, b in RANGE.findall(quote))


def unit_fits(unit: str, quote: str) -> bool:
    """The quote says the unit ("a night", "per day") and nothing that contradicts it ("per person", "weekly")."""
    if unit not in WRONG_UNIT:
        return False
    said, wrong = UNIT_SAID[unit], WRONG_UNIT[unit]
    return (said is None or bool(said.search(quote))) and not (wrong and wrong.search(quote))


def item_named(item: str, quote: str) -> bool:
    """The quote says what it prices: a hotel or room, food, transport, a car, a guide, a ticket."""
    return bool(ITEM_WORDS["hotel" if item.endswith(" hotel") else item].search(quote))


def fee_supported(basis: str, quote: str) -> bool:
    """A quote backs a tourist fee only if it names a fee, levy or tax and how it is charged: per night, or once
    per person (per visitor, on arrival). Anything less leaves the fee unconfirmed."""
    if not FEE_WORDS.search(quote):
        return False
    per_night = bool(NIGHT.search(quote))
    return per_night if basis == "per_night" else bool(PER_TRIP.search(quote)) and not per_night


def about(source: Source, quote: str, names: list[str]) -> bool:
    """The page, around the quote or in its title, names this place (the stop, a city in it, or its country)."""
    around = squash(source.title + " " + (context_of(quote, source.text) or source.text))
    return any(squash(n) in around for n in names if n)


def price_problem(amount: float, currency: str, unit: str, quote: str, source: Source, names: list[str],
                  rate: float | None = None, rate_currency: str | None = None, item: str | None = None) -> str:
    """Why this quote doesn't back the price, or "" when it does. rate: the amount recorded for it, in
    rate_currency; the quoted amount must match it."""
    if not stated(amount, quote):
        return f"{amount:g} is not stated in the quote"
    if not currency_agrees(currency, quote):
        return f"the quote is not in {currency}"
    if item and not item_named(item, quote):
        return f"the quote doesn't say it prices {item}"
    if not unit_fits(unit, quote):
        return f"the quote is not a price {unit}"
    if not about(source, quote, names):
        return f"the source doesn't say this is about {names[0]}"
    if rate is not None:
        try:
            in_rate_currency = fx.convert(amount, currency, rate_currency or currency)
        except fx.FxError:
            return f"unknown currency {currency}"
        if not near(in_rate_currency, rate, quote):
            return f"{amount:g} {currency} is not the {rate:g} {rate_currency} recorded"
    return ""
