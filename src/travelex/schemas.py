"""Pydantic models shared by the agents, graph, report and UI."""
import datetime as dt
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

TravellerType = Literal["adult", "child", "infant"]
Verdict = Literal["good", "ok_with_caveats", "avoid", "unknown"]
Mode = Literal["flight", "flight+overland", "surface"]
LegMode = Literal["flight", "train", "bus", "road"]
CostCategory = Literal["transport", "destination_fee", "visa", "lodging", "food", "guide", "vehicle",
                       "local_transport", "other"]
Provenance = Literal["seed", "learned"]  # seed: curated in code; learned: extracted from cited sources by an agent


def _country_code(value: str | None) -> str | None:
    if not value:
        return value
    value = value.strip().upper()
    if value != "*" and (len(value) != 2 or not value.isalpha()):
        raise ValueError(f'"{value}" is not an ISO 3166 alpha-2 country code, e.g. IN, GB, JP')
    return value


# ---------- request ----------

class Traveller(BaseModel):
    label: str = Field(description='Short unique name, e.g. "Adult 1", "Child (9)", "Grandma"')
    type: TravellerType
    age: int | None = Field(None, ge=0, le=120, description="Age in years, when known")
    age_months: int | None = Field(None, ge=0, le=35, description="Infants only")
    passport: str | None = Field(None, description="ISO 3166 alpha-2 country code, e.g. IN, GB, AE")
    origin: str | None = Field(None, description="Departure city")
    needs: list[str] = Field(default_factory=list, description='e.g. "senior", "wheelchair"')
    assumed: bool = Field(False, description='True only for a person the user never mentioned, guessed from '
                                             'words like "my family" or "we"')

    @field_validator("passport")
    @classmethod
    def _code(cls, value):
        return _country_code(value)


class Stop(BaseModel):
    """One country or region on the route. A single-country trip has one stop."""
    place: str = Field(description='Country or region as the user said it, e.g. "Japan", "Bali", "Galápagos"')
    country: str = Field(description="ISO 3166 alpha-2 code of the country it is in, e.g. JP, ID, EC")
    cities: list[str] = Field(default_factory=list, description="Cities or sights the user named here")
    nights: int | None = Field(None, ge=1, description="Only when the user said how long to stay here")

    @field_validator("country")
    @classmethod
    def _code(cls, value):
        return _country_code(value)


class TravelGroup(BaseModel):
    id: str  # "G1", "G2"…
    origin: str
    travellers: list[str]  # Traveller labels


class Money(BaseModel):
    amount: float = Field(ge=0)
    currency: str = Field(description="ISO 4217 code, e.g. USD, INR, AED")

    @field_validator("currency")
    @classmethod
    def _upper(cls, value):
        return value.strip().upper()


class DestinationIdea(BaseModel):
    """A place to suggest when the user hasn't picked one. Code screens it before it is offered."""
    place: str = Field(description='As a traveller would say it, e.g. "Bali" or "Sri Lanka"')
    country: str = Field("", description="ISO 3166 alpha-2 code, e.g. ID")
    why: str = Field("", description='Why it fits, at most 6 words, e.g. "dry season, monasteries, cheap"')
    note: str = Field("", description="Filled by code; leave empty")

    @model_validator(mode="before")
    @classmethod
    def _from_text(cls, value):
        return {"place": value} if isinstance(value, str) else value  # older requests stored plain names

    def label(self) -> str:
        """What the traveller sees, e.g. "Sri Lanka (dry season, beaches) — a visa in advance for IN passports"."""
        return self.place + (f" ({self.why})" if self.why else "") + (f" — {self.note}" if self.note else "")


class TripRequest(BaseModel):
    raw_query: str | None = None
    stops: list[Stop] = Field(default_factory=list, description="In travel order; empty if the user hasn't chosen")
    trip_type: Literal["land", "cruise", "around_cruise"] = Field(
        "land", description='"land": an ordinary trip (a boat tour or day cruise as one activity is still land). '
                            '"cruise": the user wants a cruise holiday, which TravelEx cannot plan. '
                            '"around_cruise": plan only the time around the user\'s own cruise')
    cruise_nights: int | None = Field(None, ge=1, description="The cruise's own length, for around_cruise")
    destination_ideas: list["DestinationIdea"] = Field(
        default_factory=list, description="2-3 suggestions when the user hasn't chosen a destination")
    travellers: list[Traveller] = Field(default_factory=list)
    groups: list[TravelGroup] = Field(default_factory=list, description="Filled by code from origins")
    start_date: date | None = None
    end_date: date | None = None
    exact_dates: bool = Field(False, description="True only if the user typed specific calendar days")
    month: str | None = Field(None, description='"YYYY-MM" when only the month is known')
    duration_text: str | None = Field(None, description='The user\'s own words for the length, copied exactly, '
                                                       'e.g. "4-5 days", "two weeks", "5 nights"')
    duration_range: list[int] | None = Field(None, min_length=2, max_length=2,
                                             description='[fewest, most] days, e.g. [4, 5] for "4-5 days"')
    nights: int | None = Field(None, ge=1)
    budget: Money | None = None
    flights_in_budget: bool = Field(True, description='False when the user says the budget leaves out flights, '
                                                      'e.g. "excluding airfare", "plus flights"')
    diet: list[str] = Field(default_factory=list, description='e.g. "vegetarian", "halal", "nut allergy"')
    interests: list[str] = Field(default_factory=list)
    lodging_style: Literal["budget", "mid-range", "luxury"] | None = Field(
        None, description='Only when the user asks for it, e.g. "cheapest possible" → budget, '
                          '"high-end hotels" or "villa" → luxury. Else empty: code picks a level to fit the budget. '
                          '"Budget GBP 9000" or "our budget is $5000" is the money, not a hotel level')
    missing: list[str] = Field(default_factory=list, description="Fields to ask the user for")
    flags: list[str] = Field(default_factory=list, description='e.g. "date_in_past"')
    clarification_needed: str | None = Field(None, description="A question for the user, if anything is missing")

    @property
    def destination(self) -> str:
        """For display: "Italy → Switzerland"."""
        where = " → ".join(s.place for s in self.stops) or "destination to be chosen"
        return f"{where} (around your cruise)" if self.trip_type == "around_cruise" and self.stops else where


# ---------- sources ----------

class Source(BaseModel):
    id: str
    url: str
    title: str
    kind: Literal["web", "news", "page_extract"]
    published: date | None = None
    retrieved_at: datetime
    text: str

    @field_validator("published", mode="before")
    @classmethod
    def _parse_published(cls, value):
        """Tavily gives ISO dates for web results and RFC 2822 ('Thu, 01 Oct 2026 ...') for news."""
        if not value or isinstance(value, date):
            return value or None
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            pass
        try:
            return parsedate_to_datetime(value).date()
        except (TypeError, ValueError):
            return None

    def render(self) -> str:
        """What the LLM sees for this source."""
        when = f", {self.published}" if self.published else ""
        return f"[{self.id}] {self.title} ({self.url}{when})\n{self.text}"


# ---------- season ----------

class MonthNote(BaseModel):
    verdict: Literal["good", "ok_with_caveats", "avoid"]
    reason: str = ""


class ClimateProfile(BaseModel):
    """Twelve months of travel advice for one place. Seeded in code or researched by the season agent."""
    place: str
    months: dict[int, MonthNote] = Field(description="Month number 1-12 → verdict; all 12 months")
    vulnerable_avoid: dict[int, str] = Field(default_factory=dict,
                                             description="Months to avoid only with children or seniors → why")
    vulnerable_note: str = Field("", description='Always relevant for kids or seniors, e.g. "3,500 m altitude"')
    source_ids: list[str] = Field(default_factory=list)
    provenance: Provenance = "learned"

    @field_validator("months")
    @classmethod
    def _all_twelve(cls, months: dict[int, MonthNote]) -> dict[int, MonthNote]:
        bad = [m for m in months if not 1 <= m <= 12]
        if bad:
            raise ValueError(f"months must be 1-12, got {bad}")
        return {m: months.get(m) or MonthNote(verdict="ok_with_caveats", reason="no data") for m in range(1, 13)}


class MonthEntry(BaseModel):
    month: int = Field(ge=1, le=12)
    verdict: Literal["good", "ok_with_caveats", "avoid"]
    reason: str = ""


class MonthReason(BaseModel):
    month: int = Field(ge=1, le=12)
    reason: str


class ClimateResearch(BaseModel):
    """What the season agent extracts. Lists, not dicts: OpenAI structured output rejects dict fields."""
    months: list[MonthEntry] = Field(description="One entry per month, 1-12")
    vulnerable_avoid: list[MonthReason] = Field(default_factory=list,
                                                description="Months to avoid only with children or seniors")
    vulnerable_note: str = Field("", description='Always relevant for kids or seniors, e.g. "3,500 m altitude"')
    source_ids: list[str] = Field(default_factory=list)

    def to_profile(self, place: str) -> ClimateProfile:
        return ClimateProfile(place=place,
                              months={e.month: MonthNote(verdict=e.verdict, reason=e.reason) for e in self.months},
                              vulnerable_avoid={e.month: e.reason for e in self.vulnerable_avoid},
                              vulnerable_note=self.vulnerable_note, source_ids=self.source_ids)


class SeasonVerdict(BaseModel):
    place: str
    month: int | None = Field(None, ge=1, le=12, description="None when the travel month is unknown")
    verdict: Verdict
    reasons: list[str] = Field(default_factory=list)
    stricter_for_vulnerable: bool = False  # True when kids or seniors changed the verdict


# ---------- visas & fees ----------

SourceKind = Literal["destination_government", "home_government", "other_government", "third_party"]
Relevance = Literal["applies", "all_visitors", "unchecked"]


class Citation(BaseModel):
    """A quote code found word for word in a fetched page (grounded), judged to be about this traveller's
    passport and destination (relevant). Stored with the URL, so it survives caching."""
    url: str
    title: str = ""
    quote: str
    supports: str = ""  # which claim: "requirement", "fee", "stay", "how_to_apply"
    retrieved_at: datetime | None = None
    context: str = ""  # the text around the quote on the page
    source_kind: SourceKind = "third_party"
    relevance: Relevance = "unchecked"  # "applies": to this passport; "all_visitors": to every nationality
    relevance_note: str = ""


class QuoteVerdict(BaseModel):
    index: int = Field(description="The quote's number")
    verdict: Literal["applies", "all_visitors", "other_nationality", "other_destination", "other_purpose",
                     "contradicts", "unclear"]
    reason: str = Field(description="A few words, e.g. 'about US citizens' or 'names Indian nationals'")


class RelevanceCheck(BaseModel):
    verdicts: list[QuoteVerdict]


class Evidence(BaseModel):
    source_id: str = Field(description="ID of a numbered source, e.g. G1.4")
    quote: str = Field(max_length=300, description="Copied character for character from that source, "
                                                   "one sentence or less")
    supports: Literal["requirement", "fee", "stay", "how_to_apply"]


class VisaRule(BaseModel):
    passport: str = Field(description='ISO code of the passport, or "*" for every passport without its own rule')
    country: str = Field(description="ISO code of the country being entered")
    zone: str | None = Field(None, description='A shared visa area, e.g. "Schengen": one visa covers all of it')
    visa_required: bool
    document: str = Field(description='"visa", "e_visa", "visa_on_arrival", "eta", "entry_permit" or "none"')
    fee: float | None = None
    fee_currency: str = "USD"
    max_stay_days: int | None = None
    accepted_ids: list[str] = Field(default_factory=list)
    notes: str = ""
    source_url: str = ""
    verified_on: date | None = None  # None: not yet checked by a person against source_url
    provenance: Provenance = "learned"
    confirmed: bool = Field(False, description="Filled by code; leave false")
    unconfirmed: list[str] = Field(default_factory=list, description="Filled by code; leave empty")  # "fee", "stay"
    citations: list[Citation] = Field(default_factory=list, description="Filled by code; leave empty")
    cited_for: str | None = Field(None, description="Filled by code; leave empty")  # country the quotes were checked for

    @field_validator("passport", "country")
    @classmethod
    def _code(cls, value):
        return _country_code(value)


class VisaResearch(BaseModel):
    """What the visa agent extracts: the rule, and a quote for each claim. Code keeps only quotes it finds."""
    rule: VisaRule
    evidence: list[Evidence] = Field(description="A quote for the requirement, and for the fee, stay and how to "
                                                 "apply when the rule states them")


class VisaCheck(BaseModel):
    traveller: str
    passport: str | None
    country: str
    rule: VisaRule | None  # None: nothing known, see summary and sources
    summary: str = ""
    confirmed_by_search: bool = False
    source_ids: list[str] = Field(default_factory=list)


class AgeBand(BaseModel):
    min_age: int = Field(ge=0)
    max_age: int = Field(ge=0)
    share: float = Field(ge=0, le=1, description="Share of the adult amount, e.g. 0.5 for half price, 0 if exempt")
    label: str


class FeeRule(BaseModel):
    """A per-person tourist fee or levy, e.g. Bhutan's SDF, the Galápagos park fee or Bali's tourist levy."""
    country: str
    place: str | None = Field(None, description="Region or city it applies to; None for the whole country")
    name: str
    basis: Literal["per_night", "per_trip"]
    amount: float = Field(ge=0, description="Adult amount")
    currency: str
    passports: list[str] | None = Field(None, description="Only these passports; None = everyone else")
    age_bands: list[AgeBand] = Field(default_factory=list, description="Reduced rates; older ages pay the adult amount")
    source_url: str = ""
    verified_on: date | None = None
    provenance: Provenance = "learned"
    source_id: str = Field("", description="The source that states this fee")
    quote: str = Field("", max_length=300, description="The source's words that state the fee, copied exactly")
    confirmed: bool = Field(True, description="Filled by code: a source's exact words state this amount")


class FeeLine(BaseModel):
    traveller: str
    name: str
    place: str
    basis: Literal["per_night", "per_trip"]
    unit_amount: float  # after the age-band share
    currency: str
    units: int  # nights, or 1 for per_trip
    total: float
    band: str  # e.g. "adult", "child 6-12 (50%)", "under 6 (exempt)"
    source_url: str = ""
    confirmed: bool = True  # False: researched, but no source's words state the amount


# ---------- transport ----------

class Leg(BaseModel):
    origin: str
    destination: str
    mode: LegMode
    carrier: str | None = None
    depart: datetime | None = None
    arrive: datetime | None = None
    overnight: bool = False


class RouteOption(BaseModel):
    mode: Mode
    route: str = Field(description='IATA codes or places joined by "-", e.g. "MAA-CCU-PBH"')
    legs: list[Leg] = Field(default_factory=list)
    adult_fare: Money = Field(description="Price for one adult, as the source gives it (one-way for a leg between stops)")
    fare_covers: Literal["return", "one_way", "unclear"] = Field(
        "return", description='What the source\'s price covers: "return", "one_way", or "unclear" if it doesn\'t say')
    child_fare: Money | None = Field(None, description="Quoted child fare, only when a source gives one")
    fare_basis: Literal["quoted", "estimated"] = Field(
        "estimated", description='"quoted" only when a source states this price for this route')
    evidence: str = Field("", max_length=300, description="Exact words copied from one source that show this "
                                                          "route exists (and its price, if quoted)")
    warnings: list[str] = Field(default_factory=list, description="Filled by code; leave empty")
    duration_hours: float | None = None
    max_layover_hours: float | None = None
    kid_note: str = ""
    accessibility_note: str = ""
    source_ids: list[str] = Field(default_factory=list)

    def arrival(self) -> datetime | None:
        return self.legs[-1].arrive if self.legs else None


class FareQuote(BaseModel):
    """One flight price a source states, pooled with the others into a typical fare."""
    source_id: str
    amount: float
    currency: str
    covers: Literal["return", "one_way", "unclear"]
    words: str = Field(max_length=200, description="The words around the price, copied exactly from the source")


class RouteList(BaseModel):
    home_airports: list[str] = Field(description="IATA codes of the airports serving the group's home city, "
                                                 'e.g. ["DXB", "DWC"] for Dubai')
    options: list[RouteOption] = Field(description="2-3 options, cheapest first; empty if nothing was found")
    fare_quotes: list[FareQuote] = Field(default_factory=list,
                                         description="Every flight price the sources state for this journey")


class GroupResult(BaseModel):
    group_id: str
    options: list[RouteOption] = Field(default_factory=list)
    chosen: int = 0  # index into options
    visas: list[VisaCheck] = Field(default_factory=list)
    status: Literal["ok", "failed"] = "ok"
    key: str = ""  # the inputs it was researched for; equal keys mean the result can be reused

    def chosen_option(self) -> RouteOption | None:
        return self.options[self.chosen] if 0 <= self.chosen < len(self.options) else None


# ---------- destination costs ----------

LodgingStyle = Literal["budget", "mid-range", "luxury"]


class LodgingRate(BaseModel):
    style: LodgingStyle
    per_room_night: float = Field(ge=0)


class LodgingCost(BaseModel):
    style: LodgingStyle
    counted: float  # what the budget would cover at this hotel level


CostItem = Literal["budget hotel", "mid-range hotel", "luxury hotel", "food", "local transport", "vehicle", "guide",
                   "other"]


CostUnit = Literal["per room-night", "per person-day", "per day", "per trip"]


class CostEvidence(BaseModel):
    """The words behind one of a stop's amounts. Code keeps it only if the source contains the words and they
    state this amount, in this currency and unit, for this place, and it matches the rate recorded."""
    item: CostItem
    source_id: str
    quote: str = Field(max_length=200, description="The words that state the price, copied exactly")
    amount: float = Field(description="The price the quote states, as a number")
    currency: str = Field(description="ISO code of the quote's price")
    unit: CostUnit = Field(description='"per room-night" for hotels, "per person-day" for food and local '
                                       'transport, "per day" for a vehicle or guide, "per trip" for other')


class StopCosts(BaseModel):
    """One stop's rates found by research. Totals are computed by feasibility, never by the LLM."""
    place: str
    currency: str = Field(description="ISO code; every amount below is in it")
    lodging_per_room_night: float = Field(ge=0)
    lodging_rates: list[LodgingRate] = Field(default_factory=list,
                                             description="A typical room at each level: budget, mid-range, luxury")
    rooms: int = Field(ge=1)
    room_config: str = Field(description='e.g. "1 family room + 1 double"')
    food_per_person_day: float = Field(ge=0)
    guide_per_day: float = Field(0, ge=0)
    guide_required: bool = Field(False, description="True where visitors must have a licensed guide")
    vehicle_needed: bool = Field(False, description="True only where visitors can't get around by public transport, "
                                                    "taxis or day tours, or must travel with a guide and driver")
    vehicle_per_day: float = Field(0, ge=0, description="Car or van with driver, if vehicle_needed")
    local_transport_per_person_day: float = Field(0, ge=0, description="Buses, trains, metro and taxis for one "
                                                                       "traveller, when no vehicle is needed")
    vehicle_capacity: int | None = None
    vehicle_accessible: bool = False
    other_per_trip: float = Field(0, ge=0, description="Entry tickets, permits and similar, for the whole party")
    fees: list[FeeRule] = Field(default_factory=list, description="Tourist fees and levies; empty if there are none")
    onward: RouteOption | None = Field(None, description="Getting from this stop to the next stop, one way")
    notes: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    evidence: list[CostEvidence] = Field(default_factory=list,
                                         description="For each amount a source states, the words that state it")
    key: str = Field("", description="Filled by code; leave empty")

    def room_rate(self, style: str | None) -> float:
        """A room at that level, falling back to the rate researched for the requested level."""
        return next((r.per_room_night for r in self.lodging_rates if r.style == style), self.lodging_per_room_night)


class Gap(BaseModel):
    """Something the research couldn't back with a source: for the gap-filling agent, then the traveller."""
    id: str
    kind: Literal["cost", "fare", "route"]
    key: str  # the stop's place, or the travel group's id
    item: str  # "budget hotel", "food", "local transport", "fare"…
    text: str
    estimate_usd: float | None = None  # what the research estimated: a sourced price far from it is a misread
    status: Literal["open", "filled", "gave_up"] = "open"
    note: str = ""  # the source that filled it, or why it couldn't be filled


# ---------- meeting & feasibility ----------

class MeetingOption(BaseModel):
    kind: Literal["meet_at_destination", "meet_at_hub_fly_together"]
    place: str
    note: str = ""


class MeetingPlan(BaseModel):
    meeting_point: str
    day1: date | None
    arrivals: dict[str, date | None]  # group id → arrival date at the destination
    buffer_nights: dict[str, int]  # group id → nights spent waiting for the others
    options: list[MeetingOption]


class CostLine(BaseModel):
    category: CostCategory
    label: str
    traveller: str | None = None  # None = shared by the whole party
    amount: float
    currency: str
    amount_report: float  # in the budget currency
    source_ids: list[str] = Field(default_factory=list)  # this run's sources behind the amount
    urls: list[str] = Field(default_factory=list)  # pages behind it when there is no source ID (fee and visa rules)
    quote: str = ""  # the words that state the amount, or how it was worked out


class Feasibility(BaseModel):
    ok: bool
    currency: str  # the budget currency; every total is in it
    total: float  # the whole trip
    budget: float | None
    excluded: float = 0  # in total but outside the budget: flights to and from home, when the user leaves them out
    gap: float  # counted - budget; > 0 means over budget
    lodging_style: str | None = None  # the hotel level these costs use
    lodging_chosen: bool = False  # True when code picked that level to fit the budget
    lodging_others: list[LodgingCost] = Field(default_factory=list)  # the same trip at the other levels

    @property
    def counted(self) -> float:
        """What the budget has to cover."""
        return round(self.total - self.excluded, 2)

    @property
    def spare(self) -> float:
        """Budget left for activities and extras."""
        return max(round(-self.gap, 2), 0) if self.budget is not None else 0

    @property
    def complete(self) -> bool:
        """Every journey and stop the trip needs has a price."""
        return not self.unpriced

    def status(self) -> str:
        """The budget verdict in a few words. A total with missing prices never just "fits"."""
        if self.gap > 0:
            return f"over budget by {self.gap:,.0f} {self.currency}"
        if self.reasons:
            return "does not fit"
        if self.unpriced:
            n = len(self.unpriced)
            return f"{'fits so far' if self.budget is not None else 'costed so far'}, but {n} item{'s' * (n > 1)} not priced"
        return "fits the budget" if self.budget is not None else "costed"
    reasons: list[str] = Field(default_factory=list)  # why the plan doesn't fit, budget-wise
    unpriced: list[str] = Field(default_factory=list)  # what the trip needs but research couldn't price
    lines: list[CostLine] = Field(default_factory=list)
    per_traveller: dict[str, float] = Field(default_factory=dict)
    shared: float = 0
    conversions: list[str] = Field(default_factory=list)  # e.g. "INR->AED"
    fx_as_of: date | None = None
    fx_source: str | None = None


# ---------- alternatives ----------

class TripChanges(BaseModel):
    """What an alternative changes. Leave a field empty to keep it as it is."""
    stops: list[Stop] | None = Field(None, description="A different destination or route, in travel order")
    month: str | None = Field(None, description='A different travel month, "YYYY-MM"')
    duration_text: str | None = Field(None, description='A different length, e.g. "4 days", "6 nights"')
    lodging_style: Literal["budget", "mid-range", "luxury"] | None = None
    budget: Money | None = Field(None, description="Only if the fix is to spend more")


class PlanComment(BaseModel):
    """A traveller's comment on the finished plan, sorted into what it changes."""
    summary: str = Field(description='The change asked for, in a few words, e.g. "slower day 3, add a fjord cruise"')
    trip_changes: TripChanges = Field(description="Only changes to the trip itself: destination or stops, month, "
                                                  "length, hotel level or a bigger budget. Empty for changes to "
                                                  "the days.")

    def changes_trip(self) -> bool:
        return any(v is not None for v in self.trip_changes.model_dump().values())


class AlternativeIdea(BaseModel):
    title: str = Field(description='Short, e.g. "Nepal instead of Bhutan" or "Go in October, not July"')
    why: str = Field(description="Which problem it fixes and what the traveller gives up, in one sentence")
    changes: TripChanges


class AlternativeIdeas(BaseModel):
    ideas: list[AlternativeIdea] = Field(min_length=1, max_length=3)


class Alternative(BaseModel):
    """An idea after code applied it and re-ran the research: everything needed to switch to it."""
    title: str
    why: str
    changes: TripChanges
    request: TripRequest
    feasibility: Feasibility | None = None
    season: dict[str, SeasonVerdict] = Field(default_factory=dict)
    group_results: dict[str, GroupResult] = Field(default_factory=dict)
    costs: dict[str, StopCosts] = Field(default_factory=dict)
    meeting: MeetingPlan | None = None
    sources: dict[str, Source] = Field(default_factory=dict)
    error: str | None = None
    trade_offs: list[str] = Field(default_factory=list)  # worked out by code: what changes, and what is still wrong

    def fixes(self) -> bool:
        """Fits the budget and no stop is in a season to avoid."""
        return bool(self.feasibility and self.feasibility.ok) and all(
            v.verdict != "avoid" for v in self.season.values())


# ---------- itinerary & food ----------

class Activity(BaseModel):
    name: str
    strenuous: bool = False
    kid_option: str = ""  # e.g. "pony ride up, or skip and visit the base café"


class DayPlan(BaseModel):
    day: int = Field(ge=1)
    date: dt.date | None = None  # dt.date: the field name shadows `date` inside this class
    city: str
    title: str
    activities: list[Activity] = Field(default_factory=list)
    transfer: str = ""  # e.g. "Drive Paro → Thimphu, 1.5 h"
    rest_day: bool = False


class Itinerary(BaseModel):
    days: list[DayPlan]
    trimmed_variant: str = ""  # how to cut it to the shorter end of a duration range
    notes: list[str] = Field(default_factory=list)


class Dish(BaseModel):
    name: str
    description: str
    diet_tags: list[str] = Field(default_factory=list)  # "vegetarian", "contains nuts"… checked against the ingredients
    kid_friendly: bool = False
    # Filled by code: a dish that names an allergen on the trip is dropped; every dish kept lists the trip's
    # allergens here as unknown, because no source confirms it is free of them.
    allergens_unknown: list[str] = Field(default_factory=list, description="Filled by code; leave empty")


class Place(BaseModel):
    name: str
    area: str = ""
    why: str = ""
    source_ids: list[str] = Field(default_factory=list)


class FoodGuide(BaseModel):
    city: str
    dishes: list[Dish]
    places: list[Place] = Field(default_factory=list)
    diet_notes: str = ""


# ---------- QA ----------

class QAReport(BaseModel):  # built by code; only `notes` may come from the LLM
    passed: bool
    issues: list[str] = Field(default_factory=list)  # e.g. "day_count", "food_missing:Punakha"
    notes: list[str] = Field(default_factory=list)
