"""Month-by-month season verdicts for any place, stricter when kids or seniors travel.

climate_for() returns a curated seed profile, else one the season agent learned, else None (research it).
"""
from langsmith import traceable

from travelex.schemas import ClimateProfile, MonthNote, SeasonVerdict
from travelex.tools import knowledge


def months(reason: str, *numbers: int) -> dict[int, str]:
    return dict.fromkeys(numbers, reason)


def profile(place: str, good: set[int], caveats: dict[int, str], avoid: dict[int, str] | None = None,
            vulnerable_avoid: dict[int, str] | None = None, vulnerable_note: str = "") -> ClimateProfile:
    notes = {m: MonthNote(verdict="ok_with_caveats", reason=caveats.get(m, "")) for m in range(1, 13)}
    notes |= {m: MonthNote(verdict="good") for m in good}
    notes |= {m: MonthNote(verdict="avoid", reason=r) for m, r in (avoid or {}).items()}
    return ClimateProfile(place=place, months=notes, vulnerable_avoid=vulnerable_avoid or {},
                          vulnerable_note=vulnerable_note, provenance="seed")


HIMALAYA_EAST_CAVEATS = {**months("cold; high passes may close with snow", 12, 1, 2),
                         **months("late monsoon showers", 9),
                         **months("monsoon: landslides can cut mountain roads", 6, 7, 8)}
LADAKH = profile(
    "Ladakh", {6, 7, 8, 9},
    {**months("cold nights; some high passes may already be shut", 5, 10),
     **months("extreme cold (-15 to -25 °C); the Manali and Srinagar roads are closed, fly in only",
              11, 12, 1, 2, 3, 4)},
    vulnerable_avoid=months("too cold at 3,500 m for young children or seniors", 11, 12, 1, 2, 3, 4),
    vulnerable_note="3,500 m altitude: plan two easy days to acclimatise")

SEED_PROFILES = [
    profile("Bhutan", {3, 4, 5, 10, 11},
            {**months("cold; high passes such as Chele La may close with snow", 12, 1, 2),
             **months("tail of the monsoon: some rain and cloud", 9)},
            avoid=months("monsoon: heavy rain, landslides and low cloud that delays Paro flights", 6, 7, 8)),
    profile("Nepal", {3, 4, 10, 11},
            {**months("cold in the hills; high treks are snowbound", 12, 1, 2),
             **months("hot and hazy before the monsoon", 5),
             **months("monsoon: leeches on trails and landslides on mountain roads", 6, 7, 8, 9)}),
    LADAKH,
    profile("Kerala", {10, 11, 12, 1, 2, 3},
            {**months("hot and humid", 4, 5),
             **months("south-west monsoon: heavy rain, though houseboats still run", 6, 7, 8, 9)}),
    profile("Sikkim", {3, 4, 5, 10, 11}, HIMALAYA_EAST_CAVEATS),
    profile("Darjeeling", {3, 4, 5, 10, 11}, HIMALAYA_EAST_CAVEATS),
    profile("Thailand", {11, 12, 1, 2},
            {**months("hot season, often above 35 °C", 3, 4, 5),
             **months("rainy season; rough seas on the Andaman coast (Phuket)", 6, 7, 8, 9, 10)}),
    profile("Japan", {3, 4, 5, 10, 11},
            {**months("rainy season (tsuyu)", 6), **months("hot and humid", 7, 8),
             **months("typhoon season", 9), **months("cold; snow in the north and the Alps", 12, 1, 2)}),
    profile("Iceland", {6, 7, 8},
            {**months("short days and icy roads; highland roads closed, but northern lights", 11, 12, 1, 2),
             **months("shoulder season: changeable weather, most highland roads still closed", 3, 4, 5, 9, 10)}),
    profile("Inca Trail", {5, 6, 7, 8, 9},
            {**months("rainy season: muddy, slippery trail", 12, 1, 3),
             **months("shoulder season: some rain", 4, 10, 11)},
            avoid=months("the Inca Trail is closed every February for maintenance", 2)),
]
SEEDS = {p.place.casefold(): p for p in SEED_PROFILES} | {"leh": LADAKH}


def climate_for(place: str) -> ClimateProfile | None:
    key = place.strip().casefold()
    return SEEDS.get(key) or knowledge.recall("climate", key, ClimateProfile)


def remember(profile: ClimateProfile, evidence_at=None) -> None:
    """evidence_at: when the oldest page behind the profile was fetched."""
    profile = profile.model_copy(update={"provenance": "learned"})
    knowledge.remember("climate", profile.place.strip().casefold(), profile, ClimateProfile, evidence_at=evidence_at)


@traceable(name="climate.season_verdict")
def season_verdict(profile: ClimateProfile, month: int, vulnerable: bool = False) -> SeasonVerdict:
    note = profile.months.get(month) or MonthNote(verdict="ok_with_caveats", reason="no data for this month")
    verdict, reasons, stricter = note.verdict, [note.reason], False
    if verdict != "avoid" and vulnerable and month in profile.vulnerable_avoid:
        verdict, stricter = "avoid", True
        reasons.insert(0, profile.vulnerable_avoid[month])
    if vulnerable and profile.vulnerable_note:
        reasons.append(profile.vulnerable_note)
    return SeasonVerdict(place=profile.place, month=month, verdict=verdict,
                         reasons=[r for r in reasons if r], stricter_for_vulnerable=stricter)
