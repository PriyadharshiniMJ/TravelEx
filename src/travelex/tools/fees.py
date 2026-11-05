"""Per-person tourist fees and levies (Bhutan's SDF, park fees, city taxes…), for any country.

rules_for() returns curated seeds, else what the costs agent learned, else None (unknown, so research it).
An empty list means "researched: no such fees". Seed rules have verified_on=None until a person checks them.
"""
from langsmith import traceable

from travelex.schemas import AgeBand, FeeLine, FeeRule, Traveller
from travelex.tools import knowledge

SDF = "Sustainable Development Fee (SDF)"
BHUTAN = "https://www.bhutan.travel"
BHUTAN_BANDS = [AgeBand(min_age=0, max_age=5, share=0, label="under 6 (exempt)"),
                AgeBand(min_age=6, max_age=12, share=0.5, label="child 6-12 (50%)")]

SEEDS: dict[str, list[FeeRule]] = {
    "BT": [
        FeeRule(country="BT", name=SDF, basis="per_night", amount=1200, currency="INR", passports=["IN", "BD", "MV"],
                age_bands=BHUTAN_BANDS, source_url=BHUTAN, provenance="seed"),
        FeeRule(country="BT", name=SDF, basis="per_night", amount=100, currency="USD",
                age_bands=BHUTAN_BANDS, source_url=BHUTAN, provenance="seed"),
    ],
}

LIST = list[FeeRule]


def _key(country: str, place: str | None) -> str:
    return f"{country.upper()}-{place}" if place else country.upper()


def rules_for(country: str, place: str | None = None) -> list[FeeRule] | None:
    """Seeds cover a whole country. Learned rules are cached per stop, because a country-level search
    can miss a regional fee (the Galápagos park fee is not charged in Quito)."""
    if country.upper() in SEEDS:
        return SEEDS[country.upper()]
    return knowledge.recall("fees", _key(country, place), LIST)


def remember(country: str, rules: list[FeeRule], place: str | None = None, evidence_at=None) -> None:
    """Only fees a source's words confirm are cached; evidence_at: when the oldest page behind them was fetched."""
    rules = [r.model_copy(update={"provenance": "learned"}) for r in rules]
    if any(not r.confirmed for r in rules):
        raise ValueError("Unconfirmed fees are used for one trip only, never cached.")
    knowledge.remember("fees", _key(country, place), rules, LIST, evidence_at=evidence_at)


def applies_here(rule: FeeRule, places: list[str]) -> bool:
    return rule.place is None or rule.place.casefold() in {p.casefold() for p in places}


def applicable(rules: list[FeeRule], passport: str | None) -> list[FeeRule]:
    """Per fee, the passport-specific rule if one matches, else the default. A missing passport gets the default."""
    chosen: dict[tuple, FeeRule] = {}
    for rule in rules:
        key = (rule.place, rule.name)
        if rule.passports is None:
            chosen.setdefault(key, rule)
        elif passport in rule.passports:
            chosen[key] = rule  # beats the default, whichever comes first
    return list(chosen.values())


def band(t: Traveller, rule: FeeRule) -> tuple[float, str]:
    age = 0 if t.type == "infant" and t.age is None else t.age
    if age is None:
        return 1.0, "adult" if t.type == "adult" else "child, age unknown (charged full)"
    for b in rule.age_bands:
        if b.min_age <= age <= b.max_age:
            return b.share, b.label
    return 1.0, "adult" if t.type == "adult" else "full rate"


@traceable(name="fees.fee_lines")
def fee_lines(travellers: list[Traveller], rules: list[FeeRule], place: str, nights: int,
              places: list[str] | None = None) -> list[FeeLine]:
    """One line per traveller per fee that applies at this stop (`places`: the stop and its cities).
    Exempt travellers get a zero line so the report can show why."""
    rules = [r for r in rules if applies_here(r, places or [place])]
    lines = []
    for t in travellers:
        for rule in applicable(rules, t.passport):
            share, label = band(t, rule)
            units = nights if rule.basis == "per_night" else 1
            unit = rule.amount * share
            lines.append(FeeLine(traveller=t.label, name=rule.name, place=rule.place or place, basis=rule.basis,
                                 unit_amount=unit, currency=rule.currency, units=units, total=unit * units,
                                 band=label, source_url=rule.source_url, confirmed=rule.confirmed))
    return lines
