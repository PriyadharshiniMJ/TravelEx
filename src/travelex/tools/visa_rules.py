"""Visa and entry rules by passport and country, for any country in the world.

Lookup order: domestic or Schengen-internal travel → a curated seed rule for this passport → a rule the
visa agent learned from cited sources → the seed default ("*") for the country → None (unknown, so the
visa agent researches it and calls remember()). Seed rules have verified_on=None until a person checks them.
"""
from langsmith import traceable

from travelex.schemas import VisaRule
from travelex.tools import knowledge

SCHENGEN = frozenset("AT BE BG HR CZ DK EE FI FR DE GR HU IS IT LV LI LT LU MT NL NO PL PT RO SK SI ES SE CH".split())

BHUTAN = "https://www.bhutan.travel"
NEPAL = "https://www.immigration.gov.np"


def seed(passport: str, country: str, **kwargs) -> VisaRule:
    return VisaRule(passport=passport, country=country, provenance="seed", **kwargs)


SEEDS = [
    seed("IN", "BT", visa_required=False, document="entry_permit", accepted_ids=["passport", "voter_id"],
         source_url=BHUTAN, notes="Free entry permit issued on arrival at Paro airport or Phuentsholing."),
    seed("BD", "BT", visa_required=False, document="entry_permit", accepted_ids=["passport"], source_url=BHUTAN),
    seed("MV", "BT", visa_required=False, document="entry_permit", accepted_ids=["passport"], source_url=BHUTAN),
    seed("*", "BT", visa_required=True, document="e_visa", fee=40, source_url=BHUTAN,
         notes="Apply online before travel; the SDF is paid with the application."),
    seed("IN", "NP", visa_required=False, document="none", accepted_ids=["passport", "voter_id"], source_url=NEPAL),
    seed("*", "NP", visa_required=True, document="visa_on_arrival", fee=30, max_stay_days=15, source_url=NEPAL,
         notes="15-day tourist visa; 30 days is $50 and 90 days is $125."),
]


def zone_of(country: str) -> str | None:
    return "Schengen" if country in SCHENGEN else None


DOMESTIC = "Domestic travel."
FREE_MOVEMENT = "Free movement inside the Schengen area."


def _key(passport: str, country: str) -> str:
    return f"{passport}-{country}"


def _check_key(passport: str, country: str) -> str:
    return f"{passport}-{country}-seedcheck"


def by_law(rule: VisaRule) -> bool:
    """No visa because of who you are, not a policy to cite: your own country, or Schengen to Schengen."""
    return rule.notes in (DOMESTIC, FREE_MOVEMENT)


@traceable(name="visa_rules.lookup")
def lookup(passport: str | None, country: str | None) -> VisaRule | None:
    """By law → a seed rule (with its cited confirmation, if one is cached) → a learned rule → None."""
    if not passport or not country:
        return None
    passport, country = passport.upper(), country.upper()
    if passport == country:
        return VisaRule(passport=passport, country=country, visa_required=False, document="none",
                        notes=DOMESTIC, provenance="seed", confirmed=True)
    if passport in SCHENGEN and country in SCHENGEN:
        return VisaRule(passport=passport, country=country, zone="Schengen", visa_required=False, document="none",
                        notes=FREE_MOVEMENT, provenance="seed", confirmed=True)
    seeds = [r for r in SEEDS if r.country == country]
    zone = zone_of(country)
    learned = knowledge.recall("visa", _key(passport, country), VisaRule)
    if learned is None and zone:  # one Schengen answer covers every Schengen country
        learned = knowledge.recall("visa", _key(passport, zone), VisaRule)
        # remember which country the quotes were checked for, so the caller re-checks them for this one
        learned = learned and learned.model_copy(update={"country": country,
                                                         "cited_for": learned.cited_for or learned.country})
    seed = next((r for r in seeds if r.passport == passport), None)
    if seed is None and learned is None:
        seed = next((r for r in seeds if r.passport == "*"), None)
    if seed is not None:
        return knowledge.recall("visa", _check_key(passport, country), VisaRule) or seed
    return learned


def evidence_at(rule: VisaRule):
    """When the oldest page the rule cites was fetched (None: no dated citation, so now)."""
    return min((c.retrieved_at for c in rule.citations if c.retrieved_at), default=None, key=str)


def remember(rule: VisaRule) -> VisaRule:
    """Store a rule the visa agent extracted from cited sources, under its country and its visa zone."""
    rule = rule.model_copy(update={"zone": rule.zone or zone_of(rule.country), "provenance": "learned"})
    when = evidence_at(rule)
    knowledge.remember("visa", _key(rule.passport, rule.country), rule, VisaRule, evidence_at=when)
    if rule.zone:
        knowledge.remember("visa", _key(rule.passport, rule.zone), rule, VisaRule, evidence_at=when)
    return rule


def remember_check(passport: str, merged: VisaRule) -> VisaRule:
    """Cache a seed rule with the research that checked it, so the next trip with this passport and country gets
    the cited rule without searching. The checking itself, claim by claim, is the visa agent's confirm_seed."""
    knowledge.remember("visa", _check_key(passport, merged.country), merged, VisaRule, evidence_at=evidence_at(merged))
    return merged
