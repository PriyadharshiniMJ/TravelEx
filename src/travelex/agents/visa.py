"""Visa check: one entry rule per traveller per country, every claim backed by a quote found word for word in a
fetched page. Unknown pairs are researched; curated (seed) rules are confirmed by research once; both are cached
with their citations."""
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage

from travelex.agents.common import event, host_of, register, render, source_kind, verify_evidence
from travelex.agents.evidence import currencies_in
from travelex.agents.transport import choose_option
from travelex.llm import get_llm, structured
from travelex.schemas import (
    Citation, GroupResult, RelevanceCheck, Source, TripRequest, VisaCheck, VisaResearch, VisaRule,
)
from travelex.state import GroupState
from travelex.tools import fx, search, visa_rules
from travelex.tools.countries import country_name
from travelex.trip import group_key

VISA_PROMPT = """Extract the entry rule for a tourist with this passport visiting this country, using only the
numbered sources. Prefer official government sources over agencies and blogs.
- visa_required: false only if the traveller can enter for tourism without any visa. An electronic travel
  authorisation (ESTA, eTA, ETA, K-ETA…) is not a visa: set document "eta" and visa_required false.
- document: "none", "eta", "e_visa", "visa_on_arrival", "entry_permit" or "visa".
- fee and fee_currency: the standard adult tourist fee for that document, only if a source states it.
- max_stay_days: if stated. notes: how to apply and key conditions, in one or two sentences; say so if the
  sources disagree or look out of date.
- source_url: the most official source you used.
- evidence: for each claim, a quote copied character for character from one source, one sentence or less,
  with the source's ID and what it supports: "requirement" (whether a visa or which document is needed),
  "fee", "stay" or "how_to_apply". Never fix typos or join sentences. A claim without a quote is removed."""

DOCUMENTS = {"none": "No visa needed", "eta": "Travel authorisation (eTA) needed", "e_visa": "e-Visa needed",
             "visa_on_arrival": "Visa on arrival", "entry_permit": "Entry permit on arrival", "visa": "Visa needed"}
UNCONFIRMED = " (not confirmed: no source we found states this word for word)"


VERIFY_PROMPT = """You check whether quotes support a visa rule for one specific traveller: a {who} passport
holder visiting {where} as a tourist. Each quote comes with the text around it on its page and who published it.
For each quote decide:
- applies: it states the rule for {who} citizens or passport holders entering {where}{zone}.
- all_visitors: it states a rule for every visitor to {where}, whatever their nationality.
- other_nationality: it is written for citizens of another country (e.g. a US embassy page for Americans).
- other_destination: it is about entering a different country.
- other_purpose: it is about work, study, residence or transit, not a tourist visit.
- contradicts: it is about this traveller and destination but says something different from the rule.
- unclear: the quote and its context don't say who it applies to.
Judge only from the quote and its context, never from what you know yourself."""

NUMBERS = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


def states_number(quote: str, value: float | None) -> bool:
    """A fee or stay quote must contain the number it supports ("EUR 90", "90 days", "1,200")."""
    return value is None or any(abs(float(n.replace(",", "")) - value) < 0.01 for n in NUMBERS.findall(quote))


def other_member(c: Citation, country: str, on_trip: frozenset[str] = frozenset()) -> str | None:
    """The zone country, not on this trip, that a page is about, judged by its address and title
    ("india-iceland", "Visa to Iceland"), when it doesn't name this country too. A Schengen rule is the same
    everywhere, but Iceland's application page and its visa centre's checklist don't help someone travelling to
    Austria. On a France + Italy trip France's pages do: one visa, applied for at the main destination."""
    zone = visa_rules.zone_of(country)
    if not zone:
        return None
    page = re.sub(r"[-_/.]+", " ", f"{c.url} {c.title}").casefold()
    names = lambda code: re.search(rf"\b{re.escape(country_name(code).casefold())}\b", page)
    if names(country):
        return None
    return next((country_name(code) for code in sorted(visa_rules.SCHENGEN - on_trip)
                 if code != country and names(code)), None)


def check_relevance(rule: VisaRule, citations: list[Citation], passport: str, country: str,
                    on_trip: frozenset[str] = frozenset()) -> tuple[list[Citation], list[str]]:
    """Keep citations that are about this passport, this destination and a tourist visit.
    Code first (whose page, does the quote hold the number), then one judgement by a fast model.
    Returns (kept, notes about contradicting sources)."""
    zone = visa_rules.zone_of(country)
    candidates = []
    for c in citations:
        if other_member(c, country, on_trip):  # about a zone country not on this trip: not a source here
            continue
        c = c.model_copy(update={"source_kind": source_kind(c.url, passport, country, zone)})
        if c.supports == "fee" and not states_number(c.quote, rule.fee):
            continue
        if c.supports == "stay" and not states_number(c.quote, rule.max_stay_days):
            continue
        candidates.append(c)
    if not candidates:
        return [], []
    who, where = country_name(passport), country_name(country)
    claims = (f"Rule: visa_required={rule.visa_required}, document={rule.document}, fee={rule.fee} "
              f"{rule.fee_currency}, max_stay_days={rule.max_stay_days}")
    quotes = "\n\n".join(f"[{i}] supports {c.supports}; published by: {c.source_kind.replace('_', ' ')} "
                         f"({host_of(c.url)})\nQuote: {c.quote}\nContext: {c.context or '(none)'}"
                         for i, c in enumerate(candidates))
    prompt = VERIFY_PROMPT.format(who=who, where=where, zone=f" or the {zone} area" if zone else "")
    try:
        verdicts = structured(get_llm("verify"), RelevanceCheck,
                              [SystemMessage(prompt), HumanMessage(f"{claims}\n\n{quotes}")]).verdicts
    except Exception:  # fail soft: unchecked quotes are kept but can't confirm the rule
        return candidates, []
    by_index = {v.index: v for v in verdicts}
    kept, notes = [], []
    for i, c in enumerate(candidates):
        v = by_index.get(i)
        if v is None:
            continue
        if v.verdict == "contradicts":
            notes.append(f"{host_of(c.url)} says otherwise: “{c.quote}”")
        foreign = c.source_kind == "other_government"  # written for another country's citizens
        if (v.verdict == "applies" and not foreign) or v.verdict == "all_visitors":
            kept.append(c.model_copy(update={"relevance": v.verdict, "relevance_note": v.reason}))
    return kept, notes


def ground(rule: VisaRule, citations: list[Citation]) -> VisaRule:
    """Keep only what a relevant, verified quote supports: no fee or stay without one, and a rule with no such
    quote for the requirement itself is marked unconfirmed (confirmed=False, shown in the summary).
    Quotes that were never judged for relevance can't confirm anything."""
    checked = [c for c in citations if c.relevance != "unchecked"]
    supports = {c.supports for c in checked}
    update: dict = {"citations": citations, "confirmed": "requirement" in supports, "verified_on": None}
    if rule.fee is not None and "fee" not in supports:
        update["fee"] = None
    if rule.max_stay_days is not None and "stay" not in supports:
        update["max_stay_days"] = None
    return rule.model_copy(update=update)


def research_visa(passport: str, country: str, request: TripRequest, prefix: str, sources: dict[str, Source]
                  ) -> tuple[VisaRule, dict[str, Source], list[str], int]:
    """Three searches, one extraction, then the quotes are checked against the pages.
    Returns (rule with citations, new sources, queries, quotes dropped)."""
    year = (request.month or str(request.start_date or ""))[:4]
    who, where = country_name(passport), country_name(country)
    queries = [f"{who} passport holders {where} tourist visa requirements {year}".strip(),
               f"{where} official visa policy {who} citizens",
               f"{where} tourist visa fee for {who} applicants"]
    new, found = register(search.search_all(queries), prefix, sources)
    messages = [SystemMessage(VISA_PROMPT),
                HumanMessage(f"Passport: {who} ({passport})\nCountry visited: {where} ({country})\n\n"
                             f"Sources:\n{render(found)}")]
    research = structured(get_llm("visa"), VisaResearch, messages)
    rule = research.rule.model_copy(update={"passport": passport, "country": country})
    grounded, not_found = verify_evidence(research.evidence, {s.id: s for s in found})
    relevant, contradictions = check_relevance(rule, grounded, passport, country, trip_countries(request))
    rule = ground(rule, relevant).model_copy(update={"cited_for": country})
    if contradictions:
        rule = rule.model_copy(update={"notes": f"Some sources disagree: {' '.join(contradictions)} {rule.notes}".strip()})
    return rule, new, queries, not_found + len(grounded) - len(relevant)


def confirm_seed(seed: VisaRule, passport: str, researched: VisaRule) -> VisaRule:
    """Check a stored (seed) rule against fresh research, one claim at a time. The requirement (visa or not, and
    which document) is confirmed only if the sources say the same; the fee only by a quote stating that amount in
    that currency; the stay only by a quote stating that number of days. A claim the sources contradict is noted,
    and its quotes are not shown as if they backed it. Cached with what it found."""
    quotes = researched.citations
    same_requirement = (researched.visa_required, researched.document) == (seed.visa_required, seed.document)
    fee_quotes = [c for c in quotes if c.supports == "fee" and seed.fee is not None
                  and states_number(c.quote, seed.fee) and seed.fee_currency.upper() in currencies_in(c.quote)]
    stay_quotes = [c for c in quotes if c.supports == "stay" and seed.max_stay_days is not None
                   and states_number(c.quote, seed.max_stay_days)]
    differs = []
    if researched.confirmed and researched.visa_required != seed.visa_required:
        differs.append(f"sources found now say {'a visa is needed' if researched.visa_required else 'no visa is needed'}")
    elif researched.confirmed and not same_requirement:
        differs.append(f"sources found now say: {DOCUMENTS.get(researched.document, researched.document).lower()}")
    if researched.fee is not None and seed.fee is not None and \
            (researched.fee, researched.fee_currency.upper()) != (seed.fee, seed.fee_currency.upper()):
        differs.append(f"sources found now give the fee as {researched.fee:g} {researched.fee_currency}")
    if researched.max_stay_days and seed.max_stay_days and researched.max_stay_days != seed.max_stay_days:
        differs.append(f"sources found now give the stay as {researched.max_stay_days} days")
    notes = seed.notes
    if differs:
        notes = f"Unlike our stored rule, {'; '.join(differs)}: check before you travel. {notes}".strip()
    backing = [c for c in quotes if (c.supports == "requirement" and same_requirement) or c in fee_quotes
               or c in stay_quotes or c.supports == "how_to_apply"]
    unconfirmed = [claim for claim, value, quoted in (("fee", seed.fee, fee_quotes), ("stay", seed.max_stay_days, stay_quotes))
                   if value is not None and not quoted]
    merged = seed.model_copy(update={"citations": backing, "notes": notes, "cited_for": seed.country,
                                     "confirmed": researched.confirmed and same_requirement,
                                     "unconfirmed": unconfirmed})
    return visa_rules.remember_check(passport, merged)


def fee_in(amount: float, currency: str, target: str | None) -> str:
    """A visa fee in the plan's currency ("≈ 9,184 INR"); the cited quote keeps the official amount."""
    if not target or target == currency:
        return f"{amount:g} {currency}"
    try:
        return f"≈ {fx.convert(amount, currency, target):,.0f} {target}"
    except fx.FxError:
        return f"{amount:g} {currency}"


def trip_countries(request: TripRequest) -> frozenset[str]:
    return frozenset(s.country for s in request.stops)


def borrowed(rule: VisaRule, country: str, on_trip: frozenset[str] = frozenset()) -> bool:
    """A zone rule researched for another country (France's Schengen answer, reused for Portugal), or one still
    citing another zone country's pages (saved before those were left out)."""
    return bool(rule.citations) and ((rule.cited_for is not None and rule.cited_for != country)
                                     or any(other_member(c, country, on_trip) for c in rule.citations))


def recheck_for(rule: VisaRule, passport: str, country: str,
                on_trip: frozenset[str] = frozenset()) -> tuple[VisaRule, bool]:
    """Judge a borrowed rule's quotes again for this country: Schengen-wide quotes still count, France-only ones
    ("visit France as their main destination… VFS centres") don't. One judge call, no new search.
    Returns (the rule with only the quotes that apply here, whether the requirement is still supported)."""
    fresh = [c.model_copy(update={"relevance": "unchecked", "relevance_note": ""}) for c in rule.citations]
    kept, contradictions = check_relevance(rule, fresh, passport, country, on_trip)
    rechecked = ground(rule.model_copy(update={"country": country}), kept).model_copy(update={"cited_for": country})
    if rechecked.confirmed and not contradictions:
        visa_rules.remember(rechecked)  # next time this country has its own entry
        return rechecked, True
    return rechecked, False


def assemble_group(state: GroupState) -> dict:
    """Join the two branches: the routes found and the visa checks, as one GroupResult."""
    request, group = state["request"], state["group"]
    by_label = {t.label: t for t in request.travellers}
    options = state.get("options") or []
    result = GroupResult(group_id=group.id, options=options,
                         chosen=choose_option(options, [by_label[label] for label in group.travellers]),
                         visas=state.get("visas") or [], status="ok" if options else "failed",
                         key=group_key(request, group))
    return {"result": result}


def needs_confirmation(rule: VisaRule) -> bool:
    return rule.provenance == "seed" and not rule.citations and not visa_rules.by_law(rule)


def summarise(rule: VisaRule | None, currency: str | None = None) -> str:
    """One line: what's needed, the fee and stay, and the first sentence of the notes. A zone rule names the
    zone, not the country it was researched for ("Schengen visa", not "visa for France")."""
    if rule is None:
        return "Not known: check with the embassy."
    what = DOCUMENTS.get(rule.document, rule.document)
    if rule.zone and rule.visa_required:
        what = f"{what} ({rule.zone} visa, one covers the whole area)"
    fee = f", {fee_in(rule.fee, rule.fee_currency, currency)}" if rule.fee else ""
    fee += " (fee not confirmed)" if fee and "fee" in rule.unconfirmed else ""
    stay = f", up to {rule.max_stay_days} days" if rule.max_stay_days else ""
    stay += " (not confirmed)" if stay and "stay" in rule.unconfirmed else ""
    # the first sentence: a full stop after a lowercase letter or digit, so "U.S. citizens…" isn't cut at "U.S."
    first = re.split(r"(?<=[a-z0-9)])\.\s+", rule.notes)[0].rstrip(".") if rule.notes and not rule.zone else ""
    unconfirmed = UNCONFIRMED if not (rule.confirmed or visa_rules.by_law(rule)) else ""
    return f"{what}{fee}{stay}{unconfirmed}" + (f". {first}." if first else "")


def visa_check(state: GroupState) -> dict:
    started = time.perf_counter()
    request, group = state["request"], state["group"]
    prefix = f"{state['source_prefix']}V"  # runs beside the transport search, so its sources are numbered apart
    by_label = {t.label: t for t in request.travellers}
    countries = list(dict.fromkeys(s.country for s in request.stops))
    sources = {**(state.get("taken") or {}), **(state.get("sources") or {})}  # number after IDs already given out
    budget_currency = request.budget.currency if request.budget else None
    new_sources, errors, queries, dropped = {}, [], [], 0
    known: dict[tuple[str, str], tuple[VisaRule | None, bool]] = {}
    checks = []
    for label in group.travellers:
        passport = by_label[label].passport
        for country in countries:
            if not passport:
                checks.append(VisaCheck(traveller=label, passport=None, country=country, rule=None,
                                        summary="Passport not known yet, so visas can't be checked."))
                continue
            if (passport, country) not in known:
                rule, researched, fallback = visa_rules.lookup(passport, country), False, None
                if rule is not None and borrowed(rule, country, trip_countries(request)):
                    rule, still_holds = recheck_for(rule, passport, country, trip_countries(request))
                    if not still_holds:  # nothing quoted is about this country: research it, keep this as a fallback
                        fallback, rule = rule, None
                if rule is None or needs_confirmation(rule):
                    try:
                        found, added, asked, lost = research_visa(passport, country, request, prefix, sources)
                        sources |= added
                        new_sources |= added
                        queries += asked
                        dropped += lost
                        researched = True
                        rule = visa_rules.remember(found) if rule is None else confirm_seed(rule, passport, found)
                    except search.SearchConfigError:
                        raise
                    except Exception as e:  # fail soft: the visa is shown as unknown or uncited, not fatal
                        errors.append(f"visa_check ({passport}->{country}): {type(e).__name__}: {e}")
                        rule = rule or fallback
                known[(passport, country)] = (rule, researched)
            rule, researched = known[(passport, country)]
            cited = [s.id for s in sources.values() if rule and any(c.url == s.url for c in rule.citations)]
            checks.append(VisaCheck(traveller=label, passport=passport, country=country, rule=rule,
                                    summary=summarise(rule, budget_currency), confirmed_by_search=researched,
                                    source_ids=cited))

    detail = "; ".join(f"{c.traveller} → {c.country}: {c.summary.split('.')[0]}"
                       f" ({len(c.rule.citations) if c.rule else 0} quotes)" for c in checks)
    note = f"; dropped {dropped} quote(s): not on the page, or not about this passport and trip" if dropped else ""
    return {"visas": checks, "sources": new_sources, "errors": errors,
            "events": [event("visa_check", "checked", f"{group.origin}: {detail}{note}", queries, started)]}
