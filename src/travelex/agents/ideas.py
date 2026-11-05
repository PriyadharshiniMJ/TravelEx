"""Destination ideas, when the user hasn't picked a place: the intake model suggests them; code screens each one
against what TravelEx already knows, for free (no search): the stored season profile for the travel month, the
stored tourist fees against the budget, and the stored visa rules for each passport. An idea in a season to avoid,
or whose fees alone break the budget, is left out; the rest carry a short warning when something needs care."""
from travelex.agents.season import MONTH_NAMES, stop_verdict
from travelex.schemas import DestinationIdea, TripRequest
from travelex.tools import climate, fx, visa_rules
from travelex.tools.countries import country_name
from travelex.tools.fees import fee_lines, rules_for
from travelex.trip import has_vulnerable, resolve_nights, trip_months

FEES_WARN_SHARE = 0.5  # fees over half the budget: say so


def season_problem(idea: DestinationIdea, request: TripRequest) -> tuple[str | None, str | None]:
    """(why to leave it out, a warning): from the stored season profile, if there is one."""
    months = trip_months(request)
    names = [idea.place, country_name(idea.country)] if idea.country else [idea.place]
    profile = next(filter(None, map(climate.climate_for, names)), None)
    if profile is None or not months:
        return None, None
    verdict = stop_verdict(profile, months, has_vulnerable(request))
    when = MONTH_NAMES[months[0] - 1]
    why = verdict.reasons[0].rstrip(".") if verdict.reasons else ""
    if verdict.verdict == "avoid":
        return f"{when} is a season to avoid there" + (f" ({why})" if why else ""), None
    if verdict.verdict == "ok_with_caveats":
        return None, f"{when}: {why}" if why else f"{when} needs care"
    return None, None


def fee_problem(idea: DestinationIdea, request: TripRequest) -> tuple[str | None, str | None]:
    """(why to leave it out, a warning): known per-person tourist fees against the budget."""
    rules = rules_for(idea.country) if idea.country else None
    nights, _ = resolve_nights(request)
    if not (rules and nights and request.budget and request.travellers):
        return None, None
    try:
        total = sum(fx.convert(line.total, line.currency, request.budget.currency)
                    for line in fee_lines(request.travellers, rules, idea.place, nights))
    except fx.FxError:
        return None, None
    cost = f"{total:,.0f} {request.budget.currency}"
    if total > request.budget.amount:
        return f"its tourist fees alone ({cost} for {nights} nights) are more than your budget", None
    if total > FEES_WARN_SHARE * request.budget.amount:
        return None, f"tourist fees about {cost}"
    return None, None


def visa_warning(idea: DestinationIdea, request: TripRequest) -> str | None:
    """Passports that need a visa arranged in advance (not on arrival, not online), from the stored rules."""
    if not idea.country:
        return None
    needs = sorted({t.passport for t in request.travellers if t.passport
                    and (rule := visa_rules.lookup(t.passport, idea.country))
                    and rule.visa_required and rule.document == "visa"})
    return f"a visa in advance for {', '.join(needs)} passports" if needs else None


def screen_ideas(request: TripRequest) -> tuple[list[DestinationIdea], list[str]]:
    """(ideas to offer, each with any warning as its note; why the others were left out). If every idea fails,
    all are offered, each saying what's wrong, rather than none."""
    kept, dropped, failed = [], [], []
    for idea in request.destination_ideas:
        (season_out, season_warn), (fee_out, fee_warn) = season_problem(idea, request), fee_problem(idea, request)
        warnings = [w for w in (season_warn, fee_warn, visa_warning(idea, request)) if w]
        if reason := season_out or fee_out:
            dropped.append(f"{idea.place}: {reason}")
            failed.append(idea.model_copy(update={"note": reason}))
        else:
            kept.append(idea.model_copy(update={"note": "; ".join(warnings)}))
    if not kept:
        return failed, []
    return kept, dropped
