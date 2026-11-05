from datetime import date

import pytest

import travelex.agents.season as season_module
from builders import person, trip
from fakes import page, tool_call
from travelex.agents.season import season_check
from travelex.schemas import Stop
from travelex.tools import climate

ZANZIBAR = Stop(place="Zanzibar", country="TZ")
ZANZIBAR_PROFILE = {  # a ClimateResearch, as the LLM returns it
    "months": [{"month": m, "verdict": "good"} for m in (1, 2, 3, 6, 7, 8, 9, 10, 11, 12)]
              + [{"month": 4, "verdict": "avoid", "reason": "long rains: heavy daily downpours"},
                 {"month": 5, "verdict": "ok_with_caveats", "reason": "end of the long rains"}],
    "source_ids": ["S1.1", "S99"],
}


def state(request, sources=None):
    return {"request": request, "sources": sources or {}}


def test_seed_profile_needs_no_search_or_llm(use_fake_llm):
    llm = use_fake_llm(season_module, responder=lambda m: pytest.fail("no LLM call expected"))
    out = season_check(state(trip(person("A"), month="2027-07", nights=5)))
    assert out["season"]["Bhutan"].verdict == "avoid"
    assert "monsoon" in out["season"]["Bhutan"].reasons[0]
    assert llm.calls == [] and out["sources"] == {}


def test_city_can_match_a_seed_profile():
    peru = Stop(place="Peru", country="PE", cities=["Cusco", "Inca Trail"])
    out = season_check(state(trip(person("A", passport="GB"), stops=[peru], month="2027-02")))
    assert out["season"]["Peru"].verdict == "avoid"
    assert "February" in out["season"]["Peru"].reasons[0]


def test_new_place_is_researched_cited_and_cached(use_fake_llm, fake_search, search_calls):
    fake_search.append(page("https://example.org/zanzibar-weather", "April brings the long rains..."))
    llm = use_fake_llm(season_module, tool_call("ClimateResearch", ZANZIBAR_PROFILE))
    request = trip(person("A", passport="NG", origin="Lagos"), stops=[ZANZIBAR], month="2027-04", nights=5)

    out = season_check(state(request))
    verdict = out["season"]["Zanzibar"]
    assert (verdict.verdict, verdict.reasons) == ("avoid", ["long rains: heavy daily downpours"])
    assert list(out["sources"]) == ["S1.1"]
    assert len(search_calls) == 2 and "Zanzibar" in search_calls[0]
    cached = climate.climate_for("zanzibar")
    assert cached.provenance == "learned" and cached.source_ids == ["S1.1"]  # S99 was never a source

    out = season_check(state(request.model_copy(update={"month": "2027-08"})))
    assert out["season"]["Zanzibar"].verdict == "good"
    assert len(search_calls) == 2 and len(llm.calls) == 1  # second trip reused the cache


def test_trip_across_months_takes_the_worse_month():
    request = trip(person("A"), start_date=date(2027, 5, 28), end_date=date(2027, 6, 3), nights=None)
    verdict = season_check(state(request))["season"]["Bhutan"]
    assert (verdict.verdict, verdict.month) == ("avoid", 6)
    assert verdict.reasons[0].startswith("June: monsoon")


def test_kids_make_the_verdict_stricter():
    ladakh = Stop(place="Ladakh", country="IN")
    adults = season_check(state(trip(person("A"), stops=[ladakh], month="2027-01")))["season"]["Ladakh"]
    family = season_check(state(trip(person("A"), person("Kid", type="child", age=3), stops=[ladakh],
                                     month="2027-01")))["season"]["Ladakh"]
    assert adults.verdict == "ok_with_caveats"
    assert family.verdict == "avoid" and family.stricter_for_vulnerable


def test_unknown_month_and_failed_research_fail_soft(use_fake_llm, fake_search):
    assert season_check(state(trip(person("A"), nights=5)))["season"]["Bhutan"].verdict == "unknown"

    use_fake_llm(season_module, responder=lambda m: (_ for _ in ()).throw(RuntimeError("model down")))
    out = season_check(state(trip(person("A", passport="NG"), stops=[ZANZIBAR], month="2027-04")))
    assert out["season"]["Zanzibar"].verdict == "unknown"
    assert "model down" in out["errors"][0]
