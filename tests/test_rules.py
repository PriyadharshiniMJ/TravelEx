"""The deterministic rule layer: trip helpers, fx, fees, visas, seasons and the learned-rules cache."""
from datetime import date

import pytest

from builders import person, trip
from travelex.schemas import AgeBand, ClimateProfile, FeeRule, MonthNote, Stop, TripRequest, VisaRule
from travelex.tools import climate, fees, fx, visa_rules
from travelex.tools.climate import climate_for, season_verdict
from travelex.tools.fees import fee_lines
from travelex.tools.visa_rules import lookup
from travelex.trip import allocate_nights, group_by_origin, is_vulnerable, parse_duration, resolve_nights


# ---------- trip helpers ----------

def test_group_by_origin_ignores_case_and_keeps_first_seen_order():
    travellers = [person("Mum", origin="Toronto"), person("Grandma", origin="Mumbai"),
                  person("Kid", type="child", age=5, origin="toronto ")]
    assert [(g.id, g.origin, g.travellers) for g in group_by_origin(travellers)] == [
        ("G1", "Toronto", ["Mum", "Kid"]), ("G2", "Mumbai", ["Grandma"])]


@pytest.mark.parametrize("kwargs, expected", [
    ({"start_date": date(2027, 4, 10), "end_date": date(2027, 4, 20)}, (10, None)),
    ({"duration_range": (4, 5)}, (4, 3)),  # golden A1: "4-5 days"
    ({"duration_range": (5, 5)}, (4, None)),
    ({"nights": 6}, (6, None)),
    ({"stops": [Stop(place="Italy", country="IT", nights=5), Stop(place="Switzerland", country="CH", nights=3)]},
     (8, None)),
    ({}, (None, None)),
])
def test_resolve_nights(kwargs, expected):
    assert resolve_nights(TripRequest(**kwargs)) == expected


@pytest.mark.parametrize("text, expected", [
    ("6 days", ([6, 6], None)),
    ("4-5 days", ([4, 5], None)),
    ("4 to 5 days", ([4, 5], None)),
    ("5 nights", (None, 5)),
    ("a week", ([7, 7], None)),
    ("Two weeks", ([14, 14], None)),
    ("a fortnight", ([14, 14], None)),
    ("about 10 days in Japan", ([10, 10], None)),
    ("a long weekend", (None, None)),
    (None, (None, None)),
])
def test_parse_duration(text, expected):
    assert parse_duration(text) == expected


def test_allocate_nights_keeps_user_numbers_and_splits_the_rest():
    stops = [Stop(place="France", country="FR"), Stop(place="Italy", country="IT", nights=4),
             Stop(place="Switzerland", country="CH")]
    request = TripRequest(stops=stops, nights=9)
    assert [(s.place, n) for s, n in allocate_nights(request)] == [("France", 3), ("Italy", 4), ("Switzerland", 2)]


def test_country_codes_are_validated():
    with pytest.raises(ValueError, match="ISO 3166"):
        Stop(place="Japan", country="Japan")
    assert Stop(place="Japan", country="jp").country == "JP"


def test_vulnerable_travellers():
    assert is_vulnerable(person("Kid", type="child", age=11))
    assert is_vulnerable(person("Baby", type="infant"))
    assert is_vulnerable(person("Grandma", age=68))
    assert is_vulnerable(person("Grandpa", needs=["senior"]))
    assert not is_vulnerable(person("Adult", age=40))


# ---------- fx ----------

def test_fx_pegs_and_cross_rates():
    assert fx.convert(3.6725, "AED", "USD") == pytest.approx(1.0)
    assert fx.convert(100, "usd", "USD") == 100
    assert fx.convert(1.6, "NPR", "INR") == pytest.approx(1.0)
    assert fx.convert(fx.SEED_PER_USD["INR"], "INR", "AED") == pytest.approx(3.6725)


def test_fx_offline_uses_the_seed_table_and_says_so():
    assert fx.rates().source == fx.SEED_SOURCE


def test_fx_unknown_currency():
    with pytest.raises(fx.FxError, match="XYZ"):
        fx.convert(1, "XYZ", "USD")


def test_fx_reads_a_cached_day_without_the_network(tmp_path, monkeypatch):
    monkeypatch.delenv("TRAVELEX_FX_OFFLINE")
    monkeypatch.setenv("TRAVELEX_FX_DIR", str(tmp_path))
    monkeypatch.setattr(fx, "_today", {})
    monkeypatch.setattr(fx, "_fetch", lambda: pytest.fail("should read the cache"))
    (tmp_path / f"{date.today()}.json").write_text(
        '{"per_usd": {"USD": 1, "JPY": 150}, "as_of": "2026-10-27", "source": "test"}', encoding="utf-8")
    assert fx.convert(300, "JPY", "USD") == pytest.approx(2.0)


def test_fx_falls_back_to_seed_when_the_service_fails(tmp_path, monkeypatch):
    monkeypatch.delenv("TRAVELEX_FX_OFFLINE")
    monkeypatch.setenv("TRAVELEX_FX_DIR", str(tmp_path))
    monkeypatch.setattr(fx, "_today", {})
    monkeypatch.setattr(fx, "_fetch", lambda: (_ for _ in ()).throw(OSError("offline")))
    assert fx.rates() is fx.SEED


# ---------- fees ----------

def totals(lines):
    return {l.traveller: (l.total, l.currency, l.band) for l in lines}


def test_bhutan_sdf_by_age_band_for_most_passports():
    travellers = [person("A", passport="GB"), person("C9", type="child", age=9, passport="GB"),
                  person("C4", type="child", age=4, passport="GB"), person("B", type="infant", passport="GB"),
                  person("C?", type="child", passport="GB")]
    assert totals(fee_lines(travellers, fees.rules_for("BT"), "Bhutan", 6)) == {
        "A": (600, "USD", "adult"), "C9": (300, "USD", "child 6-12 (50%)"),
        "C4": (0, "USD", "under 6 (exempt)"), "B": (0, "USD", "under 6 (exempt)"),
        "C?": (600, "USD", "child, age unknown (charged full)")}


def test_bhutan_sdf_regional_rate_for_indian_passports():
    lines = fee_lines([person("A"), person("C8", type="child", age=8)], fees.rules_for("bt"), "Bhutan", 4)
    assert totals(lines) == {"A": (4800, "INR", "adult"), "C8": (2400, "INR", "child 6-12 (50%)")}


def test_golden_c2_and_c3_sdf_totals():
    c2 = [person(f"A{i}", passport="US", origin="Dubai") for i in range(3)]
    assert sum(l.total for l in fee_lines(c2, fees.rules_for("BT"), "Bhutan", 4)) == 1200
    c3 = [person("A1", passport="GB"), person("A2", passport="GB"),
          person("K9", type="child", age=9, passport="GB"), person("K4", type="child", age=4, passport="GB")]
    assert sum(l.total for l in fee_lines(c3, fees.rules_for("BT"), "Bhutan", 6)) == 1500


def test_per_trip_fee_for_a_region_with_child_band():
    park = FeeRule(country="EC", place="Galápagos", name="National park entry", basis="per_trip", amount=200,
                   currency="USD", age_bands=[AgeBand(min_age=0, max_age=11, share=0.5, label="under 12 (50%)")])
    travellers = [person("A", passport="DE"), person("K", type="child", age=9, passport="DE")]
    lines = fee_lines(travellers, [park], "Galápagos", 4)
    assert totals(lines) == {"A": (200, "USD", "adult"), "K": (100, "USD", "under 12 (50%)")}
    assert fee_lines(travellers, [park], "Ecuador", 5, ["Ecuador", "Quito"]) == []  # not charged on the mainland
    assert len(fee_lines(travellers, [park], "Ecuador", 9, ["Ecuador", "Galápagos"])) == 2  # a city match counts


def test_unknown_fees_are_none_and_learned_fees_are_recalled():
    assert fees.rules_for("ID") is None
    levy = FeeRule(country="ID", place="Bali", name="Bali tourist levy", basis="per_trip", amount=150000,
                   currency="IDR")
    fees.remember("ID", [levy])
    assert fees.rules_for("ID")[0].model_dump() == levy.model_copy(update={"provenance": "learned"}).model_dump()
    fees.remember("IS", [])
    assert fees.rules_for("IS") == []  # researched: no fees, which is different from unknown


# ---------- visas ----------

@pytest.mark.parametrize("passport, country, required, document, fee", [
    ("IN", "BT", False, "entry_permit", None),
    ("in", "BT", False, "entry_permit", None),
    ("AE", "BT", True, "e_visa", 40),
    ("US", "BT", True, "e_visa", 40),
    ("IN", "NP", False, "none", None),
    ("MY", "NP", True, "visa_on_arrival", 30),
    ("IN", "IN", False, "none", None),
    ("DE", "CH", False, "none", None),  # Schengen-internal
])
def test_visa_lookup(passport, country, required, document, fee):
    rule = lookup(passport, country)
    assert (rule.visa_required, rule.document, rule.fee) == (required, document, fee)


def test_visa_lookup_unknown_then_learned():
    assert lookup("IN", "JP") is None
    assert lookup(None, "BT") is None
    visa_rules.remember(VisaRule(passport="IN", country="FR", visa_required=True, document="visa", fee=90,
                                 fee_currency="EUR"))
    learned = lookup("IN", "FR")
    assert (learned.zone, learned.provenance, learned.fee) == ("Schengen", "learned", 90)


def test_seed_rule_for_the_passport_beats_a_learned_one():
    visa_rules.remember(VisaRule(passport="IN", country="BT", visa_required=True, document="visa"))
    assert lookup("IN", "BT").provenance == "seed"


# ---------- seasons ----------

@pytest.mark.parametrize("place, month, vulnerable, verdict, stricter", [
    ("Bhutan", 7, False, "avoid", False),
    ("Bhutan", 10, False, "good", False),
    ("Ladakh", 1, False, "ok_with_caveats", False),
    ("Leh", 1, True, "avoid", True),  # golden D3: avoid because of the kids
    ("Ladakh", 7, True, "good", False),
    ("Inca Trail", 2, False, "avoid", False),  # golden D2: closed every February
    ("Iceland", 12, True, "ok_with_caveats", False),  # golden D4: caveats, no alternatives
    ("Japan", 4, True, "good", False),
])
def test_season_verdict(place, month, vulnerable, verdict, stricter):
    result = season_verdict(climate_for(place), month, vulnerable)
    assert (result.verdict, result.stricter_for_vulnerable) == (verdict, stricter)


def test_season_reasons_name_the_problem():
    assert "monsoon" in season_verdict(climate_for("Bhutan"), 7).reasons[0]
    assert any("altitude" in r for r in season_verdict(climate_for("Ladakh"), 7, vulnerable=True).reasons)
    assert "children" in season_verdict(climate_for("Ladakh"), 1, vulnerable=True).reasons[0]


def test_unknown_place_then_learned_profile():
    assert climate_for("Zanzibar") is None
    months = {m: MonthNote(verdict="good") for m in range(1, 13)} | {4: MonthNote(verdict="avoid", reason="long rains")}
    climate.remember(ClimateProfile(place="Zanzibar", months=months))
    assert season_verdict(climate_for("zanzibar"), 4).verdict == "avoid"


def test_learned_rules_expire(rules_dir, monkeypatch):
    visa_rules.remember(VisaRule(passport="IN", country="JP", visa_required=True, document="e_visa"))
    assert lookup("IN", "JP") is not None
    monkeypatch.setitem(visa_rules.knowledge.TTL_DAYS, "visa", -1)
    assert lookup("IN", "JP") is None


def test_trip_builder_groups_travellers():
    request = trip(person("A", origin="Dubai"), person("B"))
    assert [g.origin for g in request.groups] == ["Dubai", "Chennai"]
    assert request.destination == "Bhutan"
