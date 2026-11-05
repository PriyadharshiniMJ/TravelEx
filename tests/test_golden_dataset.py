"""The golden dataset is in sync with its source, well formed, evenly split, and agrees with the seed rules."""
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

from travelex.schemas import Stop
from travelex.tools.visa_rules import lookup

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "evals" / "golden"
GOLDEN = GOLDEN_DIR / "travelex_golden_v1.jsonl"
CASES = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_jsonl_matches_build_golden():
    spec = importlib.util.spec_from_file_location("build_golden", GOLDEN_DIR / "build_golden.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.CASES == CASES, "run: python evals/golden/build_golden.py"


def test_25_unique_cases_split_evenly_across_5_focus_areas():
    assert len(CASES) == 25
    assert len({c["metadata"]["id"] for c in CASES}) == 25
    assert set(Counter(c["metadata"]["focus"] for c in CASES).values()) == {5}


def test_destinations_are_worldwide():
    countries = {s["country"] for c in CASES for s in c["outputs"].get("request", {}).get("stops", [])}
    assert len(countries) >= 10


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["metadata"]["id"])
def test_case_shape(case):
    assert set(case) == {"inputs", "outputs", "metadata"}
    assert case["inputs"]["today"] == "2026-10-27"
    assert case["inputs"]["query"].strip()
    assert isinstance(case["inputs"]["resume"], list)
    assert case["outputs"], "a case with no expectations tests nothing"
    assert case["metadata"]["nodes"]
    for s in case["outputs"].get("request", {}).get("stops", []):
        Stop(**s)  # valid place and ISO country code


VISAS = [(c["metadata"]["id"], v) for c in CASES for v in c["outputs"].get("visa", [])]


@pytest.mark.parametrize("case_id, expected", VISAS, ids=[f"{i}-{v['passport']}-{v['country']}" for i, v in VISAS])
def test_visa_expectations_agree_with_known_rules(case_id, expected):
    """Pairs without a seed rule are researched at eval time; the evaluator checks those against the run."""
    rule = lookup(expected["passport"], expected["country"])
    if rule is None:
        pytest.skip("no seed rule: checked against the researched rule at eval time")
    assert rule.visa_required == expected["visa_required"]
    for field in ("fee", "fee_currency", "document", "zone"):
        if field in expected:
            assert getattr(rule, field) == expected[field]
