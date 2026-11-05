"""Every schema an LLM fills must be valid for OpenAI structured output, or the live call fails with a 400.

The rules checked are the ones that have bitten us: no dict fields (objects without fixed properties) and
no tuples (arrays without "items").
"""
import pytest

from travelex.schemas import (
    AlternativeIdeas, ClimateResearch, FoodGuide, Itinerary, RelevanceCheck, RouteList, StopCosts, TripRequest,
    VisaResearch,
)

LLM_SCHEMAS = [TripRequest, ClimateResearch, RouteList, VisaResearch, RelevanceCheck, StopCosts, AlternativeIdeas,
               Itinerary, FoodGuide]


def problems(node, path="$") -> list[str]:
    found = []
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" not in node and "$ref" not in node:
            found.append(f"{path}: object without fixed properties (a dict field)")
        if node.get("type") == "array" and "items" not in node:
            found.append(f"{path}: array without items (a tuple field)")
        for key, value in node.items():
            found += problems(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            found += problems(value, f"{path}[{i}]")
    return found


@pytest.mark.parametrize("schema", LLM_SCHEMAS, ids=lambda s: s.__name__)
def test_schema_is_openai_compatible(schema):
    assert problems(schema.model_json_schema()) == []
