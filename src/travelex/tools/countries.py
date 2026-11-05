"""ISO country codes → names for search queries and prompts ("KR" → "South Korea")."""
from functools import cache

import pycountry


@cache
def country_name(code: str | None) -> str:
    if not code:
        return "unknown country"
    country = pycountry.countries.get(alpha_2=code.upper())
    if country is None:
        return code
    return getattr(country, "common_name", None) or country.name
