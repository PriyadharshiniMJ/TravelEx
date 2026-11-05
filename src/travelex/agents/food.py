"""Food: per city, a short guide of local dishes and places to eat that fit the travellers' diet.
Code drops places no source names, dishes that break the diet, and dishes that name an allergen a traveller must
avoid. Code never calls a dish safe for an allergy: what's left is marked "unknown", to ask about."""
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage

from travelex.agents.common import event, register, render
from travelex.llm import get_llm, structured
from travelex.schemas import Dish, FoodGuide, Place, Source
from travelex.state import FoodInput
from travelex.tools import search
from travelex.tools.countries import country_name
from travelex.trip import party

FOOD_PROMPT = """You write a short food guide for one city, from the numbered sources and your knowledge of the
local cuisine. The city is in the country given: use only dishes and places from there, and ignore any source
about a place of the same name in another country.
- dishes: 3-6 specialities of this city or region, not dishes from elsewhere. diet_tags only where the dish's
  usual ingredients make them true: "vegetarian", "vegan", "contains nuts", "contains pork", "spicy". Beef,
  buffalo, veal, lamb, guinea pig and fish are not pork. Never tag a dish "halal": that depends on the kitchen.
  {kids}
- The travellers' diet: {diet}. Recommend only dishes that fit it, or a version of a local dish that does,
  and say in diet_notes what to ask for or avoid (fish stock in "vegetarian" soups, nuts in sauces).
- Allergies: {allergies}. Leave out dishes usually made with these, name every main ingredient in the
  description, and say in diet_notes how to ask for it in the local language. Leave allergens_unknown empty.
- places: 2-3 restaurants, markets or food streets that a source names, with area, why, and source_ids.
  Never invent a place."""
KIDS = "kid_friendly for mild, familiar food."
NO_KIDS = "No children are travelling: kid_friendly false, and never mention children."

MEAT = ("chicken", "beef", "pork", "mutton", "lamb", "goat", "fish", "prawn", "shrimp", "crab", "squid", "octopus",
        "bacon", "ham", "sausage", "yak meat", "duck", "veal", "anchovy", "meat", "phaksha", "jasha maru", "sikam",
        "tonkatsu", "samgyeopsal", "bulgogi", "galbi", "katsu", "salmon", "tuna", "eel", "oyster", "clam", "mussel",
        "cod", "baccalà", "baccala", "bacalao", "bacalhau", "seafood", "lobster", "scallop", "buffalo", "cuy",
        "guinea pig", "rabbit", "venison", "turkey", "quail", "tripe", "liver", "steak", "suadero")
PORK = ("pork", "bacon", "ham", "lard", "phaksha", "sikam", "tonkatsu", "samgyeopsal", "moo hong", "char siu",
        "babi guling", "chorizo", "prosciutto", "salami")
FORBIDDEN = {"vegetarian": MEAT, "vegan": MEAT + ("cheese", "milk", "butter", "ghee", "egg", "honey", "yoghurt",
                                                   "yogurt", "cream", "datshi"),
             "halal": PORK, "jain": MEAT, "pescatarian": tuple(m for m in MEAT if m not in
                                                              ("fish", "prawn", "shrimp", "crab", "squid", "octopus",
                                                               "anchovy", "salmon", "tuna", "eel", "oyster", "clam",
                                                               "mussel", "cod", "baccalà", "baccala", "bacalao",
                                                               "bacalhau", "seafood", "lobster", "scallop"))}


# Ingredients and dishes that usually contain each allergen. "nut allergy" covers peanuts and tree nuts.
ALLERGENS = {
    "peanut": ("peanut", "groundnut", "satay", "sate", "kare-kare", "gado-gado", "pad thai", "massaman", "kung pao",
               "mafe", "pecel"),
    "tree nut": ("nut", "almond", "cashew", "walnut", "pecan", "pistachio", "hazelnut", "macadamia", "pine nut",
                 "praline", "marzipan", "pesto", "baklava", "nougat", "frangipane"),
    "sesame": ("sesame", "tahini", "hummus", "halva", "halwa", "gomae"),
    "shellfish": ("shrimp", "prawn", "crab", "lobster", "crayfish", "langoustine", "scallop", "oyster", "clam",
                  "mussel", "squid", "octopus", "shrimp paste", "belacan"),
    "fish": ("fish", "salmon", "tuna", "anchovy", "cod", "eel", "dashi", "bonito", "fish sauce", "nam pla",
             "sardine", "mackerel", "katsuobushi"),
    "egg": ("egg", "omelette", "omelet", "mayonnaise", "meringue", "custard", "tamagoyaki", "carbonara"),
    "dairy": ("milk", "cheese", "butter", "ghee", "cream", "yoghurt", "yogurt", "paneer", "curd", "lassi", "gelato",
              "datshi"),
    "gluten": ("wheat", "flour", "bread", "udon", "ramen", "pasta", "couscous", "seitan", "soy sauce", "tempura",
               "barley", "rye", "bulgur", "naan", "roti", "croissant", "baguette", "pizza", "dumpling"),
    "soy": ("soy", "soya", "tofu", "edamame", "miso", "tempeh", "natto"),
}
ALLERGY_NAMES = (("peanut", ("peanut", "groundnut")), ("tree nut", ("tree nut",)), ("sesame", ("sesame",)),
                 ("shellfish", ("shellfish", "crustacean", "seafood")), ("fish", ("fish", "seafood")),
                 ("egg", ("egg",)), ("dairy", ("dairy", "lactose", "milk")),
                 ("gluten", ("gluten", "wheat", "coeliac", "celiac")), ("soy", ("soy",)))
ALLERGY_WORDS = re.compile(r"allerg|intoleran|coeliac|celiac|\bfree\b|-free", re.IGNORECASE)
PLAIN_NUTS = re.compile(r"(?<!pea)(?<!tree )\bnuts?\b", re.IGNORECASE)  # "nut allergy", not "peanut" or "tree nut"
# A mention cleared by the words around it: "without peanuts", "no egg", "peanut-free", "unlike the chicken version"
NEGATION_BEFORE = re.compile(r"\b(no|without|minus|free of|instead of|unlike|not|skip|hold the)\W+(\w+\W+){0,2}$",
                             re.IGNORECASE)
NEGATION_AFTER = re.compile(r"^\W?free\b", re.IGNORECASE)


def allergies(diet: list[str]) -> list[str]:
    """The allergens a traveller must avoid, from entries such as "peanut allergy" or "gluten-free".
    "nut allergy" means peanuts and tree nuts: people allergic to nuts are usually told to avoid both."""
    found: list[str] = []
    for entry in (d.casefold() for d in diet if ALLERGY_WORDS.search(d)):
        names = [name for name, words in ALLERGY_NAMES if any(re.search(rf"\b{w}s?\b", entry) for w in words)]
        if PLAIN_NUTS.search(entry):
            names += ["peanut", "tree nut"]
        found += [n for n in names if n not in found]
    return found


def allergen_terms(names: list[str]) -> dict[str, tuple[str, ...]]:
    return {n: ALLERGENS[n] for n in names}


def mentions(text: str, term: str) -> bool:
    """The term appears in the text (plural too) at least once without a negation around it."""
    for m in re.finditer(rf"\b{re.escape(term)}(?:s|es)?\b", text, re.IGNORECASE):
        if not (NEGATION_BEFORE.search(text[max(0, m.start() - 40):m.start()]) or NEGATION_AFTER.search(text[m.end():])):
            return True
    return False


def conflicts(dish: Dish, allergens: dict[str, tuple[str, ...]]) -> str | None:
    """The allergen this dish names, in its name, description or tags ("contains nuts"), or None. A tag can
    never clear an ingredient the description names; only a negation next to the word does ("peanut-free")."""
    text = " | ".join([dish.name, dish.description, *dish.diet_tags])
    return next((name for name, terms in allergens.items() if any(mentions(text, t) for t in terms)), None)


def diets(diet: list[str]) -> list[str]:
    return [d for d in (x.lower() for x in diet) if any(key in d for key in FORBIDDEN)]


def forbidden_terms(diet: list[str]) -> set[str]:
    terms: set[str] = set()
    for d in diets(diet):
        for key, words in FORBIDDEN.items():
            if key in d:
                terms |= set(words)
    return terms


def has_term(text: str, term: str) -> bool:
    return re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE) is not None


def breaks_diet(dish: Dish, terms: set[str]) -> str | None:
    """The forbidden word that makes this dish unsuitable, or None. A "veg" name clears a word that appears only
    in the description ("veg momos: like the chicken ones, but with cabbage"), and so does a negation ("no fish
    sauce"). A "vegetarian" tag doesn't: the model's tags never override the ingredients it describes."""
    veg_name = any(has_term(dish.name, w) for w in ("veg", "vegetarian", "vegan"))
    for term in terms:
        if mentions(dish.name, term) or (mentions(dish.description, term) and not veg_name):
            return term
    return None


def checked_tags(dish: Dish) -> list[str]:
    """The model's diet tags, checked against the ingredients the dish names. "contains pork" stays only on a dish
    with pork in it (and is added when the description names pork); "vegetarian" and "vegan" go from a dish that
    names meat, fish or (vegan) animal products; "halal" always goes, since it depends on the kitchen."""
    text = f"{dish.name} | {dish.description}"
    veg_name = any(has_term(dish.name, w) for w in ("veg", "vegetarian", "vegan"))
    pork = any(mentions(text, t) for t in PORK)
    names = lambda diet: any(mentions(dish.name, t) or (mentions(dish.description, t) and not veg_name)
                             for t in FORBIDDEN[diet])
    tags = []
    for tag in dish.diet_tags:
        low = tag.strip().casefold()
        if "halal" in low or ("pork" in low and not pork) or (low in ("vegetarian", "veg") and names("vegetarian")) \
                or (low == "vegan" and names("vegan")):
            continue
        tags.append(tag)
    if pork and not any("pork" in t.casefold() for t in tags):
        tags.append("contains pork")
    return tags


def in_country(place: Place, sources: dict[str, Source], names: list[str]) -> bool:
    """A source the place cites names the city or its country: a search for Iceland's "South Coast Region" can
    return South Coast Plaza in California, a page that names neither."""
    return not names or any(n.casefold() in f"{sources[i].title} {sources[i].text}".casefold()
                            for i in place.source_ids if i in sources for n in names)


def grounded(place: Place, sources: dict[str, Source]) -> bool:
    """A place counts only if a source it cites names it."""
    name = place.name.casefold()
    return any(name in sources[i].text.casefold() or name in sources[i].title.casefold()
               for i in place.source_ids if i in sources)


def food_agent(state: FoodInput) -> dict:
    started = time.perf_counter()
    city, request = state["city"], state["request"]
    country = country_name(state["country"]) if state.get("country") else ""
    where = f"{city}, {country}" if country and country.casefold() not in city.casefold() else city
    avoid = allergies(request.diet)
    diet = ", ".join(d for d in request.diet if not ALLERGY_WORDS.search(d)) or "no restrictions"
    queries = [f"{where} local food must try dishes",
               f"best {' '.join(diets(request.diet)) or 'local'} restaurants in {where}"]
    children = any(t.type != "adult" for t in request.travellers)
    if children:
        queries.append(f"family friendly restaurants {where}")
    try:
        new, found = register(search.search_all(queries), state["source_prefix"], {}, state.get("taken"))
        prompt = FOOD_PROMPT.format(diet=diet, allergies=", ".join(avoid) or "none",
                                    kids=KIDS if children else NO_KIDS)
        task = f"City: {city}\nCountry: {country or 'not known'}\nParty: {party(request)}\n\nSources:\n{render(found)}"
        guide = structured(get_llm("food"), FoodGuide, [SystemMessage(prompt), HumanMessage(task)])
    except search.SearchConfigError:
        raise
    except Exception as e:  # fail soft: the QA step reports the missing city and may retry it
        return {"errors": [f"food_agent ({city}): {type(e).__name__}: {e}"],
                "events": [event("food_agent", "failed", f"{city}: {e}", queries, started)]}

    sources = {s.id: s for s in found}
    terms, allergens = forbidden_terms(request.diet), allergen_terms(avoid)
    # nothing here says a kept dish is free of the allergens: the traveller must still ask
    kept_dishes = [d.model_copy(update={"kid_friendly": d.kid_friendly and children, "allergens_unknown": avoid,
                                        "diet_tags": checked_tags(d)})
                   for d in guide.dishes if not breaks_diet(d, terms) and not conflicts(d, allergens)]
    places = [p.model_copy(update={"source_ids": [i for i in p.source_ids if i in sources]}) for p in guide.places]
    names = [country, city] if country else []
    kept_places = [p for p in places if grounded(p, sources) and in_country(p, sources, names)]
    dropped = len(guide.dishes) - len(kept_dishes), len(places) - len(kept_places)
    guide = guide.model_copy(update={"city": city, "dishes": kept_dishes,
                                     "places": kept_places})
    note = (f" (dropped {dropped[0]} dish(es) off-diet or with an allergen, {dropped[1]} unsourced or "
            f"out-of-country place(s))"
            if any(dropped) else "")
    return {"food": {city: guide}, "sources": new,
            "events": [event("food_agent", "guide", f"{city}: {', '.join(d.name for d in kept_dishes)}{note}",
                             queries, started)]}
