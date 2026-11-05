"""Exchange rates for any currency: today's rates from open.er-api.com (no key needed), cached per day.

Falls back to an approximate seed table when offline or when TRAVELEX_FX_OFFLINE is set (tests).
"""
import json
import os
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

RATES_URL = "https://open.er-api.com/v6/latest/USD"
LIVE_SOURCE = "ExchangeRate-API (open.er-api.com)"
SEED_SOURCE = "approximate seed table"
SEED_AS_OF = date(2026, 10, 27)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
TIMEOUT_SECONDS = 10

# Units of each currency per US dollar, copied from the live service on SEED_AS_OF. Fallback only.
SEED_PER_USD = {
    "USD": 1.0, "INR": 95.27, "BTN": 95.27, "NPR": 95.27 * 1.6,"AED": 3.6725, "EUR": 0.8939, "GBP": 0.7492,
    "CHF": 0.8271, "SGD": 1.2807, "MYR": 4.0873, "THB": 33.79, "JPY": 157.4, "KRW": 1325.2, "IDR": 18148.8,
    "CAD": 1.4016, "AUD": 1.4246, "NZD": 1.7587, "BRL": 5.4319, "PEN": 3.4399, "MXN": 18.59, "KES": 129.9,
    "TZS": 2642.6, "NGN": 1359.9, "ZAR": 16.64, "MAD": 9.828, "ISK": 121.9, "MVR": 15.455, "LKR": 332.0,
}


@dataclass(frozen=True)
class Rates:
    per_usd: dict[str, float]
    as_of: date
    source: str


SEED = Rates(SEED_PER_USD, SEED_AS_OF, SEED_SOURCE)
_today: dict[date, Rates] = {}


class FxError(KeyError):
    pass


def _fetch() -> Rates:
    with urllib.request.urlopen(RATES_URL, timeout=TIMEOUT_SECONDS) as response:
        data = json.load(response)
    if data.get("result") != "success":
        raise ValueError(f"exchange-rate service said: {data.get('error-type') or data.get('result')}")
    as_of = datetime.fromtimestamp(data["time_last_update_unix"], timezone.utc).date()
    return Rates(data["rates"], as_of, LIVE_SOURCE)


def rates() -> Rates:
    """Today's rates: memory, then the disk cache, then the live service, then the seed table."""
    if os.getenv("TRAVELEX_FX_OFFLINE"):
        return SEED
    today = date.today()
    if today in _today:
        return _today[today]
    path = Path(os.getenv("TRAVELEX_FX_DIR", PROJECT_ROOT / ".cache" / "fx")) / f"{today}.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        table = Rates(data["per_usd"], date.fromisoformat(data["as_of"]), data["source"])
    else:
        try:
            table = _fetch()
        except Exception:  # offline or the service is down: rough numbers beat no plan
            return SEED
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"per_usd": table.per_usd, "as_of": table.as_of.isoformat(),
                                    "source": table.source}), encoding="utf-8")
    _today[today] = table
    return table


def known(code: str, table: Rates | None = None) -> bool:
    """An ISO currency code we have a rate for."""
    return code.upper() in (table or rates()).per_usd


def convert(amount: float, source: str, target: str, table: Rates | None = None) -> float:
    source, target = source.upper(), target.upper()
    if source == target:
        return amount
    table = table or rates()
    missing = [c for c in (source, target) if c not in table.per_usd]
    if missing:
        raise FxError(f"No exchange rate for {', '.join(missing)} in the {table.source}.")
    return amount / table.per_usd[source] * table.per_usd[target]
