"""Rules the research agents learned (visas, fees, seasons), cached on disk so later trips reuse them."""
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TTL_DAYS = {"visa": 30, "fees": 30, "climate": 365}  # visa and fee rules change more often than seasons


def rules_dir() -> Path:
    return Path(os.getenv("TRAVELEX_RULES_DIR", PROJECT_ROOT / ".cache" / "rules"))


def _path(kind: str, key: str) -> Path:
    return rules_dir() / kind / f"{re.sub(r'[^a-z0-9*-]+', '_', key.casefold()).replace('*', 'any')}.json"


def aware(when: datetime | str) -> datetime:
    when = datetime.fromisoformat(when) if isinstance(when, str) else when
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def recall(kind: str, key: str, type_: Any) -> Any | None:
    """The stored value, or None if it was never learned or is older than its TTL. Its age counts from when its
    evidence was fetched, not when it was saved: a rule rebuilt today from a month-old page is a month old."""
    path = _path(kind, key)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    as_of = min(aware(data["saved_at"]), aware(data.get("evidence_at") or data["saved_at"]))
    if datetime.now(timezone.utc) - as_of > timedelta(days=TTL_DAYS[kind]):
        return None
    return TypeAdapter(type_).validate_python(data["value"])


def remember(kind: str, key: str, value: Any, type_: Any, evidence_at: datetime | str | None = None) -> None:
    """evidence_at: when the oldest page behind the value was fetched (None: now)."""
    path = _path(kind, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    payload = {"saved_at": now.isoformat(), "evidence_at": aware(evidence_at or now).isoformat(),
               "value": TypeAdapter(type_).dump_python(value, mode="json")}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
