"""Card database loader: consumes a kardscm JSON export, enriches board reads.

The database file is NOT shipped with the plugin (KARDS card data belongs to
1939 Games; the kardscm project intentionally ships no card data either). The
owner generates it locally for personal use:

    git clone https://github.com/AABur/kardscm && cd kardscm
    uv sync
    uv run kardscm sync --yes          # may need HTTPS_PROXY on some networks
    uv run kardscm export -f json -o cards.json

and drops ``cards.json`` into this plugin's data directory (``data_path()``).
Without the file the plugin keeps its vision-only behaviour.

The export shape (kardscm ``export -f json``, raw API dump, verified against
kardscm 0.9.0 ``export/collection.py:card_to_api_dict``):

    {"cardId": ..., "importId": ..., "faction": "soviet", "type": "tank",
     "rarity": ..., "set": ..., "title": {"en": "...", "zh": "..."},
     "text": {...}, "kredits": 4, "attack": 5, "defense": 5,
     "attributes": ["guard"], "extra_abilities": [...],
     "operationCost": 1, "reserved": false, "exile": false, ...}

Orders and countermeasures carry null attack/defense/operationCost.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

DB_FILENAME = "cards.json"

_UNIT_TYPES = frozenset({"infantry", "tank", "artillery", "fighter", "bomber"})

# kardscm attribute keys -> solver keyword ids.
_KEYWORD_MAP = {
    "ambush": "ambush",
    "blitz": "blitz",
    "fury": "fury",
    "guard": "guard",
    "shock": "shock",
    "smokescreen": "smokescreen",
    "covert": "concealed",
    "mobilize": "mobilize",
}

_HEAVY_ARMOR_KEYS = {"heavyArmor1": 1, "heavyArmor2": 2, "heavyArmor3": 3}

_MAX_DB_BYTES = 32 * 1024 * 1024


def _normalize_name(value: str) -> str:
    """Fold a card name for matching: casefold, strip, collapse spaces/NBSP."""
    cleaned = " ".join(value.replace(" ", " ").split())
    return cleaned.casefold()


class CardDb:
    """Name-indexed view over a kardscm export; thread-safe read-only."""

    def __init__(self, cards: list[dict[str, Any]]) -> None:
        self._by_name: dict[str, dict[str, Any]] = {}
        self._by_import_id: dict[str, dict[str, Any]] = {}
        self.card_count = 0
        for card in cards:
            if not isinstance(card, dict):
                continue
            self.card_count += 1
            import_id = card.get("importId")
            if isinstance(import_id, str) and import_id:
                self._by_import_id[import_id] = card
            # Index every locale of the title so any UI language matches.
            title = card.get("title")
            candidates: list[str] = []
            if isinstance(title, dict):
                candidates = [v for v in title.values() if isinstance(v, str) and v.strip()]
            elif isinstance(title, str) and title.strip():
                candidates = [title]
            for name in candidates:
                self._by_name.setdefault(_normalize_name(name), card)

    # -- lookup -----------------------------------------------------------

    def lookup(self, name: str) -> dict[str, Any] | None:
        if not name:
            return None
        return self._by_name.get(_normalize_name(str(name)))

    def unit_type(self, card: dict[str, Any]) -> str | None:
        value = card.get("type")
        return value if value in _UNIT_TYPES else None

    def op_cost(self, card: dict[str, Any]) -> int | None:
        value = card.get("operationCost")
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return value if 0 <= value <= 12 else None

    def deploy_cost(self, card: dict[str, Any]) -> int | None:
        value = card.get("kredits")
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return value if 0 <= value <= 24 else None

    def attack(self, card: dict[str, Any]) -> int | None:
        value = card.get("attack")
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return value if 0 <= value <= 99 else None

    def keywords(self, card: dict[str, Any]) -> tuple[list[str], int]:
        """Solver keywords plus heavy armor level from raw attributes."""
        found: list[str] = []
        heavy_armor = 0
        raw = card.get("attributes")
        if isinstance(raw, list):
            for attr in raw:
                token = str(attr)
                if token in _KEYWORD_MAP:
                    found.append(_KEYWORD_MAP[token])
                elif token in _HEAVY_ARMOR_KEYS:
                    heavy_armor = max(heavy_armor, _HEAVY_ARMOR_KEYS[token])
        extra = card.get("extra_abilities")
        if isinstance(extra, list):
            for attr in extra:
                token = str(attr)
                if token in _KEYWORD_MAP and token not in found:
                    found.append(_KEYWORD_MAP[token])
        return found, heavy_armor


class CardDbLoadError(Exception):
    """Raised when a cards.json exists but cannot be parsed."""


def load_card_db(path: Path) -> CardDb | None:
    """Load a kardscm export; None when the file is absent (normal state).

    Raises CardDbLoadError for a present-but-invalid file so the caller can
    surface the problem instead of silently running without the database.
    """
    try:
        if not path.is_file():
            return None
        if path.stat().st_size > _MAX_DB_BYTES:
            raise CardDbLoadError(f"{path.name} 超过 32MB，疑似不是 kardscm 导出文件")
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CardDbLoadError(f"读取 {path.name} 失败：{exc}") from exc
    except json.JSONDecodeError as exc:
        raise CardDbLoadError(f"{path.name} 不是有效 JSON：{exc.msg}") from exc
    if isinstance(data, dict):
        cards = data.get("cards")
    else:
        cards = data
    if not isinstance(cards, list) or not cards:
        raise CardDbLoadError(f"{path.name} 里没有卡牌数组")
    return CardDb(cards)


class CardDbHolder:
    """Lazy, once-only loader for the plugin's data directory."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._db: CardDb | None = None
        self._error: str | None = None
        self._loaded = False

    def get(self, data_dir: Path) -> tuple[CardDb | None, str | None]:
        """Return (db, error); error is set only for an invalid file."""
        with self._lock:
            if not self._loaded:
                self._loaded = True
                try:
                    self._db = load_card_db(data_dir / DB_FILENAME)
                except CardDbLoadError as exc:
                    self._error = str(exc)
            return self._db, self._error
