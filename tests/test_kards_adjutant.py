"""Unit tests for the KARDS adjutant three-stage pipeline.

Covers the orchestration (capture -> vision JSON -> solver -> persona push),
the verdict composition, and the JSON transcription layer's guards.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

import plugin.plugins.kards_adjutant as kards_module
import plugin.plugins.kards_adjutant.board_read as board_read_module
from plugin.plugins.kards_adjutant import KardsAdjutantPlugin, compose_verdict
from plugin.plugins.kards_adjutant.board_read import (
    BoardReadError,
    board_state_from_json,
    extract_json,
    parse_board_text,
)
from plugin.plugins.kards_adjutant.card_db import (
    CardDb,
    CardDbHolder,
    CardDbLoadError,
    load_card_db,
)
from plugin.plugins.kards_adjutant.lethal_solver import solve

# A data dir that never exists: the card db stays absent (normal state).
_EMPTY_DATA_DIR = Path(tempfile.gettempdir()) / "kards-adjutant-test-nodata"


class _Logger:
    def info(self, *args, **kwargs) -> None:
        pass

    def warning(self, *args, **kwargs) -> None:
        pass

    def exception(self, *args, **kwargs) -> None:
        pass


class _Models:
    def __init__(self, response_content: str | None) -> None:
        self._response_content = response_content
        self.calls: list[dict[str, object]] = []

    async def get_client(self) -> "_FakeClient":
        return _FakeClient(self)


class _FakeClient:
    def __init__(self, owner: "_Models") -> None:
        self._owner = owner
        self.chat = self

    @property
    def completions(self) -> "_FakeClient":
        return self

    async def create(self, **kwargs: object) -> object:
        self._owner.calls.append(kwargs)
        content = self._owner._response_content

        class _Message:
            pass

        class _Choice:
            pass

        class _Response:
            pass

        message = _Message()
        message.content = content
        choice = _Choice()
        choice.message = message
        response = _Response()
        response.choices = [] if content is None else [choice]
        return response


class _Context:
    plugin_id = "kards_adjutant"

    def __init__(self, models: _Models) -> None:
        self.models = models
        self.pushed: list[dict[str, object]] = []

    def push_message(self, **kwargs: object) -> dict[str, object]:
        self.pushed.append(kwargs)
        return {"ok": True}

    async def get_own_effective_config(self) -> dict[str, object]:
        return {
            "kards": {
                "screenshot_max_edge_px": 1234,
                "screenshot_jpeg_quality": 77,
                "vision_max_tokens": 999,
                "solver_node_limit": 12345,
            }
        }


def _plugin(models: _Models) -> KardsAdjutantPlugin:
    plugin = KardsAdjutantPlugin.__new__(KardsAdjutantPlugin)
    plugin.ctx = _Context(models)
    plugin.logger = _Logger()
    plugin._card_dbs = CardDbHolder()
    plugin.data_path = lambda *parts: _EMPTY_DATA_DIR.joinpath(*parts)
    return plugin


def _sync_capture(data_url: str):
    def _capture(max_edge: int, quality: int) -> tuple[str, str]:
        return data_url, ""

    return _capture


_VALID_BOARD_JSON = {
    "read_ok": True,
    "issues": [],
    "my_kreds": 3,
    "my_hq_hp": 20,
    "enemy_hq_hp": 5,
    "enemy_hq_slot": 4,
    "enemy_hq_smokescreen": False,
    "enemy_hq_immune": False,
    "my_units": [
        {"name": "我步兵", "type": "infantry", "attack": 4, "hp": 3, "line": "front",
         "slot": -1, "op_cost": 1, "can_act": True, "keywords": [], "heavy_armor": 0,
         "confidence": "clear"},
        {"name": "我坦克", "type": "tank", "attack": 5, "hp": 5, "line": "support",
         "slot": 1, "op_cost": 1, "can_act": True, "keywords": [], "heavy_armor": 0,
         "confidence": "clear"},
    ],
    "enemy_units": [
        {"name": "敌守卫", "type": "infantry", "attack": 1, "hp": 3, "line": "support",
         "slot": 3, "op_cost": 1, "can_act": True, "keywords": ["guard"], "heavy_armor": 0,
         "confidence": "clear"},
    ],
    "hand": [
        {"name": "炮击", "cost": 2, "damage": 3, "target": "hq", "confidence": "clear"},
    ],
}


# ---------------------------------------------------------------------------
# Transcription layer
# ---------------------------------------------------------------------------

@pytest.mark.plugin_unit
def test_extract_json_strips_fences_and_prose() -> None:
    fenced = "```json\n" + json.dumps(_VALID_BOARD_JSON) + "\n```"
    assert extract_json(fenced)["my_kreds"] == 3
    chatty = "好的，以下是读数：\n" + json.dumps(_VALID_BOARD_JSON) + "\n以上。"
    assert extract_json(chatty)["enemy_hq_hp"] == 5


@pytest.mark.plugin_unit
def test_extract_json_rejects_garbage() -> None:
    with pytest.raises(BoardReadError):
        extract_json("")
    with pytest.raises(BoardReadError):
        extract_json("这不是 JSON")


@pytest.mark.plugin_unit
def test_board_state_validation_rejects_bad_values() -> None:
    bad = dict(_VALID_BOARD_JSON, my_kreds=99)
    with pytest.raises(BoardReadError):
        board_state_from_json(bad)
    bad_type = json.loads(json.dumps(_VALID_BOARD_JSON))
    bad_type["my_units"][0]["type"] = "战列舰"
    with pytest.raises(BoardReadError):
        board_state_from_json(bad_type)
    unreadable = dict(_VALID_BOARD_JSON, read_ok=False, issues=["画面模糊"])
    with pytest.raises(BoardReadError, match="画面模糊"):
        board_state_from_json(unreadable)


@pytest.mark.plugin_unit
def test_board_state_collects_low_confidence_and_unknown_keywords() -> None:
    data = json.loads(json.dumps(_VALID_BOARD_JSON))
    data["my_units"][0]["confidence"] = "blurry"
    data["enemy_units"][0]["keywords"] = ["guard", "unknown:海啸"]
    state = board_state_from_json(data)
    assert state.low_confidence_fields == ("我步兵",)
    assert state.unknown_keywords == ("unknown:海啸",)
    sol = solve(state)
    assert any("低把握" in c for c in sol.caveats)
    assert any("未知关键词" in c for c in sol.caveats)


@pytest.mark.plugin_unit
def test_missing_critical_number_is_rejected() -> None:
    data = dict(_VALID_BOARD_JSON, enemy_hq_hp=None)
    with pytest.raises(BoardReadError, match="敌方总部血量"):
        parse_board_text(json.dumps(data))


# ---------------------------------------------------------------------------
# Card database (kardscm export)
# ---------------------------------------------------------------------------

_KARDSCM_CARDS = [
    {
        "cardId": 1, "importId": "germany_grenadier_guards", "faction": "germany",
        "type": "infantry", "rarity": "Standard", "set": "Base",
        "title": {"en": "Grenadier Guards", "zh": "掷弹兵卫队"},
        "text": {"en": "Guard.", "zh": "守护。"},
        "kredits": 3, "attack": 2, "defense": 5,
        "attributes": ["guard"], "extra_abilities": [],
        "operationCost": 1, "reserved": False, "exile": False, "quantity": 4,
    },
    {
        "cardId": 2, "importId": "soviet_il2", "faction": "soviet",
        "type": "bomber", "rarity": "Limited", "set": "Base",
        "title": {"en": "Il-2 Sturmovik", "zh": "Il-2 Sturmovik"},
        "text": {"en": "", "zh": ""},
        "kredits": 4, "attack": 5, "defense": 3,
        "attributes": ["heavyArmor2"], "extra_abilities": [],
        "operationCost": 2, "reserved": False, "exile": False, "quantity": 2,
    },
    {
        "cardId": 3, "importId": "soviet_barrage", "faction": "soviet",
        "type": "order", "rarity": "Standard", "set": "Base",
        "title": {"en": "Artillery Barrage", "zh": "炮兵掩护"},
        "text": {"en": "Deal 3 damage.", "zh": "造成 3 点伤害。"},
        "kredits": 3, "attack": None, "defense": None,
        "attributes": [], "extra_abilities": [],
        "operationCost": None, "reserved": False, "exile": False, "quantity": 3,
    },
]


@pytest.mark.plugin_unit
def test_card_db_loads_kardscm_export(tmp_path: Path) -> None:
    path = tmp_path / "cards.json"
    path.write_text(json.dumps(_KARDSCM_CARDS), encoding="utf-8")
    db = load_card_db(path)
    assert db is not None and db.card_count == 3
    # Both locales and case/NBSP-insensitive matching resolve to the card.
    assert db.lookup("掷弹兵卫队")["importId"] == "germany_grenadier_guards"
    assert db.lookup("grenadier  guards")["importId"] == "germany_grenadier_guards"
    assert db.lookup("IL-2 sturmovik")["type"] == "bomber"
    assert db.lookup("不存在的卡") is None
    # Orders are not unit types; units expose op cost and keywords.
    assert db.unit_type(db.lookup("炮兵掩护")) is None
    assert db.op_cost(db.lookup("Il-2 Sturmovik")) == 2
    assert db.keywords(db.lookup("掷弹兵卫队")) == (["guard"], 0)
    assert db.keywords(db.lookup("Il-2 Sturmovik")) == ([], 2)
    assert db.deploy_cost(db.lookup("炮兵掩护")) == 3


@pytest.mark.plugin_unit
def test_card_db_missing_file_is_none_invalid_raises(tmp_path: Path) -> None:
    assert load_card_db(tmp_path / "cards.json") is None
    bad = tmp_path / "cards.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(CardDbLoadError):
        load_card_db(bad)


@pytest.mark.plugin_unit
def test_prefill_rescues_unreadable_type_and_fills_base_facts() -> None:
    db = CardDb(list(_KARDSCM_CARDS))
    raw = {
        "read_ok": True, "issues": [], "my_kreds": 5, "my_hq_hp": 20,
        "enemy_hq_hp": 20, "enemy_hq_slot": 4,
        "enemy_hq_smokescreen": False, "enemy_hq_immune": False,
        "my_units": [
            # Type unreadable and op cost null — the db must rescue both,
            # while hp (damaged in game) and attack (buffed) stay as read.
            {"name": "掷弹兵卫队", "type": None, "attack": 4, "hp": 2, "line": "front",
             "slot": -1, "op_cost": None, "can_act": True, "keywords": [],
             "heavy_armor": 0, "confidence": "clear"},
        ],
        "enemy_units": [],
        "hand": [
            # Cost unreadable; damage stays vision-only.
            {"name": "炮兵掩护", "cost": None, "damage": 3, "target": "any", "confidence": "clear"},
        ],
    }
    state = board_state_from_json(raw, card_db=db)
    unit = state.my_units[0]
    assert unit.type == "infantry"
    assert unit.op_cost == 1
    assert unit.keywords == ("guard",)
    assert unit.attack == 4  # live buff preserved
    assert unit.hp == 2  # current damage state preserved
    assert state.hand[0].cost == 3


@pytest.mark.plugin_unit
def test_without_db_unreadable_type_still_rejected() -> None:
    raw = {
        "read_ok": True, "issues": [], "my_kreds": 5, "my_hq_hp": 20,
        "enemy_hq_hp": 20, "enemy_hq_slot": 4,
        "enemy_hq_smokescreen": False, "enemy_hq_immune": False,
        "my_units": [
            {"name": "掷弹兵卫队", "type": None, "attack": 4, "hp": 2, "line": "front",
             "slot": -1, "op_cost": 1, "can_act": True, "keywords": [],
             "heavy_armor": 0, "confidence": "clear"},
        ],
        "enemy_units": [], "hand": [],
    }
    with pytest.raises(BoardReadError, match="类型不可读"):
        board_state_from_json(raw)


# ---------------------------------------------------------------------------
# Verdict composition
# ---------------------------------------------------------------------------

@pytest.mark.plugin_unit
def test_compose_verdict_lethal_first_line() -> None:
    state = board_state_from_json(_VALID_BOARD_JSON)
    sol = solve(state)
    text = compose_verdict(sol, state)
    first = text.splitlines()[0]
    # The guard (3hp, slot 3) covers the HQ (slot 4): infantry clears it for
    # 1 op, the tank pushes in for 2 op and lands 5 on a 5hp HQ -> lethal.
    assert first.startswith("结论：最大伤害")
    assert "斩杀：能" in first
    assert "读数：Kreds 3" in text
    assert "先清目标：敌守卫" in text
    assert "未计入敌方反制指令" in text


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

@pytest.mark.plugin_unit
@pytest.mark.asyncio
async def test_capture_failure_returns_err_without_model_call(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(max_edge: int, quality: int) -> tuple[None, str]:
        return None, "无法捕获屏幕: 测试"

    monkeypatch.setattr(kards_module, "_capture_screen_jpeg", _fail)
    models = _Models(json.dumps(_VALID_BOARD_JSON))
    plugin = _plugin(models)

    result = await plugin.lethal_check()

    assert result.is_err()
    assert "无法捕获屏幕" in str(result.error)
    assert models.calls == []
    assert plugin.ctx.pushed == []


@pytest.mark.plugin_unit
@pytest.mark.asyncio
async def test_lethal_check_pipeline_pushes_verdict_for_persona(monkeypatch: pytest.MonkeyPatch) -> None:
    data_url = "data:image/jpeg;base64,AA"
    monkeypatch.setattr(kards_module, "_capture_screen_jpeg", _sync_capture(data_url))
    models = _Models(json.dumps(_VALID_BOARD_JSON))
    plugin = _plugin(models)

    result = await plugin.lethal_check()

    assert result.is_ok()
    call = models.calls[0]
    assert call["model"] == "lethal_vision"
    assert call["max_tokens"] == 999  # from effective config
    user_content = call["messages"][1]["content"]
    assert user_content[1]["image_url"]["url"] == data_url
    assert isinstance(user_content[0]["text"], str)

    # The persona gets the computed verdict and must repeat, not recompute.
    assert len(plugin.ctx.pushed) == 1
    pushed = plugin.ctx.pushed[0]
    assert pushed["ai_behavior"] == "respond"
    assert pushed["visibility"] == []
    text = pushed["parts"][0]["text"]
    assert text.startswith("【KARDS 斩杀计算结果")
    assert "禁止重新计算" in text
    assert "斩杀：能" in text
    assert result.value["headline"].startswith("最大伤害")


@pytest.mark.plugin_unit
@pytest.mark.asyncio
async def test_unparsable_vision_reply_is_a_clean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kards_module, "_capture_screen_jpeg", _sync_capture("data:image/jpeg;base64,AA"))
    models = _Models("我觉得你能赢！加油！")  # no JSON at all
    plugin = _plugin(models)

    result = await plugin.lethal_check()

    assert result.is_err()
    assert "读图失败" in str(result.error)
    assert plugin.ctx.pushed == []


@pytest.mark.plugin_unit
@pytest.mark.asyncio
async def test_empty_model_answer_is_a_clean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kards_module, "_capture_screen_jpeg", _sync_capture("data:image/jpeg;base64,AA"))
    models = _Models(None)
    plugin = _plugin(models)

    result = await plugin.lethal_check()

    assert result.is_err()
    assert "读图失败" in str(result.error)
