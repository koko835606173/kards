"""Vision transcription layer: screenshot -> strict-JSON board state.

The vision model is a readout clerk, nothing more: it transcribes visible
numbers into a fixed JSON shape and is explicitly forbidden from calculating,
advising, or inventing values. All game logic lives in lethal_solver.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .lethal_solver import BoardState, DamageOrder, Unit

UNIT_TYPES = frozenset({"infantry", "tank", "artillery", "fighter", "bomber"})
KNOWN_KEYWORDS = frozenset(
    {"ambush", "blitz", "fury", "guard", "shock", "smokescreen", "immune", "pinned"}
)

# plugin/ is outside the repo-wide English-prompt lint scope by convention;
# the owner-facing game prompt stays in Simplified Chinese on purpose.
SYSTEM_PROMPT = """你是 KARDS 读数员，只把画面如实转录成一个 JSON，禁止任何计算、推理、建议或寒暄。
主人（我方）是屏幕下方的玩家。只输出一个 JSON 对象，前后不许有任何别的文字或代码块标记。

输出格式（字段缺一不可，读不到的数值用 null）：
{
  "read_ok": true,
  "issues": [],
  "my_kreds": 0,
  "my_hq_hp": 0,
  "enemy_hq_hp": 0,
  "enemy_hq_slot": 4,
  "enemy_hq_smokescreen": false,
  "enemy_hq_immune": false,
  "my_units": [],
  "enemy_units": [],
  "hand": []
}

单位对象字段（my_units / enemy_units 数组的元素）：
{"name": "名称或牌面文字", "type": "infantry|tank|artillery|fighter|bomber",
 "attack": 0, "hp": 0, "line": "front|support", "slot": -1,
 "op_cost": 1, "can_act": true, "keywords": [], "heavy_armor": 0, "confidence": "clear"}

手牌对象字段（hand 数组的元素）——只列「直接伤害类指令」且费用与伤害数值都看得见的牌：
{"name": "名称", "cost": 1, "damage": 1, "target": "hq|unit|any", "confidence": "clear"}

铁律：
1. 所有数字只来自画面；被手指/特效遮挡或模糊的，取保守值并把该字段 confidence 标为 "blurry"、在 issues 里写明；完全看不到的用 null 并写进 issues。
2. type 对照：步兵=infantry，坦克=tank，炮兵=artillery，战斗机=fighter，轰炸机=bomber；看不准类型就写 null。
3. keywords 只能用这些英文 id：ambush(伏击) blitz(闪击) fury(奋战) guard(守护) shock(冲击) smokescreen(烟幕) immune(免疫) pinned(压制)；画面上有不在这个列表里的关键词，写成 "unknown:中文或原文"。
4. line：前线单位 line="front"、slot=-1；支援线单位 line="support"、slot=从该玩家视角从左到右 0-4。
5. enemy_hq_slot：敌方支援线 5 个槽位里 HQ 所在的位置(0-4)。
6. can_act：画面显示该单位本回合还能行动（未被压制/未耗尽）为 true。
7. hand 里不要放单位牌、增益牌、看不清效果的牌——它们不参与本次计算。
8. 画面不是 KARDS 对局时：read_ok=false，issues 写一句原因，其余字段保持默认。"""

USER_BRIEF = "转录这张 KARDS 画面的当前局面。只输出 JSON。"

_MAX_JSON_CHARS = 20_000


class BoardReadError(Exception):
    """Raised when the model's JSON cannot be turned into a BoardState."""


def extract_json(text: str) -> dict[str, Any]:
    """Strip code fences / prose and parse the first JSON object."""
    if not text or not text.strip():
        raise BoardReadError("视觉模型没有返回内容")
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise BoardReadError("返回内容里没有 JSON 对象")
    snippet = cleaned[start : end + 1]
    if len(snippet) > _MAX_JSON_CHARS:
        raise BoardReadError("返回的 JSON 过长，疑似不是纯读数")
    try:
        data = json.loads(snippet)
    except json.JSONDecodeError as exc:
        raise BoardReadError(f"JSON 解析失败：{exc.msg}（第 {exc.lineno if exc.lineno else 0} 行）") from exc
    if not isinstance(data, dict):
        raise BoardReadError("JSON 顶层不是对象")
    return data


def _require_int(data: dict[str, Any], key: str, *, lo: int, hi: int, label: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BoardReadError(f"{label} 不是数字：{value!r}")
    if not (lo <= value <= hi):
        raise BoardReadError(f"{label} 超出合理范围：{value}")
    return int(value)


def _unit_from(raw: Any, side: str, idx: int, issues: list[str]) -> tuple[Unit | None, str | None, list[str]]:
    """Parse one unit; returns (unit, unreadable_reason, unknown_keywords)."""
    if not isinstance(raw, dict):
        raise BoardReadError(f"{side} 单位 #{idx} 不是对象")
    name = str(raw.get("name") or f"{side}单位{idx + 1}")
    conf = "blurry" if raw.get("confidence") == "blurry" else "clear"

    utype = raw.get("type")
    if utype not in UNIT_TYPES:
        # An unreadable type blocks every targeting rule for that unit.
        return None, f"{name}（类型不可读）", []

    attack = _require_int(raw, "attack", lo=0, hi=99, label=f"{name} 攻击力")
    hp = _require_int(raw, "hp", lo=1, hi=99, label=f"{name} 血量")
    line = raw.get("line")
    if line not in ("front", "support"):
        return None, f"{name}（所在线不可读）", []
    slot = _require_int(raw, "slot", lo=-1, hi=4, label=f"{name} 槽位")
    if slot is None:
        slot = -1 if line == "front" else 0
    op_cost = _require_int(raw, "op_cost", lo=0, hi=12, label=f"{name} 操作费")
    if op_cost is None:
        op_cost = 1
        issues.append(f"{name} 操作费不可读，按 1 计")
    keywords_raw = raw.get("keywords")
    keywords: list[str] = []
    unknown: list[str] = []
    if isinstance(keywords_raw, list):
        for kw in keywords_raw:
            token = str(kw).strip().lower()
            if token in KNOWN_KEYWORDS:
                keywords.append(token)
            elif token and token != "none":
                unknown.append(str(kw).strip())
    heavy_armor = _require_int(raw, "heavy_armor", lo=0, hi=3, label=f"{name} 重甲")
    if heavy_armor is None:
        heavy_armor = 0
    if attack is None or hp is None:
        return None, f"{name}（攻击/血量不可读）", []

    unit = Unit(
        name=name,
        type=str(utype),
        attack=attack,
        hp=hp,
        line=line,
        slot=slot,
        op_cost=op_cost,
        keywords=tuple(keywords),
        heavy_armor=heavy_armor,
        can_act=bool(raw.get("can_act", True)),
    )
    low = name if conf == "blurry" else None
    return unit, low, unknown


def _prefill_from_db(raw: Any, db: Any) -> Any:
    """Fill unreadable base facts from the card database, in place.

    Runs before validation, so a unit whose type/operation cost the vision
    model could not read is rescued instead of rejected. Only base facts are
    filled; live state (hp, position, attack with buffs) stays untouched.
    """
    if not isinstance(raw, dict) or db is None:
        return raw
    card = db.lookup(raw.get("name"))
    if card is None:
        return raw

    db_type = db.unit_type(card)
    if db_type is not None and raw.get("type") not in UNIT_TYPES:
        raw["type"] = db_type

    db_op = db.op_cost(card)
    if db_op is not None and raw.get("op_cost") is None:
        raw["op_cost"] = db_op

    db_cost = db.deploy_cost(card)
    if db_cost is not None and raw.get("cost") is None:
        raw["cost"] = db_cost

    keywords, heavy_armor = db.keywords(card)
    if keywords:
        existing = raw.get("keywords")
        existing_list = existing if isinstance(existing, list) else []
        raw["keywords"] = list(keywords) + [k for k in existing_list if k not in keywords]
    if heavy_armor and not raw.get("heavy_armor"):
        raw["heavy_armor"] = heavy_armor
    return raw


def board_state_from_json(data: dict[str, Any], card_db: Any = None) -> BoardState:
    """Validate the transcribed JSON into a BoardState (or raise)."""
    issues: list[str] = []
    raw_issues = data.get("issues")
    if isinstance(raw_issues, list):
        issues.extend(str(i) for i in raw_issues if str(i).strip())

    if data.get("read_ok") is not True:
        reason = "; ".join(issues) or "画面不是 KARDS 对局或基础信息不可读"
        raise BoardReadError(f"读图失败：{reason}")

    my_kreds = _require_int(data, "my_kreds", lo=0, hi=24, label="我方 Kreds")
    enemy_hq = _require_int(data, "enemy_hq_hp", lo=1, hi=99, label="敌方总部血量")
    my_hq = _require_int(data, "my_hq_hp", lo=1, hi=99, label="我方总部血量")
    if my_kreds is None or enemy_hq is None or my_hq is None:
        missing = [
            label for label, value in (
                ("我方 Kreds", my_kreds), ("敌方总部血量", enemy_hq), ("我方总部血量", my_hq),
            ) if value is None
        ]
        raise BoardReadError(f"关键数值不可读：{', '.join(missing)}")

    enemy_hq_slot = _require_int(data, "enemy_hq_slot", lo=0, hi=4, label="敌方 HQ 槽位")
    if enemy_hq_slot is None:
        enemy_hq_slot = 4
        issues.append("敌方 HQ 槽位不可读，按最右(4)计")

    low_conf: list[str] = []
    unknown_kws: list[str] = []

    def _parse_units(key: str, side: str) -> list[Unit]:
        units: list[Unit] = []
        raw_list = data.get(key)
        if not isinstance(raw_list, list):
            raise BoardReadError(f"{side}单位列表缺失")
        for idx, raw in enumerate(raw_list):
            unit, reason, unknown = _unit_from(raw, side, idx, issues)
            if unit is None:
                raise BoardReadError(f"{side}单位读数不完整：{reason}")
            units.append(unit)
            if reason:
                low_conf.append(reason)
            if unknown:
                unknown_kws.extend(unknown)
        return units

    if card_db is not None:
        for key in ("my_units", "enemy_units", "hand"):
            raw_list = data.get(key)
            if isinstance(raw_list, list):
                for item in raw_list:
                    _prefill_from_db(item, card_db)

    my_units = _parse_units("my_units", "我方")
    enemy_units = _parse_units("enemy_units", "敌方")

    hand: list[DamageOrder] = []
    raw_hand = data.get("hand")
    if isinstance(raw_hand, list):
        for idx, raw in enumerate(raw_hand):
            if not isinstance(raw, dict):
                continue
            cost = _require_int(raw, "cost", lo=0, hi=24, label=f"手牌#{idx} 费用")
            damage = _require_int(raw, "damage", lo=1, hi=30, label=f"手牌#{idx} 伤害")
            if cost is None or damage is None:
                continue  # unreadable damage cards simply don't participate
            target = raw.get("target")
            if target not in ("hq", "unit", "any"):
                target = "any"
            hand.append(DamageOrder(
                name=str(raw.get("name") or f"指令{idx + 1}"),
                cost=cost,
                damage=damage,
                target=target,
            ))
            if raw.get("confidence") == "blurry":
                low_conf.append(f"手牌:{raw.get('name') or idx}")

    return BoardState(
        my_kreds=my_kreds,
        my_hq_hp=my_hq,
        enemy_hq_hp=enemy_hq,
        my_units=my_units,
        enemy_units=enemy_units,
        hand=hand,
        enemy_hq_slot=enemy_hq_slot,
        enemy_hq_smokescreen=bool(data.get("enemy_hq_smokescreen", False)),
        enemy_hq_immune=bool(data.get("enemy_hq_immune", False)),
        low_confidence_fields=tuple(low_conf),
        unknown_keywords=tuple(unknown_kws),
    )


def parse_board_text(text: str, card_db: Any = None) -> BoardState:
    return board_state_from_json(extract_json(text), card_db)


async def read_board(client: Any, data_url: str, *, max_tokens: int = 1400, card_db: Any = None) -> BoardState:
    """Send the screenshot to the vision slot and parse the JSON reply."""
    response = await client.chat.completions.create(
        model="lethal_vision",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_BRIEF},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        max_tokens=max_tokens,
    )
    choices = getattr(response, "choices", None)
    content = ""
    if choices:
        content = (getattr(choices[0].message, "content", None) or "").strip()
    return parse_board_text(content, card_db)
