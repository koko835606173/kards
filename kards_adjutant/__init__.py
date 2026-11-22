"""KARDS adjutant plugin: in-turn lethal calculation, three-stage pipeline.

Stage 1 (vision) transcribes the live screen into a strict-JSON board state.
Stage 2 (solver) exhaustively enumerates this turn's lines with pure code.
Stage 3 (persona) receives the computed verdict and repeats it in its own
voice — it never computes anything itself.
"""

from __future__ import annotations

import asyncio
import base64
import io
from typing import Any

from plugin.sdk.plugin import (
    Err,
    NekoPluginBase,
    Ok,
    SdkError,
    neko_plugin,
    plugin_entry,
    quick_action,
)

from . import board_read, lethal_solver
from .card_db import CardDbHolder
from .lethal_solver import Solution

_DEFAULT_MAX_EDGE = 1600
_DEFAULT_JPEG_QUALITY = 85
_DEFAULT_VISION_TOKENS = 1400
_DEFAULT_NODE_LIMIT = 200_000


def _capture_screen_jpeg(max_edge: int, quality: int) -> tuple[str | None, str]:
    """Grab the full virtual screen and return a JPEG data URL.

    Runs inside a worker thread. Returns ``(None, reason)`` on any failure so
    the entry can surface a readable error instead of a stack trace.
    """
    try:
        from PIL import ImageGrab
    except Exception as exc:  # pragma: no cover - depends on host deps
        return None, f"PIL 不可用: {exc}"
    try:
        image = ImageGrab.grab(all_screens=True).convert("RGB")
    except Exception as exc:
        return None, f"无法捕获屏幕: {exc}"
    try:
        width, height = image.size
        scale = max_edge / float(max(width, height))
        if scale < 1.0:
            image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}", ""
    except Exception as exc:
        return None, f"截图编码失败: {exc}"


def _format_steps(solution: Solution) -> list[str]:
    lines: list[str] = []
    for i, step in enumerate(solution.best_line, start=1):
        verb = {
            "attack": "攻击",
            "order": "使用指令",
            "move_attack": "开进前线后攻击",
        }.get(step.kind, "行动")
        note = f"（{step.note}）" if step.note else ""
        lines.append(f"{i}. {step.actor} {verb} {step.target} → {step.damage} 点{note}")
    return lines


def compose_verdict(solution: Solution, state: lethal_solver.BoardState) -> str:
    """Build the structured verdict the persona will repeat verbatim."""
    if solution.lethal:
        headline = f"最大伤害 {solution.max_hq_damage} ／ 斩杀：能（敌方总部 {solution.enemy_hq_hp} 血）"
    else:
        gap = solution.enemy_hq_hp - solution.max_hq_damage
        headline = f"最大伤害 {solution.max_hq_damage} ／ 斩杀：不能（还差 {gap}）"
    blocks = [f"结论：{headline}"]
    if solution.best_line:
        blocks.append("出牌路线：\n" + "\n".join(_format_steps(solution)))
    if solution.clears:
        blocks.append("先清目标：" + "、".join(solution.clears))
    readout = (
        f"读数：Kreds {state.my_kreds}｜敌 HQ {state.enemy_hq_hp} 血｜"
        f"我方 {len(state.my_units)} 单位（前线 {sum(1 for u in state.my_units if u.line == 'front')}）｜"
        f"敌方 {len(state.enemy_units)} 单位｜手牌直伤 {len(state.hand)} 张"
    )
    blocks.append(readout)
    blocks.extend(solution.caveats)
    return "\n".join(blocks)


_PERSONA_INSTRUCTION = (
    "【KARDS 斩杀计算结果｜插件已算好，你只负责转述】\n"
    "用你的语气把下面的结论说给主人听。规则：第一句必须是结论本身（最大伤害与能否斩杀），"
    "然后按顺序念出牌路线。禁止重新计算、禁止修改任何数字、禁止增删路线步骤。\n\n"
)


@neko_plugin
class KardsAdjutantPlugin(NekoPluginBase):
    """Capture -> transcribe -> solve -> persona-repeat pipeline."""

    def __init__(self, ctx):
        super().__init__(ctx)
        self._card_dbs = CardDbHolder()

    async def _settings(self) -> dict[str, int]:
        """Read the optional [kards] config section with safe defaults."""
        section: Any = None
        try:
            config = await self.ctx.get_own_effective_config()
            if isinstance(config, dict):
                section = config.get("kards")
        except Exception:
            section = None
        if not isinstance(section, dict):
            section = {}

        def _int(key: str, default: int) -> int:
            try:
                value = int(section.get(key, default))
            except (TypeError, ValueError):
                return default
            return value

        return {
            "max_edge": _int("screenshot_max_edge_px", _DEFAULT_MAX_EDGE),
            "quality": _int("screenshot_jpeg_quality", _DEFAULT_JPEG_QUALITY),
            "vision_tokens": _int("vision_max_tokens", _DEFAULT_VISION_TOKENS),
            "node_limit": _int("solver_node_limit", _DEFAULT_NODE_LIMIT),
        }

    @plugin_entry(
        id="lethal_check",
        name="KARDS 斩杀计算",
        description=(
            "主人正在玩 KARDS（二战卡牌游戏）时，凡问到「能不能斩杀／这回合最多打多少伤害／"
            "能不能打死他／帮我看下场面」一类问题就调用此工具。不要向主人索要截图——"
            "插件会自动截取当前屏幕：先让视觉模型把局面转录成数据，再用纯代码穷举本回合所有"
            "攻击顺序，结论会推送给角色去说。调用成功后不要自己复算或改数字，"
            "也不要复述完整路线（角色会说），最多一句简短确认。主人问的不是 KARDS 时不要调用。"
        ),
        timeout=150.0,
        llm_result_fields=["pushed", "headline"],
    )
    @quick_action(icon="⚔️", priority=3)
    async def lethal_check(self, **_):
        settings = await self._settings()
        data_url, reason = await asyncio.to_thread(
            _capture_screen_jpeg, settings["max_edge"], settings["quality"]
        )
        if not data_url:
            return Err(SdkError(f"KARDS 斩杀计算取消：{reason}"))

        try:
            client = await self.ctx.models.get_client()
        except Exception as exc:
            return Err(SdkError(f"视觉模型网关不可用：{exc}"))

        db, db_error = self._card_dbs.get(self.data_path())

        try:
            state = await board_read.read_board(
                client, data_url, max_tokens=settings["vision_tokens"], card_db=db
            )
        except board_read.BoardReadError as exc:
            return Err(SdkError(f"读图失败：{exc}"))
        except Exception as exc:
            return Err(SdkError(f"读图阶段模型调用失败：{exc}"))

        solution = lethal_solver.solve(state, node_limit=settings["node_limit"])
        verdict = compose_verdict(solution, state)
        if db is not None:
            verdict += f"\n牌库：已启用（{db.card_count} 张卡，按名称补全基础数值）"
        elif db_error:
            verdict += f"\n牌库：加载失败（{db_error}），本次仅凭画面读数"

        self.ctx.push_message(
            source=self.plugin_id,
            visibility=[],
            ai_behavior="respond",
            parts=[{"type": "text", "text": _PERSONA_INSTRUCTION + verdict}],
            metadata={"context_type": "kards_lethal_check"},
        )
        headline = verdict.splitlines()[0].removeprefix("结论：")[:200]
        return Ok({"pushed": True, "headline": headline})
