"""Exhaustive this-turn lethal solver for KARDS.

Pure calculation over a transcribed board state: no I/O, no randomness, no
model calls. The vision layer only transcribes the board into
:class:`BoardState`; every rule decision happens here so it can be unit
tested.

Rule sources (checked 2026-11):
- Frontline occupancy/targets: official "Battlefield elements: The Frontline"
  https://support.kards.com/hc/en-us/articles/360026464712
- Support line capacity / long-range attacks:
  https://support.kards.com/hc/en-us/articles/360026757171
- Operation costs (attack/move cost Kreds): official "Cards"
  https://support.kards.com/hc/en-us/articles/360026768151
- Keywords: official Guard / Fury / Ambush / Blitz / Heavy Armor / Smokescreen
  articles on support.kards.com plus the community wiki keyword page
  (kards.fandom.com/wiki/Keywords). Bomber/artillery retaliation exceptions:
  kards.fandom.com/wiki/Tips_for_KARDS.
- Fighter long-range and fighter can hit an unguarded HQ: owner-confirmed.
- Tank move+attack pays the operation cost twice: owner-confirmed.

Encoded combat matrix (attacker -> legal targets):
- my front unit (I control the frontline): enemy support-line units + enemy HQ
- my support melee (infantry/tank): enemy frontline units only
- my support long-range (artillery/fighter/bomber): enemy frontline units,
  enemy support-line units, and the enemy HQ
- Guard blocks attacks on adjacent support-line units / the HQ unless the
  attacker is artillery or bomber; fighters do not pierce guard
- bombers must clear enemy fighters in a line before attacking it
- tanks sitting in support may move into an unoccupied frontline and attack
  in the same turn, paying the operation cost twice
"""

from __future__ import annotations

from dataclasses import dataclass, field

LONG_RANGE_TYPES = frozenset({"artillery", "fighter", "bomber"})
GUARD_PIERCING_TYPES = frozenset({"artillery", "bomber"})
MELEE_TYPES = frozenset({"infantry", "tank"})

# Keywords the solver understands. Anything else is surfaced as a caveat
# instead of silently changing the maths.
COMBAT_KEYWORDS = frozenset(
    {"ambush", "blitz", "fury", "guard", "shock", "smokescreen", "immune", "pinned"}
)

DEFAULT_NODE_LIMIT = 200_000


@dataclass(frozen=True)
class Unit:
    """One unit exactly as transcribed from the screen."""

    name: str
    type: str  # infantry|tank|artillery|fighter|bomber
    attack: int
    hp: int
    line: str  # front|support
    slot: int = -1  # support-line position (0-4); -1 on the frontline
    op_cost: int = 1
    keywords: tuple[str, ...] = ()
    heavy_armor: int = 0
    can_act: bool = True

    def has(self, keyword: str) -> bool:
        return keyword in self.keywords


@dataclass(frozen=True)
class DamageOrder:
    """A hand order with readable direct damage."""

    name: str
    cost: int
    damage: int
    target: str = "any"  # hq|unit|any


@dataclass
class BoardState:
    my_kreds: int
    my_hq_hp: int
    enemy_hq_hp: int
    my_units: list[Unit] = field(default_factory=list)
    enemy_units: list[Unit] = field(default_factory=list)
    hand: list[DamageOrder] = field(default_factory=list)
    enemy_hq_slot: int = 4
    enemy_hq_smokescreen: bool = False
    enemy_hq_immune: bool = False
    low_confidence_fields: tuple[str, ...] = ()
    unknown_keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class Step:
    kind: str  # attack|order|move_attack
    actor: str
    target: str  # unit name or "敌方总部"
    damage: int
    note: str = ""


@dataclass
class Solution:
    lethal: bool
    max_hq_damage: int
    enemy_hq_hp: int
    best_line: list[Step] = field(default_factory=list)
    clears: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    truncated: bool = False


class SolverAborted(Exception):
    """Raised when the node budget is exhausted; partial results are kept."""


def _damage_taken(defender: Unit, raw: int) -> int:
    """Heavy Armor reduces damage taken per attack, floored at zero."""
    return max(0, raw - defender.heavy_armor)


class _Sim:
    """Hashable DFS snapshot of the simulation."""

    __slots__ = (
        "enemy_hp", "my_hp", "attacks_left", "ambush_used", "shock_used",
        "kreds", "orders_used", "hq_damage", "moves_done",
    )

    def __init__(self, state: BoardState) -> None:
        self.enemy_hp = tuple(u.hp for u in state.enemy_units)
        self.my_hp = tuple(u.hp for u in state.my_units)
        attacks = []
        for unit in state.my_units:
            if not unit.can_act or unit.has("pinned"):
                attacks.append(0)
            elif unit.has("fury"):
                attacks.append(2)  # Fury: two attacks, each paid separately.
            else:
                attacks.append(1)
        self.attacks_left = tuple(attacks)
        self.ambush_used: frozenset[int] = frozenset()
        self.shock_used: frozenset[int] = frozenset()
        self.kreds = state.my_kreds
        self.orders_used: frozenset[int] = frozenset()
        self.hq_damage = 0
        self.moves_done: frozenset[int] = frozenset()

    def clone(self) -> "_Sim":
        new = _Sim.__new__(_Sim)
        for name in _Sim.__slots__:
            setattr(new, name, getattr(self, name))
        return new

    def key(self) -> tuple:
        return (
            self.enemy_hp, self.my_hp, self.attacks_left, tuple(sorted(self.ambush_used)),
            tuple(sorted(self.shock_used)), self.kreds, tuple(sorted(self.orders_used)),
            self.hq_damage, tuple(sorted(self.moves_done)),
        )


class Solver:
    """Depth-first exhaustive search maximising capped HQ damage."""

    def __init__(self, state: BoardState, node_limit: int = DEFAULT_NODE_LIMIT) -> None:
        self.state = state
        self.my = list(state.my_units)
        self.enemy = list(state.enemy_units)
        self.node_limit = node_limit
        self.nodes = 0
        self.truncated = False
        self._memo: dict[tuple, tuple[int, tuple[Step, ...]]] = {}
        self._best = 0
        self._target = state.enemy_hq_hp

    # -- targeting ---------------------------------------------------------

    def _alive(self, sim: _Sim, i: int) -> bool:
        return sim.enemy_hp[i] > 0

    def _enemy_front(self, sim: _Sim) -> list[int]:
        return [
            i for i, u in enumerate(self.enemy)
            if u.line == "front" and self._alive(sim, i)
        ]

    def _guards(self, sim: _Sim) -> list[int]:
        return [
            i for i, u in enumerate(self.enemy)
            if u.has("guard") and u.line == "support" and self._alive(sim, i)
        ]

    def _guarded(self, sim: _Sim, idx: int) -> bool:
        target = self.enemy[idx]
        if target.slot < 0:
            return False
        for gi in self._guards(sim):
            guard = self.enemy[gi]
            if guard.slot >= 0 and abs(guard.slot - target.slot) == 1:
                return True
        return False

    def _hq_guarded(self, sim: _Sim) -> bool:
        for gi in self._guards(sim):
            guard = self.enemy[gi]
            if guard.slot >= 0 and abs(guard.slot - self.state.enemy_hq_slot) == 1:
                return True
        return False

    def _fighters_in(self, sim: _Sim, line: str) -> list[int]:
        return [
            i for i, u in enumerate(self.enemy)
            if u.type == "fighter" and u.line == line and self._alive(sim, i)
        ]

    def _effective_line(self, ai: int, sim: _Sim) -> str:
        if ai in sim.moves_done:
            return "front"
        return self.my[ai].line

    def _legal_targets(self, sim: _Sim, ai: int) -> list[tuple[str, int]]:
        """Legal targets for my unit ai as ("hq", -1) / ("unit", idx)."""
        attacker = self.my[ai]
        line = self._effective_line(ai, sim)
        targets: list[tuple[str, int]] = []

        if line == "front":
            reach = "support_only"
        elif attacker.type in LONG_RANGE_TYPES:
            reach = "any"
        else:
            reach = "front_only"

        for i, defender in enumerate(self.enemy):
            if not self._alive(sim, i):
                continue
            if defender.has("immune") or defender.has("smokescreen"):
                continue  # cannot be attacked by units (orders still can)
            if defender.line == "front":
                if reach in ("any", "front_only"):
                    if attacker.type == "bomber":
                        fighters = self._fighters_in(sim, "front")
                        if fighters and i not in fighters:
                            continue
                    targets.append(("unit", i))
            else:
                if reach in ("any", "support_only"):
                    if attacker.type == "bomber":
                        fighters = self._fighters_in(sim, "support")
                        if fighters and i not in fighters:
                            continue
                    if self._guarded(sim, i) and attacker.type not in GUARD_PIERCING_TYPES:
                        continue
                    targets.append(("unit", i))

        hq_ok = reach in ("any", "support_only")
        if hq_ok and not self.state.enemy_hq_smokescreen and not self.state.enemy_hq_immune:
            if attacker.type == "bomber" and self._fighters_in(sim, "support"):
                hq_ok = False  # interception: clear the line's fighters first
            if hq_ok and self._hq_guarded(sim) and attacker.type not in GUARD_PIERCING_TYPES:
                hq_ok = False
            if hq_ok:
                targets.append(("hq", -1))
        return targets

    # -- action application --------------------------------------------------

    def _apply_attack(self, sim: _Sim, ai: int, tkind: str, tidx: int, *, move: bool) -> tuple[_Sim, Step]:
        attacker = self.my[ai]
        nsim = sim.clone()
        op = attacker.op_cost * (2 if move else 1)
        nsim.kreds = sim.kreds - op
        left = list(nsim.attacks_left)
        left[ai] -= 1
        nsim.attacks_left = tuple(left)
        if move:
            nsim.moves_done = sim.moves_done | {ai}

        if tkind == "hq":
            nsim.hq_damage = min(self._target, sim.hq_damage + attacker.attack)
            # Attacking the HQ does not consume Shock (official keyword text).
            step = Step("move_attack" if move else "attack", attacker.name, "敌方总部",
                        min(attacker.attack, self._target - sim.hq_damage) if sim.hq_damage < self._target else 0)
            return nsim, step

        defender = self.enemy[tidx]
        dmg = _damage_taken(defender, attacker.attack)
        eh = list(nsim.enemy_hp)
        mh = list(nsim.my_hp)
        note = ""

        attacker_takes_nothing = attacker.type in ("artillery", "bomber")
        shock_now = attacker.has("shock") and ai not in sim.shock_used

        if shock_now:
            # Shock negates retaliation — even against Ambush (official text).
            nsim.shock_used = sim.shock_used | {ai}
            eh[tidx] -= dmg
            note = "冲击：不受反击"
        elif defender.has("ambush") and tidx not in sim.ambush_used:
            nsim.ambush_used = sim.ambush_used | {tidx}
            amb = _damage_taken(attacker, defender.attack)
            mh[ai] -= amb
            if mh[ai] <= 0:
                dmg = 0
                note = f"伏击先手 {amb} 点：攻击者阵亡，本次攻击落空"
            else:
                eh[tidx] -= dmg
                note = f"伏击先手 {amb} 点反击"
        else:
            eh[tidx] -= dmg
            if not attacker_takes_nothing:
                retaliation = _damage_taken(attacker, defender.attack)
                if defender.type == "bomber":
                    retaliation = 0  # bombers never retaliate
                if retaliation:
                    mh[ai] -= retaliation
                    note = f"受反击 {retaliation} 点"

        nsim.enemy_hp = tuple(eh)
        nsim.my_hp = tuple(mh)
        step = Step("move_attack" if move else "attack", attacker.name, defender.name, dmg, note)
        return nsim, step

    def _apply_order(self, sim: _Sim, oi: int, tkind: str, tidx: int) -> tuple[_Sim, Step]:
        order = self.state.hand[oi]
        nsim = sim.clone()
        nsim.kreds = sim.kreds - order.cost
        nsim.orders_used = sim.orders_used | {oi}
        if tkind == "hq":
            gain = min(order.damage, self._target - sim.hq_damage) if sim.hq_damage < self._target else 0
            nsim.hq_damage = min(self._target, sim.hq_damage + order.damage)
            step = Step("order", order.name, "敌方总部", gain)
        else:
            eh = list(nsim.enemy_hp)
            eh[tidx] -= order.damage  # orders ignore Heavy Armor (official)
            nsim.enemy_hp = tuple(eh)
            step = Step("order", order.name, self.enemy[tidx].name, order.damage)
        return nsim, step

    # -- search ----------------------------------------------------------------

    def _actions(self, sim: _Sim) -> list[tuple[str, int, str, int, bool]]:
        """Applicable actions as (kind, actor, target_kind, target_idx, move)."""
        actions: list[tuple[str, int, str, int, bool]] = []
        enemy_front = self._enemy_front(sim)
        my_front_count = sum(
            1 for i, u in enumerate(self.my)
            if self._effective_line(i, sim) == "front" and sim.my_hp[i] > 0
        )
        for ai, unit in enumerate(self.my):
            if sim.my_hp[ai] <= 0 or sim.attacks_left[ai] <= 0:
                continue
            if sim.kreds < unit.op_cost:
                continue
            for tk, ti in self._legal_targets(sim, ai):
                actions.append(("attack", ai, tk, ti, False))
            # Tank push: support -> (empty) frontline -> attack, two op costs.
            if (
                unit.type == "tank"
                and unit.line == "support"
                and ai not in sim.moves_done
                and not enemy_front
                and sim.kreds >= 2 * unit.op_cost
                and my_front_count < 5
            ):
                # _effective_line reports "front" for moved units, so compute
                # targets with the move pre-applied.
                moved = sim.clone()
                moved.moves_done = sim.moves_done | {ai}
                for mtk, mti in self._legal_targets(moved, ai):
                    actions.append(("attack", ai, mtk, mti, True))
        for oi, order in enumerate(self.state.hand):
            if oi in sim.orders_used or sim.kreds < order.cost:
                continue
            if order.target in ("any", "hq"):
                actions.append(("order", oi, "hq", -1, False))
            if order.target in ("any", "unit"):
                for i, defender in enumerate(self.enemy):
                    if self._alive(sim, i) and not defender.has("immune"):
                        actions.append(("order", oi, "unit", i, False))
        return actions

    def _potential(self, sim: _Sim) -> int:
        """Cheap upper bound on future HQ damage, for pruning."""
        bound = 0
        for ai, unit in enumerate(self.my):
            if sim.my_hp[ai] <= 0:
                continue
            affordable = sim.kreds // unit.op_cost if unit.op_cost > 0 else sim.attacks_left[ai]
            bound += min(sim.attacks_left[ai], max(0, affordable)) * unit.attack
        for oi, order in enumerate(self.state.hand):
            if oi not in sim.orders_used and sim.kreds >= order.cost:
                bound += order.damage
        return bound

    def _dfs(self, sim: _Sim) -> tuple[int, tuple[Step, ...]]:
        key = sim.key()
        cached = self._memo.get(key)
        if cached is not None:
            return cached
        self.nodes += 1
        if self.nodes > self.node_limit:
            raise SolverAborted()

        best_value = sim.hq_damage
        best_chain: tuple[Step, ...] = ()
        if sim.hq_damage + self._potential(sim) <= self._best:
            # Cannot beat the incumbent even in the best case.
            self._memo[key] = (best_value, best_chain)
            return best_value, best_chain

        for kind, actor, tkind, tidx, move in self._actions(sim):
            if kind == "order":
                nsim, step = self._apply_order(sim, actor, tkind, tidx)
            else:
                nsim, step = self._apply_attack(sim, actor, tkind, tidx, move=move)
            if nsim.hq_damage < self._best and nsim.hq_damage + self._potential(nsim) <= self._best:
                continue
            value, chain = self._dfs(nsim)
            if value > best_value or (value == best_value and not best_chain and chain):
                best_value = value
                best_chain = (step, *chain)
                self._best = max(self._best, best_value)
            if self._best >= self._target:
                break  # lethal found; stop the whole search
        self._memo[key] = (best_value, best_chain)
        return best_value, best_chain

    # -- entry -----------------------------------------------------------------

    def solve(self) -> Solution:
        if any(u.line == "front" for u in self.my) and any(u.line == "front" for u in self.enemy):
            return Solution(
                lethal=False, max_hq_damage=0, enemy_hq_hp=self.state.enemy_hq_hp,
                caveats=["读图矛盾：双方同时占据前线（规则上不可能），请重截一张画面"],
            )
        if self.state.enemy_hq_immune:
            return Solution(
                lethal=False, max_hq_damage=0, enemy_hq_hp=self.state.enemy_hq_hp,
                caveats=["敌方总部处于免疫状态，本回合无法造成伤害"],
            )
        try:
            sim = _Sim(self.state)
            _, chain = self._dfs(sim)
            best_line = list(chain)
        except SolverAborted:
            self.truncated = True
            best_line = [
                Step(*s) for s in self._reconstruct_best()
            ]

        total = sum(s.damage for s in best_line if s.target == "敌方总部")
        clears: list[str] = []
        for s in best_line:
            if s.target != "敌方总部" and s.target not in clears:
                clears.append(s.target)

        caveats: list[str] = []
        if self.state.low_confidence_fields:
            caveats.append(
                f"含 {len(self.state.low_confidence_fields)} 个低把握读数"
                f"（{', '.join(self.state.low_confidence_fields[:5])}）"
            )
        if self.state.unknown_keywords:
            unique = sorted(set(self.state.unknown_keywords))[:5]
            caveats.append(f"未知关键词按无效果计：{', '.join(unique)}")
        caveats.append("未计入敌方反制指令（不可见）")
        if self.truncated:
            caveats.append("搜索超出节点上限，展示的是已找到的最优解")

        return Solution(
            lethal=total >= self.state.enemy_hq_hp,
            max_hq_damage=total,
            enemy_hq_hp=self.state.enemy_hq_hp,
            best_line=best_line,
            clears=clears,
            caveats=caveats,
            truncated=self.truncated,
        )

    def _reconstruct_best(self) -> list[tuple]:
        """Best chain seen before an abort; walks memo from the root state."""
        chain: list[tuple] = []
        sim = _Sim(self.state)
        for _ in range(200):
            cached = self._memo.get(sim.key())
            if not cached or not cached[1]:
                break
            step = cached[1][0]
            # Re-derive the action that produced this step to advance the sim.
            advanced = False
            for kind, actor, tkind, tidx, move in self._actions(sim):
                if kind == "order":
                    nsim, s = self._apply_order(sim, actor, tkind, tidx)
                else:
                    nsim, s = self._apply_attack(sim, actor, tkind, tidx, move=move)
                if s == step:
                    chain.append((s.kind, s.actor, s.target, s.damage, s.note))
                    sim = nsim
                    advanced = True
                    break
            if not advanced:
                break
        return chain


def solve(state: BoardState, node_limit: int = DEFAULT_NODE_LIMIT) -> Solution:
    """Compute the best this-turn line against the enemy HQ."""
    return Solver(state, node_limit).solve()
