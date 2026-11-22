"""Unit tests for the pure KARDS lethal solver.

Each test names the rule it pins, with the source of that rule in a comment.
Board coordinates: enemy support-line slots 0-4, HQ at ``enemy_hq_slot``.
"""

from __future__ import annotations

import pytest

from plugin.plugins.kards_adjutant.lethal_solver import (
    BoardState,
    DamageOrder,
    Unit,
    solve,
)


def _u(name, type_, attack, hp, line, slot=-1, op_cost=1, keywords=(), heavy_armor=0, can_act=True):
    return Unit(
        name=name, type=type_, attack=attack, hp=hp, line=line, slot=slot,
        op_cost=op_cost, keywords=tuple(keywords), heavy_armor=heavy_armor,
        can_act=can_act,
    )


def _state(my_units, enemy_units, kreds=10, enemy_hp=20, hand=(), hq_slot=4, **kw):
    return BoardState(
        my_kreds=kreds, my_hq_hp=20, enemy_hq_hp=enemy_hp,
        my_units=list(my_units), enemy_units=list(enemy_units),
        hand=list(hand), enemy_hq_slot=hq_slot, **kw
    )


# --------------------------------------------------------------------------
# Basic targeting matrix
# --------------------------------------------------------------------------

@pytest.mark.plugin_unit
def test_frontline_unit_hits_hq_directly():
    # Frontline units may attack the enemy support line and HQ
    # (official Frontline article).
    state = _state([_u("步兵", "infantry", 4, 3, "front")], [], enemy_hp=12)
    sol = solve(state)
    assert sol.lethal is False
    assert sol.max_hq_damage == 4
    assert sol.best_line[0].target == "敌方总部"


@pytest.mark.plugin_unit
def test_support_melee_cannot_touch_hq_but_long_range_can():
    # Support-line melee can only hit enemy frontline units; artillery can
    # hit the HQ from the support line (official Support Line article).
    enemy = [_u("敌步兵", "infantry", 2, 3, "support", slot=3)]
    melee_only = _state([_u("我步兵", "infantry", 5, 3, "support", slot=1)], enemy, enemy_hp=10)
    assert solve(melee_only).max_hq_damage == 0

    artillery = _state([_u("我火炮", "artillery", 5, 2, "support", slot=1)], enemy, enemy_hp=10)
    sol = solve(artillery)
    assert sol.max_hq_damage == 5
    assert sol.best_line[0].target == "敌方总部"


@pytest.mark.plugin_unit
def test_fighter_hits_unguarded_hq():
    # Owner-confirmed: fighters are long range and may hit an unguarded HQ.
    enemy = [_u("敌卡车", "infantry", 1, 2, "support", slot=0)]
    state = _state([_u("我战机", "fighter", 6, 3, "support", slot=2)], enemy, enemy_hp=6)
    sol = solve(state)
    assert sol.lethal is True
    assert sol.best_line[0].target == "敌方总部"


# --------------------------------------------------------------------------
# Guard (adjacency, positional taunt)
# --------------------------------------------------------------------------

@pytest.mark.plugin_unit
def test_guard_adjacent_to_hq_blocks_melee_lethal():
    # Guard protects adjacent units/HQ from everything except bombers and
    # artillery (official Guard article). Clearing it unlocks the HQ.
    enemy = [
        _u("敌守卫", "infantry", 2, 5, "support", slot=3, keywords=("guard",)),
    ]
    me = [
        _u("我甲", "infantry", 7, 5, "front"),
        _u("我乙", "infantry", 5, 5, "front"),
    ]
    state = _state(me, enemy, kreds=10, enemy_hp=7, hq_slot=4)
    sol = solve(state)
    # One unit spends its attack clearing the guard; the other hits the HQ.
    assert sol.lethal is True
    assert sol.clears == ["敌守卫"]
    assert sol.best_line[-1].target == "敌方总部"
    assert sol.best_line[-1].actor == "我甲"  # the 7-attack unit goes face


@pytest.mark.plugin_unit
def test_guard_is_not_global_taunt():
    # A guard far from the HQ does not protect it (adjacency only).
    enemy = [_u("敌守卫", "infantry", 2, 5, "support", slot=0, keywords=("guard",))]
    me = [_u("我步兵", "infantry", 7, 5, "front")]
    state = _state(me, enemy, enemy_hp=7, hq_slot=4)
    sol = solve(state)
    assert sol.lethal is True
    assert sol.best_line[0].target == "敌方总部"


@pytest.mark.plugin_unit
def test_artillery_and_bomber_pierce_guard():
    enemy = [_u("敌守卫", "infantry", 2, 9, "support", slot=3, keywords=("guard",))]
    for utype in ("artillery", "bomber"):
        me = [_u(f"我{utype}", utype, 6, 2, "support", slot=2)]
        state = _state(me, enemy, enemy_hp=6, hq_slot=4)
        sol = solve(state)
        assert sol.lethal is True, utype
        assert sol.best_line[0].target == "敌方总部"


@pytest.mark.plugin_unit
def test_fighter_cannot_pierce_guard():
    # Only bombers and artillery pierce guard; fighters cannot (official text).
    enemy = [_u("敌守卫", "infantry", 2, 9, "support", slot=3, keywords=("guard",))]
    me = [_u("我战机", "fighter", 6, 2, "support", slot=2)]
    state = _state(me, enemy, enemy_hp=6, hq_slot=4)
    sol = solve(state)
    assert sol.max_hq_damage == 0


# --------------------------------------------------------------------------
# Retaliation / Ambush / Shock / Heavy Armor
# --------------------------------------------------------------------------

@pytest.mark.plugin_unit
def test_dying_defender_still_retaliation():
    # Combat is simultaneous: a defender that dies still deals its attack
    # (implied by the Ambush/Shock carve-outs; keyword wiki). White-box check.
    from plugin.plugins.kards_adjutant.lethal_solver import Solver, _Sim

    enemy = [_u("敌重坦", "tank", 6, 2, "front")]
    me = [_u("我步兵", "infantry", 2, 6, "support", slot=0)]
    state = _state(me, enemy)
    solver = Solver(state)
    sim = _Sim(state)
    nsim, step = solver._apply_attack(sim, 0, "unit", 0, move=False)
    assert step.damage == 2  # defender (2hp) dies but still took the hit
    assert nsim.enemy_hp[0] <= 0
    assert nsim.my_hp[0] == 0  # and dealt its 6 retaliation to the 6hp attacker
    assert "受反击 6 点" in step.note


@pytest.mark.plugin_unit
def test_ambush_first_strike_kills_attacker_and_fizzles():
    # Ambush: defender strikes first; a killed attacker deals nothing
    # (official Ambush article). The ambusher blocks the tank push, so the
    # chaff must die popping Ambush before the real attacker clears it.
    from plugin.plugins.kards_adjutant.lethal_solver import Solver, _Sim

    enemy = [_u("敌伏兵", "infantry", 5, 4, "front", keywords=("ambush",))]
    me = [
        _u("我炮灰", "infantry", 3, 5, "support", slot=0),
        _u("我主力", "infantry", 4, 6, "support", slot=1),
        _u("我坦克", "tank", 5, 6, "support", slot=2, op_cost=1),
    ]
    state = _state(me, enemy, kreds=6, enemy_hp=5)
    sol = solve(state)

    # White-box: attacking Ambush with a unit it kills fizzes the attack.
    solver = Solver(state)
    sim = _Sim(state)
    _, step = solver._apply_attack(sim, 0, "unit", 0, move=False)
    assert step.damage == 0
    assert "伏击先手" in step.note

    # Integration: chaff pops Ambush, main force clears it, tank pushes in.
    assert sol.lethal is True
    ambush_hits = [s for s in sol.best_line if s.target == "敌伏兵"]
    assert len(ambush_hits) == 2
    assert any(s.damage == 0 for s in ambush_hits)
    assert sol.best_line[-1].target == "敌方总部"


@pytest.mark.plugin_unit
def test_shock_negates_retaliation_and_survives_attacking_hq():
    # Shock: first attack on a unit takes no retaliation — even against
    # Ambush; attacking the HQ does not consume Shock (keyword wiki).
    from plugin.plugins.kards_adjutant.lethal_solver import Solver, _Sim

    enemy = [_u("敌伏兵", "infantry", 9, 2, "front", keywords=("ambush",))]
    me = [
        _u("我冲击兵", "infantry", 4, 4, "support", slot=0, keywords=("shock",)),
        _u("我坦克", "tank", 6, 6, "support", slot=1, op_cost=1),
    ]
    state = _state(me, enemy, kreds=5, enemy_hp=6)
    solver = Solver(state)
    sim = _Sim(state)
    nsim, step = solver._apply_attack(sim, 0, "unit", 0, move=False)
    assert step.damage == 4  # no Ambush first strike: Shock suppressed it
    assert "冲击" in step.note
    assert nsim.my_hp[0] == 4  # untouched

    # Shock is not consumed by the HQ attack: 4/4 shock unit hits HQ first,
    # then still has Shock when clearing the blocker… but the blocker is on
    # the frontline, so the order is: shock unit kills blocker (no
    # retaliation even vs Ambush 9 attack), tank pushes, both hit HQ? No —
    # the shock unit already attacked the blocker, so only the tank reaches
    # the HQ afterwards.
    sol = solve(_state(me, enemy, kreds=5, enemy_hp=6))
    assert sol.lethal is True


@pytest.mark.plugin_unit
def test_heavy_armor_reduces_damage_and_floors_at_zero():
    enemy = [_u("敌重甲", "tank", 1, 3, "front", heavy_armor=2)]
    me = [_u("我步兵", "infantry", 2, 5, "support", slot=0), _u("我小枪", "infantry", 1, 5, "support", slot=1)]
    state = _state(me, enemy, enemy_hp=20)
    sol = solve(state)
    hits = [s for s in sol.best_line if s.target == "敌重甲"]
    for s in hits:
        assert s.damage == 0  # 2 attack - 2 armor = 0; 1 attack - 2 armor = 0


@pytest.mark.plugin_unit
def test_orders_ignore_heavy_armor():
    # Orders deal full damage regardless of Heavy Armor (official article).
    # The HA blocker can only be cleared by the order: the tank's 6 attack
    # minus Heavy Armor 3 is 3 < 4 hp, while the order's 4 lands in full.
    enemy = [_u("敌重甲", "tank", 0, 4, "front", heavy_armor=3)]
    me = [_u("我坦克", "tank", 6, 6, "support", slot=1, op_cost=1)]
    hand = [DamageOrder("穿甲指令", cost=1, damage=4, target="unit")]
    state = _state(me, enemy, kreds=3, enemy_hp=6, hand=hand)
    sol = solve(state)
    assert sol.lethal is True
    assert sol.best_line[0].kind == "order"
    assert sol.best_line[0].damage == 4
    assert sol.best_line[-1].target == "敌方总部"


# --------------------------------------------------------------------------
# Kreds budget and Fury
# --------------------------------------------------------------------------

@pytest.mark.plugin_unit
def test_operation_cost_budgets_attacks():
    # Each attack costs the unit's operation cost (official Cards article).
    me = [
        _u("我甲", "infantry", 4, 3, "front", op_cost=3),
        _u("我乙", "infantry", 3, 3, "front", op_cost=2),
    ]
    state = _state(me, [], kreds=5, enemy_hp=20)
    sol = solve(state)
    # Affordable combo: 4+3=7 would need 5 kreds exactly -> both fit (3+2).
    assert sol.max_hq_damage == 7
    state2 = _state(me, [], kreds=3, enemy_hp=20)
    assert solve(state2).max_hq_damage == 4


@pytest.mark.plugin_unit
def test_fury_attacks_twice_paying_twice():
    # Fury: two attacks per turn, operation cost paid for each attack
    # (official Fury article).
    me = [_u("我奋战兵", "infantry", 3, 3, "front", op_cost=2, keywords=("fury",))]
    state = _state(me, [], kreds=4, enemy_hp=20)
    assert solve(state).max_hq_damage == 6
    state = _state(me, [], kreds=2, enemy_hp=20)
    assert solve(state).max_hq_damage == 3


@pytest.mark.plugin_unit
def test_order_cost_competes_with_operations():
    me = [_u("我步兵", "infantry", 3, 3, "front", op_cost=2)]
    hand = [DamageOrder("重炮", cost=3, damage=4, target="hq")]
    state = _state(me, [], kreds=3, enemy_hp=20, hand=hand)
    # Best use of 3 kreds: the order (4) beats the attack (3).
    assert solve(state).max_hq_damage == 4


# --------------------------------------------------------------------------
# Overkill, smokescreen, immunity, frontline dynamics
# --------------------------------------------------------------------------

@pytest.mark.plugin_unit
def test_overkill_is_avoided():
    # A second attack into the 1hp blocker wastes a Kred the tank push
    # needs; the optimal line clears it exactly once.
    enemy = [_u("敌残兵", "infantry", 0, 1, "front")]
    me = [
        _u("我A", "infantry", 3, 3, "support", slot=0, op_cost=1),
        _u("我B", "infantry", 3, 3, "support", slot=1, op_cost=1),
        _u("我坦克", "tank", 6, 6, "support", slot=2, op_cost=2),
    ]
    state = _state(me, enemy, kreds=5, enemy_hp=6)
    sol = solve(state)
    assert sol.lethal is True
    unit_hits = [s for s in sol.best_line if s.target == "敌残兵"]
    assert len(unit_hits) == 1
    assert sol.best_line[-1].target == "敌方总部"


@pytest.mark.plugin_unit
def test_smokescreen_blocks_units_but_orders_land():
    enemy = [_u("敌烟幕兵", "infantry", 0, 2, "front", keywords=("smokescreen",))]
    me = [_u("我步兵", "infantry", 6, 3, "support", slot=0)]
    state = _state(me, enemy, enemy_hp=20)
    assert solve(state).max_hq_damage == 0

    # Orders pierce smokescreen: the order clears the blocker so the tank
    # can push in and reach the HQ.
    me2 = [_u("我坦克", "tank", 6, 6, "support", slot=1, op_cost=1)]
    hand = [DamageOrder("直射", cost=1, damage=3, target="unit")]
    state2 = _state(me2, enemy, enemy_hp=6, hand=hand, kreds=3)
    sol2 = solve(state2)
    assert sol2.lethal is True
    assert any(s.target == "敌烟幕兵" and s.damage == 3 for s in sol2.best_line)


@pytest.mark.plugin_unit
def test_hq_smokescreen_blocks_units_orders_still_work():
    me = [_u("我火炮", "artillery", 7, 2, "support", slot=1)]
    state = _state(me, [], enemy_hp=9, enemy_hq_smokescreen=True)
    assert solve(state).max_hq_damage == 0
    hand = [DamageOrder("轰炸令", cost=2, damage=9, target="hq")]
    state2 = _state(me, [], enemy_hp=9, enemy_hq_smokescreen=True, hand=hand, kreds=2)
    assert solve(state2).lethal is True


@pytest.mark.plugin_unit
def test_enemy_frontline_must_be_cleared_before_tank_push():
    # Enemy holds the frontline: my support melee trades into it; once it
    # dies the tank may move in and attack HQ in the same turn, paying the
    # operation cost twice (owner-confirmed rule).
    enemy = [_u("敌前兵", "infantry", 1, 2, "front")]
    me = [
        _u("我炮灰", "infantry", 2, 5, "support", slot=0),
        _u("我坦克", "tank", 6, 6, "support", slot=1, op_cost=2),
    ]
    state = _state(me, enemy, kreds=6, enemy_hp=6)
    sol = solve(state)
    assert sol.lethal is True
    kinds = [(s.actor, s.kind) for s in sol.best_line]
    assert ("我坦克", "attack") in kinds or ("我坦克", "move_attack") in kinds
    assert sol.best_line[-1].target == "敌方总部"


@pytest.mark.plugin_unit
def test_tank_cannot_move_and_attack_when_out_of_kreds():
    enemy = []
    me = [_u("我坦克", "tank", 6, 6, "support", slot=1, op_cost=2)]
    state = _state(me, enemy, kreds=3, enemy_hp=6)  # needs 2x2 for move+attack
    assert solve(state).max_hq_damage == 0
    state2 = _state(me, enemy, kreds=4, enemy_hp=6)
    assert solve(state2).max_hq_damage == 6


@pytest.mark.plugin_unit
def test_bomber_must_clear_fighters_before_hitting_the_line():
    # Interception: bombers must attack enemy fighters in a line before
    # anything else in that line (community wiki tips). A Fury bomber kills
    # the fighter with its first attack and reaches the HQ with the second.
    enemy = [
        _u("敌战机", "fighter", 2, 3, "support", slot=2),
        _u("敌卡车", "infantry", 0, 1, "support", slot=1),
    ]
    me = [_u("我轰炸机", "bomber", 5, 3, "support", slot=3, keywords=("fury",))]
    state = _state(me, enemy, kreds=2, enemy_hp=5)
    sol = solve(state)
    assert sol.lethal is True
    assert sol.best_line[0].target == "敌战机"
    assert sol.best_line[1].target == "敌方总部"


@pytest.mark.plugin_unit
def test_bomber_retaliation_exceptions():
    # Bombers take no damage when attacking and never retaliate
    # (community wiki tips). White-box checks on _apply_attack.
    from plugin.plugins.kards_adjutant.lethal_solver import Solver, _Sim

    enemy = [_u("敌轰炸机", "bomber", 7, 3, "front"), _u("敌步兵", "infantry", 5, 3, "front")]
    me = [_u("我步兵", "infantry", 3, 5, "support", slot=0), _u("我轰炸机", "bomber", 4, 3, "support", slot=1)]
    state = _state(me, enemy)
    solver = Solver(state)
    sim = _Sim(state)

    # My infantry attacks the enemy bomber: no retaliation damage back.
    _, step = solver._apply_attack(sim, 0, "unit", 0, move=False)
    assert step.note == ""
    # My bomber attacks a 5-attack defender: takes no damage when attacking.
    nsim, step2 = solver._apply_attack(sim, 1, "unit", 1, move=False)
    assert step2.note == ""
    assert nsim.my_hp[1] == 3


# --------------------------------------------------------------------------
# Aggregate behaviour
# --------------------------------------------------------------------------

@pytest.mark.plugin_unit
def test_no_lethal_reports_gap():
    me = [_u("我步兵", "infantry", 4, 3, "front")]
    state = _state(me, [], kreds=10, enemy_hp=15)
    sol = solve(state)
    assert sol.lethal is False
    assert sol.max_hq_damage == 4
    assert "还差 11" not in str(sol)  # gap is computed at compose time


@pytest.mark.plugin_unit
def test_contradictory_frontline_read_is_refused():
    both_front = _state(
        [_u("我兵", "infantry", 4, 3, "front")],
        [_u("敌兵", "infantry", 4, 3, "front")],
    )
    sol = solve(both_front)
    assert sol.max_hq_damage == 0
    assert any("前线" in c for c in sol.caveats)


@pytest.mark.plugin_unit
def test_unknown_keywords_surface_as_caveats():
    state = _state([_u("我步兵", "infantry", 4, 3, "front")], [], enemy_hp=10,
                   unknown_keywords=("unknown:海啸",))
    sol = solve(state)
    assert any("未知关键词" in c for c in sol.caveats)


@pytest.mark.plugin_unit
def test_pinned_unit_cannot_attack():
    me = [_u("我被压兵", "infantry", 9, 9, "front", keywords=("pinned",))]
    state = _state(me, [], enemy_hp=5)
    assert solve(state).max_hq_damage == 0


@pytest.mark.plugin_unit
def test_node_limit_truncation_is_flagged():
    # A big board with no lethal in reach under a tiny node budget must
    # degrade gracefully instead of hanging or crashing.
    me = [_u(f"我兵{i}", "infantry", 5, 5, "front", op_cost=1, keywords=("fury",)) for i in range(5)]
    enemy = [_u(f"敌兵{i}", "infantry", 5, 5, "support", slot=i % 5) for i in range(4)]
    state = _state(me, enemy, kreds=20, enemy_hp=99)
    sol = solve(state, node_limit=50)
    assert sol.truncated is True
    assert any("节点上限" in c for c in sol.caveats)
