"""Tests for ``play_lab.coordinator`` hand helpers."""

from __future__ import annotations

import random

from poker_core.models import ActionType, Player, Street
from poker_core.parser import make_hand_config, parse_card
from poker_core.transitions import apply_action, deal_flop, post_blinds

from play_lab.coordinator import (
    HandCoordinator,
    ScenarioState,
    needs_flop_deal,
    needs_river_deal,
    needs_turn_deal,
)


def _flop_triple():
    return parse_card("Ah"), parse_card("7c"), parse_card("2d")


def test_needs_flop_deal_after_limp_line() -> None:
    cfg = make_hand_config("As Kd", "BTN_SB", 100.0, villain_cards="Qc Jd")
    hist, _ = post_blinds(cfg)
    s = apply_action(cfg, hist, action_type=ActionType.CALL, player=Player.HERO, amount_to_bb=1.0)
    s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK, player=Player.VILLAIN)
    assert needs_flop_deal(s)


def test_needs_turn_after_flop_check_check() -> None:
    cfg = make_hand_config("As Kd", "BTN_SB", 100.0, villain_cards="Qc Jd")
    hist, _ = post_blinds(cfg)
    s = apply_action(cfg, hist, action_type=ActionType.RAISE, player=Player.HERO, amount_to_bb=2.5)
    s = apply_action(cfg, s.action_history, action_type=ActionType.CALL, player=Player.VILLAIN, amount_to_bb=2.5)
    assert needs_flop_deal(s)
    s = deal_flop(cfg, s.action_history, _flop_triple())
    assert s.current_street == Street.FLOP
    s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK, player=Player.VILLAIN)
    s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK, player=Player.HERO)
    assert needs_turn_deal(s)
    assert not needs_river_deal(s)


def test_note_hand_completed() -> None:
    sc = ScenarioState(label="t", rng_seed=1, rng=random.Random(1), effective_stack_bb=100.0)
    assert sc.hands_completed == 0
    HandCoordinator.note_hand_completed(sc)
    assert sc.hands_completed == 1
