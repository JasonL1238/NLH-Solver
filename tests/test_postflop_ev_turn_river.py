"""Turn/river EV policy: legality, determinism, and debug key parity."""

from __future__ import annotations

import pytest

from poker_core.models import ActionType, Player, Street
from poker_core.parser import make_hand_config, parse_card
from poker_core.transitions import apply_action, deal_flop, deal_river, deal_turn, post_blinds

from postflop_equity.integration import (
    recommend_postflop_action_with_equity,
    recommend_river_action_with_equity,
    recommend_turn_action_with_equity,
)


def _flop_board():
    return (parse_card("Ah"), parse_card("7c"), parse_card("2d"))


def _turn_state_hero_to_act():
    """BTN hero, check-check flop, villain checks turn → hero to act."""
    cfg = make_hand_config("As Kd", hero_position="BTN_SB", effective_stack_bb=100.0)
    h, _ = post_blinds(cfg)
    s = apply_action(cfg, h, action_type=ActionType.RAISE, player=Player.HERO, amount_to_bb=2.5)
    s = apply_action(cfg, s.action_history, action_type=ActionType.CALL, player=Player.VILLAIN, amount_to_bb=2.5)
    s = deal_flop(cfg, s.action_history, _flop_board())
    s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK, player=Player.VILLAIN)
    s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK, player=Player.HERO)
    s = deal_turn(cfg, s.action_history, parse_card("9s"))
    s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK, player=Player.VILLAIN)
    assert s.current_actor == Player.HERO
    assert s.current_street == Street.TURN
    return cfg, s


def _river_state_hero_to_act():
    """Check through to river; villain checks river → hero to act."""
    cfg, s = _turn_state_hero_to_act()
    s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK, player=Player.HERO)
    s = deal_river(cfg, s.action_history, parse_card("4h"))
    s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK, player=Player.VILLAIN)
    assert s.current_actor == Player.HERO
    assert s.current_street == Street.RIVER
    return cfg, s


def _assert_recommendation_legal(dec) -> None:
    rec = dec.recommended_action
    la = rec.legal_action
    matches = [a for a in dec.legal_actions if a.action_type == la.action_type]
    assert matches, f"No legal row for {la.action_type}"
    legal = matches[0]
    if la.action_type in (ActionType.BET, ActionType.RAISE):
        assert rec.size_bb is not None
        assert legal.min_to_bb is not None and legal.max_to_bb is not None
        assert float(legal.min_to_bb) <= rec.size_bb <= float(legal.max_to_bb)


def test_turn_ev_recommendation_is_legal() -> None:
    _, s = _turn_state_hero_to_act()
    dec = recommend_turn_action_with_equity(s, samples=400, seed=11)
    _assert_recommendation_legal(dec)


def test_river_ev_recommendation_is_legal() -> None:
    _, s = _river_state_hero_to_act()
    dec = recommend_river_action_with_equity(s, samples=1, seed=0)
    _assert_recommendation_legal(dec)


def test_turn_ev_deterministic_with_seed() -> None:
    _, s = _turn_state_hero_to_act()
    a = recommend_turn_action_with_equity(s, samples=600, seed=99)
    b = recommend_turn_action_with_equity(s, samples=600, seed=99)
    assert repr(a.recommended_action) == repr(b.recommended_action)
    assert a.debug["equity_estimate"] == b.debug["equity_estimate"]


def test_postflop_debug_keys_turn() -> None:
    _, s = _turn_state_hero_to_act()
    dec = recommend_turn_action_with_equity(s, samples=200, seed=3)
    dbg = dec.debug
    assert dbg["current_street"] == "TURN"
    assert dbg["board_cards_count"] == 4
    assert dbg["classification_uses_flop_subset"] is True
    assert "range_note" in dbg
    assert "equity_estimate" in dbg
    assert "baseline_action" in dbg and "final_action" in dbg
    assert dbg["baseline_action"] == dbg["final_action"]


def test_postflop_debug_keys_river() -> None:
    _, s = _river_state_hero_to_act()
    dec = recommend_river_action_with_equity(s, seed=0)
    dbg = dec.debug
    assert dbg["current_street"] == "RIVER"
    assert dbg["board_cards_count"] == 5
    assert dbg["classification_uses_flop_subset"] is True


def test_recommend_postflop_action_with_equity_flop_delegates() -> None:
    """FLOP street dispatches to existing flop EV path."""
    cfg = make_hand_config("As Kd", hero_position="BTN_SB", effective_stack_bb=100.0)
    h, _ = post_blinds(cfg)
    s = apply_action(cfg, h, action_type=ActionType.RAISE, player=Player.HERO, amount_to_bb=2.5)
    s = apply_action(cfg, s.action_history, action_type=ActionType.CALL, player=Player.VILLAIN, amount_to_bb=2.5)
    s = deal_flop(cfg, s.action_history, _flop_board())
    s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK, player=Player.VILLAIN)
    assert s.current_street == Street.FLOP
    dec = recommend_postflop_action_with_equity(s, street=Street.FLOP, samples=150, seed=5)
    assert dec.debug.get("ev_policy") == "flop_policy.ev_recommender"


def test_wrong_street_raises() -> None:
    _, s = _turn_state_hero_to_act()
    with pytest.raises(ValueError, match="current_street"):
        recommend_postflop_action_with_equity(s, street=Street.RIVER, samples=10, seed=1)
