"""Tests for ``play_lab.preflop_bridge``."""

from __future__ import annotations

from baseline_preflop.parser import make_preflop_state
from poker_core.models import ActionType, Player
from poker_core.parser import make_hand_config
from poker_core.transitions import apply_action, post_blinds

from play_lab.preflop_bridge import poker_actions_to_preflop_raw


def test_bridge_open_call_preflop_matches_make_preflop_state() -> None:
    cfg = make_hand_config("As Kd", "BTN_SB", 100.0, villain_cards="Qc Jd")
    hist, _ = post_blinds(cfg)
    s1 = apply_action(
        cfg,
        hist,
        action_type=ActionType.RAISE,
        player=Player.HERO,
        amount_to_bb=2.5,
    )
    s2 = apply_action(
        cfg,
        s1.action_history,
        action_type=ActionType.CALL,
        player=Player.VILLAIN,
        amount_to_bb=2.5,
    )
    raw = poker_actions_to_preflop_raw(s2.action_history)
    st = make_preflop_state(
        "As Kd",
        "BTN_SB",
        100.0,
        raw,
    )
    assert st.current_actor is None or st.betting_round_closed


def test_bridge_bb_hero_order() -> None:
    cfg = make_hand_config("As Kd", "BB", 100.0, villain_cards="Qc Jd")
    hist, _ = post_blinds(cfg)
    s1 = apply_action(
        cfg,
        hist,
        action_type=ActionType.RAISE,
        player=Player.VILLAIN,
        amount_to_bb=2.5,
    )
    s2 = apply_action(
        cfg,
        s1.action_history,
        action_type=ActionType.CALL,
        player=Player.HERO,
        amount_to_bb=2.5,
    )
    raw = poker_actions_to_preflop_raw(s2.action_history)
    make_preflop_state("As Kd", "BB", 100.0, raw)
