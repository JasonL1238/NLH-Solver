"""Auto runout when both players are all-in (Play Lab)."""

from __future__ import annotations

import random

from poker_core.models import ActionType, Player
from poker_core.parser import make_hand_config, parse_card
from poker_core.transitions import apply_action, deal_flop, post_blinds
from poker_core.validation import validate_hand

from play_lab.runout import auto_runout_board_if_needed


def _flop():
    return parse_card("2h"), parse_card("3d"), parse_card("4c")


def test_auto_runout_flop_all_in_reaches_showdown() -> None:
    cfg = make_hand_config("As Kd", "BTN_SB", 30.0, villain_cards="Qc Jd")
    rng = random.Random(123)
    h, s = post_blinds(cfg)
    s = apply_action(cfg, h, action_type=ActionType.RAISE, player=Player.HERO, amount_to_bb=2.5)
    s = apply_action(cfg, s.action_history, action_type=ActionType.CALL, player=Player.VILLAIN, amount_to_bb=2.5)
    s = deal_flop(cfg, s.action_history, _flop())
    s = apply_action(cfg, s.action_history, action_type=ActionType.BET, player=Player.VILLAIN, amount_to_bb=27.5)
    s = apply_action(cfg, s.action_history, action_type=ActionType.CALL, player=Player.HERO, amount_to_bb=27.5)
    assert s.awaiting_runout is True
    hist2 = auto_runout_board_if_needed(cfg, s.action_history, rng)
    assert len(hist2) > len(s.action_history)
    s2 = validate_hand(cfg, hist2)
    assert not s2.awaiting_runout
    assert s2.hand_over
    assert s2.showdown_ready
    assert len(s2.board_cards) == 5
