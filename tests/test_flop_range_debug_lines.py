"""Range model debug strings for UI / traces."""

from __future__ import annotations

from poker_core.models import Player
from poker_core.parser import make_hand_config
from poker_core.transitions import post_blinds

from flop_equity.range_model import villain_flop_range_debug_lines


def test_debug_lines_after_blinds_only() -> None:
    cfg = make_hand_config("As Kd", hero_position="BTN_SB", effective_stack_bb=100.0, villain_cards="Qc Jd")
    hist, st = post_blinds(cfg)
    assert st.current_actor == Player.HERO
    lines = villain_flop_range_debug_lines(st)
    assert any("preflop line" in x.lower() for x in lines)
    assert any("blockers" in x.lower() for x in lines)
