"""Unit tests for play_lab stack carry helpers (no Streamlit)."""

from __future__ import annotations

from poker_core.models import ActionType, Player
from poker_core.parser import make_hand_config
from poker_core.transitions import apply_action, post_blinds

from play_lab.stack_carry import is_busted_for_next_hand, stacks_after_completed_hand


def _cfg(*, hh: str, vh: str, pos: str = "BTN_SB", hs: float = 100.0, vs: float = 100.0):
    return make_hand_config(
        hh,
        hero_position=pos,
        effective_stack_bb=min(hs, vs),
        villain_cards=vh,
        hero_starting_bb=hs,
        villain_starting_bb=vs,
    )


def test_fold_hero_mucks_villain_takes_pot() -> None:
    cfg = _cfg(hh="As Kd", vh="2c 3d", pos="BTN_SB", hs=40.0, vs=100.0)
    hist, st0 = post_blinds(cfg)
    assert st0.current_actor == Player.HERO
    st = apply_action(cfg, hist, action_type=ActionType.FOLD, player=Player.HERO)
    assert st.hand_over and st.fold_winner == Player.VILLAIN
    nh, nv = stacks_after_completed_hand(cfg, st)
    ch = float(st.hero_contribution_bb)
    cv = float(st.villain_contribution_bb)
    assert nh == 40.0 - ch
    assert nv == 100.0 + ch


def test_busted_below_one_bb() -> None:
    assert is_busted_for_next_hand(0.5, 50.0, big_blind_bb=1.0) is True
    assert is_busted_for_next_hand(1.0, 1.0, big_blind_bb=1.0) is False
