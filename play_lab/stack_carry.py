"""Carry chip stacks across completed hands (Play Lab; display + next-hand config)."""

from __future__ import annotations

from poker_core.models import HandConfig, HandState, Player

from play_lab.showdown_display import ShowdownWinner, hu_showdown_result


def stacks_after_completed_hand(cfg: HandConfig, state: HandState) -> tuple[float, float]:
    """Return ``(hero_bb, villain_bb)`` after this hand is fully decided.

    Assumes ``state.hand_over`` and valid contributions. No rake.
    """
    h0 = float(cfg.hero_starting_bb)
    v0 = float(cfg.villain_starting_bb)
    ch = float(state.hero_contribution_bb)
    cv = float(state.villain_contribution_bb)
    pot = ch + cv

    if state.fold_winner == Player.HERO:
        return h0 + cv, v0 - cv
    if state.fold_winner == Player.VILLAIN:
        return h0 - ch, v0 + ch

    if len(state.board_cards) == 5 and cfg.villain_hole_cards is not None:
        res = hu_showdown_result(cfg.hero_hole_cards, cfg.villain_hole_cards, list(state.board_cards))
        if res is None:
            return h0, v0
        if res.winner == ShowdownWinner.HERO:
            return h0 + cv, v0 - cv
        if res.winner == ShowdownWinner.VILLAIN:
            return h0 - ch, v0 + ch
        return h0 - ch + pot / 2.0, v0 - cv + pot / 2.0

    return h0, v0


def is_busted_for_next_hand(hero_bb: float, villain_bb: float, *, big_blind_bb: float) -> bool:
    """True if either player cannot cover a full big blind (cannot post / play next hand)."""
    bb = float(big_blind_bb)
    return hero_bb < bb - 1e-9 or villain_bb < bb - 1e-9
