"""Append board runout cards when both players are all-in (``awaiting_runout``)."""

from __future__ import annotations

import random
from typing import List

from poker_core.models import Action, HandConfig
from poker_core.transitions import deal_river, deal_turn
from poker_core.validation import validate_hand

from play_lab.deck import blocked_for_runout, draw_street_card


def auto_runout_board_if_needed(
    config: HandConfig,
    action_history: List[Action],
    rng: random.Random,
) -> List[Action]:
    """Return history with ``DEAL_TURN`` / ``DEAL_RIVER`` while ``awaiting_runout`` is true."""
    hist = list(action_history)
    for _ in range(4):
        state = validate_hand(config, hist)
        if not state.awaiting_runout:
            break
        blocked = blocked_for_runout(
            config.hero_hole_cards,
            config.villain_hole_cards,
            list(state.board_cards),
        )
        card = draw_street_card(rng, blocked)
        n = len(state.board_cards)
        if n == 3:
            ns = deal_turn(config, hist, card)
        elif n == 4:
            ns = deal_river(config, hist, card)
        else:
            break
        hist = ns.action_history
    return hist
