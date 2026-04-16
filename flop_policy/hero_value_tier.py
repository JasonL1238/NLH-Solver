"""Hero flop value tier for EV policy (check-raise / re-raise gating and EV stubs)."""

from __future__ import annotations

from typing import Literal

from poker_core.models import HandState

from flop_spot.classification import classify_hand
from flop_spot.models import MadeHandCategory

HeroValueTier = Literal["NUTS_NEAR", "STRONG", "OTHER"]

_NUTS_NEAR_MADE = frozenset(
    {MadeHandCategory.NUTS_OR_NEAR_NUTS, MadeHandCategory.SET},
)
_STRONG_MADE = frozenset(
    {
        MadeHandCategory.TWO_PAIR,
        MadeHandCategory.OVERPAIR,
        MadeHandCategory.TOP_PAIR_STRONG_KICKER,
    },
)


def hero_flop_value_tier(state: HandState) -> HeroValueTier:
    """Coarse tier from flop-spot hand class (first three board cards if more dealt).

    Matches ``spot_debug`` convention: on turn/river only the flop subset is used
    for this classifier until a street-aware made-hand bucket exists.
    """
    board = list(state.board_cards)
    if len(board) < 3:
        return "OTHER"
    flop_board = board[:3]
    hand = classify_hand(state.config.hero_hole_cards, flop_board)
    if hand.made_hand in _NUTS_NEAR_MADE:
        return "NUTS_NEAR"
    if hand.made_hand in _STRONG_MADE:
        return "STRONG"
    return "OTHER"


def hero_has_pressure_draw(state: HandState) -> bool:
    """Strong semibluff shape: combo draw or strong draw (flop subset board)."""
    board = list(state.board_cards)
    if len(board) < 3:
        return False
    flop_board = board[:3]
    hand = classify_hand(state.config.hero_hole_cards, flop_board)
    return bool(hand.has_combo_draw or hand.has_strong_draw)
