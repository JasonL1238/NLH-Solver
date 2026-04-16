"""v1 villain range on later streets: reuse flop range + full-board dead filter."""

from __future__ import annotations

from typing import List, Tuple

from poker_core.models import HandState, HoleCards

from flop_equity.range_model import build_villain_flop_range

RANGE_NOTE_V1 = (
    "v1 postflop range: same preflop+flop inference as build_villain_flop_range; "
    "dead cards use full board; turn/river line conditioning not applied yet."
)


def build_villain_postflop_range(
    state: HandState,
) -> Tuple[List[Tuple[HoleCards, float]], str]:
    """Weighted villain combos for turn/river EV (v1 carry-forward).

    Internally delegates to :func:`flop_equity.range_model.build_villain_flop_range`,
    which already excludes ``hero`` hole cards and all ``state.board_cards``.
    """
    combos, summary = build_villain_flop_range(state)
    return combos, summary
