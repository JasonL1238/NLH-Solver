"""Deterministic metrics over weighted villain hole-card ranges."""

from __future__ import annotations

from collections import Counter
from typing import List, Tuple

from poker_core.models import Card, HoleCards

from flop_spot.classification import classify_hand
from flop_spot.models import MadeHandCategory


_NUT_MADE = frozenset(
    {MadeHandCategory.NUTS_OR_NEAR_NUTS, MadeHandCategory.SET},
)


def villain_range_nut_metrics(
    flop_board: List[Card],
    v_range: List[Tuple[HoleCards, float]],
) -> Tuple[float, str]:
    """Return (nut_weight / total_weight, short debug summary).

    ``nut`` counts villain combos whose flop made-hand is nuts/near-nuts or a set,
    using the same ``classify_hand`` path as hero (3-card flop only).
    """
    if len(flop_board) != 3:
        raise ValueError("villain_range_nut_metrics requires exactly 3 flop cards")
    total_w = 0.0
    nut_w = 0.0
    counter: Counter[str] = Counter()
    for hole, w in v_range:
        if w < 0 or w > 1e9:
            continue
        total_w += float(w)
        cls = classify_hand(hole, flop_board)
        cat = cls.made_hand.value
        counter[cat] += float(w)
        if cls.made_hand in _NUT_MADE:
            nut_w += float(w)
    if total_w < 1e-12:
        return 0.0, "empty_range"
    frac = nut_w / total_w
    top3 = counter.most_common(3)
    summary = ", ".join(f"{k}:{v:.3g}" for k, v in top3)
    return frac, summary
