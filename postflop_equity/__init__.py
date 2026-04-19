"""Postflop range carry-forward and equity integration helpers.

``integration`` is loaded lazily so ``from postflop_equity.range_carryforward import …``
does not pull in ``postflop_policy.ev_recommender`` during package init (avoids cycles).
"""

from __future__ import annotations

from typing import Any

from .range_carryforward import RANGE_NOTE_V1, build_villain_postflop_range

_LAZY = frozenset({
    "recommend_turn_action_with_equity",
    "recommend_river_action_with_equity",
    "recommend_postflop_action_with_equity",
})

__all__ = [
    "RANGE_NOTE_V1",
    "build_villain_postflop_range",
    "recommend_postflop_action_with_equity",
    "recommend_river_action_with_equity",
    "recommend_turn_action_with_equity",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from . import integration

        val = getattr(integration, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
