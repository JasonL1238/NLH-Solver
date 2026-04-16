"""Adaptive / exploit layer for flop decisions (Phase E-style).

Pattern for future streets (do not skip baseline or equity):
  turn_baseline + turn_equity -> turn_adaptive
  river_baseline + river_equity -> river_adaptive

Public API:
  - FlopOpponentProfile, record_flop_hand, smoothing helpers
  - RangeAdjustmentHints, compute_range_adjustment_hints
  - apply_flop_exploit_adjustment
  - AdaptiveFlopDecision, recommend_adaptive_flop_action, compare helpers
"""

from __future__ import annotations

from .adaptive_ranges import RangeAdjustmentHints, compute_range_adjustment_hints
from .adaptive_recommender import (
    AdaptiveFlopDecision,
    compare_flop_baseline_vs_equity_vs_adaptive,
    recommend_adaptive_flop_action,
    simulate_flop_overfolder,
    simulate_flop_raise_heavy,
    simulate_flop_stab_heavy,
    simulate_flop_sticky_caller,
)
from .exploit_adjuster import apply_flop_exploit_adjustment
from .opponent_model import (
    FlopOpponentProfile,
    flop_archetypes,
    record_flop_hand,
    villain_flop_profile_summary,
)

__all__ = [
    "AdaptiveFlopDecision",
    "FlopOpponentProfile",
    "RangeAdjustmentHints",
    "apply_flop_exploit_adjustment",
    "compare_flop_baseline_vs_equity_vs_adaptive",
    "compute_range_adjustment_hints",
    "flop_archetypes",
    "recommend_adaptive_flop_action",
    "record_flop_hand",
    "simulate_flop_overfolder",
    "simulate_flop_raise_heavy",
    "simulate_flop_stab_heavy",
    "simulate_flop_sticky_caller",
    "villain_flop_profile_summary",
]
