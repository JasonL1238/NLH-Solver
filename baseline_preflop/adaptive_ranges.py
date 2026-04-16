"""Adaptive villain range assumptions (Phase B).

This does not solve poker. It converts opponent profile evidence into simple,
explicit range/tendency assumptions to drive exploit rules and debugging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .models import PokerState, Position
from .opponent_model import OpponentPreflopProfile, PRIORS


@dataclass(frozen=True)
class AdaptiveRangeAssumptions:
    villain_profile_summary: str
    # Key assumed rates
    bb_fold_to_steal_assumed: float
    bb_3bet_vs_open_assumed: float
    btn_open_rate_assumed: float
    btn_limp_rate_assumed: float
    btn_fold_to_3bet_assumed: float
    bb_raise_vs_limp_assumed: float
    # Sizing distributions (bucket freqs)
    btn_open_size_bucket_dist: Dict[str, float] = field(default_factory=dict)
    bb_3bet_size_bucket_dist: Dict[str, float] = field(default_factory=dict)
    bb_iso_size_bucket_dist: Dict[str, float] = field(default_factory=dict)
    # Notes for explainability
    notes: List[str] = field(default_factory=list)


def adjusted_villain_assumptions(profile: OpponentPreflopProfile, state: PokerState) -> AdaptiveRangeAssumptions:
    """Return adjusted villain assumptions for the current preflop spot."""
    archetypes = profile.archetypes()
    summary = ",".join(archetypes) if archetypes else "UNKNOWN"

    bb_fold = profile.stat_bb_fold_to_steal()
    bb_3b = profile.stat_bb_3bet_vs_open()
    btn_open = profile.stat_btn_open_rate()
    btn_limp = profile.stat_btn_limp_rate()
    btn_fold3 = profile.stat_btn_fold_to_3bet()
    bb_iso = profile.stat_bb_raise_vs_limp()

    notes: List[str] = []

    # High-signal notes
    if bb_fold.confidence >= 0.4:
        if bb_fold.smoothed_rate > PRIORS["bb_fold_to_steal_rate"] + 0.10:
            notes.append("BB folds to steals more than baseline.")
        if bb_fold.smoothed_rate < PRIORS["bb_fold_to_steal_rate"] - 0.10:
            notes.append("BB defends steals more than baseline (stickier).")

    if bb_3b.confidence >= 0.4:
        if bb_3b.smoothed_rate > PRIORS["bb_3bet_vs_open_rate"] + 0.06:
            notes.append("BB 3bets more than baseline (aggressive).")
        if bb_3b.smoothed_rate < PRIORS["bb_3bet_vs_open_rate"] - 0.05:
            notes.append("BB 3bets less than baseline (passive).")

    if btn_limp.confidence >= 0.4 and btn_limp.smoothed_rate > PRIORS["btn_limp_rate"] + 0.10:
        notes.append("BTN limps more than baseline.")

    if state.hero_position == Position.BTN_SB:
        notes.append("Hero is BTN (steal logic relevant).")
    else:
        notes.append("Hero is BB (defense/3bet logic relevant).")

    return AdaptiveRangeAssumptions(
        villain_profile_summary=summary,
        bb_fold_to_steal_assumed=bb_fold.smoothed_rate,
        bb_3bet_vs_open_assumed=bb_3b.smoothed_rate,
        btn_open_rate_assumed=btn_open.smoothed_rate,
        btn_limp_rate_assumed=btn_limp.smoothed_rate,
        btn_fold_to_3bet_assumed=btn_fold3.smoothed_rate,
        bb_raise_vs_limp_assumed=bb_iso.smoothed_rate,
        btn_open_size_bucket_dist=profile.dist_btn_open_size(),
        bb_3bet_size_bucket_dist=profile.dist_bb_3bet_size(),
        bb_iso_size_bucket_dist=profile.dist_bb_iso_size(),
        notes=notes,
    )

