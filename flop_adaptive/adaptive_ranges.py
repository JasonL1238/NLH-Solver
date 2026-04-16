"""Confidence-weighted numeric hints for flop exploit (no solver).

``RangeAdjustmentHints`` feeds ``exploit_adjuster`` only.  A future version
may map hints to ``villain_range_override`` for a second equity pass; v1
keeps a single Monte Carlo run inside ``flop_policy.recommend_flop_action_ev``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from poker_core.models import HandState

from .opponent_model import FlopOpponentProfile, flop_archetypes


@dataclass(frozen=True)
class RangeAdjustmentHints:
    """Scalar multipliers in ``[0, 1]``-ish band around 1.0 for exploit rules."""

    cbet_frequency_scalar: float = 1.0
    defend_vs_bet_scalar: float = 1.0
    fold_vs_raise_scalar: float = 1.0
    notes: Tuple[str, ...] = field(default_factory=tuple)  # type: ignore[assignment]


def compute_range_adjustment_hints(
    state: HandState,
    profile: FlopOpponentProfile,
    spot_debug: Dict[str, Any],
) -> RangeAdjustmentHints:
    """Derive deterministic hints from ``FlopOpponentProfile`` + spot debug.

    Uses smoothed rates and confidence from the profile; low confidence
    pulls scalars toward 1.0 (no adjustment).
    """
    notes: List[str] = []
    arch = set(flop_archetypes(profile))

    ft_cb = profile.smoothed_fold_to_flop_cbet()
    cb_pfr = profile.smoothed_villain_cbet_when_pfr()
    ft_st = profile.smoothed_fold_to_flop_stab()
    rz = profile.smoothed_hero_raise_vs_cbet()
    stab = profile.smoothed_villain_bet_when_checked_to()

    cbet_scalar = 1.0
    defend_scalar = 1.0
    fold_vs_raise_scalar = 1.0

    w_ft = ft_cb.confidence
    w_cb = cb_pfr.confidence
    w_st = ft_st.confidence
    w_rz = rz.confidence
    w_stab = stab.confidence

    # --- Widen / tighten c-bet region (hero as PFR checked to / first OOP) ---
    ctx = str(spot_debug.get("action_context_label") or "")
    if ctx in ("PFR_IP_CHECKED_TO", "PFR_OOP_FIRST_TO_ACT"):
        if "FLOP_OVERFOLDER_VS_CBET" in arch and ft_cb.smoothed_rate > ft_cb.prior + 0.08:
            delta = 0.12 * w_ft
            cbet_scalar = 1.0 + delta
            notes.append("High fold-to-cbet read: slight c-bet widen scalar.")
        if "FLOP_STICKY_CALLER" in arch or ft_cb.smoothed_rate < ft_cb.prior - 0.08:
            delta = 0.10 * max(w_ft, 0.25)
            cbet_scalar = 1.0 - delta
            notes.append("Sticky vs cbet: slight c-bet tighten scalar.")
        if "AUTO_CBETS_TOO_MUCH" in arch and cb_pfr.smoothed_rate > cb_pfr.prior + 0.12:
            delta = 0.06 * w_cb
            cbet_scalar = min(cbet_scalar, 1.0 - delta)
            notes.append("Villain auto-cbets a lot: marginal c-bet frequency trim.")
        if "PASSIVE_ON_FLOP" in arch:
            delta = 0.05 * w_cb
            cbet_scalar = max(cbet_scalar, 1.0 + delta)
            notes.append("Passive flop c-bet frequency: tiny widen.")

    # --- Defend vs flop bet (hero facing bet) ---
    if ctx.startswith("FACING_") and "RAISE" not in ctx:
        if ft_cb.smoothed_rate > ft_cb.prior + 0.10:
            defend_scalar = 1.0 + 0.08 * w_ft
            notes.append("Villain overfolds vs pressure historically: defend scalar up.")
        if ft_cb.smoothed_rate < ft_cb.prior - 0.10:
            defend_scalar = 1.0 - 0.08 * w_ft
            notes.append("Villain underfolds: defend scalar down.")
        if ft_st.smoothed_rate > ft_st.prior + 0.10:
            defend_scalar = max(defend_scalar, 1.0 + 0.05 * w_st)
            notes.append("Overfolds vs stab: small extra defend.")

    # --- Facing raise after we bet ---
    if "FACING_RAISE" in ctx:
        if "FLOP_RAISES_LIGHT" in arch:
            fold_vs_raise_scalar = 1.0 - 0.08 * w_rz
            notes.append("Raise-light read: fold slightly less vs flop raises.")
        if "FLOP_RAISES_VALUE_HEAVY" in arch:
            fold_vs_raise_scalar = 1.0 + 0.08 * w_rz
            notes.append("Value-heavy raise read: fold slightly more.")

    # --- Caller stab tendency when checked to IP ---
    if "PFC_IP_CHECKED_TO" in ctx and stab.smoothed_rate > stab.prior + 0.12:
        cbet_scalar = max(cbet_scalar, 1.0 + 0.04 * w_stab)
        notes.append("Villain stabs a lot when checked to: value stab scalar nudge.")

    # Clamp scalars to sane band
    cbet_scalar = max(0.75, min(1.25, cbet_scalar))
    defend_scalar = max(0.75, min(1.25, defend_scalar))
    fold_vs_raise_scalar = max(0.75, min(1.25, fold_vs_raise_scalar))

    if not notes:
        notes.append("No strong flop exploit read; scalars neutral.")

    return RangeAdjustmentHints(
        cbet_frequency_scalar=round(cbet_scalar, 4),
        defend_vs_bet_scalar=round(defend_scalar, 4),
        fold_vs_raise_scalar=round(fold_vs_raise_scalar, 4),
        notes=tuple(notes),
    )
