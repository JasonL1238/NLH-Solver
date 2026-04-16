"""Configuration for flop EV policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvPolicyConfig:
    """Tuning knobs for ``recommend_flop_action_ev``."""

    grid_points: int = 7
    """Number of sizes between min and max (inclusive) for BET/RAISE."""

    street_policy_version: str = "v1_flop_shallow_stub"
    """Documented in FlopDecision.debug; later streets use same MC as flop."""

    thin_raise_min_bar: float = 0.28
    """Lower bound blended with pot odds for thin-raise filtering."""

    reraised_eq_discount_default: float = 0.82
    """Multiplier on hero equity in the villain-reraise branch of ``ev_aggression_line``."""

    reraised_eq_discount_nuts: float = 1.0
    """Same multiplier when hero is ``NUTS_NEAR`` (value raise / re-raise)."""

    eq_pressure_min: float = 0.34
    """Minimum MC equity to allow thin raises under pressure (scenario 2)."""

    villain_nut_frac_max: float = 0.16
    """Max villain range nut/set weight fraction for pressure thin-raises."""

    p_fold_pressure_min: float = 0.36
    """Median ``p_fold`` over the raise grid must meet this for scenario 2."""
