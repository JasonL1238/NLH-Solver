"""Turn/river EV integration (mirrors ``flop_equity.equity_integration`` debug keys)."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from poker_core.models import HoleCards, HandState, Street

from flop_policy.config import EvPolicyConfig
from flop_spot.models import FlopDecision

from postflop_range.range_tracker import VillainParticleTracker

from postflop_policy.ev_recommender import (
    recommend_postflop_action_ev,
    recommend_river_action_ev,
    recommend_turn_action_ev,
)

_EQUITY_MARGIN = 0.05


def recommend_turn_action_with_equity(
    state: HandState,
    samples: int = 5000,
    seed: Optional[int] = None,
    villain_range_override: Optional[List[Tuple[HoleCards, float]]] = None,
    particle_tracker: Optional[VillainParticleTracker] = None,
    config: Optional[EvPolicyConfig] = None,
    profile: Any = None,
) -> FlopDecision:
    """EV-first turn recommendation (Monte Carlo river runouts)."""
    dec = recommend_turn_action_ev(
        state,
        profile=profile,
        samples=samples,
        seed=seed,
        villain_range_override=villain_range_override,
        particle_tracker=particle_tracker,
        config=config,
    )
    return _attach_equity_integration_debug(dec)


def recommend_river_action_with_equity(
    state: HandState,
    samples: int = 5000,
    seed: Optional[int] = None,
    villain_range_override: Optional[List[Tuple[HoleCards, float]]] = None,
    particle_tracker: Optional[VillainParticleTracker] = None,
    config: Optional[EvPolicyConfig] = None,
    profile: Any = None,
) -> FlopDecision:
    """EV-first river recommendation (exact equity vs range; ``samples`` unused)."""
    dec = recommend_river_action_ev(
        state,
        profile=profile,
        samples=samples,
        seed=seed,
        villain_range_override=villain_range_override,
        particle_tracker=particle_tracker,
        config=config,
    )
    return _attach_equity_integration_debug(dec)


def recommend_postflop_action_with_equity(
    state: HandState,
    *,
    street: Street,
    samples: int = 5000,
    seed: Optional[int] = None,
    villain_range_override: Optional[List[Tuple[HoleCards, float]]] = None,
    particle_tracker: Optional[VillainParticleTracker] = None,
    config: Optional[EvPolicyConfig] = None,
    profile: Any = None,
) -> FlopDecision:
    """Dispatch by ``street`` (FLOP/TURN/RIVER) with parity debug keys."""
    dec = recommend_postflop_action_ev(
        state,
        profile=profile,
        street=street,
        samples=samples,
        seed=seed,
        villain_range_override=villain_range_override,
        particle_tracker=particle_tracker,
        config=config,
    )
    return _attach_equity_integration_debug(dec)


def _attach_equity_integration_debug(dec: FlopDecision) -> FlopDecision:
    dbg = dict(dec.debug)
    ch = dec.recommended_action
    dbg.setdefault("monte_carlo_samples", dbg.get("samples_used"))
    dbg["baseline_action"] = repr(ch)
    dbg["final_action"] = repr(ch)
    dbg["equity_changed_recommendation"] = False
    dbg["equity_adjustment_notes"] = [
        "EV-first postflop policy; no fold/call overlay vs separate baseline.",
    ]
    dbg["equity_overlay_margin"] = _EQUITY_MARGIN
    return FlopDecision(
        legal_actions=dec.legal_actions,
        recommended_action=ch,
        explanation=dec.explanation,
        debug=dbg,
    )
