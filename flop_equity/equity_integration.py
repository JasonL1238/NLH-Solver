"""Flop decision integration: EV-first policy + equity debug fields.

``recommend_flop_action_with_equity`` is the stable entrypoint name; it runs
``flop_policy.recommend_flop_action_ev`` (Monte Carlo vs villain range, response
model, discrete sizing grid).  Legacy pot-odds overlay on a separate baseline
policy has been removed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from poker_core.models import HoleCards, HandState

from flop_policy.config import EvPolicyConfig
from flop_policy.ev_recommender import recommend_flop_action_ev
from flop_spot.models import FlopDecision


_EQUITY_MARGIN = 0.05


def recommend_flop_action_with_equity(
    state: HandState,
    samples: int = 5000,
    seed: Optional[int] = None,
    villain_range_override: Optional[List[Tuple[HoleCards, float]]] = None,
    config: Optional[EvPolicyConfig] = None,
    profile: Any = None,
) -> FlopDecision:
    """Return EV-first flop recommendation with equity-rich debug.

    This replaces the former ``baseline + MC pot-odds nudge`` pipeline.
    Optional ``profile`` feeds the flop response model (same as adaptive).
    """
    dec = recommend_flop_action_ev(
        state,
        profile=profile,
        samples=samples,
        seed=seed,
        villain_range_override=villain_range_override,
        config=config,
    )
    dbg = dict(dec.debug)
    # Legacy / test-expected keys (EV-first: no secondary adjustment)
    ch = dec.recommended_action
    dbg.setdefault("monte_carlo_samples", dbg.get("samples_used"))
    dbg["baseline_action"] = repr(ch)
    dbg["final_action"] = repr(ch)
    dbg["equity_changed_recommendation"] = False
    dbg["equity_adjustment_notes"] = [
        "EV-first policy; no fold/call overlay vs separate baseline.",
    ]
    dbg["equity_overlay_margin"] = _EQUITY_MARGIN

    explanation = dec.explanation
    return FlopDecision(
        legal_actions=dec.legal_actions,
        recommended_action=ch,
        explanation=explanation,
        debug=dbg,
    )


def compare_flop_baseline_vs_equity(
    state: HandState,
    samples: int = 5000,
    seed: Optional[int] = None,
) -> Dict:
    """Inspect EV policy output (legacy name: no separate Phase D baseline)."""
    ev_dec = recommend_flop_action_with_equity(state, samples=samples, seed=seed)
    notes = ev_dec.debug.get("equity_adjustment_notes") or []

    return {
        "baseline_action": ev_dec.debug.get("baseline_action"),
        "baseline_rule_id": ev_dec.debug.get("baseline_rule_id"),
        "equity_action": repr(ev_dec.recommended_action),
        "equity_estimate": ev_dec.debug.get("equity_estimate"),
        "pot_odds_threshold": ev_dec.debug.get("pot_odds_threshold"),
        "equity_changed": ev_dec.debug.get("equity_changed_recommendation"),
        "notes": notes,
    }
