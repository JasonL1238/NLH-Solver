"""Compatibility shim: ``recommend_flop_action`` → EV-first policy.

Phase D rule engine was removed; callers should use
``flop_policy.recommend_flop_action_ev`` or
``flop_equity.equity_integration.recommend_flop_action_with_equity`` directly.
"""

from __future__ import annotations

from poker_core.models import HandState

from flop_policy.ev_recommender import recommend_flop_action_ev
from flop_spot.models import FlopDecision


def recommend_flop_action(
    state: HandState,
    *,
    samples: int = 2000,
    seed: int = 0,
) -> FlopDecision:
    """Delegate to ``recommend_flop_action_ev`` (neutral profile, fixed seed)."""
    return recommend_flop_action_ev(
        state, profile=None, samples=samples, seed=seed,
    )
