"""Shared postflop EV math and (turn/river) policy entrypoints."""

from .ev_core import (
    apply_thin_raise_filter,
    build_ev_candidates,
    ev_aggression_line,
    ev_call_fold_pot,
    linspace_sizes,
    pick_best_ev_candidate,
    pot_fraction_bet_targets,
)

__all__ = [
    "apply_thin_raise_filter",
    "build_ev_candidates",
    "ev_aggression_line",
    "ev_call_fold_pot",
    "linspace_sizes",
    "pick_best_ev_candidate",
    "pot_fraction_bet_targets",
    "recommend_postflop_action_ev",
    "recommend_river_action_ev",
    "recommend_turn_action_ev",
]


def __getattr__(name: str):
    # Lazy: avoids import cycle (flop_policy.ev_recommender → postflop_policy.ev_core
    # while postflop_policy.__init__ would otherwise pull ev_recommender → flop_policy).
    if name == "recommend_postflop_action_ev":
        from .ev_recommender import recommend_postflop_action_ev

        return recommend_postflop_action_ev
    if name == "recommend_river_action_ev":
        from .ev_recommender import recommend_river_action_ev

        return recommend_river_action_ev
    if name == "recommend_turn_action_ev":
        from .ev_recommender import recommend_turn_action_ev

        return recommend_turn_action_ev
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
