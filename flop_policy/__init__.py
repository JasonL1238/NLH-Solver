"""EV-first flop policy (legal_actions grid + MC + response model)."""

from .config import EvPolicyConfig

__all__ = ["EvPolicyConfig", "recommend_flop_action_ev"]


def __getattr__(name: str):
    # Lazy: ``flop_policy.ev_recommender`` imports ``postflop_policy.ev_core``; loading
    # ``ev_recommender`` from package ``__init__`` would re-enter while ``ev_core`` is loading.
    if name == "recommend_flop_action_ev":
        from .ev_recommender import recommend_flop_action_ev

        return recommend_flop_action_ev
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
