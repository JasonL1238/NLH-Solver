"""Shallow villain response probabilities for flop EV (no solver).

Uses optional ``FlopOpponentProfile``-like objects with smoothed stats +
confidence; falls back to neutral priors when absent or low confidence.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from flop_spot.models import FlopContext, FlopDerivedContext

_PRIOR_FOLD_VS_BET = 0.40
_PRIOR_CALL_VS_BET = 0.52
_PRIOR_RAISE_VS_BET = 0.08


def _stat(profile: Any, name: str) -> Optional[Tuple[float, float]]:
    """Return (smoothed_rate, confidence) or None."""
    if profile is None:
        return None
    fn = getattr(profile, name, None)
    if fn is None or not callable(fn):
        return None
    s = fn()
    r = getattr(s, "smoothed_rate", None)
    c = getattr(s, "confidence", None)
    if r is None or c is None:
        return None
    return float(r), float(c)


def villain_response_vs_hero_bet(
    ctx: FlopDerivedContext,
    profile: Any,
    bet_frac_of_pot: float,
) -> Tuple[float, float, float]:
    """Return ``(p_fold, p_call, p_raise)`` after hero bets/raises on flop."""
    _ = bet_frac_of_pot
    pf = float(_PRIOR_FOLD_VS_BET)
    pr = float(_PRIOR_RAISE_VS_BET)

    st = _stat(profile, "smoothed_fold_to_flop_cbet")
    if st is not None and ctx.flop_context in (
        FlopContext.PFR_IP_CHECKED_TO,
        FlopContext.PFR_OOP_FIRST_TO_ACT,
    ):
        rate, conf = st
        pf = (1.0 - conf) * _PRIOR_FOLD_VS_BET + conf * rate

    pc = max(0.0, 1.0 - pf - pr)
    s = pf + pc + pr
    if s < 1e-9:
        return _PRIOR_FOLD_VS_BET, _PRIOR_CALL_VS_BET, _PRIOR_RAISE_VS_BET
    return pf / s, pc / s, pr / s


def villain_response_vs_hero_check(ctx: FlopDerivedContext, profile: Any) -> float:
    """Probability villain stabs (bets) after hero checks (CHECK EV stub)."""
    _ = ctx
    st = _stat(profile, "smoothed_villain_bet_when_checked_to")
    if st is None:
        return 0.38
    rate, conf = st
    return (1.0 - conf) * 0.38 + conf * rate
