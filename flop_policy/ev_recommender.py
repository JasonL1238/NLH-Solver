"""EV-first flop recommender: legal_actions grid + MC equity + response model."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from poker_core.legal_actions import legal_actions
from poker_core.models import HandState, HoleCards, Player, Street

from flop_equity.monte_carlo import estimate_flop_equity
from flop_equity.range_model import build_villain_flop_range
from flop_spot.context import derive_flop_context
from flop_spot.models import FlopActionChoice, FlopDecision
from flop_spot.spot_debug import build_spot_debug

from postflop_policy.ev_core import (
    apply_thin_raise_filter,
    build_ev_candidates,
    median_p_fold_over_raise_grid,
    pick_best_ev_candidate,
)

from .config import EvPolicyConfig
from .hero_value_tier import hero_flop_value_tier, hero_has_pressure_draw
from .range_metrics import villain_range_nut_metrics
from .response_model import villain_response_vs_hero_bet, villain_response_vs_hero_check


def recommend_flop_action_ev(
    state: HandState,
    profile: Any = None,
    *,
    samples: int = 3000,
    seed: Optional[int] = None,
    villain_range_override: Optional[List[Tuple[HoleCards, float]]] = None,
    config: Optional[EvPolicyConfig] = None,
) -> FlopDecision:
    """Pick argmax EV over legal flop actions (discrete BET/RAISE grid)."""
    if state.current_street != Street.FLOP:
        raise ValueError(f"Expected FLOP, got {state.current_street.value}")
    if state.hand_over:
        raise ValueError("Hand is already over")
    if state.current_actor != Player.HERO:
        raise ValueError("Not hero's turn to act")

    cfg = config or EvPolicyConfig()
    la_list = legal_actions(state)
    if not la_list:
        raise ValueError("No legal actions available on flop")

    ctx = derive_flop_context(state)
    if villain_range_override is not None:
        v_range = villain_range_override
        range_summary = "User-provided override range"
    else:
        v_range, range_summary = build_villain_flop_range(state)

    hero = state.config.hero_hole_cards
    board = list(state.board_cards)
    mc = estimate_flop_equity(hero, board, v_range, samples=samples, seed=seed)
    eq = float(mc["equity_estimate"])

    pot = float(state.pot_size_bb)
    to_call = float(state.current_bet_to_call_bb)
    facing = to_call > 1e-9

    pot_odds_threshold: Optional[float] = None
    if facing:
        pot_odds_threshold = round(to_call / (pot + to_call), 4)

    hero_tier = hero_flop_value_tier(state)
    pressure_draw = hero_has_pressure_draw(state)
    flop_board = board[:3] if len(board) >= 3 else []
    if len(flop_board) == 3:
        villain_nut_frac, nut_range_top = villain_range_nut_metrics(flop_board, v_range)
    else:
        villain_nut_frac, nut_range_top = 1.0, "no_flop_board"

    p_fold_raise_median = 0.0
    if facing:
        p_fold_raise_median = median_p_fold_over_raise_grid(
            la_list=la_list,
            state=state,
            cfg=cfg,
            ctx=ctx,
            profile=profile,
            bet_response=villain_response_vs_hero_bet,
        )

    raw = build_ev_candidates(
        la_list=la_list,
        state=state,
        eq=eq,
        cfg=cfg,
        ctx=ctx,
        profile=profile,
        bet_response=villain_response_vs_hero_bet,
        check_stab_prob=villain_response_vs_hero_check,
        hero_value_tier=hero_tier,
    )
    candidates, thin_raise_branch = apply_thin_raise_filter(
        raw,
        facing=facing,
        pot_odds_threshold=pot_odds_threshold,
        eq=eq,
        cfg=cfg,
        hero_value_tier=hero_tier,
        villain_nut_frac=villain_nut_frac,
        p_fold_raise_median=p_fold_raise_median,
        has_pressure_draw=pressure_draw,
    )
    choice, best_ev, best_note = pick_best_ev_candidate(
        candidates,
        la_list,
        facing=facing,
        hero_value_tier=hero_tier,
    )

    explanation = (
        f"EV-first: argmax over {len(candidates)} candidates (EV={best_ev:.3f}bb). "
        f"{best_note}"
    )

    extra_debug: Dict[str, Any] = {
        "equity_estimate": eq,
        "win_rate": mc["win_rate"],
        "tie_rate": mc["tie_rate"],
        "samples_used": mc["samples_used"],
        "monte_carlo_samples": mc["samples_used"],
        "villain_range_summary": range_summary,
        "pot_odds_threshold": pot_odds_threshold,
        "street_policy_version": cfg.street_policy_version,
        "ev_policy": "flop_policy.ev_recommender",
        "ev_best": round(best_ev, 4),
        "hero_value_tier": hero_tier,
        "villain_nut_weight": round(villain_nut_frac, 4),
        "villain_nut_range_top": nut_range_top,
        "p_fold_raise_median": round(p_fold_raise_median, 4),
        "thin_raise_policy_branch": thin_raise_branch,
        "hero_has_pressure_draw": pressure_draw,
        "ev_candidates": [
            {"repr": repr(c), "ev": round(ev, 4), "note": n} for c, ev, n in candidates
        ],
    }

    dbg = build_spot_debug(
        state,
        policy_rule_id="EV_ARGMAX",
        explanation=explanation,
        legal=la_list,
        recommended_action_repr=repr(choice),
        extra=extra_debug,
    )

    return FlopDecision(
        legal_actions=la_list,
        recommended_action=choice,
        explanation=explanation,
        debug=dbg,
    )
