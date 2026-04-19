"""Unified postflop EV recommender (flop delegates; turn/river MC/exact equity)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from poker_core.legal_actions import legal_actions
from poker_core.models import HandState, HoleCards, Player, Street

from flop_equity.monte_carlo import estimate_showdown_equity
from flop_policy.config import EvPolicyConfig
from flop_policy.ev_recommender import recommend_flop_action_ev
from flop_policy.hero_value_tier import hero_flop_value_tier, hero_has_pressure_draw
from flop_policy.range_metrics import villain_range_nut_metrics
from flop_policy.response_model import villain_response_vs_hero_bet, villain_response_vs_hero_check
from flop_spot.context import derive_flop_context
from flop_spot.models import FlopActionChoice, FlopDecision
from flop_spot.spot_debug import build_spot_debug

from postflop_range.range_tracker import VillainParticleTracker

from .ev_core import (
    apply_thin_raise_filter,
    build_ev_candidates,
    median_p_fold_over_raise_grid,
    pick_best_ev_candidate,
)


def recommend_postflop_action_ev(
    state: HandState,
    profile: Any = None,
    *,
    street: Street,
    samples: int = 3000,
    seed: Optional[int] = None,
    villain_range_override: Optional[List[Tuple[HoleCards, float]]] = None,
    particle_tracker: Optional[VillainParticleTracker] = None,
    config: Optional[EvPolicyConfig] = None,
) -> FlopDecision:
    """Argmax EV over legal postflop actions for the given ``street``."""
    if street == Street.FLOP:
        return recommend_flop_action_ev(
            state,
            profile=profile,
            samples=samples,
            seed=seed,
            villain_range_override=villain_range_override,
            config=config,
        )

    if street not in (Street.TURN, Street.RIVER):
        raise ValueError(f"Unsupported street: {street!r}")

    if state.current_street != street:
        raise ValueError(
            f"state.current_street is {state.current_street!r}, expected {street!r}",
        )
    if state.hand_over:
        raise ValueError("Hand is already over")
    if state.current_actor != Player.HERO:
        raise ValueError("Not hero's turn to act")

    cfg = config or EvPolicyConfig()
    la_list = legal_actions(state)
    if not la_list:
        raise ValueError(f"No legal actions available on {street.value}")

    # v1: neutral response priors on turn/river (flop profile not applied here).
    profile_effective: Any = None
    range_note_for_debug: str

    if villain_range_override is not None:
        v_range = villain_range_override
        range_summary = "User-provided override range"
        range_note_for_debug = "User-provided override range"
    elif particle_tracker is not None:
        v_range = particle_tracker.export_weighted_range_for_rollouts()
        range_summary = "postflop_range.VillainParticleTracker (particle export)"
        range_note_for_debug = "Particle-based range (see particle_range_debug)"
    else:
        # Local import: avoids ``postflop_equity`` package init pulling ``integration``
        # while this module is still loading (circular import with ``integration``).
        from postflop_equity.range_carryforward import RANGE_NOTE_V1, build_villain_postflop_range

        v_range, range_summary = build_villain_postflop_range(state)
        range_note_for_debug = RANGE_NOTE_V1

    hero = state.config.hero_hole_cards
    board = list(state.board_cards)
    mc = estimate_showdown_equity(
        hero,
        board,
        v_range,
        street=street,
        samples=samples,
        seed=seed,
    )
    eq = float(mc["equity_estimate"])

    ctx = derive_flop_context(state)

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
            profile=profile_effective,
            bet_response=villain_response_vs_hero_bet,
        )

    raw = build_ev_candidates(
        la_list=la_list,
        state=state,
        eq=eq,
        cfg=cfg,
        ctx=ctx,
        profile=profile_effective,
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
        f"EV-first ({street.value}): argmax over {len(candidates)} candidates "
        f"(EV={best_ev:.3f}bb). {best_note}"
    )

    extra_debug: Dict[str, Any] = {
        "equity_estimate": eq,
        "win_rate": mc["win_rate"],
        "tie_rate": mc["tie_rate"],
        "samples_used": mc["samples_used"],
        "monte_carlo_samples": mc["samples_used"],
        "villain_range_summary": range_summary,
        "range_note": range_note_for_debug,
        "pot_odds_threshold": pot_odds_threshold,
        "street_policy_version": cfg.street_policy_version,
        "ev_policy": "postflop_policy.ev_recommender",
        "response_model_profile": "neutral_v1_turn_river",
        "user_profile_ignored_turn_river": profile is not None,
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
    if particle_tracker is not None:
        from postflop_range.debug import build_debug_dict

        extra_debug["particle_range_debug"] = build_debug_dict(particle_tracker)

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


def recommend_turn_action_ev(
    state: HandState,
    profile: Any = None,
    *,
    samples: int = 3000,
    seed: Optional[int] = None,
    villain_range_override: Optional[List[Tuple[HoleCards, float]]] = None,
    particle_tracker: Optional[VillainParticleTracker] = None,
    config: Optional[EvPolicyConfig] = None,
) -> FlopDecision:
    """Thin wrapper: ``street=Street.TURN``."""
    return recommend_postflop_action_ev(
        state,
        profile=profile,
        street=Street.TURN,
        samples=samples,
        seed=seed,
        villain_range_override=villain_range_override,
        particle_tracker=particle_tracker,
        config=config,
    )


def recommend_river_action_ev(
    state: HandState,
    profile: Any = None,
    *,
    samples: int = 3000,
    seed: Optional[int] = None,
    villain_range_override: Optional[List[Tuple[HoleCards, float]]] = None,
    particle_tracker: Optional[VillainParticleTracker] = None,
    config: Optional[EvPolicyConfig] = None,
) -> FlopDecision:
    """Thin wrapper: ``street=Street.RIVER`` (``samples`` ignored for exact equity)."""
    return recommend_postflop_action_ev(
        state,
        profile=profile,
        street=Street.RIVER,
        samples=samples,
        seed=seed,
        villain_range_override=villain_range_override,
        particle_tracker=particle_tracker,
        config=config,
    )
