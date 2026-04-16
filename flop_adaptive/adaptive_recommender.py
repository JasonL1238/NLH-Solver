"""Orchestrate EV-first flop policy + adaptive exploit (flop-only).

1. ``flop_policy.recommend_flop_action_ev`` — MC vs range, response model, grid.
2. ``flop_adaptive`` — ``compute_range_adjustment_hints`` reads profile + spot
   debug; ``apply_flop_exploit_adjustment`` nudges the EV action when allowed.

``AdaptiveFlopDecision`` still exposes ``baseline_*`` / ``equity_*`` fields for
callers; both are filled from the same EV ``FlopDecision`` for compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from poker_core.models import HandState, Player, Street

from flop_policy.ev_recommender import recommend_flop_action_ev
from flop_spot.models import FlopActionChoice, FlopDecision

from .adaptive_ranges import RangeAdjustmentHints, compute_range_adjustment_hints
from .exploit_adjuster import apply_flop_exploit_adjustment
from .opponent_model import (
    FlopOpponentProfile,
    confidence_summary_dict,
    flop_archetypes,
    key_smoothed_flop_stats_dict,
    villain_flop_profile_summary,
)


@dataclass
class AdaptiveFlopDecision:
    """Layered flop output: baseline, equity-aware, adapted."""

    legal_actions: List
    baseline_recommendation: FlopActionChoice
    equity_aware_recommendation: FlopActionChoice
    adapted_recommendation: FlopActionChoice
    adaptation_changed: bool
    explanation: str
    baseline_decision: FlopDecision
    equity_decision: FlopDecision
    debug: Dict[str, Any] = field(default_factory=dict)


def _actions_equal(a: FlopActionChoice, b: FlopActionChoice) -> bool:
    if a.legal_action.action_type != b.legal_action.action_type:
        return False
    if a.size_bb is None and b.size_bb is None:
        return True
    if a.size_bb is None or b.size_bb is None:
        return False
    return abs(a.size_bb - b.size_bb) < 0.02


def recommend_adaptive_flop_action(
    state: HandState,
    profile: FlopOpponentProfile,
    samples: int = 5000,
    seed: Optional[int] = None,
) -> AdaptiveFlopDecision:
    """Run EV policy then exploit; return structured ``AdaptiveFlopDecision``."""
    if state.current_street != Street.FLOP:
        raise ValueError(f"Expected FLOP, got {state.current_street.value}")
    if state.hand_over:
        raise ValueError("Hand is already over")
    if state.current_actor != Player.HERO:
        raise ValueError("Not hero's turn to act")

    ev_dec = recommend_flop_action_ev(
        state, profile, samples=samples, seed=seed,
    )

    hints = compute_range_adjustment_hints(state, profile, ev_dec.debug)
    adapted_choice, exploit_notes = apply_flop_exploit_adjustment(
        ev_dec, state, hints, ev_dec.debug,
    )

    adaptation_changed = not _actions_equal(
        adapted_choice, ev_dec.recommended_action,
    )

    eq_notes = list(ev_dec.debug.get("equity_adjustment_notes") or [])
    equity_changed_baseline = bool(ev_dec.debug.get("equity_changed_recommendation"))

    parts = [ev_dec.explanation]
    if equity_changed_baseline:
        parts.append(f"[Equity overlay: {eq_notes[-1] if eq_notes else 'adjusted'}]")
    if adaptation_changed:
        parts.append(f"[Exploit: {exploit_notes[-1] if exploit_notes else 'adjusted'}]")
    explanation = " ".join(parts)

    dbg: Dict[str, Any] = {}
    for k in (
        "action_context_label",
        "hero_preflop_role",
        "hero_position_relation_on_flop",
        "made_hand_category",
        "draw_category",
        "board_texture_label",
        "flop_bet_size_bucket",
        "spr_bucket",
        "baseline_rule_id",
    ):
        dbg[k] = ev_dec.debug.get(k)

    dbg["villain_flop_profile_summary"] = villain_flop_profile_summary(profile)
    dbg["key_smoothed_flop_stats"] = key_smoothed_flop_stats_dict(profile)
    dbg["confidence_summary"] = confidence_summary_dict(profile)
    dbg["range_adjustment_notes"] = list(hints.notes)
    dbg["exploit_adjustment_notes"] = exploit_notes
    dbg["baseline_action"] = repr(ev_dec.recommended_action)
    dbg["equity_action"] = repr(ev_dec.recommended_action)
    dbg["final_action"] = repr(adapted_choice)
    dbg["equity_changed_vs_baseline"] = equity_changed_baseline
    dbg["equity_adjustment_notes_baseline"] = eq_notes
    dbg["adaptation_changed_vs_equity"] = adaptation_changed
    dbg["equity_estimate"] = ev_dec.debug.get("equity_estimate")
    dbg["pot_odds_threshold"] = ev_dec.debug.get("pot_odds_threshold")
    dbg["monte_carlo_samples"] = ev_dec.debug.get("monte_carlo_samples")
    dbg["villain_range_summary"] = ev_dec.debug.get("villain_range_summary")
    dbg["flop_archetypes"] = flop_archetypes(profile)

    return AdaptiveFlopDecision(
        legal_actions=ev_dec.legal_actions,
        baseline_recommendation=ev_dec.recommended_action,
        equity_aware_recommendation=ev_dec.recommended_action,
        adapted_recommendation=adapted_choice,
        adaptation_changed=adaptation_changed,
        explanation=explanation,
        baseline_decision=ev_dec,
        equity_decision=ev_dec,
        debug=dbg,
    )


def simulate_flop_overfolder(profile: FlopOpponentProfile, n: int) -> None:
    """Deterministically inflate fold-to-flop-cbet stats (testing helper)."""
    for _ in range(max(0, n)):
        profile.hero_fold_to_flop_cbet_opportunities += 1
        profile.hero_fold_to_flop_cbet_count += 1


def simulate_flop_raise_heavy(profile: FlopOpponentProfile, n: int) -> None:
    """Inflate raise-vs-cbet counts (testing helper)."""
    for _ in range(max(0, n)):
        profile.hero_fold_to_flop_cbet_opportunities += 1
        profile.hero_raise_vs_flop_cbet_count += 1


def simulate_flop_stab_heavy(profile: FlopOpponentProfile, n: int) -> None:
    """Inflate stab-when-checked-to stats (testing helper)."""
    for _ in range(max(0, n)):
        profile.villain_flop_bet_when_checked_to_opportunities += 1
        profile.villain_flop_bet_when_checked_to_count += 1


def simulate_flop_sticky_caller(profile: FlopOpponentProfile, n: int) -> None:
    """Inflate call-vs-cbet stats (low fold-to-cbet) for testing."""
    for _ in range(max(0, n)):
        profile.hero_fold_to_flop_cbet_opportunities += 1
        profile.hero_call_flop_cbet_count += 1


def compare_flop_baseline_vs_equity_vs_adaptive(
    state: HandState,
    profile: FlopOpponentProfile,
    samples: int = 5000,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Side-by-side comparison for manual inspection."""
    dec = recommend_adaptive_flop_action(state, profile, samples=samples, seed=seed)
    return {
        "baseline_action": repr(dec.baseline_recommendation),
        "equity_action": repr(dec.equity_aware_recommendation),
        "adapted_action": repr(dec.adapted_recommendation),
        "adaptation_changed": dec.adaptation_changed,
        "equity_estimate": dec.debug.get("equity_estimate"),
        "pot_odds_threshold": dec.debug.get("pot_odds_threshold"),
        "profile_summary": dec.debug.get("villain_flop_profile_summary"),
        "key_smoothed_flop_stats": dec.debug.get("key_smoothed_flop_stats"),
        "exploit_notes": dec.debug.get("exploit_adjustment_notes"),
        "range_notes": dec.debug.get("range_adjustment_notes"),
    }
