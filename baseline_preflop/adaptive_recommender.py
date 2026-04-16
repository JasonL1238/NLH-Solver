"""Adaptive preflop recommender (Phase B).

Orchestrates:
  1) baseline Phase A recommendation (must remain canonical)
  2) opponent profile stats + derived archetypes
  3) exploit-adjusted action distribution
  4) sampling a final action

Preflop-only, HU-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import random

from .adaptive_ranges import adjusted_villain_assumptions
from .exploit_adjuster import (
    ExploitResult,
    choose_raise_option,
    exploit_adjust_action_distribution,
    sample_action,
)
from .models import ActionType, Decision, LegalActionOption, PokerState
from .opponent_model import OpponentPreflopProfile
from .recommender import recommend_preflop_action


@dataclass(frozen=True)
class AdaptiveDecision:
    legal_actions: List[LegalActionOption]
    baseline_recommendation: LegalActionOption
    action_frequencies: Dict[str, float]
    roll: float
    adapted_recommendation: LegalActionOption
    adaptation_changed: bool
    explanation: str
    debug: dict = field(default_factory=dict)


def recommend_adaptive_preflop_action(
    state: PokerState,
    profile: OpponentPreflopProfile,
    rng: Optional[random.Random] = None,
) -> AdaptiveDecision:
    rng = rng or random.Random()

    baseline_decision: Decision = recommend_preflop_action(state)
    legal = baseline_decision.legal_actions

    # If baseline has no legal actions (hand over/closed), pass through.
    if not legal:
        return AdaptiveDecision(
            legal_actions=[],
            baseline_recommendation=baseline_decision.recommended_action,
            action_frequencies={},
            roll=0.0,
            adapted_recommendation=baseline_decision.recommended_action,
            adaptation_changed=False,
            explanation=baseline_decision.explanation,
            debug={
                **(baseline_decision.debug or {}),
                "villain_profile_summary": ",".join(profile.archetypes()),
                "baseline_action": repr(baseline_decision.recommended_action),
                "final_action": repr(baseline_decision.recommended_action),
                "adaptation_changed": False,
            },
        )

    assumptions = adjusted_villain_assumptions(profile, state)
    exploit: ExploitResult = exploit_adjust_action_distribution(
        state=state,
        profile=profile,
        baseline=baseline_decision,
    )

    token, roll = sample_action(exploit.action_frequencies, rng)

    # Convert token to a concrete legal action option
    chosen: Optional[LegalActionOption] = None
    if token == "RAISE":
        chosen = choose_raise_option(legal, exploit.raise_size_frequencies, rng)
    else:
        # pick the first legal action of that type
        for a in legal:
            if a.action_type.value == token:
                chosen = a
                break

    if chosen is None:
        # fallback: baseline recommendation (should not happen)
        chosen = baseline_decision.recommended_action

    adaptation_changed = (chosen != baseline_decision.recommended_action)

    explanation = "Adaptive preflop decision based on baseline + opponent model."
    if exploit.notes:
        explanation += " " + " ".join(exploit.notes)

    debug = {
        **(baseline_decision.debug or {}),
        "villain_profile_summary": assumptions.villain_profile_summary,
        "key_smoothed_stats": {
            "bb_fold_to_steal": round(profile.stat_bb_fold_to_steal().smoothed_rate, 4),
            "bb_3bet_vs_open": round(profile.stat_bb_3bet_vs_open().smoothed_rate, 4),
            "btn_open_rate": round(profile.stat_btn_open_rate().smoothed_rate, 4),
            "btn_limp_rate": round(profile.stat_btn_limp_rate().smoothed_rate, 4),
        },
        "key_sizing_stats": {
            "btn_open_size_bucket_dist": assumptions.btn_open_size_bucket_dist,
            "bb_3bet_size_bucket_dist": assumptions.bb_3bet_size_bucket_dist,
            "bb_iso_size_bucket_dist": assumptions.bb_iso_size_bucket_dist,
        },
        "confidence_summary": {
            "bb_fold_to_steal_confidence": round(profile.stat_bb_fold_to_steal().confidence, 4),
            "bb_3bet_confidence": round(profile.stat_bb_3bet_vs_open().confidence, 4),
            "btn_open_confidence": round(profile.stat_btn_open_rate().confidence, 4),
        },
        "range_adjustment_notes": assumptions.notes,
        "exploit_adjustment_notes": exploit.notes,
        "baseline_raise_size_filter_applied": exploit.baseline_raise_size_filter_applied,
        "baseline_raise_size_note": exploit.baseline_raise_size_note,
        "baseline_action": repr(baseline_decision.recommended_action),
        "final_action": repr(chosen),
        "action_frequencies": exploit.action_frequencies,
        "raise_size_frequencies": exploit.raise_size_frequencies,
        "roll": roll,
        "adaptation_changed": adaptation_changed,
    }

    return AdaptiveDecision(
        legal_actions=legal,
        baseline_recommendation=baseline_decision.recommended_action,
        action_frequencies=exploit.action_frequencies,
        roll=roll,
        adapted_recommendation=chosen,
        adaptation_changed=adaptation_changed,
        explanation=explanation,
        debug=debug,
    )

