"""Chart-based preflop recommendation engine for heads-up NLHE.

Replaces the earlier score-based system with explicit hand chart lookups.
Charts live in charts.py and define RAISE / CALL / FOLD sets per context.
"""

from __future__ import annotations

from typing import List, Optional

from .charts import get_chart_action
from .classification import hand_features
from .legal_actions import legal_actions_for_hero
from .models import (
    ActionType, Decision, DerivedState, HandFeatures,
    LegalActionOption, Player, PokerState, Position, StackDepthBucket,
)
from .validation import validate_preflop_state


# ---------------------------------------------------------------------------
# Context mapping
# ---------------------------------------------------------------------------

def _determine_context(d: DerivedState, hero_pos: Position) -> str:
    """Map derived-state flags to the chart context key."""
    if d.facing_4bet:
        return "BB_VS_4BET"
    if d.facing_3bet:
        if hero_pos == Position.BTN_SB:
            return "BTN_VS_3BET"
        return "BB_VS_4BET"
    if d.facing_open_raise:
        if hero_pos == Position.BTN_SB:
            return "BTN_VS_ISO"
        return "BB_VS_OPEN"
    if d.facing_limp:
        return "BB_VS_LIMP"
    return "BTN_OPEN"


def _context_label(d: DerivedState, hero_pos: Position) -> str:
    """Human-readable context label for debug output."""
    pos = hero_pos.value
    if d.facing_4bet:
        return f"{pos}_VS_4BET"
    if d.facing_3bet:
        return f"{pos}_VS_3BET"
    if d.facing_open_raise:
        size = ""
        if d.raise_size_as_multiple_of_bb is not None:
            size = f"_{d.raise_size_as_multiple_of_bb:.1f}BB"
        return f"{pos}_VS{size}_OPEN"
    if d.facing_limp:
        return f"{pos}_VS_LIMP"
    if d.unopened_pot:
        return f"{pos}_UNOPENED"
    return f"{pos}_UNKNOWN"


# ---------------------------------------------------------------------------
# Action selectors
# ---------------------------------------------------------------------------

def _find_action(
    legal: List[LegalActionOption], at: ActionType
) -> Optional[LegalActionOption]:
    for a in legal:
        if a.action_type == at:
            return a
    return None


def _find_jam(legal: List[LegalActionOption]) -> Optional[LegalActionOption]:
    raises = [a for a in legal if a.action_type == ActionType.RAISE]
    if not raises:
        return None
    return max(raises, key=lambda a: a.raise_to_bb or 0)


def _find_best_raise(
    legal: List[LegalActionOption], target: float
) -> Optional[LegalActionOption]:
    raises = [a for a in legal if a.action_type == ActionType.RAISE]
    if not raises:
        return None
    return min(raises, key=lambda a: abs((a.raise_to_bb or 0) - target))


def _pick_raise_size(
    legal: List[LegalActionOption],
    d: DerivedState,
    hf: HandFeatures,
) -> Optional[LegalActionOption]:
    """Select the best raise-to amount from legal raise options."""
    raises = [a for a in legal if a.action_type == ActionType.RAISE]
    if not raises:
        return None

    if d.stack_depth_bucket in (
        StackDepthBucket.ULTRA_SHORT, StackDepthBucket.VERY_SHORT
    ):
        return _find_jam(legal)

    if d.unopened_pot:
        return _find_best_raise(legal, 2.5)

    if d.facing_limp:
        return _find_best_raise(legal, 3.5)

    if d.facing_open_raise:
        target = (d.raise_size_in_bb or 2.5) * 3.0
        return _find_best_raise(legal, target)

    if d.facing_3bet:
        target = (d.raise_size_in_bb or 7.5) * 2.5
        return _find_best_raise(legal, target)

    if d.facing_4bet:
        return _find_jam(legal)

    return _find_best_raise(legal, 2.5)


# ---------------------------------------------------------------------------
# Decision builder
# ---------------------------------------------------------------------------

def _build_decision(
    legal: List[LegalActionOption],
    action: LegalActionOption,
    explanation: str,
    hf: HandFeatures,
    d: Optional[DerivedState],
    ctx: str,
    rule_id: str,
) -> Decision:
    raise_bucket = None
    if action.action_type == ActionType.RAISE and action.raise_to_bb and d:
        if d.raise_size_in_bb and d.raise_size_in_bb > 0:
            raise_bucket = f"{action.raise_to_bb / d.raise_size_in_bb:.1f}x_prev"
        else:
            raise_bucket = f"{action.raise_to_bb}bb"

    debug = {
        "hand_class_label": hf.hand_class_label,
        "hand_bucket": hf.hand_bucket.value,
        "stack_depth_bucket": d.stack_depth_bucket.value if d else "UNKNOWN",
        "action_context_label": ctx,
        "raise_size_bucket": raise_bucket,
        "baseline_rule_id": rule_id,
        "legal_actions": [repr(a) for a in legal],
        "recommended_action": repr(action),
        "explanation": explanation,
    }
    return Decision(
        legal_actions=legal,
        recommended_action=action,
        explanation=explanation,
        debug=debug,
    )


# ---------------------------------------------------------------------------
# Main recommender
# ---------------------------------------------------------------------------

def recommend_preflop_action(state: PokerState) -> Decision:
    """Return the baseline preflop recommendation for HERO."""
    validate_preflop_state(state)

    legal = legal_actions_for_hero(state)
    hf = hand_features(state.hero_hole_cards)
    d = state.derived
    is_btn = state.hero_position == Position.BTN_SB
    ctx = _context_label(d, state.hero_position) if d else "UNKNOWN"

    if not legal:
        return Decision(
            legal_actions=[],
            recommended_action=LegalActionOption(action_type=ActionType.FOLD),
            explanation="No legal actions available -- hand is over or round is closed.",
            debug={
                "hand_class_label": hf.hand_class_label,
                "stack_depth_bucket": d.stack_depth_bucket.value if d else "UNKNOWN",
                "action_context_label": ctx,
                "baseline_rule_id": "NO_ACTION",
                "legal_actions": [],
                "recommended_action": "NONE",
                "explanation": "No legal actions.",
            },
        )

    # --- Chart lookup ---
    chart_ctx = _determine_context(d, state.hero_position)
    label = hf.hand_class_label

    facing_raise = d.facing_open_raise or d.facing_3bet or d.facing_4bet
    chart_action = get_chart_action(
        label, chart_ctx, d.stack_depth_bucket, facing_raise=facing_raise,
    )

    if chart_action == "RAISE":
        raise_opt = _pick_raise_size(legal, d, hf)
        if raise_opt:
            explanation = (
                f"Chart raise {label} in {chart_ctx} "
                f"({d.stack_depth_bucket.value})."
            )
            return _build_decision(
                legal, raise_opt, explanation, hf, d, ctx, "CHART_RAISE",
            )
        call = _find_action(legal, ActionType.CALL)
        if call:
            explanation = (
                f"Chart raise {label} but no raise available -- calling."
            )
            return _build_decision(
                legal, call, explanation, hf, d, ctx, "CHART_RAISE_FALLBACK_CALL",
            )

    if chart_action == "CALL":
        call = _find_action(legal, ActionType.CALL)
        if call:
            explanation = (
                f"Chart call {label} in {chart_ctx} "
                f"({d.stack_depth_bucket.value})."
            )
            return _build_decision(
                legal, call, explanation, hf, d, ctx, "CHART_CALL",
            )
        check = _find_action(legal, ActionType.CHECK)
        if check:
            explanation = f"Chart call {label} -- checking (no bet to call)."
            return _build_decision(
                legal, check, explanation, hf, d, ctx, "CHART_CHECK",
            )

    if chart_action == "CHECK":
        check = _find_action(legal, ActionType.CHECK)
        if check:
            explanation = f"Chart check {label} in {chart_ctx}."
            return _build_decision(
                legal, check, explanation, hf, d, ctx, "CHART_CHECK",
            )

    # FOLD (or fallback)
    fold = _find_action(legal, ActionType.FOLD)
    check = _find_action(legal, ActionType.CHECK)
    if fold:
        explanation = (
            f"Chart fold {label} in {chart_ctx} "
            f"({d.stack_depth_bucket.value})."
        )
        return _build_decision(
            legal, fold, explanation, hf, d, ctx, "CHART_FOLD",
        )
    if check:
        explanation = f"Chart fold {label} -- checking (no fold option)."
        return _build_decision(
            legal, check, explanation, hf, d, ctx, "CHART_CHECK_FREE",
        )

    return _build_decision(
        legal, legal[0],
        f"Fallback action with {label}.", hf, d, ctx, "FALLBACK",
    )
