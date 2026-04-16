"""Chart-based preflop recommendation engine for heads-up NLHE.

Replaces the earlier score-based system with explicit hand chart lookups.
Charts live in charts.py and define RAISE / CALL / FOLD sets per context.
"""

from __future__ import annotations

from typing import List, Optional

from .charts import ALL_HANDS, get_chart_action, get_chart_sets, hand_strength
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

_STANDARD_MDF = {
    # Charts assume standard sizing per context.
    # MDF = pot_before_raise / (pot_before_raise + raise_to_bb)
    "BB_VS_OPEN": 1.5 / (1.5 + 2.5),      # 0.375
    "BTN_VS_ISO": 2.0 / (2.0 + 3.5),      # ~0.364
    "BTN_VS_3BET": 3.5 / (3.5 + 8.0),     # ~0.304
    "BB_VS_4BET": 10.5 / (10.5 + 20.0),   # ~0.344
}

_PREMIUM_IMMUNE = frozenset({"AA", "KK", "QQ", "AKs"})


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
# MDF / price sensitivity helpers
# ---------------------------------------------------------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _last_aggressive_record(state: PokerState):
    for rec in reversed(state.action_history):
        if rec.action_type in (ActionType.RAISE, ActionType.BET):
            return rec
    return None


def _compute_defense_scalar(
    state: PokerState,
    d: DerivedState,
    chart_ctx: str,
) -> tuple[float, float | None]:
    """Return (defense_scalar, actual_mdf_or_None)."""
    standard_mdf = _STANDARD_MDF.get(chart_ctx)
    if standard_mdf is None:
        return 1.0, None
    if not (d.facing_open_raise or d.facing_3bet or d.facing_4bet):
        return 1.0, None
    if d.raise_size_in_bb is None or d.raise_size_in_bb <= 0:
        return 1.0, None

    last_aggr = _last_aggressive_record(state)
    if last_aggr is None:
        return 1.0, None

    pot_before_raise = state.pot_size_bb - last_aggr.amount_added_bb
    if pot_before_raise <= 0:
        return 1.0, None

    actual_mdf = pot_before_raise / (pot_before_raise + d.raise_size_in_bb)
    defense_scalar = actual_mdf / standard_mdf
    defense_scalar = _clamp(defense_scalar, 0.4, 1.5)
    return defense_scalar, actual_mdf


def _percentile_in_set(label: str, hand_set: frozenset[str]) -> float:
    """Fraction of the set weaker than label (0.0=weakest, 1.0=strongest)."""
    if not hand_set:
        return 0.0
    s = hand_strength(label)
    below = sum(1 for h in hand_set if hand_strength(h) < s)
    return below / len(hand_set)


def _apply_mdf_filter(
    *,
    label: str,
    chart_ctx: str,
    chart_action: str,
    defense_scalar: float,
) -> tuple[str, str]:
    """Return (new_action, rule_suffix) after MDF-based range scaling."""
    if label in _PREMIUM_IMMUNE:
        return chart_action, "MDF_PREMIUM_IMMUNE"

    raise_set, call_set = get_chart_sets(chart_ctx)
    if not raise_set and not call_set:
        return chart_action, "MDF_NO_CONTEXT"

    # Tighten when villain sizes up (scalar < 1.0)
    if defense_scalar < 1.0 - 1e-9:
        cutoff = 1.0 - defense_scalar  # fold bottom cutoff fraction
        if chart_action == "CALL" and call_set is not None:
            pct = _percentile_in_set(label, call_set)
            if pct < cutoff:
                return "FOLD", "MDF_TIGHTEN_FOLD_FROM_CALL"
        if chart_action == "RAISE":
            pct = _percentile_in_set(label, raise_set)
            if pct < cutoff:
                return "FOLD", "MDF_TIGHTEN_FOLD_FROM_RAISE"
        return chart_action, "MDF_TIGHTEN_NO_CHANGE"

    # Optional widening when villain sizes down (scalar > 1.1)
    if defense_scalar > 1.1 and chart_action == "FOLD" and call_set is not None:
        upgrade = min(defense_scalar - 1.0, 0.5)
        fold_set = frozenset(ALL_HANDS - raise_set - call_set)
        pct = _percentile_in_set(label, fold_set)
        if pct >= 1.0 - upgrade:
            return "CALL", "MDF_WIDEN_CALL_FROM_FOLD"
        return chart_action, "MDF_WIDEN_NO_CHANGE"

    return chart_action, "MDF_NEUTRAL"


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
    extra_debug: Optional[dict] = None,
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
    if extra_debug:
        debug.update(extra_debug)
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

    defense_scalar, actual_mdf = _compute_defense_scalar(state, d, chart_ctx)
    filtered_action, mdf_rule = _apply_mdf_filter(
        label=label,
        chart_ctx=chart_ctx,
        chart_action=chart_action,
        defense_scalar=defense_scalar,
    )
    extra_debug = {
        "defense_scalar": round(defense_scalar, 4),
        "actual_mdf": round(actual_mdf, 4) if actual_mdf is not None else None,
        "standard_mdf": round(_STANDARD_MDF.get(chart_ctx, 0.0), 4) if chart_ctx in _STANDARD_MDF else None,
        "mdf_rule": mdf_rule,
        "chart_context": chart_ctx,
        "chart_action_raw": chart_action,
        "chart_action_filtered": filtered_action,
    }
    chart_action = filtered_action

    if chart_action == "RAISE":
        raise_opt = _pick_raise_size(legal, d, hf)
        if raise_opt:
            explanation = (
                f"Chart raise {label} in {chart_ctx} "
                f"({d.stack_depth_bucket.value})."
            )
            return _build_decision(
                legal, raise_opt, explanation, hf, d, ctx, "CHART_RAISE", extra_debug,
            )
        call = _find_action(legal, ActionType.CALL)
        if call:
            explanation = (
                f"Chart raise {label} but no raise available -- calling."
            )
            return _build_decision(
                legal, call, explanation, hf, d, ctx, "CHART_RAISE_FALLBACK_CALL", extra_debug,
            )

    if chart_action == "CALL":
        call = _find_action(legal, ActionType.CALL)
        if call:
            explanation = (
                f"Chart call {label} in {chart_ctx} "
                f"({d.stack_depth_bucket.value})."
            )
            return _build_decision(
                legal, call, explanation, hf, d, ctx, "CHART_CALL", extra_debug,
            )
        check = _find_action(legal, ActionType.CHECK)
        if check:
            explanation = f"Chart call {label} -- checking (no bet to call)."
            return _build_decision(
                legal, check, explanation, hf, d, ctx, "CHART_CHECK", extra_debug,
            )

    if chart_action == "CHECK":
        check = _find_action(legal, ActionType.CHECK)
        if check:
            explanation = f"Chart check {label} in {chart_ctx}."
            return _build_decision(
                legal, check, explanation, hf, d, ctx, "CHART_CHECK", extra_debug,
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
            legal, fold, explanation, hf, d, ctx, "CHART_FOLD", extra_debug,
        )
    if check:
        explanation = f"Chart fold {label} -- checking (no fold option)."
        return _build_decision(
            legal, check, explanation, hf, d, ctx, "CHART_CHECK_FREE", extra_debug,
        )

    return _build_decision(
        legal, legal[0],
        f"Fallback action with {label}.", hf, d, ctx, "FALLBACK", extra_debug,
    )
