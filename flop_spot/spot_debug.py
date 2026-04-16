"""Structured debug dict for flop spots (exploit + UI).

Independent of any rule-based recommender; used by EV policy and exploit layer.
"""

from __future__ import annotations

from typing import Any, Dict, List

from poker_core.models import HandState, LegalAction, Street

from .classification import classify_board, classify_hand
from .context import derive_flop_context


def build_spot_debug(
    state: HandState,
    *,
    policy_rule_id: str,
    explanation: str,
    legal: List[LegalAction],
    recommended_action_repr: str,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Mirror legacy recommender debug keys for ``FlopDecision.debug``."""
    from postflop_policy.context_labels import postflop_action_context_label

    hc = state.config.hero_hole_cards
    board = list(state.board_cards)
    ctx = derive_flop_context(state)

    if len(board) >= 3:
        classify_board_cards = board[:3] if len(board) > 3 else board
    else:
        classify_board_cards = board

    hand = classify_hand(hc, classify_board_cards)
    board_ft = classify_board(classify_board_cards)

    if state.current_street == Street.FLOP:
        action_ctx = ctx.flop_context.value
    elif state.current_street in (Street.TURN, Street.RIVER):
        action_ctx = postflop_action_context_label(state)
    else:
        action_ctx = ctx.flop_context.value

    out: Dict[str, Any] = {
        "action_context_label": action_ctx,
        "hero_position_relation_on_flop": "IP" if ctx.hero_is_ip else "OOP",
        "hero_preflop_role": "PFR" if ctx.hero_is_pfr else "PFC",
        "made_hand_category": hand.made_hand.value,
        "draw_category": hand.draw.value,
        "board_texture_label": board_ft.texture.value,
        "flop_bet_size_bucket": ctx.bet_size_bucket.value if ctx.bet_size_bucket else None,
        "spr_bucket": ctx.spr_bucket.value,
        "spr": ctx.spr,
        "baseline_rule_id": policy_rule_id,
        "policy_rule_id": policy_rule_id,
        "legal_actions": [repr(a) for a in legal],
        "recommended_action": recommended_action_repr,
        "explanation": explanation,
        "has_showdown_value": hand.has_showdown_value,
        "has_strong_draw": hand.has_strong_draw,
        "has_combo_draw": hand.has_combo_draw,
        "has_backdoor_equity": hand.has_backdoor_equity,
        "overcards_to_board_count": hand.overcards_to_board_count,
        "current_street": state.current_street.value,
        "board_cards_count": len(board),
        "classification_uses_flop_subset": len(board) > 3,
    }
    if len(board) > 3:
        out["classification_note"] = (
            "v1: hand/board texture uses first three board cards only "
            "(flop-spot classifier); not full turn/river texture."
        )
    if extra:
        out.update(extra)
    return out
