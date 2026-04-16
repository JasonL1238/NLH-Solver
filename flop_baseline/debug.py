"""Human-readable flop decision output."""

from __future__ import annotations

from poker_core.models import HandState

from flop_spot.models import FlopDecision


def pretty_print_flop_decision(
    decision: FlopDecision,
    state: HandState,
) -> str:
    """Return a compact multi-line summary of a flop decision."""
    d = decision.debug
    hc = state.config.hero_hole_cards
    board = state.board_cards

    hero_str = f"{hc.high.rank}{hc.high.suit} {hc.low.rank}{hc.low.suit}"
    board_str = " ".join(f"{c.rank}{c.suit}" for c in board)

    lines = [
        "=" * 50,
        f"  FLOP DECISION",
        "=" * 50,
        f"  Hero hand   : {hero_str}",
        f"  Board       : {board_str}",
        f"  Pot (bb)    : {state.pot_size_bb}",
        f"  To call (bb): {state.current_bet_to_call_bb}",
        "-" * 50,
        f"  Position    : {d.get('hero_position_relation_on_flop', '?')}",
        f"  PF role     : {d.get('hero_preflop_role', '?')}",
        f"  Context     : {d.get('action_context_label', '?')}",
        f"  SPR         : {d.get('spr', '?')} ({d.get('spr_bucket', '?')})",
        f"  Bet bucket  : {d.get('flop_bet_size_bucket', '-')}",
        "-" * 50,
        f"  Made hand   : {d.get('made_hand_category', '?')}",
        f"  Draw        : {d.get('draw_category', '?')}",
        f"  SDV?        : {d.get('has_showdown_value', '?')}",
        f"  Strong draw?: {d.get('has_strong_draw', '?')}",
        f"  Combo draw? : {d.get('has_combo_draw', '?')}",
        f"  Backdoor?   : {d.get('has_backdoor_equity', '?')}",
        f"  Overcards   : {d.get('overcards_to_board_count', '?')}",
        "-" * 50,
        f"  Rule ID     : {d.get('baseline_rule_id', '?')}",
        f"  Action      : {d.get('recommended_action', '?')}",
        f"  Explanation : {d.get('explanation', '?')}",
        "-" * 50,
        f"  Legal acts  : {', '.join(d.get('legal_actions', []))}",
        "=" * 50,
    ]
    return "\n".join(lines)
