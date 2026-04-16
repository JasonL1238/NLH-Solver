"""Compact human-readable state summaries for manual inspection."""

from __future__ import annotations

from .legal_actions import legal_actions
from .models import HandState, Player


def format_state(state: HandState) -> str:
    """Return a compact multi-line summary of the hand state."""
    lines: list[str] = []
    lines.append("=" * 55)
    lines.append("HAND STATE")
    lines.append("=" * 55)

    cfg = state.config
    lines.append(f"  Hero position:   {cfg.hero_position.value}")
    lines.append(f"  Hero cards:      {cfg.hero_hole_cards}")
    lines.append(f"  Eff. stack:      {cfg.effective_stack_bb} bb")
    lines.append(f"  Street:          {state.current_street.value}")

    board_str = " ".join(repr(c) for c in state.board_cards) if state.board_cards else "—"
    lines.append(f"  Board:           {board_str}")

    lines.append(f"  Pot:             {state.pot_size_bb:.2f} bb")
    lines.append(f"  Hero contrib:    {state.hero_contribution_bb:.2f} bb")
    lines.append(f"  Villain contrib: {state.villain_contribution_bb:.2f} bb")
    lines.append(f"  To call:         {state.current_bet_to_call_bb:.2f} bb")
    lines.append(f"  Current actor:   {state.current_actor.value if state.current_actor else '—'}")

    agg = state.last_aggressor.value if state.last_aggressor else "—"
    lines.append(f"  Last aggressor:  {agg}")
    lines.append(f"  Raises (street): {state.number_of_raises_this_street}")
    lines.append(f"  Betting closed:  {state.betting_round_closed}")

    ai_parts: list[str] = []
    if state.hero_all_in:
        ai_parts.append("HERO")
    if state.villain_all_in:
        ai_parts.append("VILLAIN")
    lines.append(f"  All-in:          {', '.join(ai_parts) if ai_parts else '—'}")

    lines.append(f"  Showdown ready:  {state.showdown_ready}")
    lines.append(f"  Awaiting runout: {state.awaiting_runout}")
    lines.append(f"  Hand over:       {state.hand_over}")

    if state.fold_winner:
        lines.append(f"  Fold winner:     {state.fold_winner.value}")

    # Legal actions
    la = state.legal_actions_list
    if la is None:
        la = legal_actions(state)
    lines.append("")
    if la:
        lines.append("  Legal actions:")
        for a in la:
            lines.append(f"    - {a!r}")
    else:
        lines.append("  Legal actions:   (none)")

    lines.append("=" * 55)
    return "\n".join(lines)


def print_state(state: HandState) -> None:
    """Print the compact state summary to stdout."""
    print(format_state(state))
