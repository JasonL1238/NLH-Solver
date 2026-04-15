"""Debug printing helpers for preflop decisions."""

from __future__ import annotations

from .models import Decision, PokerState


def pretty_print_decision(decision: Decision, state: PokerState | None = None) -> str:
    """Return a compact multi-line summary of a preflop decision."""
    lines = []
    d = decision.debug

    lines.append("=" * 50)
    lines.append("PREFLOP DECISION SUMMARY")
    lines.append("=" * 50)

    if state:
        lines.append(f"  Hero cards:    {state.hero_hole_cards}")
        lines.append(f"  Position:      {state.hero_position.value}")
        lines.append(f"  Eff. stack:    {state.effective_stack_bb}bb")
        lines.append(f"  Pot:           {state.pot_size_bb}bb")
        lines.append(f"  To call:       {state.current_bet_to_call_bb}bb")
        lines.append("")

    lines.append(f"  Hand class:    {d.get('hand_class_label', '?')}")
    lines.append(f"  Hand bucket:   {d.get('hand_bucket', '?')}")
    lines.append(f"  Stack bucket:  {d.get('stack_depth_bucket', '?')}")
    lines.append(f"  Context:       {d.get('action_context_label', '?')}")
    if d.get("raise_size_bucket"):
        lines.append(f"  Raise bucket:  {d['raise_size_bucket']}")
    lines.append(f"  Rule ID:       {d.get('baseline_rule_id', '?')}")
    lines.append("")

    lines.append("  Legal actions:")
    for a in decision.legal_actions:
        marker = " <<" if a == decision.recommended_action else ""
        lines.append(f"    - {a!r}{marker}")

    lines.append("")
    lines.append(f"  >>> {decision.recommended_action!r}")
    lines.append(f"  {decision.explanation}")
    lines.append("=" * 50)

    return "\n".join(lines)


def print_decision(decision: Decision, state: PokerState | None = None) -> None:
    """Print a compact decision summary to stdout."""
    print(pretty_print_decision(decision, state))
