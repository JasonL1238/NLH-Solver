#!/usr/bin/env python3
"""Run all scenarios defined in scenarios.py and print a results table.

Usage (from project root):
    python -m scenario_runner.run
"""

from baseline_preflop.parser import (
    bb_vs_4bet_decision,
    bb_vs_limp_decision,
    bb_vs_open_decision,
    btn_vs_3bet_decision,
    btn_vs_iso_after_limp_decision,
    unopened_btn_decision,
)
from baseline_preflop.recommender import recommend_preflop_action
from scenario_runner.scenarios import SCENARIOS


_BUILDERS = {
    "btn_open":    lambda kw: unopened_btn_decision(kw["hero_cards"], kw["stack"]),
    "bb_vs_limp":  lambda kw: bb_vs_limp_decision(kw["hero_cards"], kw["stack"]),
    "bb_vs_open":  lambda kw: bb_vs_open_decision(kw["hero_cards"], kw["open"], kw["stack"]),
    "btn_vs_iso":  lambda kw: btn_vs_iso_after_limp_decision(kw["hero_cards"], kw["raise_to"], kw["stack"]),
    "btn_vs_3bet": lambda kw: btn_vs_3bet_decision(kw["hero_cards"], kw["open"], kw["threeb"], kw["stack"]),
    "bb_vs_4bet":  lambda kw: bb_vs_4bet_decision(kw["hero_cards"], kw["open"], kw["threeb"], kw["fourb"], kw["stack"]),
}


def _action_str(action) -> str:
    if action.raise_to_bb is not None:
        return f"RAISE to {action.raise_to_bb}bb"
    if action.call_amount_bb is not None:
        return f"CALL {action.call_amount_bb}bb"
    return action.action_type.value


def run():
    hdr = f"{'#':>3}  {'Scenario':<32} {'Hand':>5} {'Stack':>6} {'Bucket':<12} {'Context':<24} {'Action':<20} Rule ID"
    sep = "-" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)

    for i, (label, helper, kwargs) in enumerate(SCENARIOS, 1):
        builder = _BUILDERS.get(helper)
        if builder is None:
            print(f"{i:>3}  {label:<32}  *** UNKNOWN HELPER: {helper} ***")
            continue
        try:
            state = builder(kwargs)
            dec = recommend_preflop_action(state)
            d = dec.debug
            bucket = d.get("stack_depth_bucket", "?")
            ctx = d.get("action_context_label", "?")
            hand = d.get("hand_class_label", "?")
            rule = d.get("baseline_rule_id", "?")
            act = _action_str(dec.recommended_action)
            print(f"{i:>3}  {label:<32} {hand:>5} {state.effective_stack_bb:>5.0f}bb {bucket:<12} {ctx:<24} {act:<20} {rule}")
        except Exception as e:
            print(f"{i:>3}  {label:<32}  *** ERROR: {e} ***")

    print(sep)
    print(f"  {len(SCENARIOS)} scenarios complete.")


if __name__ == "__main__":
    run()
