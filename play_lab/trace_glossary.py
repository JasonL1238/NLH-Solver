"""Short definitions for metrics shown in Play Lab decision traces.

Keys are stable IDs referenced from ``decision_trace.TraceMetric.glossary_key``.
"""

from __future__ import annotations

from typing import Dict

GLOSSARY: Dict[str, str] = {
    "effective_stack_bb": (
        "**Effective stack (bb)** is the smaller of the two players’ starting stacks "
        "for this hand, in big blinds. Preflop charts and postflop SPR use this depth."
    ),
    "pot_bb": (
        "**Pot (bb)** is the current pot size in big blinds before your decision, "
        "derived from the action history (canonical `poker_core` state)."
    ),
    "to_call_bb": (
        "**To call (bb)** is how many more big blinds you must put in to continue "
        "if you choose call (or the call portion of a raise). Zero means you can check."
    ),
    "hero_position": (
        "**Hero position** is the engine’s seat: `BTN_SB` (button + small blind, acts first "
        "preflop) or `BB` (big blind, acts second preflop). Charts are different per seat."
    ),
    "hand_class_label": (
        "**Chart label** is the canonical two-card label used by the chart tables "
        "(e.g. `AKs`, `98o`). It is derived from exact hole cards."
    ),
    "hand_bucket": (
        "**Fine bucket** is an internal strength/shape category (A–N) used for diagnostics; "
        "the baseline chart still keys off the chart label."
    ),
    "stack_depth_bucket": (
        "**Stack depth bucket** groups effective stack into bands (e.g. SHORT, MEDIUM, DEEP). "
        "The chart lookup uses the bucket to pick which cell applies."
    ),
    "derived_facing": (
        "**Spot shape:** the boolean rows together describe where you are in the preflop "
        "line (unopened, vs limp, vs open, vs 3-bet, vs 4-bet, all-in). Exactly one primary "
        "\"facing\" pattern should read as true at a decision node; that pattern selects the "
        "**chart context** (BTN open, BB defend, etc.). Sizes like last raise (bb) feed MDF math."
    ),
    "raise_size_in_bb": (
        "**Last raise size (bb)** is how many big blinds the last aggressive raise added "
        "to the pot from the raiser’s perspective; used for MDF vs actual price."
    ),
    "chart_context": (
        "**Chart context** selects which static RAISE/CALL/FOLD chart slice we read "
        "(e.g. `BB_VS_OPEN`, `BTN_VS_3BET`). It is computed only from derived flags + seat."
    ),
    "chart_action_raw": (
        "**Raw chart action** is RAISE/CALL/FOLD from the chart before any MDF-based "
        "range tightening or widening."
    ),
    "chart_action_filtered": (
        "**After MDF filter** is the chart action after adjusting for villain’s actual "
        "raise size vs the chart’s assumed standard size."
    ),
    "mdf_rule": (
        "**MDF rule id** names which branch of the defense filter ran "
        "(e.g. tighten from call, widen from fold, premium immune)."
    ),
    "standard_mdf": (
        "**Standard MDF** (minimum defense frequency) implied by the chart’s assumed "
        "standard sizing for this context: roughly `pot_before / (pot_before + standard_raise)`."
    ),
    "actual_mdf": (
        "**Actual MDF** uses the same pot-odds idea but with the **real** last raise size: "
        "`pot_before / (pot_before + raise_to_bb)`. Bigger raises → lower MDF → defend less."
    ),
    "defense_scalar": (
        "**Defense scalar** is `actual_mdf / standard_mdf` (then clamped). "
        "Below 1.0 means villain raised larger than the chart standard → we fold more marginal "
        "combos; above 1.0 can slightly widen weak folds."
    ),
    "raise_size_bucket": (
        "**Size bucket on pick** describes how the engine chose among legal raise sizes "
        "(e.g. multiple of villain’s raise, target open size, or jam when very short)."
    ),
    "baseline_rule_id": (
        "**Baseline rule id** identifies which code path produced the pick "
        "(e.g. `CHART_CALL`, `CHART_RAISE`, `EV_ARGMAX`)."
    ),
    "recommended_action": (
        "**Recommended action** is the concrete legal choice (including raise-to amount when applicable)."
    ),
    "flop_board": (
        "**Board** lists the three flop cards. Texture and hand classification depend on these."
    ),
    "facing_bet": (
        "**Facing a bet** is true when you must put chips in to continue (to call > 0)."
    ),
    "pot_odds_threshold": (
        "**Pot odds** here is `to_call / (pot + to_call)`: the fraction of the final pot "
        "you invest with a call. Used inside the EV builder for thin raises / continue thresholds."
    ),
    "flop_context": (
        "**Flop context** encodes the line (e.g. PFR OOP first to act, facing medium bet). "
        "It drives response-model parameters in the EV estimate."
    ),
    "hero_ip_oop": (
        "**IP / OOP** is in-position vs out-of-position on the flop: who acts last on later streets."
    ),
    "hero_preflop_role": (
        "**PFR / PFC** is preflop raiser vs caller (who had the last aggressive raise preflop)."
    ),
    "spr": (
        "**SPR** (stack-to-pot ratio) is effective stack behind divided by the current pot; "
        "low SPR favors committing with draws and strong made hands."
    ),
    "spr_bucket": (
        "**SPR bucket** discretizes SPR for policy (e.g. shallow vs deep)."
    ),
    "flop_bet_size_bucket": (
        "**Facing bet size bucket** classifies villain’s bet relative to the pot for this node."
    ),
    "board_texture": (
        "**Board texture** summarizes connectivity and monotone risk (dry, wet, paired, etc.)."
    ),
    "made_hand_category": (
        "**Made hand** classifies hero’s pair/straight/flush strength relative to the board."
    ),
    "draw_category": (
        "**Draw** lists primary draws (flush draw, OESD, gutshot, combo, or none)."
    ),
    "villain_range_summary": (
        "**Villain range** is a weighted set of hands used for Monte Carlo equity. "
        "The summary string describes how that range was built for this spot."
    ),
    "monte_carlo_samples": (
        "**Monte Carlo samples** is how many random runouts were simulated. "
        "More samples → less noise in equity, slower compute."
    ),
    "equity_estimate": (
        "**Equity estimate** is approximate P(hero wins or splits) vs the villain range on this flop, "
        "from repeated random completions of the board and showdowns."
    ),
    "win_rate": (
        "**Win rate** is the fraction of MC trials where hero scoops the whole pot (ties excluded)."
    ),
    "tie_rate": (
        "**Tie rate** is the fraction of MC trials where hero splits the pot with at least one opponent."
    ),
    "ev_best": (
        "**Best EV (bb)** is the highest expected-value estimate among candidate lines, "
        "in big blinds, under the simplified response model."
    ),
    "policy_rule_id": (
        "**Policy rule** names the decision rule (`EV_ARGMAX` = pick highest-EV candidate)."
    ),
    "street_policy_version": (
        "**Config version** identifies which policy parameter set / version string was used."
    ),
}
