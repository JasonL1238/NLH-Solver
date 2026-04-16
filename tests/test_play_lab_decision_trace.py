"""Tests for play_lab.decision_trace (no Streamlit)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from baseline_preflop.models import ActionType, Decision, LegalActionOption, Position
from poker_core.parser import parse_card
from play_lab.decision_trace import (
    flop_trace_sections,
    postflop_ev_trace_steps,
    preflop_trace_sections,
    preflop_trace_steps,
)
from play_lab.deck import validate_flop_input


def test_preflop_trace_no_action() -> None:
    dec = Decision(
        legal_actions=[],
        recommended_action=LegalActionOption(ActionType.FOLD),
        explanation="No legal actions.",
        debug={
            "baseline_rule_id": "NO_ACTION",
            "explanation": "No legal actions.",
            "hand_class_label": "AKs",
            "stack_depth_bucket": "DEEP",
            "action_context_label": "BTN_SB_UNOPENED",
        },
    )
    secs = preflop_trace_sections(dec, None)
    assert len(secs) == 1
    assert "No recommendation" in secs[0][0]
    assert "NO_ACTION" in secs[0][1] or "Reason" in secs[0][1]


def test_preflop_trace_steps_has_glossary_metrics() -> None:
    call = LegalActionOption(ActionType.CALL, call_amount_bb=1.5)
    dec = Decision(
        legal_actions=[call],
        recommended_action=call,
        explanation="ok",
        debug={
            "hand_class_label": "AKs",
            "hand_bucket": "B_SUITED_BIG_BROADWAY",
            "stack_depth_bucket": "DEEP",
            "action_context_label": "X",
            "baseline_rule_id": "CHART_CALL",
            "legal_actions": ["CALL(1.5bb)"],
            "recommended_action": "CALL(1.5bb)",
            "explanation": "ok",
            "defense_scalar": 1.0,
            "actual_mdf": 0.37,
            "standard_mdf": 0.375,
            "mdf_rule": "MDF_NEUTRAL",
            "chart_context": "BB_VS_OPEN",
            "chart_action_raw": "CALL",
            "chart_action_filtered": "CALL",
        },
    )
    steps = preflop_trace_steps(dec, None)
    mdf = next(s for s in steps if s.title == "MDF vs sizing (math)")
    keys = [m.glossary_key for m in mdf.metrics if m.glossary_key]
    assert "standard_mdf" in keys and "defense_scalar" in keys


def test_validate_flop_input() -> None:
    assert not validate_flop_input("")[0]
    assert not validate_flop_input("  ")[0]
    assert not validate_flop_input("Ah 7c")[0]
    ok, msg = validate_flop_input("Ah 7c 2d")
    assert ok and msg == ""


def test_preflop_trace_full_path() -> None:
    call = LegalActionOption(ActionType.CALL, call_amount_bb=1.5)
    dec = Decision(
        legal_actions=[call],
        recommended_action=call,
        explanation="Chart call 98o in BB_VS_OPEN (MEDIUM).",
        debug={
            "hand_class_label": "98o",
            "hand_bucket": "N_WEAK_OFFSUIT_TRASH",
            "stack_depth_bucket": "MEDIUM",
            "action_context_label": "BB_VS_2.5BB_OPEN",
            "raise_size_bucket": None,
            "baseline_rule_id": "CHART_CALL",
            "legal_actions": ["CALL(1.5bb)"],
            "recommended_action": "CALL(1.5bb)",
            "explanation": "Chart call 98o in BB_VS_OPEN (MEDIUM).",
            "defense_scalar": 0.92,
            "actual_mdf": 0.35,
            "standard_mdf": 0.375,
            "mdf_rule": "MDF_TIGHTEN_NO_CHANGE",
            "chart_context": "BB_VS_OPEN",
            "chart_action_raw": "CALL",
            "chart_action_filtered": "CALL",
        },
    )
    d = SimpleNamespace(
        stack_depth_bucket=SimpleNamespace(value="MEDIUM"),
        unopened_pot=False,
        facing_limp=False,
        facing_open_raise=True,
        facing_3bet=False,
        facing_4bet=False,
        facing_all_in=False,
        raise_size_in_bb=2.5,
        raise_size_as_multiple_of_bb=1.0,
        raise_size_as_multiple_of_previous_bet=None,
        stack_to_open_ratio=40.0,
        stack_to_3bet_ratio=None,
    )
    st = SimpleNamespace(
        effective_stack_bb=100.0,
        pot_size_bb=3.5,
        current_bet_to_call_bb=1.5,
        hero_position=Position.BB,
        derived=d,
    )
    secs = preflop_trace_sections(dec, st)
    titles = [s[0] for s in secs]
    assert "Table snapshot" in titles
    assert "Derived spot (from action history)" in titles
    assert "MDF vs sizing (math)" in titles
    blob = "\n".join(b for _, b in secs)
    assert "98o" in blob
    assert "defense_scalar" in blob.lower() or "0.92" in blob


def test_postflop_ev_trace_prefixes_street() -> None:
    st = SimpleNamespace(
        config=SimpleNamespace(effective_stack_bb=80.0),
        pot_size_bb=10.0,
        current_bet_to_call_bb=0.0,
        board_cards=[parse_card("Ah"), parse_card("7c"), parse_card("2d"), parse_card("9s")],
    )
    flop_dec = SimpleNamespace(
        debug={"villain_range_summary": "x", "samples_used": 10, "equity_estimate": 0.5},
        explanation="ok",
    )
    steps = postflop_ev_trace_steps(st, flop_dec, street_label="TURN")
    assert steps[0].title.startswith("Table snapshot")
    assert "TURN" in steps[0].title


def test_flop_trace_sorts_candidates_by_ev() -> None:
    st = SimpleNamespace(
        config=SimpleNamespace(effective_stack_bb=80.0),
        pot_size_bb=10.0,
        current_bet_to_call_bb=5.0,
        board_cards=[],
    )
    flop_dec = SimpleNamespace(
        debug={
            "action_context_label": "FACING_MEDIUM_BET",
            "hero_position_relation_on_flop": "OOP",
            "hero_preflop_role": "PFR",
            "made_hand_category": "TOP_PAIR_STRONG_KICKER",
            "draw_category": "NO_REAL_DRAW",
            "board_texture_label": "TWO_TONE_BOARD",
            "flop_bet_size_bucket": "MEDIUM",
            "spr_bucket": "DEEP",
            "spr": 6.2,
            "baseline_rule_id": "EV_ARGMAX",
            "villain_range_summary": "synthetic test range",
            "samples_used": 100,
            "equity_estimate": 0.55,
            "win_rate": 0.52,
            "tie_rate": 0.06,
            "pot_odds_threshold": 0.3333,
            "street_policy_version": "test",
            "ev_best": 1.25,
            "ev_candidates": [
                {"repr": "CHECK", "ev": 0.5, "note": "a"},
                {"repr": "CALL", "ev": 0.8, "note": "b"},
                {"repr": "RAISE(to 25.0bb)", "ev": 1.25, "note": "best"},
            ],
            "equity_adjustment_notes": ["EV-first policy"],
            "final_action": "RAISE(to 25.0bb)",
            "explanation": "EV-first: argmax …",
        },
        explanation="EV-first: argmax …",
    )
    secs = flop_trace_sections(st, flop_dec)
    titles = [s[0] for s in secs]
    assert "EV grid (step-through)" in titles
    ev_section = next(b for t, b in secs if t == "EV grid (step-through)")
    # Best candidate listed first after sort
    assert ev_section.index("RAISE(to 25.0bb)") < ev_section.index("CHECK")
