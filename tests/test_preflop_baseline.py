"""Comprehensive tests for the baseline preflop decision engine."""

from __future__ import annotations

import pytest

from baseline_preflop.classification import hand_features, parse_cards
from baseline_preflop.legal_actions import legal_actions_for_hero
from baseline_preflop.models import (
    ActionType, HandBucket, LegalActionOption, Player, Position,
    StackDepthBucket, Street,
)
from baseline_preflop.parser import (
    bb_vs_4bet_decision,
    bb_vs_limp_decision,
    bb_vs_open_decision,
    btn_vs_3bet_decision,
    btn_vs_iso_after_limp_decision,
    make_preflop_state,
    unopened_btn_decision,
)
from baseline_preflop.recommender import recommend_preflop_action
from baseline_preflop.validation import ValidationError, validate_preflop_state


# ===================================================================
# Card parsing
# ===================================================================

class TestCardParsing:
    def test_parse_valid_cards(self):
        hc = parse_cards("As Kd")
        assert hc.high.rank == "A"
        assert hc.low.rank == "K"

    def test_parse_cards_reorders(self):
        hc = parse_cards("2c Ah")
        assert hc.high.rank == "A"
        assert hc.low.rank == "2"

    def test_parse_pair(self):
        hc = parse_cards("Jh Jd")
        assert hc.high.rank == "J"
        assert hc.low.rank == "J"

    def test_parse_invalid_rank(self):
        with pytest.raises(ValueError):
            parse_cards("Xs Kd")

    def test_parse_invalid_suit(self):
        with pytest.raises(ValueError):
            parse_cards("Ax Kd")

    def test_parse_duplicate_cards(self):
        with pytest.raises(ValueError):
            parse_cards("As As")

    def test_parse_wrong_token_count(self):
        with pytest.raises(ValueError):
            parse_cards("As")
        with pytest.raises(ValueError):
            parse_cards("As Kd Qh")


# ===================================================================
# Hand classification
# ===================================================================

class TestHandClassification:
    def test_pair_label(self):
        hf = hand_features(parse_cards("Ah Ad"))
        assert hf.hand_class_label == "AA"
        assert hf.is_pair is True
        assert hf.is_suited is False
        assert hf.hand_bucket == HandBucket.A_PREMIUM_PAIRS

    def test_suited_broadway(self):
        hf = hand_features(parse_cards("As Ks"))
        assert hf.hand_class_label == "AKs"
        assert hf.is_suited is True
        assert hf.hand_bucket == HandBucket.B_SUITED_BIG_BROADWAY

    def test_offsuit_broadway(self):
        hf = hand_features(parse_cards("Ad Kc"))
        assert hf.hand_class_label == "AKo"
        assert hf.hand_bucket == HandBucket.C_OFFSUIT_BIG_BROADWAY

    def test_suited_ax(self):
        hf = hand_features(parse_cards("As 5s"))
        assert hf.hand_class_label == "A5s"
        assert hf.hand_bucket == HandBucket.D_SUITED_AX_HIGH

    def test_suited_ax_low(self):
        hf = hand_features(parse_cards("Ah 2h"))
        assert hf.hand_class_label == "A2s"
        assert hf.hand_bucket == HandBucket.D_SUITED_AX_LOW

    def test_offsuit_ax(self):
        hf = hand_features(parse_cards("As 7d"))
        assert hf.hand_class_label == "A7o"
        assert hf.hand_bucket == HandBucket.E_OFFSUIT_AX_HIGH

    def test_suited_connector(self):
        hf = hand_features(parse_cards("8s 7s"))
        assert hf.hand_class_label == "87s"
        assert hf.is_connector is True
        assert hf.hand_bucket == HandBucket.J_SUITED_CONNECTORS

    def test_suited_one_gapper(self):
        hf = hand_features(parse_cards("9s 7s"))
        assert hf.hand_class_label == "97s"
        assert hf.is_one_gapper is True
        assert hf.hand_bucket == HandBucket.K_SUITED_ONE_GAPPERS

    def test_weak_offsuit_trash(self):
        hf = hand_features(parse_cards("7d 2c"))
        assert hf.hand_class_label == "72o"
        assert hf.hand_bucket == HandBucket.N_WEAK_OFFSUIT_TRASH

    def test_broadway_count(self):
        hf = hand_features(parse_cards("Ks Qs"))
        assert hf.broadway_count == 2

    def test_micro_pair(self):
        hf = hand_features(parse_cards("2d 2c"))
        assert hf.hand_bucket == HandBucket.A_MICRO_PAIRS

    def test_low_pair(self):
        hf = hand_features(parse_cards("5h 5d"))
        assert hf.hand_bucket == HandBucket.A_LOW_PAIRS

    def test_mid_pair(self):
        hf = hand_features(parse_cards("8c 8d"))
        assert hf.hand_bucket == HandBucket.A_MID_PAIRS

    def test_high_pair(self):
        hf = hand_features(parse_cards("Jh Jd"))
        assert hf.hand_bucket == HandBucket.A_HIGH_PAIRS

    def test_suited_kx_high(self):
        hf = hand_features(parse_cards("Ks 9s"))
        assert hf.hand_bucket == HandBucket.F_SUITED_KX_HIGH

    def test_suited_kx_low(self):
        hf = hand_features(parse_cards("Kh 3h"))
        assert hf.hand_bucket == HandBucket.F_SUITED_KX_LOW

    def test_offsuit_kx_high(self):
        hf = hand_features(parse_cards("Kd Tc"))
        assert hf.hand_bucket == HandBucket.G_OFFSUIT_KX_HIGH

    def test_offsuit_kx_low(self):
        hf = hand_features(parse_cards("Kd 4c"))
        assert hf.hand_bucket == HandBucket.G_OFFSUIT_KX_LOW

    def test_suited_queen_high(self):
        hf = hand_features(parse_cards("Qs 9s"))
        assert hf.hand_bucket == HandBucket.H_SUITED_QJ_HIGH

    def test_offsuit_queen_high(self):
        hf = hand_features(parse_cards("Qd Tc"))
        assert hf.hand_bucket == HandBucket.I_OFFSUIT_QJ_HIGH

    def test_offsuit_connectors(self):
        hf = hand_features(parse_cards("Td 9c"))
        assert hf.hand_bucket == HandBucket.M_OFFSUIT_CONNECTORS

    def test_wheel_ace(self):
        hf = hand_features(parse_cards("As 3d"))
        assert hf.is_wheel_ace is True

    def test_high_card_points(self):
        hf = hand_features(parse_cards("As Kd"))
        assert hf.high_card_points_simple == 7.0  # 4 + 3


# ===================================================================
# State validation
# ===================================================================

class TestStateValidation:
    def test_valid_unopened_btn(self):
        state = unopened_btn_decision("As Kd", 100)
        validate_preflop_state(state)

    def test_valid_bb_vs_open(self):
        state = bb_vs_open_decision("Qh Js", 2.5, 100)
        validate_preflop_state(state)

    def test_invalid_negative_stack(self):
        with pytest.raises(ValidationError):
            state = unopened_btn_decision("As Kd", -5)
            validate_preflop_state(state)

    def test_invalid_wrong_positions(self):
        state = unopened_btn_decision("As Kd", 100)
        state.villain_position = Position.BTN_SB
        with pytest.raises(ValidationError):
            validate_preflop_state(state)

    def test_invalid_no_history(self):
        state = unopened_btn_decision("As Kd", 100)
        state.action_history = []
        with pytest.raises(ValidationError):
            validate_preflop_state(state)

    def test_action_after_fold(self):
        history = [
            {"player": "VILLAIN", "action": "POST_BLIND", "amount": 0.5},
            {"player": "HERO", "action": "POST_BLIND", "amount": 1.0},
            {"player": "VILLAIN", "action": "FOLD"},
            {"player": "HERO", "action": "CHECK"},
        ]
        with pytest.raises(ValidationError, match="after hand is over"):
            make_preflop_state("As Kd", "BB", 100, history)

    def test_wrong_player_order(self):
        history = [
            {"player": "VILLAIN", "action": "POST_BLIND", "amount": 0.5},
            {"player": "HERO", "action": "POST_BLIND", "amount": 1.0},
            {"player": "HERO", "action": "RAISE", "amount": 2.5},
        ]
        with pytest.raises(ValidationError, match="Wrong player"):
            make_preflop_state("As Kd", "BB", 100, history)

    def test_check_when_facing_bet(self):
        history = [
            {"player": "VILLAIN", "action": "POST_BLIND", "amount": 0.5},
            {"player": "HERO", "action": "POST_BLIND", "amount": 1.0},
            {"player": "VILLAIN", "action": "RAISE", "amount": 2.5},
            {"player": "HERO", "action": "CHECK"},
        ]
        with pytest.raises(ValidationError, match="Cannot check"):
            make_preflop_state("As Kd", "BB", 100, history)


# ===================================================================
# Legal actions
# ===================================================================

class TestLegalActions:
    def test_btn_unopened_has_fold_call_raises(self):
        state = unopened_btn_decision("As Kd", 100)
        legal = legal_actions_for_hero(state)
        types = {a.action_type for a in legal}
        assert ActionType.FOLD in types
        assert ActionType.CALL in types
        assert ActionType.RAISE in types

    def test_bb_vs_open_has_fold_call_raises(self):
        state = bb_vs_open_decision("Qh Js", 2.5, 100)
        legal = legal_actions_for_hero(state)
        types = {a.action_type for a in legal}
        assert ActionType.FOLD in types
        assert ActionType.CALL in types
        assert ActionType.RAISE in types

    def test_bb_vs_limp_has_check_and_raises(self):
        state = bb_vs_limp_decision("7h 2d", 100)
        legal = legal_actions_for_hero(state)
        types = {a.action_type for a in legal}
        assert ActionType.CHECK in types
        assert ActionType.RAISE in types
        assert ActionType.FOLD not in types

    def test_round_closed_no_actions(self):
        history = [
            {"player": "VILLAIN", "action": "POST_BLIND", "amount": 0.5},
            {"player": "HERO", "action": "POST_BLIND", "amount": 1.0},
            {"player": "VILLAIN", "action": "RAISE", "amount": 2.5},
            {"player": "HERO", "action": "CALL"},
        ]
        state = make_preflop_state("As Kd", "BB", 100, history)
        legal = legal_actions_for_hero(state)
        assert legal == []

    def test_hand_over_fold_no_actions(self):
        history = [
            {"player": "HERO", "action": "POST_BLIND", "amount": 0.5},
            {"player": "VILLAIN", "action": "POST_BLIND", "amount": 1.0},
            {"player": "HERO", "action": "FOLD"},
        ]
        state = make_preflop_state("7d 2c", "BTN_SB", 100, history)
        legal = legal_actions_for_hero(state)
        assert legal == []

    def test_raises_include_jam(self):
        state = unopened_btn_decision("As Kd", 20)
        legal = legal_actions_for_hero(state)
        raises = [a for a in legal if a.action_type == ActionType.RAISE]
        max_raise = max(a.raise_to_bb for a in raises)
        assert max_raise == 20.0

    def test_raises_include_min_raise(self):
        state = unopened_btn_decision("As Kd", 100)
        legal = legal_actions_for_hero(state)
        raises = [a for a in legal if a.action_type == ActionType.RAISE]
        min_raise = min(a.raise_to_bb for a in raises)
        assert min_raise == 2.0  # min raise is 1bb increment over 1bb

    def test_all_in_player_no_actions(self):
        state = unopened_btn_decision("As Kd", 1.0)
        legal = legal_actions_for_hero(state)
        types = {a.action_type for a in legal}
        assert ActionType.FOLD in types
        assert ActionType.CALL in types
        assert ActionType.RAISE not in types


# ===================================================================
# Recommendation basics
# ===================================================================

def _assert_valid_decision(state, decision):
    """Common assertions for every recommender output."""
    assert decision.explanation
    assert decision.debug.get("hand_class_label")
    assert decision.debug.get("stack_depth_bucket")
    assert decision.debug.get("baseline_rule_id")

    if decision.legal_actions:
        assert decision.recommended_action in decision.legal_actions


class TestRecommenderBasics:
    def test_recommendation_is_legal(self):
        state = unopened_btn_decision("As Kd", 100)
        dec = recommend_preflop_action(state)
        _assert_valid_decision(state, dec)

    def test_recommendation_deterministic(self):
        state = unopened_btn_decision("Qh Js", 50)
        d1 = recommend_preflop_action(state)
        d2 = recommend_preflop_action(state)
        assert d1.recommended_action == d2.recommended_action

    def test_debug_fields_present(self):
        state = bb_vs_open_decision("Td 9d", 2.5, 80)
        dec = recommend_preflop_action(state)
        for key in ("hand_class_label", "stack_depth_bucket",
                     "action_context_label", "baseline_rule_id"):
            assert key in dec.debug


# ===================================================================
# Scenario tests
# ===================================================================

class TestScenarios:
    def test_btn_unopened_premium_pair_raises(self):
        state = unopened_btn_decision("Ah Ad", 100)
        dec = recommend_preflop_action(state)
        _assert_valid_decision(state, dec)
        assert dec.recommended_action.action_type == ActionType.RAISE

    def test_btn_unopened_weak_trash_folds(self):
        state = unopened_btn_decision("7d 2c", 100)
        dec = recommend_preflop_action(state)
        _assert_valid_decision(state, dec)
        assert dec.recommended_action.action_type == ActionType.FOLD

    def test_bb_vs_limp_suited_connector(self):
        state = bb_vs_limp_decision("8s 7s", 100)
        dec = recommend_preflop_action(state)
        _assert_valid_decision(state, dec)
        # Should at least check or raise -- not fold (no fold option vs limp)
        assert dec.recommended_action.action_type in (ActionType.CHECK, ActionType.RAISE)

    def test_bb_vs_minopen_strong_broadway(self):
        state = bb_vs_open_decision("Ad Ks", 2.0, 100)
        dec = recommend_preflop_action(state)
        _assert_valid_decision(state, dec)
        assert dec.recommended_action.action_type == ActionType.RAISE

    def test_bb_vs_large_open_weak_trash_folds(self):
        state = bb_vs_open_decision("8d 3c", 4.0, 100)
        dec = recommend_preflop_action(state)
        _assert_valid_decision(state, dec)
        assert dec.recommended_action.action_type == ActionType.FOLD

    def test_btn_limp_bb_raise_medium_pair(self):
        state = btn_vs_iso_after_limp_decision("8h 8d", 3.5, 100)
        dec = recommend_preflop_action(state)
        _assert_valid_decision(state, dec)
        assert dec.recommended_action.action_type in (ActionType.CALL, ActionType.RAISE)

    def test_btn_open_bb_3bet_with_qq(self):
        state = btn_vs_3bet_decision("Qh Qd", 2.5, 8.0, 100)
        dec = recommend_preflop_action(state)
        _assert_valid_decision(state, dec)
        assert dec.recommended_action.action_type == ActionType.RAISE

    def test_btn_open_bb_3bet_weak_suited_ace(self):
        state = btn_vs_3bet_decision("As 3s", 2.5, 8.0, 100)
        dec = recommend_preflop_action(state)
        _assert_valid_decision(state, dec)
        assert dec.recommended_action.action_type in (ActionType.FOLD, ActionType.CALL)

    def test_bb_3bet_btn_4bet_with_ak(self):
        state = bb_vs_4bet_decision("As Kd", 2.5, 8.0, 20.0, 100)
        dec = recommend_preflop_action(state)
        _assert_valid_decision(state, dec)
        assert dec.recommended_action.action_type in (ActionType.CALL, ActionType.RAISE)

    def test_round_closed_no_hero_action(self):
        history = [
            {"player": "VILLAIN", "action": "POST_BLIND", "amount": 0.5},
            {"player": "HERO", "action": "POST_BLIND", "amount": 1.0},
            {"player": "VILLAIN", "action": "CALL"},
            {"player": "HERO", "action": "CHECK"},
        ]
        state = make_preflop_state("Td 9c", "BB", 100, history)
        dec = recommend_preflop_action(state)
        assert dec.legal_actions == []


# ===================================================================
# Stack-depth sensitivity tests
# ===================================================================

class TestStackDepthSensitivity:
    """Same hand, same spot, different stack depths should potentially
    produce different recommendations."""

    def test_a5s_btn_unopened_across_stacks(self):
        results = {}
        for stack in [8, 18, 35, 70, 120]:
            state = unopened_btn_decision("As 5s", stack)
            dec = recommend_preflop_action(state)
            _assert_valid_decision(state, dec)
            results[stack] = dec.recommended_action.action_type

        # At 8bb, should be jamming or folding (short stack)
        assert results[8] in (ActionType.RAISE, ActionType.FOLD)
        # Deep stacks should open raise
        assert results[70] == ActionType.RAISE
        assert results[120] == ActionType.RAISE

    def test_kqo_bb_vs_open_across_stacks(self):
        results = {}
        for stack in [12, 25, 50, 100]:
            state = bb_vs_open_decision("Kh Qc", 2.5, stack)
            dec = recommend_preflop_action(state)
            _assert_valid_decision(state, dec)
            results[stack] = dec.recommended_action.action_type

        # Should always continue with KQo vs 2.5bb open
        for stack in [12, 25, 50, 100]:
            assert results[stack] in (ActionType.CALL, ActionType.RAISE)

    def test_76s_btn_vs_3bet_across_stacks(self):
        results = {}
        for stack in [20, 40, 100]:
            state = btn_vs_3bet_decision("7s 6s", 2.5, 8.0, stack)
            dec = recommend_preflop_action(state)
            _assert_valid_decision(state, dec)
            results[stack] = dec.recommended_action.action_type

        # At 20bb facing 3bet, 76s should fold
        assert results[20] == ActionType.FOLD
        # At 100bb, might call due to implied odds
        # (but deterministic baseline may still fold -- we just verify it's valid)

    def test_stack_bucket_changes(self):
        """Verify different stack depths map to different buckets."""
        from baseline_preflop.validation import stack_depth_bucket

        assert stack_depth_bucket(3) == StackDepthBucket.ULTRA_SHORT
        assert stack_depth_bucket(8) == StackDepthBucket.VERY_SHORT
        assert stack_depth_bucket(15) == StackDepthBucket.SHORT
        assert stack_depth_bucket(30) == StackDepthBucket.MEDIUM
        assert stack_depth_bucket(60) == StackDepthBucket.DEEP
        assert stack_depth_bucket(100) == StackDepthBucket.VERY_DEEP


# ===================================================================
# Action history ordering
# ===================================================================

class TestActionHistoryOrdering:
    def test_sequence_indices_sequential(self):
        state = bb_vs_open_decision("Td 9c", 2.5, 100)
        for i, rec in enumerate(state.action_history):
            assert rec.sequence_index == i

    def test_blinds_come_first(self):
        state = unopened_btn_decision("As Kd", 100)
        assert state.action_history[0].action_type == ActionType.POST_BLIND
        assert state.action_history[1].action_type == ActionType.POST_BLIND


# ===================================================================
# Preflop closure
# ===================================================================

class TestPreflopClosure:
    def test_btn_fold_closes(self):
        history = [
            {"player": "HERO", "action": "POST_BLIND", "amount": 0.5},
            {"player": "VILLAIN", "action": "POST_BLIND", "amount": 1.0},
            {"player": "HERO", "action": "FOLD"},
        ]
        state = make_preflop_state("7d 2c", "BTN_SB", 100, history)
        assert state.hand_over is True
        assert state.betting_round_closed is True

    def test_btn_limp_bb_check_closes(self):
        history = [
            {"player": "HERO", "action": "POST_BLIND", "amount": 0.5},
            {"player": "VILLAIN", "action": "POST_BLIND", "amount": 1.0},
            {"player": "HERO", "action": "CALL"},
            {"player": "VILLAIN", "action": "CHECK"},
        ]
        state = make_preflop_state("7s 6s", "BTN_SB", 100, history)
        assert state.betting_round_closed is True

    def test_open_call_closes(self):
        history = [
            {"player": "VILLAIN", "action": "POST_BLIND", "amount": 0.5},
            {"player": "HERO", "action": "POST_BLIND", "amount": 1.0},
            {"player": "VILLAIN", "action": "RAISE", "amount": 2.5},
            {"player": "HERO", "action": "CALL"},
        ]
        state = make_preflop_state("Td 9c", "BB", 100, history)
        assert state.betting_round_closed is True

    def test_3bet_call_closes(self):
        history = [
            {"player": "HERO", "action": "POST_BLIND", "amount": 0.5},
            {"player": "VILLAIN", "action": "POST_BLIND", "amount": 1.0},
            {"player": "HERO", "action": "RAISE", "amount": 2.5},
            {"player": "VILLAIN", "action": "RAISE", "amount": 8.0},
            {"player": "HERO", "action": "CALL"},
        ]
        state = make_preflop_state("Jh Td", "BTN_SB", 100, history)
        assert state.betting_round_closed is True


# ===================================================================
# Impossible state tests
# ===================================================================

class TestImpossibleStates:
    def test_raise_below_current_bet(self):
        history = [
            {"player": "VILLAIN", "action": "POST_BLIND", "amount": 0.5},
            {"player": "HERO", "action": "POST_BLIND", "amount": 1.0},
            {"player": "VILLAIN", "action": "RAISE", "amount": 3.0},
            {"player": "HERO", "action": "RAISE", "amount": 2.0},
        ]
        with pytest.raises(ValidationError):
            make_preflop_state("As Kd", "BB", 100, history)

    def test_raise_exceeds_stack(self):
        history = [
            {"player": "VILLAIN", "action": "POST_BLIND", "amount": 0.5},
            {"player": "HERO", "action": "POST_BLIND", "amount": 1.0},
            {"player": "VILLAIN", "action": "RAISE", "amount": 150.0},
        ]
        with pytest.raises(ValidationError):
            make_preflop_state("As Kd", "BB", 100, history)


# ===================================================================
# Derived state flags
# ===================================================================

class TestDerivedState:
    def test_unopened_flags(self):
        state = unopened_btn_decision("As Kd", 100)
        d = state.derived
        assert d.unopened_pot is True
        assert d.facing_limp is False
        assert d.facing_open_raise is False

    def test_facing_open_raise(self):
        state = bb_vs_open_decision("Td 9c", 2.5, 100)
        d = state.derived
        assert d.facing_open_raise is True
        assert d.unopened_pot is False

    def test_facing_limp(self):
        state = bb_vs_limp_decision("Td 9c", 100)
        d = state.derived
        assert d.facing_limp is True

    def test_facing_3bet(self):
        state = btn_vs_3bet_decision("Qh Qd", 2.5, 8.0, 100)
        d = state.derived
        assert d.facing_3bet is True

    def test_facing_4bet(self):
        state = bb_vs_4bet_decision("As Kd", 2.5, 8.0, 20.0, 100)
        d = state.derived
        assert d.facing_4bet is True

    def test_hero_position_btn(self):
        state = unopened_btn_decision("As Kd", 100)
        d = state.derived
        assert d.hero_is_first_to_act_preflop is True
        assert d.hero_is_in_position_postflop_future_flag is False

    def test_hero_position_bb(self):
        state = bb_vs_open_decision("Td 9c", 2.5, 100)
        d = state.derived
        assert d.hero_is_first_to_act_preflop is False
        assert d.hero_is_in_position_postflop_future_flag is True

    def test_stack_depth_bucket_assigned(self):
        state = unopened_btn_decision("As Kd", 5)
        assert state.derived.stack_depth_bucket == StackDepthBucket.ULTRA_SHORT

        state = unopened_btn_decision("As Kd", 100)
        assert state.derived.stack_depth_bucket == StackDepthBucket.VERY_DEEP


# ===================================================================
# Integration: full loop through parser -> validate -> legal -> recommend
# ===================================================================

class TestIntegration:
    def test_full_pipeline_make_preflop_state(self):
        state = make_preflop_state(
            hero_cards="As Kd",
            hero_position="BB",
            effective_stack_bb=100,
            action_history=[
                {"player": "VILLAIN", "action": "POST_BLIND", "amount": 0.5},
                {"player": "HERO", "action": "POST_BLIND", "amount": 1.0},
                {"player": "VILLAIN", "action": "RAISE", "amount": 2.5},
            ],
        )
        validate_preflop_state(state)
        legal = legal_actions_for_hero(state)
        assert len(legal) > 0
        dec = recommend_preflop_action(state)
        _assert_valid_decision(state, dec)

    def test_full_pipeline_shorthand(self):
        state = unopened_btn_decision("Jh Td", 50)
        dec = recommend_preflop_action(state)
        _assert_valid_decision(state, dec)

    def test_debug_output_string(self):
        from baseline_preflop.debug import pretty_print_decision
        state = bb_vs_open_decision("Qs Js", 2.5, 80)
        dec = recommend_preflop_action(state)
        output = pretty_print_decision(dec, state)
        assert "PREFLOP DECISION SUMMARY" in output
        assert "QJs" in output
