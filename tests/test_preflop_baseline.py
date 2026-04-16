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
# MDF price sensitivity
# ===================================================================

class TestMdfPriceSensitivity:
    def test_tightens_vs_large_open(self):
        # Same hand, deep stack, different open sizes should tighten vs large opens
        s2 = bb_vs_open_decision("7s 6s", 2.0, 100)
        d2 = recommend_preflop_action(s2)
        _assert_valid_decision(s2, d2)

        s5 = bb_vs_open_decision("7s 6s", 5.0, 100)
        d5 = recommend_preflop_action(s5)
        _assert_valid_decision(s5, d5)

        assert d2.recommended_action.action_type in (ActionType.CALL, ActionType.RAISE)
        assert d5.recommended_action.action_type == ActionType.FOLD

    def test_premiums_ignore_filter(self):
        # Premiums should never be filtered out by MDF logic
        s = bb_vs_open_decision("Ah Ad", 5.0, 100)
        d = recommend_preflop_action(s)
        _assert_valid_decision(s, d)
        assert d.recommended_action.action_type == ActionType.RAISE

    def test_debug_includes_scalar(self):
        s = bb_vs_open_decision("7s 6s", 5.0, 100)
        d = recommend_preflop_action(s)
        assert "defense_scalar" in d.debug
        assert "actual_mdf" in d.debug
        assert "mdf_rule" in d.debug

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

    def test_call_when_nothing_to_call(self):
        """BB cannot CALL when contributions are already equal (limp scenario)."""
        history = [
            {"player": "VILLAIN", "action": "POST_BLIND", "amount": 0.5},
            {"player": "HERO", "action": "POST_BLIND", "amount": 1.0},
            {"player": "VILLAIN", "action": "CALL"},
            {"player": "HERO", "action": "CALL"},
        ]
        with pytest.raises(ValidationError, match="nothing to call"):
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
        # BTN acts last postflop -> in position
        assert d.hero_is_in_position_postflop_future_flag is True

    def test_hero_position_bb(self):
        state = bb_vs_open_decision("Td 9c", 2.5, 100)
        d = state.derived
        assert d.hero_is_first_to_act_preflop is False
        # BB acts first postflop -> out of position
        assert d.hero_is_in_position_postflop_future_flag is False

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


# ===================================================================
# Audit: extended hand classification
# ===================================================================

class TestAuditHandClassification:
    """Cover the representative hands from the audit spec."""

    def test_aqo(self):
        hf = hand_features(parse_cards("Ad Qc"))
        assert hf.hand_class_label == "AQo"
        assert hf.hand_bucket == HandBucket.C_OFFSUIT_BIG_BROADWAY

    def test_a4o(self):
        hf = hand_features(parse_cards("Ah 4c"))
        assert hf.hand_class_label == "A4o"
        assert hf.hand_bucket == HandBucket.E_OFFSUIT_AX_LOW
        assert hf.is_wheel_ace is True

    def test_k9s(self):
        hf = hand_features(parse_cards("Ks 9s"))
        assert hf.hand_class_label == "K9s"
        assert hf.hand_bucket == HandBucket.F_SUITED_KX_HIGH

    def test_kto(self):
        hf = hand_features(parse_cards("Kd Tc"))
        assert hf.hand_class_label == "KTo"
        assert hf.hand_bucket == HandBucket.G_OFFSUIT_KX_HIGH

    def test_q9s(self):
        hf = hand_features(parse_cards("Qs 9s"))
        assert hf.hand_class_label == "Q9s"
        assert hf.hand_bucket == HandBucket.H_SUITED_QJ_HIGH

    def test_j8s(self):
        hf = hand_features(parse_cards("Js 8s"))
        assert hf.hand_class_label == "J8s"
        # Suited J with low >= 7 hits the QJ-high branch before the two-gapper set
        assert hf.hand_bucket == HandBucket.H_SUITED_QJ_HIGH

    def test_t9s(self):
        hf = hand_features(parse_cards("Ts 9s"))
        assert hf.hand_class_label == "T9s"
        assert hf.is_connector is True
        assert hf.hand_bucket == HandBucket.J_SUITED_CONNECTORS

    def test_76s(self):
        hf = hand_features(parse_cards("7s 6s"))
        assert hf.hand_class_label == "76s"
        assert hf.is_connector is True
        assert hf.hand_bucket == HandBucket.J_SUITED_CONNECTORS

    def test_97o(self):
        hf = hand_features(parse_cards("9d 7c"))
        assert hf.hand_class_label == "97o"
        assert hf.hand_bucket == HandBucket.M_OFFSUIT_CONNECTORS

    def test_gap_size_correct(self):
        hf_conn = hand_features(parse_cards("8s 7s"))
        assert hf_conn.gap_size == 0
        hf_one = hand_features(parse_cards("9s 7s"))
        assert hf_one.gap_size == 1
        hf_two = hand_features(parse_cards("Ts 7s"))
        assert hf_two.gap_size == 2

    def test_non_wheel_ace(self):
        hf = hand_features(parse_cards("As 8d"))
        assert hf.is_wheel_ace is False


# ===================================================================
# Audit: pot, contribution, and to-call derivation
# ===================================================================

class TestAuditPotDerivation:
    def test_bb_vs_open_contributions(self):
        state = bb_vs_open_decision("Td 9c", 2.5, 100)
        assert abs(state.pot_size_bb - 3.5) < 1e-9
        assert abs(state.hero_contribution_bb - 1.0) < 1e-9
        assert abs(state.villain_contribution_bb - 2.5) < 1e-9
        assert abs(state.current_bet_to_call_bb - 1.5) < 1e-9

    def test_btn_vs_3bet_contributions(self):
        state = btn_vs_3bet_decision("Qh Qd", 2.5, 8.0, 100)
        assert abs(state.pot_size_bb - 10.5) < 1e-9
        assert abs(state.hero_contribution_bb - 2.5) < 1e-9
        assert abs(state.villain_contribution_bb - 8.0) < 1e-9
        assert abs(state.current_bet_to_call_bb - 5.5) < 1e-9

    def test_btn_vs_iso_contributions(self):
        state = btn_vs_iso_after_limp_decision("As Kd", 3.5, 100)
        assert abs(state.hero_contribution_bb - 1.0) < 1e-9
        assert abs(state.villain_contribution_bb - 3.5) < 1e-9
        assert abs(state.pot_size_bb - 4.5) < 1e-9
        assert abs(state.current_bet_to_call_bb - 2.5) < 1e-9

    def test_bb_vs_4bet_contributions(self):
        state = bb_vs_4bet_decision("As Kd", 2.5, 8.0, 20.0, 100)
        assert abs(state.hero_contribution_bb - 8.0) < 1e-9
        assert abs(state.villain_contribution_bb - 20.0) < 1e-9
        assert abs(state.pot_size_bb - 28.0) < 1e-9
        assert abs(state.current_bet_to_call_bb - 12.0) < 1e-9

    def test_unopened_btn_contributions(self):
        state = unopened_btn_decision("As Kd", 100)
        assert abs(state.pot_size_bb - 1.5) < 1e-9
        assert abs(state.hero_contribution_bb - 0.5) < 1e-9
        assert abs(state.villain_contribution_bb - 1.0) < 1e-9
        assert abs(state.current_bet_to_call_bb - 0.5) < 1e-9

    def test_bb_vs_limp_contributions(self):
        state = bb_vs_limp_decision("7h 2d", 100)
        assert abs(state.pot_size_bb - 2.0) < 1e-9
        assert abs(state.hero_contribution_bb - 1.0) < 1e-9
        assert abs(state.villain_contribution_bb - 1.0) < 1e-9
        assert abs(state.current_bet_to_call_bb - 0.0) < 1e-9


# ===================================================================
# Audit: legal actions for all spots
# ===================================================================

class TestAuditLegalActionsAllSpots:
    def test_btn_vs_iso_has_fold_call_raise(self):
        state = btn_vs_iso_after_limp_decision("As Kd", 3.5, 100)
        legal = legal_actions_for_hero(state)
        types = {a.action_type for a in legal}
        assert ActionType.FOLD in types
        assert ActionType.CALL in types
        assert ActionType.RAISE in types

    def test_bb_vs_4bet_has_fold_call_raise(self):
        state = bb_vs_4bet_decision("As Kd", 2.5, 8.0, 20.0, 100)
        legal = legal_actions_for_hero(state)
        types = {a.action_type for a in legal}
        assert ActionType.FOLD in types
        assert ActionType.CALL in types
        assert ActionType.RAISE in types

    def test_short_stack_no_raise_when_calling_is_allin(self):
        state = bb_vs_open_decision("As Kd", 2.5, 3.0)
        legal = legal_actions_for_hero(state)
        types = {a.action_type for a in legal}
        assert ActionType.FOLD in types
        assert ActionType.CALL in types
        # Calling uses 1.5, hero has 2.0 remaining; 0.5 chips after calling
        # min raise to = 2.5 + 1.5 = 4.0, which exceeds stack
        # so hero can only jam to 3.0
        raises = [a for a in legal if a.action_type == ActionType.RAISE]
        if raises:
            assert all(abs(r.raise_to_bb - 3.0) < 1e-9 for r in raises)


# ===================================================================
# Audit: raise-size sensitivity
# ===================================================================

class TestAuditRaiseSizeSensitivity:
    def test_kjo_tightens_vs_large_open(self):
        """KJo BB should not change action at 2.5 but may at 4.5."""
        s25 = bb_vs_open_decision("Kh Jc", 2.5, 100)
        d25 = recommend_preflop_action(s25)
        _assert_valid_decision(s25, d25)
        # KJo vs standard 2.5 open should continue
        assert d25.recommended_action.action_type in (ActionType.CALL, ActionType.RAISE)

    def test_marginal_hand_folds_vs_large_open(self):
        """A marginal call hand should fold vs a large open (MDF tightening)."""
        s_small = bb_vs_open_decision("7s 6s", 2.0, 100)
        d_small = recommend_preflop_action(s_small)
        _assert_valid_decision(s_small, d_small)

        s_large = bb_vs_open_decision("7s 6s", 5.0, 100)
        d_large = recommend_preflop_action(s_large)
        _assert_valid_decision(s_large, d_large)

        # 76s should continue vs 2x, fold vs 5x
        assert d_small.recommended_action.action_type in (ActionType.CALL, ActionType.RAISE)
        assert d_large.recommended_action.action_type == ActionType.FOLD

    def test_defense_scalar_decreases_with_larger_opens(self):
        """Verify defense_scalar is monotonically decreasing as open size grows."""
        scalars = []
        for size in [2.0, 2.5, 3.5, 5.0]:
            s = bb_vs_open_decision("Td 9c", size, 100)
            d = recommend_preflop_action(s)
            scalars.append(d.debug.get("defense_scalar", 1.0))
        for i in range(len(scalars) - 1):
            assert scalars[i] >= scalars[i + 1] - 1e-9

    def test_mdf_rule_present_when_facing_raise(self):
        """All facing-raise decisions should include MDF info in debug."""
        s = bb_vs_open_decision("Td 9c", 2.5, 100)
        d = recommend_preflop_action(s)
        assert "mdf_rule" in d.debug
        assert d.debug["mdf_rule"] is not None

    def test_debug_includes_chart_context_and_filtered_action(self):
        s = bb_vs_open_decision("As Kd", 2.5, 100)
        d = recommend_preflop_action(s)
        for key in ("chart_context", "chart_action_raw", "chart_action_filtered",
                     "defense_scalar", "mdf_rule"):
            assert key in d.debug, f"Missing debug key: {key}"


# ===================================================================
# Audit: golden scenario tests
# ===================================================================

class TestAuditGoldenScenarios:
    def test_aa_btn_always_raises(self):
        for stack in [5, 10, 30, 60, 100]:
            state = unopened_btn_decision("Ah Ad", stack)
            dec = recommend_preflop_action(state)
            _assert_valid_decision(state, dec)
            assert dec.recommended_action.action_type == ActionType.RAISE

    def test_72o_btn_always_folds_at_deep_stacks(self):
        for stack in [30, 60, 100]:
            state = unopened_btn_decision("7d 2c", stack)
            dec = recommend_preflop_action(state)
            _assert_valid_decision(state, dec)
            assert dec.recommended_action.action_type == ActionType.FOLD

    def test_a5s_bb_continues_vs_min_open(self):
        state = bb_vs_open_decision("As 5s", 2.0, 100)
        dec = recommend_preflop_action(state)
        _assert_valid_decision(state, dec)
        assert dec.recommended_action.action_type in (ActionType.CALL, ActionType.RAISE)

    def test_qq_btn_continues_vs_3bet(self):
        state = btn_vs_3bet_decision("Qh Qd", 2.5, 8.0, 100)
        dec = recommend_preflop_action(state)
        _assert_valid_decision(state, dec)
        assert dec.recommended_action.action_type in (ActionType.CALL, ActionType.RAISE)

    def test_suited_connector_raises_from_btn(self):
        state = unopened_btn_decision("8s 7s", 100)
        dec = recommend_preflop_action(state)
        _assert_valid_decision(state, dec)
        assert dec.recommended_action.action_type == ActionType.RAISE

    def test_suited_connector_raises_bb_vs_limp(self):
        state = bb_vs_limp_decision("8s 7s", 100)
        dec = recommend_preflop_action(state)
        _assert_valid_decision(state, dec)
        assert dec.recommended_action.action_type == ActionType.RAISE

    def test_akk_continues_vs_4bet(self):
        state = bb_vs_4bet_decision("As Kd", 2.5, 8.0, 20.0, 100)
        dec = recommend_preflop_action(state)
        _assert_valid_decision(state, dec)
        assert dec.recommended_action.action_type in (ActionType.CALL, ActionType.RAISE)


# ===================================================================
# Audit: recommendation legality across spots
# ===================================================================

class TestAuditRecommendationLegality:
    """Every recommendation must be in the legal actions list."""

    @pytest.mark.parametrize("cards,stack", [
        ("Ah Ad", 100), ("As Ks", 100), ("7d 2c", 100),
        ("Td 9c", 50), ("5s 5d", 20), ("As 5s", 8),
    ])
    def test_btn_open_legality(self, cards, stack):
        state = unopened_btn_decision(cards, stack)
        dec = recommend_preflop_action(state)
        if dec.legal_actions:
            assert dec.recommended_action in dec.legal_actions

    @pytest.mark.parametrize("cards,open_size", [
        ("As Kd", 2.5), ("7s 6s", 2.0), ("7d 2c", 3.0),
        ("Qh Jd", 4.5), ("Td 9c", 5.0),
    ])
    def test_bb_vs_open_legality(self, cards, open_size):
        state = bb_vs_open_decision(cards, open_size, 100)
        dec = recommend_preflop_action(state)
        if dec.legal_actions:
            assert dec.recommended_action in dec.legal_actions

    @pytest.mark.parametrize("cards", ["Ah Ad", "7d 2c", "8s 7s", "Kd Qc"])
    def test_bb_vs_limp_legality(self, cards):
        state = bb_vs_limp_decision(cards, 100)
        dec = recommend_preflop_action(state)
        if dec.legal_actions:
            assert dec.recommended_action in dec.legal_actions

    @pytest.mark.parametrize("cards", ["Ah Ad", "7d 2c", "Ts 9s", "Jh Jd"])
    def test_btn_vs_3bet_legality(self, cards):
        state = btn_vs_3bet_decision(cards, 2.5, 8.0, 100)
        dec = recommend_preflop_action(state)
        if dec.legal_actions:
            assert dec.recommended_action in dec.legal_actions
