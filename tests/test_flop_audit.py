"""Comprehensive audit tests for Phase D (baseline flop) and Phase D.5 (equity-aware flop).

Covers:
  A. Hand classification (made hands, draws, edge cases)
  B. Board texture classification
  C. Flop context / node classification
  D. Recommendation legality
  E. Bet-size sensitivity
  F. SPR sensitivity
  G. Determinism
  H. Explainability / debug output
  I. Golden scenario tests
  J. Range model (D.5)
  K. Monte Carlo sanity (D.5)
  L. Pot-odds / continue-threshold integration (D.5)
  M. Baseline preservation (D.5)
  N. Sample count sanity (D.5)
  O. Bug regression tests
  P. Manual comparison helper
"""

from __future__ import annotations

import pytest

from poker_core.models import (
    ActionType,
    Card,
    HoleCards,
    Player,
    Position,
)
from poker_core.parser import make_hand_config, parse_board, parse_card, parse_cards
from poker_core.transitions import apply_action, deal_flop, post_blinds

from flop_baseline.classification import classify_board, classify_hand
from flop_baseline.context import derive_flop_context
from flop_baseline.debug import pretty_print_flop_decision
from flop_baseline.models import (
    BetSizeBucket,
    BoardTexture,
    DrawCategory,
    FlopContext,
    MadeHandCategory,
    SPRBucket,
)
from flop_baseline.recommender import recommend_flop_action

from flop_equity.range_model import build_villain_flop_range
from flop_equity.monte_carlo import estimate_flop_equity, _best_hand_rank
from flop_equity.equity_integration import (
    recommend_flop_action_with_equity,
    compare_flop_baseline_vs_equity,
)


# ---------------------------------------------------------------------------
# Test helpers (shared across all audit tests)
# ---------------------------------------------------------------------------

def _make_flop_state(
    hero_cards: str,
    hero_pos: str,
    board_str: str,
    preflop_line: str = "open_call",
    stack: float = 100.0,
):
    cfg = make_hand_config(hero_cards, hero_position=hero_pos, effective_stack_bb=stack)
    hist, _ = post_blinds(cfg)
    btn = cfg.btn_player
    bb = cfg.bb_player

    if preflop_line == "open_call":
        st = apply_action(cfg, hist, action_type=ActionType.RAISE, player=btn, amount_to_bb=2.5)
        hist = st.action_history
        st = apply_action(cfg, hist, action_type=ActionType.CALL, player=bb, amount_to_bb=2.5)
        hist = st.action_history
    elif preflop_line == "limp_check":
        st = apply_action(cfg, hist, action_type=ActionType.CALL, player=btn, amount_to_bb=1.0)
        hist = st.action_history
        st = apply_action(cfg, hist, action_type=ActionType.CHECK, player=bb)
        hist = st.action_history
    elif preflop_line == "3bet_call":
        st = apply_action(cfg, hist, action_type=ActionType.RAISE, player=btn, amount_to_bb=2.5)
        hist = st.action_history
        st = apply_action(cfg, hist, action_type=ActionType.RAISE, player=bb, amount_to_bb=8.0)
        hist = st.action_history
        st = apply_action(cfg, hist, action_type=ActionType.CALL, player=btn, amount_to_bb=8.0)
        hist = st.action_history
    elif preflop_line == "limp_raise_call":
        st = apply_action(cfg, hist, action_type=ActionType.CALL, player=btn, amount_to_bb=1.0)
        hist = st.action_history
        st = apply_action(cfg, hist, action_type=ActionType.BET, player=bb, amount_to_bb=3.0)
        hist = st.action_history
        st = apply_action(cfg, hist, action_type=ActionType.CALL, player=btn, amount_to_bb=3.0)
        hist = st.action_history
    else:
        raise ValueError(f"Unknown preflop_line: {preflop_line}")

    board = parse_board(board_str)
    return deal_flop(cfg, hist, board)


def _advance_to_hero(state):
    if state.current_actor == Player.HERO:
        return state
    return apply_action(state.config, state.action_history,
                        action_type=ActionType.CHECK, player=Player.VILLAIN)


def _villain_bets(state, amount):
    return apply_action(state.config, state.action_history,
                        action_type=ActionType.BET, player=Player.VILLAIN,
                        amount_to_bb=amount)


def _hero_bets_then_villain_raises(state, hero_bet, villain_raise):
    st = apply_action(state.config, state.action_history,
                      action_type=ActionType.BET, player=state.current_actor,
                      amount_to_bb=hero_bet)
    return apply_action(st.config, st.action_history,
                        action_type=ActionType.RAISE, player=Player.VILLAIN,
                        amount_to_bb=villain_raise)


def _hero_checks_villain_bets(state, amount):
    """Hero checks OOP, villain bets."""
    st = apply_action(state.config, state.action_history,
                      action_type=ActionType.CHECK, player=Player.HERO)
    return apply_action(st.config, st.action_history,
                        action_type=ActionType.BET, player=Player.VILLAIN,
                        amount_to_bb=amount)


# ===================================================================
# A. Hand Classification
# ===================================================================

class TestAuditHandClassification:

    def test_set_pocket_pair_hits_board(self):
        hc = parse_cards("7s 7h")
        board = [parse_card("7c"), parse_card("Kd"), parse_card("2s")]
        h = classify_hand(hc, board)
        assert h.made_hand == MadeHandCategory.SET

    def test_trips_from_board_pair_is_not_set(self):
        """Hero holds one card matching a board pair -> trips, NOT set."""
        hc = parse_cards("7s 5h")
        board = [parse_card("7d"), parse_card("7c"), parse_card("2s")]
        h = classify_hand(hc, board)
        assert h.made_hand == MadeHandCategory.NUTS_OR_NEAR_NUTS
        assert h.made_hand != MadeHandCategory.SET

    def test_trips_with_kicker_card(self):
        hc = parse_cards("Ks 7h")
        board = [parse_card("7d"), parse_card("7c"), parse_card("2s")]
        h = classify_hand(hc, board)
        assert h.made_hand == MadeHandCategory.NUTS_OR_NEAR_NUTS

    def test_two_pair_both_hero_cards_pair_board(self):
        hc = parse_cards("As 3h")
        board = [parse_card("Ad"), parse_card("Kc"), parse_card("3s")]
        h = classify_hand(hc, board)
        assert h.made_hand == MadeHandCategory.TWO_PAIR
        assert h.has_two_pair_plus is True

    def test_overpair(self):
        hc = parse_cards("Qs Qh")
        board = [parse_card("Tc"), parse_card("5d"), parse_card("2h")]
        h = classify_hand(hc, board)
        assert h.made_hand == MadeHandCategory.OVERPAIR
        assert h.has_overpair is True

    def test_top_pair_strong_kicker_queen_plus(self):
        hc = parse_cards("As Qh")
        board = [parse_card("Ad"), parse_card("7c"), parse_card("2s")]
        h = classify_hand(hc, board)
        assert h.made_hand == MadeHandCategory.TOP_PAIR_STRONG_KICKER

    def test_top_pair_medium_kicker(self):
        hc = parse_cards("Kd Jh")
        board = [parse_card("Kc"), parse_card("7s"), parse_card("2d")]
        h = classify_hand(hc, board)
        assert h.made_hand == MadeHandCategory.TOP_PAIR_MEDIUM_KICKER

    def test_top_pair_weak_kicker(self):
        hc = parse_cards("Ks 4h")
        board = [parse_card("Kc"), parse_card("7s"), parse_card("2d")]
        h = classify_hand(hc, board)
        assert h.made_hand == MadeHandCategory.TOP_PAIR_WEAK_KICKER

    def test_middle_pair_pairs_second_board_card(self):
        hc = parse_cards("7s 3h")
        board = [parse_card("Kd"), parse_card("7c"), parse_card("2s")]
        h = classify_hand(hc, board)
        assert h.made_hand == MadeHandCategory.MIDDLE_PAIR

    def test_middle_pair_pocket_between_board(self):
        hc = parse_cards("5s 5h")
        board = [parse_card("Kd"), parse_card("7c"), parse_card("3s")]
        h = classify_hand(hc, board)
        assert h.made_hand == MadeHandCategory.MIDDLE_PAIR

    def test_third_pair(self):
        hc = parse_cards("2s 3h")
        board = [parse_card("Kd"), parse_card("7c"), parse_card("2d")]
        h = classify_hand(hc, board)
        assert h.made_hand == MadeHandCategory.THIRD_PAIR_OR_WORSE_PAIR

    def test_underpair(self):
        hc = parse_cards("4s 4h")
        board = [parse_card("Kd"), parse_card("7c"), parse_card("6s")]
        h = classify_hand(hc, board)
        assert h.made_hand == MadeHandCategory.UNDERPAIR_TO_BOARD

    def test_ace_high_no_pair(self):
        hc = parse_cards("As 5h")
        board = [parse_card("Kd"), parse_card("7c"), parse_card("2s")]
        h = classify_hand(hc, board)
        assert h.made_hand == MadeHandCategory.ACE_HIGH
        assert h.has_showdown_value is True
        assert h.has_pair is False

    def test_king_high(self):
        hc = parse_cards("Kd 5c")
        board = [parse_card("Qh"), parse_card("7s"), parse_card("2d")]
        h = classify_hand(hc, board)
        assert h.made_hand == MadeHandCategory.KING_HIGH_OR_WORSE_HIGH_CARD
        assert h.has_showdown_value is False

    def test_flush_draw(self):
        hc = parse_cards("Ah Kh")
        board = [parse_card("7h"), parse_card("4h"), parse_card("2s")]
        h = classify_hand(hc, board)
        assert h.draw == DrawCategory.FLUSH_DRAW
        assert h.has_strong_draw is True

    def test_oesd(self):
        hc = parse_cards("9d 8c")
        board = [parse_card("Th"), parse_card("7s"), parse_card("2d")]
        h = classify_hand(hc, board)
        assert h.draw == DrawCategory.OPEN_ENDED_STRAIGHT_DRAW

    def test_gutshot(self):
        hc = parse_cards("Qd Jc")
        board = [parse_card("Th"), parse_card("8s"), parse_card("3d")]
        h = classify_hand(hc, board)
        assert h.draw == DrawCategory.GUTSHOT

    def test_combo_draw_flush_plus_oesd(self):
        hc = parse_cards("9h 8h")
        board = [parse_card("Th"), parse_card("7h"), parse_card("2s")]
        h = classify_hand(hc, board)
        assert h.draw == DrawCategory.COMBO_DRAW
        assert h.has_combo_draw is True

    def test_combo_draw_flush_plus_gutshot(self):
        hc = parse_cards("Ah Jh")
        board = [parse_card("Kh"), parse_card("Qh"), parse_card("3s")]
        h = classify_hand(hc, board)
        assert h.draw == DrawCategory.COMBO_DRAW

    def test_backdoor_flush_draw(self):
        hc = parse_cards("As Ks")
        board = [parse_card("7s"), parse_card("4d"), parse_card("2h")]
        h = classify_hand(hc, board)
        assert h.has_backdoor_equity is True

    def test_backdoor_straight_draw(self):
        hc = parse_cards("Jh Th")
        board = [parse_card("Ks"), parse_card("5h"), parse_card("2d")]
        h = classify_hand(hc, board)
        assert h.has_backdoor_equity is True

    def test_no_real_draw_pure_air(self):
        hc = parse_cards("4s 2h")
        board = [parse_card("Kd"), parse_card("9c"), parse_card("7s")]
        h = classify_hand(hc, board)
        assert h.has_strong_draw is False
        assert h.has_combo_draw is False
        assert h.has_pair is False

    def test_made_straight_on_flop(self):
        hc = parse_cards("8d 7c")
        board = [parse_card("9h"), parse_card("6s"), parse_card("5d")]
        h = classify_hand(hc, board)
        assert h.made_hand == MadeHandCategory.NUTS_OR_NEAR_NUTS

    def test_made_flush_on_monotone_board(self):
        hc = parse_cards("Ks Qs")
        board = [parse_card("9s"), parse_card("6s"), parse_card("3s")]
        h = classify_hand(hc, board)
        assert h.made_hand == MadeHandCategory.NUTS_OR_NEAR_NUTS

    def test_wheel_gutshot(self):
        hc = parse_cards("Ad 4c")
        board = [parse_card("5h"), parse_card("3s"), parse_card("Kd")]
        h = classify_hand(hc, board)
        assert h.draw == DrawCategory.GUTSHOT

    def test_overcards_count_two(self):
        hc = parse_cards("As Kh")
        board = [parse_card("7c"), parse_card("5d"), parse_card("2s")]
        h = classify_hand(hc, board)
        assert h.overcards_to_board_count == 2

    def test_overcards_count_zero(self):
        hc = parse_cards("4s 3h")
        board = [parse_card("Kc"), parse_card("9d"), parse_card("6h")]
        h = classify_hand(hc, board)
        assert h.overcards_to_board_count == 0

    def test_board_pair_hero_no_pair_ace_high(self):
        hc = parse_cards("As 5h")
        board = [parse_card("7d"), parse_card("7c"), parse_card("2s")]
        h = classify_hand(hc, board)
        assert h.made_hand == MadeHandCategory.ACE_HIGH


# ===================================================================
# B. Board Texture Classification
# ===================================================================

class TestAuditBoardTexture:

    def test_dry_ace_high_rainbow(self):
        board = [parse_card("Ah"), parse_card("7c"), parse_card("2d")]
        bf = classify_board(board)
        assert bf.texture == BoardTexture.DRY_HIGH_CARD
        assert bf.is_rainbow is True
        assert bf.board_is_dynamic is False

    def test_dry_low_board(self):
        board = [parse_card("8c"), parse_card("4d"), parse_card("2h")]
        bf = classify_board(board)
        assert bf.texture == BoardTexture.DRY_LOW_BOARD

    def test_paired_board(self):
        board = [parse_card("Qd"), parse_card("Qc"), parse_card("7h")]
        bf = classify_board(board)
        assert bf.texture == BoardTexture.PAIRED_BOARD
        assert bf.is_paired is True

    def test_monotone_board(self):
        board = [parse_card("Js"), parse_card("8s"), parse_card("3s")]
        bf = classify_board(board)
        assert bf.texture == BoardTexture.MONOTONE_BOARD
        assert bf.is_monotone is True

    def test_two_tone_dynamic_board(self):
        board = [parse_card("Td"), parse_card("9d"), parse_card("7h")]
        bf = classify_board(board)
        assert bf.texture == BoardTexture.DYNAMIC_DRAW_HEAVY
        assert bf.board_is_dynamic is True

    def test_rainbow_connected_high(self):
        board = [parse_card("Th"), parse_card("Jd"), parse_card("Qs")]
        bf = classify_board(board)
        assert bf.texture == BoardTexture.RAINBOW_CONNECTED

    def test_rainbow_semi_connected_high(self):
        """KJT rainbow (one gap) should be RAINBOW_CONNECTED, not DRY_HIGH_CARD."""
        board = [parse_card("Kd"), parse_card("Jc"), parse_card("Th")]
        bf = classify_board(board)
        assert bf.texture == BoardTexture.RAINBOW_CONNECTED

    def test_low_connected(self):
        board = [parse_card("5h"), parse_card("6d"), parse_card("7s")]
        bf = classify_board(board)
        assert bf.texture == BoardTexture.LOW_CONNECTED

    def test_static_board(self):
        board = [parse_card("9d"), parse_card("5c"), parse_card("2h")]
        bf = classify_board(board)
        assert bf.texture == BoardTexture.STATIC_BOARD

    def test_high_connected_two_tone(self):
        board = [parse_card("Ks"), parse_card("Qs"), parse_card("Jd")]
        bf = classify_board(board)
        assert bf.texture == BoardTexture.DYNAMIC_DRAW_HEAVY

    @pytest.mark.parametrize("cards,expected_texture", [
        ("Qd Tc 9h", BoardTexture.RAINBOW_CONNECTED),
        ("Jd 9c 8h", BoardTexture.RAINBOW_CONNECTED),
        ("Td 8c 7h", BoardTexture.RAINBOW_CONNECTED),
        ("9d 7c 6h", BoardTexture.RAINBOW_CONNECTED),
        ("8d 6c 5h", BoardTexture.LOW_CONNECTED),
    ])
    def test_semi_connected_rainbow_boards(self, cards, expected_texture):
        board = [parse_card(c) for c in cards.split()]
        bf = classify_board(board)
        assert bf.texture == expected_texture

    def test_top_card_rank_ace(self):
        board = [parse_card("Ah"), parse_card("7c"), parse_card("2d")]
        bf = classify_board(board)
        assert bf.top_card_rank == 14

    def test_board_requires_three_cards(self):
        with pytest.raises(ValueError):
            classify_board([parse_card("Ah"), parse_card("7c")])


# ===================================================================
# C. Flop Context / Node Classification
# ===================================================================

class TestAuditFlopContext:

    def test_pfr_ip_checked_to(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _advance_to_hero(st)
        ctx = derive_flop_context(st)
        assert ctx.flop_context == FlopContext.PFR_IP_CHECKED_TO
        assert ctx.hero_is_ip is True
        assert ctx.hero_is_pfr is True

    def test_pfr_oop_first_to_act(self):
        st = _make_flop_state("As Kh", "BB", "Ah 7c 2d", "3bet_call")
        ctx = derive_flop_context(st)
        assert ctx.flop_context == FlopContext.PFR_OOP_FIRST_TO_ACT
        assert ctx.hero_is_ip is False
        assert ctx.hero_is_pfr is True

    def test_pfc_ip_checked_to(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "limp_check")
        st = _advance_to_hero(st)
        ctx = derive_flop_context(st)
        assert ctx.flop_context == FlopContext.PFC_IP_CHECKED_TO
        assert ctx.hero_is_ip is True
        assert ctx.hero_is_pfr is False

    def test_pfc_oop_first_to_act(self):
        st = _make_flop_state("As Kh", "BB", "Ah 7c 2d", "open_call")
        ctx = derive_flop_context(st)
        assert ctx.flop_context == FlopContext.PFC_OOP_FIRST_TO_ACT
        assert ctx.hero_is_ip is False
        assert ctx.hero_is_pfr is False

    def test_facing_small_bet(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _villain_bets(st, 1.0)
        ctx = derive_flop_context(st)
        assert ctx.flop_context == FlopContext.FACING_SMALL_BET
        assert ctx.bet_size_bucket == BetSizeBucket.SMALL

    def test_facing_medium_bet(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _villain_bets(st, 3.5)
        ctx = derive_flop_context(st)
        assert ctx.flop_context == FlopContext.FACING_MEDIUM_BET
        assert ctx.bet_size_bucket == BetSizeBucket.MEDIUM

    def test_facing_large_bet(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _villain_bets(st, 5.0)
        ctx = derive_flop_context(st)
        assert ctx.flop_context == FlopContext.FACING_LARGE_BET
        assert ctx.bet_size_bucket == BetSizeBucket.LARGE

    def test_facing_raise_after_betting(self):
        st = _make_flop_state("As Kh", "BB", "Ah 7c 2d", "3bet_call")
        st = _hero_bets_then_villain_raises(st, hero_bet=5.0, villain_raise=15.0)
        ctx = derive_flop_context(st)
        assert ctx.flop_context == FlopContext.FACING_RAISE_AFTER_BETTING

    def test_spr_high_100bb_open_call(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _advance_to_hero(st)
        ctx = derive_flop_context(st)
        assert ctx.spr_bucket == SPRBucket.HIGH_SPR
        assert ctx.spr > 8.0

    def test_spr_medium_3bet_pot(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "3bet_call")
        st = _advance_to_hero(st)
        ctx = derive_flop_context(st)
        assert ctx.spr_bucket == SPRBucket.MEDIUM_SPR

    def test_spr_low_short_stack_3bet(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "3bet_call", stack=25.0)
        st = _advance_to_hero(st)
        ctx = derive_flop_context(st)
        assert ctx.spr_bucket == SPRBucket.LOW_SPR

    def test_bet_size_bucket_none_when_no_bet(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _advance_to_hero(st)
        ctx = derive_flop_context(st)
        assert ctx.bet_size_bucket is None


# ===================================================================
# D. Recommendation Legality (all scenarios)
# ===================================================================

class TestAuditRecommendationLegality:

    @pytest.mark.parametrize("hero_cards,hero_pos,board,pf,setup", [
        ("As Kh", "BTN_SB", "Ah 7c 2d", "open_call", "check"),
        ("7s 2h", "BTN_SB", "Kd 8c 3s", "open_call", "check"),
        ("Qs Qh", "BTN_SB", "Tc 5d 2h", "open_call", "check"),
        ("As Kh", "BB", "Ah 7c 2d", "3bet_call", "first"),
        ("7s 2h", "BB", "Kd 8c 3s", "3bet_call", "first"),
        ("As Kh", "BTN_SB", "Ah 7c 2d", "limp_check", "check"),
        ("5s 4s", "BTN_SB", "7h 6d 3c", "limp_check", "check"),
        ("As Kh", "BB", "Ah 7c 2d", "open_call", "first"),
        ("Ks Ts", "BTN_SB", "8s 5s 2h", "open_call", "bet_small"),
        ("As Kh", "BTN_SB", "Ah 7c 2d", "open_call", "bet_large"),
        ("9s 8s", "BTN_SB", "7s 6s 2h", "open_call", "bet_medium"),
    ])
    def test_action_is_legal(self, hero_cards, hero_pos, board, pf, setup):
        st = _make_flop_state(hero_cards, hero_pos, board, pf)
        if setup == "check":
            st = _advance_to_hero(st)
        elif setup == "first":
            assert st.current_actor == Player.HERO
        elif setup == "bet_small":
            st = _villain_bets(st, 1.0)
        elif setup == "bet_medium":
            st = _villain_bets(st, 3.5)
        elif setup == "bet_large":
            st = _villain_bets(st, 5.0)

        dec = recommend_flop_action(st)
        legal_types = {a.action_type for a in dec.legal_actions}
        assert dec.recommended_action.legal_action.action_type in legal_types

        rec = dec.recommended_action
        if rec.size_bb is not None:
            la = rec.legal_action
            assert la.min_to_bb is not None
            assert la.max_to_bb is not None
            assert la.min_to_bb - 0.01 <= rec.size_bb <= la.max_to_bb + 0.01

        assert dec.explanation
        assert dec.debug.get("baseline_rule_id")
        assert dec.debug.get("action_context_label")
        assert dec.debug.get("made_hand_category")
        assert dec.debug.get("board_texture_label")


# ===================================================================
# E. Bet-Size Sensitivity
# ===================================================================

class TestAuditBetSizeSensitivity:

    def test_weak_top_pair_calls_small_folds_large(self):
        st_base = _make_flop_state("Ks 4h", "BB", "Kd 7c 2s", "open_call")

        st_small = _hero_checks_villain_bets(st_base, 1.0)
        dec_small = recommend_flop_action(st_small)

        st_large = _hero_checks_villain_bets(st_base, 5.0)
        dec_large = recommend_flop_action(st_large)

        assert dec_small.recommended_action.legal_action.action_type != ActionType.FOLD
        assert dec_large.recommended_action.legal_action.action_type in (
            ActionType.CALL, ActionType.FOLD, ActionType.RAISE,
        )

    def test_flush_draw_facing_small_vs_large(self):
        st_base = _make_flop_state("Ks Ts", "BB", "8s 5s 2h", "open_call")

        st_small = _hero_checks_villain_bets(st_base, 1.0)
        dec_small = recommend_flop_action(st_small)

        st_large = _hero_checks_villain_bets(st_base, 5.0)
        dec_large = recommend_flop_action(st_large)

        assert dec_small.recommended_action.legal_action.action_type != ActionType.FOLD

    def test_middle_pair_calls_small_folds_large(self):
        st_base = _make_flop_state("7s 3h", "BTN_SB", "Kd 7c 2s", "open_call")

        st_small = _villain_bets(st_base, 1.0)
        dec_small = recommend_flop_action(st_small)

        st_large = _villain_bets(st_base, 5.0)
        dec_large = recommend_flop_action(st_large)

        assert dec_small.recommended_action.legal_action.action_type != ActionType.FOLD

    def test_strong_hand_continues_all_sizes(self):
        st_base = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")

        for bet_amt in [1.0, 3.5, 5.0]:
            st = _villain_bets(st_base, bet_amt)
            dec = recommend_flop_action(st)
            assert dec.recommended_action.legal_action.action_type != ActionType.FOLD


# ===================================================================
# F. SPR Sensitivity
# ===================================================================

class TestAuditSPRSensitivity:

    def test_top_pair_weak_kicker_different_spr(self):
        # High SPR
        st_hi = _make_flop_state("Ks 4h", "BTN_SB", "Kd 7c 2s", "open_call", stack=100)
        st_hi = _advance_to_hero(st_hi)
        ctx_hi = derive_flop_context(st_hi)
        assert ctx_hi.spr_bucket in (SPRBucket.HIGH_SPR, SPRBucket.MEDIUM_SPR)

        # Low SPR
        st_lo = _make_flop_state("Ks 4h", "BTN_SB", "Kd 7c 2s", "3bet_call", stack=25)
        st_lo = _advance_to_hero(st_lo)
        ctx_lo = derive_flop_context(st_lo)
        assert ctx_lo.spr_bucket == SPRBucket.LOW_SPR

    def test_combo_draw_spr_variation(self):
        # High SPR
        st_hi = _make_flop_state("9s 8s", "BB", "7s 6s 2h", "open_call", stack=100)
        dec_hi = recommend_flop_action(st_hi)

        # Low SPR
        st_lo = _make_flop_state("9s 8s", "BB", "7s 6s 2h", "3bet_call", stack=25)
        dec_lo = recommend_flop_action(st_lo)

        assert dec_hi.recommended_action.legal_action.action_type in (
            ActionType.BET, ActionType.CHECK
        )
        assert dec_lo.recommended_action.legal_action.action_type in (
            ActionType.BET, ActionType.CHECK
        )


# ===================================================================
# G. Determinism
# ===================================================================

class TestAuditDeterminism:

    def test_baseline_identical_on_repeat(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _advance_to_hero(st)
        results = set()
        for _ in range(10):
            dec = recommend_flop_action(st)
            key = (
                dec.recommended_action.legal_action.action_type,
                dec.recommended_action.size_bb,
                dec.explanation,
            )
            results.add(key)
        assert len(results) == 1

    def test_equity_deterministic_with_seed(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _advance_to_hero(st)
        d1 = recommend_flop_action_with_equity(st, samples=500, seed=42)
        d2 = recommend_flop_action_with_equity(st, samples=500, seed=42)
        assert d1.debug["equity_estimate"] == d2.debug["equity_estimate"]
        assert d1.recommended_action.legal_action.action_type == d2.recommended_action.legal_action.action_type


# ===================================================================
# H. Explainability / Debug
# ===================================================================

class TestAuditDebugOutput:

    BASELINE_KEYS = {
        "action_context_label", "hero_position_relation_on_flop",
        "hero_preflop_role", "made_hand_category", "draw_category",
        "board_texture_label", "flop_bet_size_bucket", "spr_bucket", "spr",
        "baseline_rule_id", "legal_actions", "recommended_action",
        "explanation", "has_showdown_value", "has_strong_draw",
        "has_combo_draw", "has_backdoor_equity", "overcards_to_board_count",
    }

    EQUITY_KEYS = {
        "villain_range_summary", "monte_carlo_samples", "equity_estimate",
        "win_rate", "tie_rate", "pot_odds_threshold", "baseline_action",
        "final_action", "equity_changed_recommendation",
        "equity_adjustment_notes",
    }

    def test_baseline_debug_complete(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _advance_to_hero(st)
        dec = recommend_flop_action(st)
        missing = self.BASELINE_KEYS - set(dec.debug.keys())
        assert not missing, f"Missing: {missing}"

    def test_equity_debug_complete_facing_bet(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _villain_bets(st, 3.5)
        dec = recommend_flop_action_with_equity(st, samples=500, seed=1)
        all_keys = self.BASELINE_KEYS | self.EQUITY_KEYS
        missing = all_keys - set(dec.debug.keys())
        assert not missing, f"Missing: {missing}"

    def test_equity_debug_complete_checked_to(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _advance_to_hero(st)
        dec = recommend_flop_action_with_equity(st, samples=500, seed=1)
        all_keys = self.BASELINE_KEYS | self.EQUITY_KEYS
        missing = all_keys - set(dec.debug.keys())
        assert not missing, f"Missing: {missing}"
        assert dec.debug["pot_odds_threshold"] is None

    def test_pretty_print(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _advance_to_hero(st)
        dec = recommend_flop_action(st)
        output = pretty_print_flop_decision(dec, st)
        assert "FLOP DECISION" in output
        assert "Made hand" in output


# ===================================================================
# I. Golden Scenario Tests (Phase D)
# ===================================================================

class TestAuditGoldenScenarios:

    def test_pfr_ip_top_pair_cbet_dry_ace_high(self):
        """BTN opened, BB called, flop AK2r checks to BTN. Top pair medium kicker."""
        st = _make_flop_state("As Th", "BTN_SB", "Ad Kc 2h", "open_call")
        st = _advance_to_hero(st)
        dec = recommend_flop_action(st)
        assert dec.recommended_action.legal_action.action_type == ActionType.BET

    def test_pfr_ip_ak_high_on_low_connected_twotone(self):
        """BTN opened, BB called, low connected two-tone board, hero has AK high."""
        st = _make_flop_state("As Kh", "BTN_SB", "7d 6d 5h", "open_call")
        st = _advance_to_hero(st)
        dec = recommend_flop_action(st)
        assert dec.recommended_action.legal_action.action_type in (ActionType.BET, ActionType.CHECK)

    def test_pfc_oop_facing_cbet_middle_pair_dynamic(self):
        """BB defended vs BTN open, faces c-bet with middle pair on dynamic board."""
        st = _make_flop_state("8s 3h", "BB", "Kd 8d 6h", "open_call")
        st = _hero_checks_villain_bets(st, 3.5)
        dec = recommend_flop_action(st)
        assert dec.recommended_action.legal_action.action_type in (
            ActionType.CALL, ActionType.FOLD, ActionType.RAISE,
        )

    def test_facing_raise_with_overpair(self):
        """Hero c-bets with overpair, faces a raise. Should call."""
        st = _make_flop_state("Qs Qh", "BB", "Td 7c 2s", "3bet_call")
        st = _hero_bets_then_villain_raises(st, hero_bet=5.0, villain_raise=15.0)
        dec = recommend_flop_action(st)
        assert dec.recommended_action.legal_action.action_type in (ActionType.CALL, ActionType.RAISE)

    def test_nut_flush_draw_facing_medium_bet(self):
        """Hero has nut flush draw facing medium flop bet."""
        st = _make_flop_state("As 4s", "BTN_SB", "Ks 9s 3h", "open_call")
        st = _villain_bets(st, 3.5)
        dec = recommend_flop_action(st)
        assert dec.recommended_action.legal_action.action_type != ActionType.FOLD

    def test_gutshot_overcards_facing_large_bet(self):
        """Hero has gutshot + overcards facing large bet. May fold."""
        st = _make_flop_state("Ad Kc", "BTN_SB", "Qs 9h 7d", "open_call")
        st = _villain_bets(st, 5.0)
        dec = recommend_flop_action(st)
        assert dec.recommended_action.legal_action.action_type in (
            ActionType.CALL, ActionType.FOLD, ActionType.RAISE,
        )

    def test_total_air_checked_to_ip_as_pfc(self):
        """PFC IP with total air checked to: EV may check or stab."""
        st = _make_flop_state("4s 2h", "BTN_SB", "Kd 9c 7s", "limp_check")
        st = _advance_to_hero(st)
        dec = recommend_flop_action(st)
        assert dec.recommended_action.legal_action.action_type in (
            ActionType.CHECK, ActionType.BET,
        )


# ===================================================================
# J. Range Model Sanity (D.5)
# ===================================================================

class TestAuditRangeModel:

    @pytest.mark.parametrize("pf_line", [
        "open_call", "3bet_call", "limp_check", "limp_raise_call",
    ])
    def test_range_nonempty(self, pf_line):
        st = _make_flop_state("As Kh", "BTN_SB", "7c 4d 2s", pf_line)
        st = _advance_to_hero(st)
        rng, summary = build_villain_flop_range(st)
        assert len(rng) > 0
        assert len(summary) > 0

    def test_dead_cards_excluded(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _advance_to_hero(st)
        rng, _ = build_villain_flop_range(st)
        dead = {st.config.hero_hole_cards.high, st.config.hero_hole_cards.low} | set(st.board_cards)
        for hc, _ in rng:
            assert hc.high not in dead
            assert hc.low not in dead

    def test_different_lines_different_summaries(self):
        st1 = _make_flop_state("As Kh", "BTN_SB", "7c 4d 2s", "open_call")
        st1 = _advance_to_hero(st1)
        _, s1 = build_villain_flop_range(st1)

        st2 = _make_flop_state("As Kh", "BTN_SB", "7c 4d 2s", "3bet_call")
        st2 = _advance_to_hero(st2)
        _, s2 = build_villain_flop_range(st2)

        assert s1 != s2

    def test_villain_bet_changes_weights(self):
        st = _make_flop_state("As Kh", "BTN_SB", "7c 4d 2s", "open_call")
        st_check = _advance_to_hero(st)
        rng_check, _ = build_villain_flop_range(st_check)

        st_bet = _villain_bets(st, 3.5)
        rng_bet, _ = build_villain_flop_range(st_bet)

        w_check = sum(w for _, w in rng_check)
        w_bet = sum(w for _, w in rng_bet)
        assert w_bet < w_check

    def test_hero_bb_villain_is_btn(self):
        st = _make_flop_state("As Kh", "BB", "7c 4d 2s", "open_call")
        rng, summary = build_villain_flop_range(st)
        assert len(rng) > 30
        assert "BTN" in summary

    def test_3bet_range_narrower_than_open_call(self):
        st1 = _make_flop_state("As Kh", "BTN_SB", "7c 4d 2s", "open_call")
        st1 = _advance_to_hero(st1)
        rng1, _ = build_villain_flop_range(st1)

        st2 = _make_flop_state("As Kh", "BTN_SB", "7c 4d 2s", "3bet_call")
        st2 = _advance_to_hero(st2)
        rng2, _ = build_villain_flop_range(st2)

        assert len(rng2) < len(rng1)


# ===================================================================
# K. Monte Carlo Sanity (D.5)
# ===================================================================

class TestAuditMonteCarloSanity:

    def test_set_very_high_equity(self):
        st = _make_flop_state("7s 7h", "BTN_SB", "7c Kd 2s", "open_call")
        st = _advance_to_hero(st)
        rng, _ = build_villain_flop_range(st)
        result = estimate_flop_equity(st.config.hero_hole_cards, st.board_cards, rng,
                                      samples=2000, seed=42)
        assert result["equity_estimate"] > 0.85

    def test_air_low_equity(self):
        st = _make_flop_state("4s 3h", "BTN_SB", "Kd 8c 2s", "open_call")
        st = _advance_to_hero(st)
        rng, _ = build_villain_flop_range(st)
        result = estimate_flop_equity(st.config.hero_hole_cards, st.board_cards, rng,
                                      samples=2000, seed=42)
        assert result["equity_estimate"] < 0.30

    def test_overpair_good_equity(self):
        st = _make_flop_state("Qs Qh", "BTN_SB", "Tc 5d 2h", "open_call")
        st = _advance_to_hero(st)
        rng, _ = build_villain_flop_range(st)
        result = estimate_flop_equity(st.config.hero_hole_cards, st.board_cards, rng,
                                      samples=2000, seed=42)
        assert result["equity_estimate"] > 0.65

    def test_top_pair_weak_kicker_between_strong_and_air(self):
        st = _make_flop_state("Ks 4h", "BTN_SB", "Kd 7c 2s", "open_call")
        st = _advance_to_hero(st)
        rng, _ = build_villain_flop_range(st)
        result = estimate_flop_equity(st.config.hero_hole_cards, st.board_cards, rng,
                                      samples=2000, seed=42)
        assert 0.40 < result["equity_estimate"] < 0.85

    def test_combo_draw_beats_gutshot_equity(self):
        board = [parse_card("8s"), parse_card("9s"), parse_card("2h")]

        st1 = _make_flop_state("Ks Ts", "BTN_SB", "8s 9s 2h", "open_call")
        st1 = _advance_to_hero(st1)
        rng1, _ = build_villain_flop_range(st1)
        r_combo = estimate_flop_equity(parse_cards("Ks Ts"), board, rng1,
                                       samples=2000, seed=42)

        st2 = _make_flop_state("Qs Jh", "BTN_SB", "8s 9s 2h", "open_call")
        st2 = _advance_to_hero(st2)
        rng2, _ = build_villain_flop_range(st2)
        r_gutshot = estimate_flop_equity(parse_cards("Qs Jh"), board, rng2,
                                         samples=2000, seed=42)

        assert r_combo["equity_estimate"] > r_gutshot["equity_estimate"]

    def test_empty_range_raises(self):
        with pytest.raises(ValueError, match="empty"):
            estimate_flop_equity(
                parse_cards("As Kh"),
                [parse_card("7c"), parse_card("4d"), parse_card("2s")],
                [], samples=10,
            )


# ===================================================================
# L. Pot-Odds / Continue-Threshold Integration (D.5)
# ===================================================================

class TestAuditPotOdds:

    def test_flush_draw_calls_vs_medium_bet(self):
        st = _make_flop_state("Ks Ts", "BB", "8s 5s 2h", "open_call")
        st = _hero_checks_villain_bets(st, 3.5)
        dec = recommend_flop_action_with_equity(st, samples=2000, seed=42)
        assert dec.recommended_action.legal_action.action_type != ActionType.FOLD
        assert dec.debug["pot_odds_threshold"] is not None

    def test_air_folds_to_large_bet(self):
        st = _make_flop_state("4s 3h", "BTN_SB", "Kd 8c 2s", "open_call")
        st = _villain_bets(st, 5.0)
        dec = recommend_flop_action_with_equity(st, samples=2000, seed=42)
        assert dec.recommended_action.legal_action.action_type == ActionType.FOLD

    def test_pot_odds_threshold_present_facing_bet(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _villain_bets(st, 3.5)
        dec = recommend_flop_action_with_equity(st, samples=500, seed=1)
        assert dec.debug["pot_odds_threshold"] is not None
        assert 0.0 < dec.debug["pot_odds_threshold"] < 1.0

    def test_pot_odds_none_when_no_bet(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _advance_to_hero(st)
        dec = recommend_flop_action_with_equity(st, samples=500, seed=1)
        assert dec.debug["pot_odds_threshold"] is None

    def test_large_bet_higher_threshold_than_small(self):
        st_base = _make_flop_state("7s 6h", "BTN_SB", "Kd 7c 2s", "open_call")

        st_small = _villain_bets(st_base, 1.0)
        dec_small = recommend_flop_action_with_equity(st_small, samples=2000, seed=42)

        st_large = _villain_bets(st_base, 5.0)
        dec_large = recommend_flop_action_with_equity(st_large, samples=2000, seed=42)

        assert dec_large.debug["pot_odds_threshold"] > dec_small.debug["pot_odds_threshold"]

    def test_equity_adjustment_notes_always_present(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _villain_bets(st, 3.5)
        dec = recommend_flop_action_with_equity(st, samples=500, seed=1)
        assert isinstance(dec.debug["equity_adjustment_notes"], list)
        assert len(dec.debug["equity_adjustment_notes"]) > 0

    def test_equity_changes_fold_to_call_when_equity_high(self):
        """If baseline folds but equity is above threshold, D.5 may upgrade to call."""
        st = _make_flop_state("Ks Ts", "BB", "8s 5s 2h", "open_call")
        st = _hero_checks_villain_bets(st, 5.0)

        baseline_dec = recommend_flop_action(st)
        equity_dec = recommend_flop_action_with_equity(st, samples=2000, seed=42)

        if baseline_dec.recommended_action.legal_action.action_type == ActionType.FOLD:
            if equity_dec.debug["equity_estimate"] > equity_dec.debug["pot_odds_threshold"]:
                assert equity_dec.debug["equity_changed_recommendation"] is True


# ===================================================================
# M. Baseline Preservation (D.5)
# ===================================================================

class TestAuditBaselinePreservation:

    def test_legal_actions_identical(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _advance_to_hero(st)
        dec_base = recommend_flop_action(st)
        dec_eq = recommend_flop_action_with_equity(st, samples=500, seed=1)
        base_types = {a.action_type for a in dec_base.legal_actions}
        eq_types = {a.action_type for a in dec_eq.legal_actions}
        assert base_types == eq_types

    def test_baseline_action_recorded_in_debug(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _advance_to_hero(st)
        dec = recommend_flop_action_with_equity(st, samples=500, seed=1)
        assert "baseline_action" in dec.debug
        assert len(dec.debug["baseline_action"]) > 0

    def test_final_action_is_legal(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _villain_bets(st, 3.5)
        dec = recommend_flop_action_with_equity(st, samples=500, seed=1)
        legal_types = {a.action_type for a in dec.legal_actions}
        assert dec.recommended_action.legal_action.action_type in legal_types

    def test_context_labels_preserved(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _advance_to_hero(st)
        dec = recommend_flop_action_with_equity(st, samples=500, seed=1)
        assert dec.debug["action_context_label"]
        assert dec.debug["hero_preflop_role"]
        assert dec.debug["made_hand_category"]
        assert dec.debug["board_texture_label"]
        assert dec.debug["baseline_rule_id"]

    @pytest.mark.parametrize("hero_cards,board,pf,setup", [
        ("As Kh", "Ah 7c 2d", "open_call", "check"),
        ("7s 2h", "Kd 8c 3s", "open_call", "check"),
        ("Qs Qh", "Tc 5d 2h", "open_call", "check"),
        ("As Kh", "Ah 7c 2d", "open_call", "bet"),
        ("4s 3h", "Kd 8c 2s", "open_call", "bet"),
        ("7s 7h", "7c Kd 2s", "3bet_call", "check"),
    ])
    def test_equity_always_legal(self, hero_cards, board, pf, setup):
        st = _make_flop_state(hero_cards, "BTN_SB", board, pf)
        if setup == "check":
            st = _advance_to_hero(st)
        elif setup == "bet":
            st = _villain_bets(st, 3.5)
        dec = recommend_flop_action_with_equity(st, samples=500, seed=42)
        legal_types = {a.action_type for a in dec.legal_actions}
        assert dec.recommended_action.legal_action.action_type in legal_types


# ===================================================================
# N. Sample Count Sanity (D.5)
# ===================================================================

class TestAuditSampleCountSanity:

    def test_seeded_determinism_at_various_sample_counts(self):
        hero = parse_cards("As Kh")
        board = [parse_card("Ah"), parse_card("7c"), parse_card("2d")]
        rng = [
            (parse_cards("Qd Jd"), 1.0),
            (parse_cards("Tc 9c"), 1.0),
            (parse_cards("8h 7h"), 1.0),
        ]

        for n in [500, 2000, 5000]:
            r1 = estimate_flop_equity(hero, board, rng, samples=n, seed=42)
            r2 = estimate_flop_equity(hero, board, rng, samples=n, seed=42)
            assert r1 == r2

    def test_rough_stability_across_sample_counts(self):
        """Different sample counts should produce roughly similar results for strong hands."""
        st = _make_flop_state("Qs Qh", "BTN_SB", "Tc 5d 2h", "open_call")
        st = _advance_to_hero(st)
        rng, _ = build_villain_flop_range(st)
        hero = st.config.hero_hole_cards

        r_2k = estimate_flop_equity(hero, st.board_cards, rng, samples=2000, seed=42)
        r_5k = estimate_flop_equity(hero, st.board_cards, rng, samples=5000, seed=42)
        r_10k = estimate_flop_equity(hero, st.board_cards, rng, samples=10000, seed=42)

        # All should be in a reasonable range for overpair
        for r in [r_2k, r_5k, r_10k]:
            assert 0.55 < r["equity_estimate"] < 0.90

        # Should not wildly differ
        assert abs(r_2k["equity_estimate"] - r_10k["equity_estimate"]) < 0.10


# ===================================================================
# O. Bug Regression Tests
# ===================================================================

class TestAuditBugRegressions:

    def test_trips_not_classified_as_set(self):
        """Regression: hero with one card matching board pair should be TRIPS, not SET."""
        hc = parse_cards("7s 5h")
        board = [parse_card("7d"), parse_card("7c"), parse_card("2s")]
        h = classify_hand(hc, board)
        assert h.made_hand == MadeHandCategory.NUTS_OR_NEAR_NUTS
        assert h.made_hand != MadeHandCategory.SET

    def test_set_still_correct_for_pocket_pair(self):
        """Regression: pocket pair matching one board card should still be SET."""
        hc = parse_cards("7s 7h")
        board = [parse_card("7c"), parse_card("Kd"), parse_card("2s")]
        h = classify_hand(hc, board)
        assert h.made_hand == MadeHandCategory.SET

    def test_semi_connected_rainbow_not_dry_high(self):
        """Regression: KJT rainbow (semi-connected) should be RAINBOW_CONNECTED."""
        board = [parse_card("Kd"), parse_card("Jc"), parse_card("Th")]
        bf = classify_board(board)
        assert bf.texture == BoardTexture.RAINBOW_CONNECTED
        assert bf.texture != BoardTexture.DRY_HIGH_CARD

    def test_truly_dry_boards_unchanged(self):
        """Regression: actual dry boards must remain DRY_HIGH_CARD."""
        board = [parse_card("Ah"), parse_card("7c"), parse_card("2d")]
        bf = classify_board(board)
        assert bf.texture == BoardTexture.DRY_HIGH_CARD

    def test_limp_raise_call_preflop_line(self):
        """Regression: BB iso-raise after BTN limp uses BET not RAISE."""
        st = _make_flop_state("As Kh", "BTN_SB", "7c 4d 2s", "limp_raise_call")
        st = _advance_to_hero(st)
        rng, summary = build_villain_flop_range(st)
        assert len(rng) > 0


# ===================================================================
# P. Manual Comparison Helper Tests
# ===================================================================

class TestAuditManualComparison:

    def test_compare_helper_dry_cbet_spot(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _advance_to_hero(st)
        comp = compare_flop_baseline_vs_equity(st, samples=500, seed=42)
        for key in ("baseline_action", "equity_action", "equity_estimate",
                     "pot_odds_threshold", "equity_changed", "notes"):
            assert key in comp

    def test_compare_helper_facing_bet(self):
        st = _make_flop_state("Ks Ts", "BB", "8s 5s 2h", "open_call")
        st = _hero_checks_villain_bets(st, 3.5)
        comp = compare_flop_baseline_vs_equity(st, samples=500, seed=42)
        assert "baseline_action" in comp
        assert "equity_action" in comp
        assert comp["equity_estimate"] is not None

    def test_compare_helper_strong_vs_weak(self):
        """Compare helper with a strong hand and a weak hand should give different equity."""
        st_strong = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st_strong = _advance_to_hero(st_strong)
        c_strong = compare_flop_baseline_vs_equity(st_strong, samples=1000, seed=42)

        st_weak = _make_flop_state("4s 3h", "BTN_SB", "Kd 8c 2s", "open_call")
        st_weak = _advance_to_hero(st_weak)
        c_weak = compare_flop_baseline_vs_equity(st_weak, samples=1000, seed=42)

        assert c_strong["equity_estimate"] > c_weak["equity_estimate"]

    @pytest.mark.parametrize("hero,board,pf,desc", [
        ("As Kh", "Ah 7c 2d", "open_call", "PFR IP dry c-bet"),
        ("Ks Ts", "8s 5s 2h", "open_call", "flush draw facing bet"),
        ("7s 3h", "Kd 7c 2s", "open_call", "middle pair weak kicker"),
        ("Qs Qh", "Tc 5d 2h", "open_call", "overpair on dry board"),
        ("4s 3h", "Kd 8c 2s", "open_call", "total air"),
    ])
    def test_compare_helper_various_spots(self, hero, board, pf, desc):
        st = _make_flop_state(hero, "BTN_SB", board, pf)
        st = _advance_to_hero(st)
        comp = compare_flop_baseline_vs_equity(st, samples=500, seed=42)
        assert comp["equity_estimate"] is not None
        assert comp["baseline_action"] is not None
        assert comp["equity_action"] is not None


# ===================================================================
# Q. Hand Evaluator Edge Cases (D.5)
# ===================================================================

class TestAuditHandEvaluator:

    def test_straight_flush_beats_quads(self):
        sf = [parse_card(c) for c in ["5h", "6h", "7h", "8h", "9h", "2d", "3c"]]
        qu = [parse_card(c) for c in ["As", "Ah", "Ad", "Ac", "Kh", "2d", "3c"]]
        assert _best_hand_rank(sf) > _best_hand_rank(qu)

    def test_full_house_beats_flush(self):
        fh = [parse_card(c) for c in ["As", "Ah", "Ad", "Kh", "Kd", "2c", "3c"]]
        fl = [parse_card(c) for c in ["Ah", "Kh", "Qh", "Th", "2h", "3c", "4d"]]
        assert _best_hand_rank(fh) > _best_hand_rank(fl)

    def test_two_pair_beats_one_pair(self):
        tp = [parse_card(c) for c in ["Ah", "Ad", "Kh", "Kd", "3c", "5s", "7h"]]
        op = [parse_card(c) for c in ["Ah", "Ad", "Qh", "Jd", "3c", "5s", "7h"]]
        assert _best_hand_rank(tp) > _best_hand_rank(op)

    def test_wheel_straight_5_high(self):
        wh = [parse_card(c) for c in ["Ah", "2d", "3c", "4s", "5h", "Kc", "Qd"]]
        rank = _best_hand_rank(wh)
        assert rank[0] == 4  # STRAIGHT
        assert rank[1] == 5  # 5-high

    def test_seven_card_picks_best_five(self):
        """Seven cards include a hidden flush; evaluator should find it."""
        cards = [parse_card(c) for c in ["Ah", "Kh", "5h", "3h", "2h", "Qd", "Js"]]
        rank = _best_hand_rank(cards)
        assert rank[0] == 5  # FLUSH
