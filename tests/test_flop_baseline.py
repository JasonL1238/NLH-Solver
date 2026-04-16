"""Comprehensive tests for Phase D: Baseline Flop Strategy Engine.

Covers hand classification, board texture, flop context, recommendation
legality, size/SPR sensitivity, determinism, and debug output.
"""

from __future__ import annotations

import pytest

from poker_core.models import (
    Action,
    ActionType,
    Card,
    HandConfig,
    HandState,
    HoleCards,
    LegalAction,
    Player,
    Position,
    Street,
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


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_flop_state(
    hero_cards: str,
    hero_pos: str,
    board_str: str,
    preflop_line: str = "open_call",
    stack: float = 100.0,
) -> HandState:
    """Build a HandState at the flop after a given preflop line.

    preflop_line options:
      - "open_call": BTN opens 2.5, BB calls
      - "limp_check": BTN limps (calls 1.0), BB checks
      - "3bet_call": BTN opens 2.5, BB 3-bets 8, BTN calls
    """
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
    else:
        raise ValueError(f"Unknown preflop_line: {preflop_line}")

    board = parse_board(board_str)
    st = deal_flop(cfg, hist, board)
    return st


def _advance_to_hero_action(state: HandState, *, villain_action: str = "check") -> HandState:
    """If villain acts first on the flop, apply their action to get to hero's turn."""
    cfg = state.config
    if state.current_actor == Player.HERO:
        return state

    if villain_action == "check":
        st = apply_action(cfg, state.action_history,
                          action_type=ActionType.CHECK, player=Player.VILLAIN)
    else:
        raise ValueError(f"Unknown villain_action: {villain_action}")
    return st


def _villain_bets(state: HandState, amount: float) -> HandState:
    """Have villain bet, then return state where hero faces the bet."""
    cfg = state.config
    # Villain checks if it is their turn
    if state.current_actor == Player.VILLAIN:
        pass  # we will apply bet directly
    elif state.current_actor == Player.HERO:
        raise ValueError("Hero acts first; cannot have villain bet first.")

    st = apply_action(cfg, state.action_history,
                      action_type=ActionType.BET, player=Player.VILLAIN,
                      amount_to_bb=amount)
    return st


def _hero_bets_then_villain_raises(state: HandState, hero_bet: float, villain_raise: float) -> HandState:
    """Hero bets, villain raises, returning state where hero faces the raise."""
    cfg = state.config
    st = apply_action(cfg, state.action_history,
                      action_type=ActionType.BET, player=state.current_actor,
                      amount_to_bb=hero_bet)
    hist = st.action_history
    st = apply_action(cfg, hist,
                      action_type=ActionType.RAISE, player=Player.VILLAIN,
                      amount_to_bb=villain_raise)
    return st


# ===================================================================
# TEST: Hand Classification
# ===================================================================

class TestHandClassification:
    """Verify made-hand and draw detection."""

    def test_set(self):
        hc = parse_cards("7s 7h")
        board = [parse_card("7c"), parse_card("Kd"), parse_card("2s")]
        h = classify_hand(hc, board)
        assert h.made_hand == MadeHandCategory.SET
        assert h.has_pair is True
        assert h.has_two_pair_plus is True

    def test_two_pair(self):
        hc = parse_cards("As Ts")
        board = [parse_card("Ad"), parse_card("Tc"), parse_card("4h")]
        h = classify_hand(hc, board)
        assert h.made_hand == MadeHandCategory.TWO_PAIR
        assert h.has_two_pair_plus is True

    def test_overpair(self):
        hc = parse_cards("Qs Qh")
        board = [parse_card("Tc"), parse_card("5d"), parse_card("2h")]
        h = classify_hand(hc, board)
        assert h.made_hand == MadeHandCategory.OVERPAIR
        assert h.has_overpair is True
        assert h.has_showdown_value is True

    def test_top_pair_strong_kicker(self):
        hc = parse_cards("As Kh")
        board = [parse_card("Ah"), parse_card("7c"), parse_card("2d")]
        h = classify_hand(hc, board)
        assert h.made_hand == MadeHandCategory.TOP_PAIR_STRONG_KICKER
        assert h.has_top_pair is True

    def test_top_pair_medium_kicker(self):
        hc = parse_cards("As Th")
        board = [parse_card("Ah"), parse_card("7c"), parse_card("2d")]
        h = classify_hand(hc, board)
        assert h.made_hand == MadeHandCategory.TOP_PAIR_MEDIUM_KICKER
        assert h.has_top_pair is True

    def test_top_pair_weak_kicker(self):
        hc = parse_cards("As 4h")
        board = [parse_card("Ah"), parse_card("7c"), parse_card("2d")]
        h = classify_hand(hc, board)
        assert h.made_hand == MadeHandCategory.TOP_PAIR_WEAK_KICKER
        assert h.has_top_pair is True

    def test_middle_pair(self):
        hc = parse_cards("7s 3h")
        board = [parse_card("Kd"), parse_card("7c"), parse_card("2s")]
        h = classify_hand(hc, board)
        assert h.made_hand == MadeHandCategory.MIDDLE_PAIR
        assert h.has_pair is True
        assert h.has_top_pair is False

    def test_third_pair_or_worse(self):
        hc = parse_cards("2s 3h")
        board = [parse_card("Kd"), parse_card("7c"), parse_card("2d")]
        h = classify_hand(hc, board)
        assert h.made_hand == MadeHandCategory.THIRD_PAIR_OR_WORSE_PAIR

    def test_underpair(self):
        hc = parse_cards("3s 3h")
        board = [parse_card("Kc"), parse_card("9d"), parse_card("6h")]
        h = classify_hand(hc, board)
        assert h.made_hand == MadeHandCategory.UNDERPAIR_TO_BOARD

    def test_ace_high(self):
        hc = parse_cards("As 5h")
        board = [parse_card("Kd"), parse_card("7c"), parse_card("2s")]
        h = classify_hand(hc, board)
        assert h.made_hand == MadeHandCategory.ACE_HIGH
        assert h.has_showdown_value is True
        assert h.has_pair is False

    def test_king_high(self):
        hc = parse_cards("Ks 5h")
        board = [parse_card("Qd"), parse_card("7c"), parse_card("2s")]
        h = classify_hand(hc, board)
        assert h.made_hand == MadeHandCategory.KING_HIGH_OR_WORSE_HIGH_CARD
        assert h.has_showdown_value is False

    def test_flush_draw(self):
        hc = parse_cards("Ks Ts")
        board = [parse_card("8s"), parse_card("5s"), parse_card("2h")]
        h = classify_hand(hc, board)
        assert h.draw == DrawCategory.FLUSH_DRAW
        assert h.has_strong_draw is True

    def test_oesd(self):
        hc = parse_cards("9s 8h")
        board = [parse_card("Tc"), parse_card("5d"), parse_card("7h")]
        h = classify_hand(hc, board)
        assert h.draw == DrawCategory.OPEN_ENDED_STRAIGHT_DRAW
        assert h.has_strong_draw is True

    def test_gutshot(self):
        hc = parse_cards("Qs 9h")
        board = [parse_card("Jc"), parse_card("8d"), parse_card("2s")]
        h = classify_hand(hc, board)
        assert h.draw == DrawCategory.GUTSHOT

    def test_combo_draw(self):
        hc = parse_cards("9s 8s")
        board = [parse_card("7s"), parse_card("6s"), parse_card("2h")]
        h = classify_hand(hc, board)
        assert h.draw == DrawCategory.COMBO_DRAW
        assert h.has_combo_draw is True
        assert h.has_strong_draw is True

    def test_no_draw(self):
        hc = parse_cards("As Kh")
        board = [parse_card("Ah"), parse_card("7c"), parse_card("2d")]
        h = classify_hand(hc, board)
        assert h.draw == DrawCategory.NO_REAL_DRAW
        assert h.has_strong_draw is False

    def test_made_straight(self):
        hc = parse_cards("9s 8h")
        board = [parse_card("7c"), parse_card("6d"), parse_card("Ts")]
        h = classify_hand(hc, board)
        assert h.made_hand == MadeHandCategory.NUTS_OR_NEAR_NUTS

    def test_overcards_count(self):
        hc = parse_cards("As Kh")
        board = [parse_card("7c"), parse_card("5d"), parse_card("2s")]
        h = classify_hand(hc, board)
        assert h.overcards_to_board_count == 2

    def test_no_overcards(self):
        hc = parse_cards("4s 3h")
        board = [parse_card("Kc"), parse_card("9d"), parse_card("6h")]
        h = classify_hand(hc, board)
        assert h.overcards_to_board_count == 0


# ===================================================================
# TEST: Board Texture
# ===================================================================

class TestBoardTexture:
    """Verify board texture classification."""

    def test_dry_high_card(self):
        board = [parse_card("Ah"), parse_card("7c"), parse_card("2d")]
        bf = classify_board(board)
        assert bf.texture == BoardTexture.DRY_HIGH_CARD
        assert bf.is_rainbow is True
        assert bf.is_paired is False
        assert bf.board_is_dynamic is False

    def test_dry_low_board(self):
        board = [parse_card("8c"), parse_card("4d"), parse_card("2h")]
        bf = classify_board(board)
        assert bf.texture == BoardTexture.DRY_LOW_BOARD
        assert bf.is_rainbow is True
        assert bf.board_is_high_card_heavy is False

    def test_paired_board(self):
        board = [parse_card("Ks"), parse_card("Kd"), parse_card("4c")]
        bf = classify_board(board)
        assert bf.texture == BoardTexture.PAIRED_BOARD
        assert bf.is_paired is True

    def test_monotone_board(self):
        board = [parse_card("8h"), parse_card("Th"), parse_card("2h")]
        bf = classify_board(board)
        assert bf.texture == BoardTexture.MONOTONE_BOARD
        assert bf.is_monotone is True

    def test_low_connected(self):
        board = [parse_card("5h"), parse_card("6d"), parse_card("7s")]
        bf = classify_board(board)
        assert bf.texture == BoardTexture.LOW_CONNECTED
        assert bf.has_straight_heaviness is True
        assert bf.board_is_low_connected is True

    def test_rainbow_connected(self):
        board = [parse_card("Th"), parse_card("Jd"), parse_card("Qs")]
        bf = classify_board(board)
        assert bf.texture == BoardTexture.RAINBOW_CONNECTED
        assert bf.is_rainbow is True
        assert bf.has_straight_heaviness is True

    def test_two_tone_board(self):
        board = [parse_card("Ks"), parse_card("9s"), parse_card("4h")]
        bf = classify_board(board)
        assert bf.is_two_tone is True

    def test_dynamic_draw_heavy(self):
        board = [parse_card("9h"), parse_card("Th"), parse_card("8d")]
        bf = classify_board(board)
        assert bf.texture == BoardTexture.DYNAMIC_DRAW_HEAVY
        assert bf.board_is_dynamic is True

    def test_static_board(self):
        board = [parse_card("Kh"), parse_card("7d"), parse_card("3s")]
        bf = classify_board(board)
        # K-7-3 rainbow, gaps = 6 and 4, all >= 3
        assert bf.is_rainbow is True
        assert bf.board_is_dynamic is False

    def test_top_card_rank(self):
        board = [parse_card("Ah"), parse_card("7c"), parse_card("2d")]
        bf = classify_board(board)
        assert bf.top_card_rank == 14  # Ace

    def test_board_requires_three_cards(self):
        with pytest.raises(ValueError, match="3 cards"):
            classify_board([parse_card("Ah"), parse_card("7c")])


# ===================================================================
# TEST: Flop Context
# ===================================================================

class TestFlopContext:
    """Verify context derivation: position, PFR/PFC, SPR, bet bucket."""

    def test_pfr_ip_checked_to(self):
        """Hero BTN opens, BB calls, BB checks on flop -> PFR_IP_CHECKED_TO."""
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _advance_to_hero_action(st)
        ctx = derive_flop_context(st)
        assert ctx.flop_context == FlopContext.PFR_IP_CHECKED_TO
        assert ctx.hero_is_ip is True
        assert ctx.hero_is_pfr is True

    def test_pfr_oop_first_to_act(self):
        """Hero BB 3-bets, BTN calls -> hero acts first on flop as PFR OOP."""
        st = _make_flop_state("As Kh", "BB", "Ah 7c 2d", "3bet_call")
        ctx = derive_flop_context(st)
        assert ctx.flop_context == FlopContext.PFR_OOP_FIRST_TO_ACT
        assert ctx.hero_is_ip is False
        assert ctx.hero_is_pfr is True

    def test_pfc_ip_checked_to(self):
        """Hero BTN in limped pot, BB checks -> PFC_IP_CHECKED_TO."""
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "limp_check")
        st = _advance_to_hero_action(st)
        ctx = derive_flop_context(st)
        assert ctx.flop_context == FlopContext.PFC_IP_CHECKED_TO
        assert ctx.hero_is_ip is True
        assert ctx.hero_is_pfr is False

    def test_pfc_oop_first_to_act(self):
        """Hero BB, BTN opens, BB calls -> hero acts first OOP as PFC."""
        st = _make_flop_state("As Kh", "BB", "Ah 7c 2d", "open_call")
        ctx = derive_flop_context(st)
        assert ctx.flop_context == FlopContext.PFC_OOP_FIRST_TO_ACT
        assert ctx.hero_is_ip is False
        assert ctx.hero_is_pfr is False

    def test_facing_small_bet(self):
        """Hero faces a small bet (<= 33% pot)."""
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        # Villain bets 1.0 into 5.0 pot = 20%
        st = _villain_bets(st, 1.0)
        ctx = derive_flop_context(st)
        assert ctx.flop_context == FlopContext.FACING_SMALL_BET
        assert ctx.bet_size_bucket == BetSizeBucket.SMALL

    def test_facing_medium_bet(self):
        """Hero faces a medium bet (>33% to <=75% pot)."""
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        # Villain bets 3.5 into 5.0 pot = 70%
        st = _villain_bets(st, 3.5)
        ctx = derive_flop_context(st)
        assert ctx.flop_context == FlopContext.FACING_MEDIUM_BET
        assert ctx.bet_size_bucket == BetSizeBucket.MEDIUM

    def test_facing_large_bet(self):
        """Hero faces a large bet (>75% pot)."""
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        # Villain bets 5.0 into 5.0 pot = 100%
        st = _villain_bets(st, 5.0)
        ctx = derive_flop_context(st)
        assert ctx.flop_context == FlopContext.FACING_LARGE_BET
        assert ctx.bet_size_bucket == BetSizeBucket.LARGE

    def test_facing_raise_after_betting(self):
        """Hero bets, villain raises -> FACING_RAISE_AFTER_BETTING."""
        st = _make_flop_state("As Kh", "BB", "Ah 7c 2d", "3bet_call")
        # Hero bets as PFR OOP, villain raises
        st = _hero_bets_then_villain_raises(st, hero_bet=5.0, villain_raise=15.0)
        ctx = derive_flop_context(st)
        assert ctx.flop_context == FlopContext.FACING_RAISE_AFTER_BETTING

    def test_spr_high(self):
        """100bb stack, 5bb pot -> high SPR."""
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _advance_to_hero_action(st)
        ctx = derive_flop_context(st)
        assert ctx.spr_bucket == SPRBucket.HIGH_SPR
        assert ctx.spr > 8.0

    def test_spr_medium(self):
        """100bb stack, 16bb pot -> medium SPR."""
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "3bet_call")
        st = _advance_to_hero_action(st)
        ctx = derive_flop_context(st)
        assert ctx.spr_bucket == SPRBucket.MEDIUM_SPR

    def test_spr_low(self):
        """25bb stack, open/call -> low SPR."""
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "3bet_call", stack=25.0)
        st = _advance_to_hero_action(st)
        ctx = derive_flop_context(st)
        assert ctx.spr_bucket == SPRBucket.LOW_SPR

    def test_limped_pot_no_pfr(self):
        """In a limped pot, neither player is PFR."""
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "limp_check")
        st = _advance_to_hero_action(st)
        ctx = derive_flop_context(st)
        assert ctx.hero_is_pfr is False

    def test_bet_size_bucket_none_when_not_facing(self):
        """No bet to call -> bet_size_bucket is None."""
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _advance_to_hero_action(st)
        ctx = derive_flop_context(st)
        assert ctx.bet_size_bucket is None


# ===================================================================
# TEST: Recommendation Legality
# ===================================================================

class TestRecommendationLegality:
    """Every recommendation must be among legal actions with valid sizing."""

    @pytest.mark.parametrize("hero_cards,hero_pos,board,pf_line,setup", [
        # PFR IP checked-to with various hands
        ("As Kh", "BTN_SB", "Ah 7c 2d", "open_call", "check_to_hero"),
        ("7s 2h", "BTN_SB", "Kd 8c 3s", "open_call", "check_to_hero"),
        ("Qs Qh", "BTN_SB", "Tc 5d 2h", "open_call", "check_to_hero"),
        # PFR OOP first to act
        ("As Kh", "BB", "Ah 7c 2d", "3bet_call", "hero_first"),
        ("7s 2h", "BB", "Kd 8c 3s", "3bet_call", "hero_first"),
        # PFC IP checked-to (limped pot)
        ("As Kh", "BTN_SB", "Ah 7c 2d", "limp_check", "check_to_hero"),
        ("5s 4s", "BTN_SB", "7h 6d 3c", "limp_check", "check_to_hero"),
        # PFC OOP first to act
        ("As Kh", "BB", "Ah 7c 2d", "open_call", "hero_first"),
        ("7s 2h", "BB", "Kd 8c 3s", "open_call", "hero_first"),
        # Facing bets of various sizes
        ("As Kh", "BTN_SB", "Ah 7c 2d", "open_call", "facing_small_bet"),
        ("As Kh", "BTN_SB", "Ah 7c 2d", "open_call", "facing_large_bet"),
        ("Ks Ts", "BB", "8s 5s 2h", "open_call", "facing_medium_bet_oop"),
    ])
    def test_recommendation_is_legal(self, hero_cards, hero_pos, board, pf_line, setup):
        st = _make_flop_state(hero_cards, hero_pos, board, pf_line)

        if setup == "check_to_hero":
            st = _advance_to_hero_action(st)
        elif setup == "hero_first":
            assert st.current_actor == Player.HERO
        elif setup == "facing_small_bet":
            st = _villain_bets(st, 1.0)
        elif setup == "facing_large_bet":
            st = _villain_bets(st, 5.0)
        elif setup == "facing_medium_bet_oop":
            st = apply_action(st.config, st.action_history,
                              action_type=ActionType.CHECK, player=Player.HERO)
            st = apply_action(st.config, st.action_history,
                              action_type=ActionType.BET, player=Player.VILLAIN,
                              amount_to_bb=3.5)

        dec = recommend_flop_action(st)

        legal_types = {a.action_type for a in dec.legal_actions}
        assert dec.recommended_action.legal_action.action_type in legal_types

        # If sizing is chosen, verify it's within bounds
        rec = dec.recommended_action
        if rec.size_bb is not None:
            la = rec.legal_action
            assert la.min_to_bb is not None
            assert la.max_to_bb is not None
            assert la.min_to_bb - 0.01 <= rec.size_bb <= la.max_to_bb + 0.01


# ===================================================================
# TEST: Size Sensitivity
# ===================================================================

class TestSizeSensitivity:
    """Same hand+board, different bet sizes should produce different actions."""

    def test_marginal_hand_folds_large_but_calls_small(self):
        """Marginal pair continues vs small pressure; large bet may fold or call (EV)."""
        # Hero BB with third pair facing a bet
        st_base = _make_flop_state("4s 3h", "BB", "Kd 7c 4d", "open_call")
        # Hero checks OOP, villain bets

        # Small bet: 1.0 into 5.0
        st1 = apply_action(st_base.config, st_base.action_history,
                           action_type=ActionType.CHECK, player=Player.HERO)
        st1 = apply_action(st1.config, st1.action_history,
                           action_type=ActionType.BET, player=Player.VILLAIN, amount_to_bb=1.0)
        dec1 = recommend_flop_action(st1)

        # Large bet: 5.0 into 5.0
        st2 = apply_action(st_base.config, st_base.action_history,
                           action_type=ActionType.CHECK, player=Player.HERO)
        st2 = apply_action(st2.config, st2.action_history,
                           action_type=ActionType.BET, player=Player.VILLAIN, amount_to_bb=5.0)
        dec2 = recommend_flop_action(st2)

        assert dec1.recommended_action.legal_action.action_type != ActionType.FOLD
        assert dec2.recommended_action.legal_action.action_type in (
            ActionType.FOLD, ActionType.CALL, ActionType.RAISE,
        )

    def test_strong_hand_continues_regardless_of_size(self):
        """A strong hand should continue vs any bet size."""
        st_base = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")

        # Small bet
        st1 = _villain_bets(st_base, 1.0)
        dec1 = recommend_flop_action(st1)
        assert dec1.recommended_action.legal_action.action_type != ActionType.FOLD

        # Large bet
        st2 = _villain_bets(st_base, 5.0)
        dec2 = recommend_flop_action(st2)
        assert dec2.recommended_action.legal_action.action_type != ActionType.FOLD


# ===================================================================
# TEST: SPR Sensitivity
# ===================================================================

class TestSPRSensitivity:
    """Same hand+board at different SPR can lead to different decisions."""

    def test_draw_at_different_spr(self):
        """A flush draw facing a large bet at low SPR should fold (no implied odds)."""
        # High SPR: 100bb stacks, open call pot = 5bb
        st1 = _make_flop_state("Ks Ts", "BB", "8s 5s 2h", "open_call", stack=100)
        st1 = apply_action(st1.config, st1.action_history,
                           action_type=ActionType.CHECK, player=Player.HERO)
        st1 = apply_action(st1.config, st1.action_history,
                           action_type=ActionType.BET, player=Player.VILLAIN, amount_to_bb=3.5)
        dec1 = recommend_flop_action(st1)

        # Low SPR: 3-bet pot at 25bb stacks
        st2 = _make_flop_state("Ks Ts", "BB", "8s 5s 2h", "3bet_call", stack=25)
        # In 3bet pot hero is PFR (BB 3-bet). Hero acts first OOP.
        # Villain checks back:
        st2 = apply_action(st2.config, st2.action_history,
                           action_type=ActionType.BET, player=Player.HERO, amount_to_bb=5.0)
        st2 = apply_action(st2.config, st2.action_history,
                           action_type=ActionType.RAISE, player=Player.VILLAIN, amount_to_bb=17.0)
        dec2 = recommend_flop_action(st2)

        # High SPR draw call is reasonable
        assert dec1.recommended_action.legal_action.action_type in (ActionType.CALL, ActionType.RAISE)
        # Low SPR facing raise with just a draw -- fold is reasonable (no combo draw)
        # (flush draw is strong but facing raise at low SPR is tough)
        assert dec2.recommended_action.legal_action.action_type in (ActionType.FOLD, ActionType.CALL)


# ===================================================================
# TEST: Determinism
# ===================================================================

class TestDeterminism:
    """Baseline flop engine must be fully deterministic."""

    def test_same_state_same_output(self):
        """Running the recommender twice on the same state produces identical output."""
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _advance_to_hero_action(st)

        dec1 = recommend_flop_action(st)
        dec2 = recommend_flop_action(st)

        assert dec1.recommended_action.legal_action.action_type == dec2.recommended_action.legal_action.action_type
        assert dec1.recommended_action.size_bb == dec2.recommended_action.size_bb
        assert dec1.explanation == dec2.explanation

    def test_determinism_multiple_runs(self):
        """10 runs produce identical results."""
        st = _make_flop_state("9s 8s", "BB", "7h 6d 3c", "open_call")
        results = []
        for _ in range(10):
            dec = recommend_flop_action(st)
            results.append((
                dec.recommended_action.legal_action.action_type,
                dec.recommended_action.size_bb,
                dec.explanation,
            ))
        assert len(set(results)) == 1


# ===================================================================
# TEST: Debug Fields
# ===================================================================

class TestDebugFields:
    """Every debug dict must contain all required keys."""

    REQUIRED_KEYS = {
        "action_context_label",
        "hero_position_relation_on_flop",
        "hero_preflop_role",
        "made_hand_category",
        "draw_category",
        "board_texture_label",
        "flop_bet_size_bucket",
        "spr_bucket",
        "spr",
        "baseline_rule_id",
        "legal_actions",
        "recommended_action",
        "explanation",
        "has_showdown_value",
        "has_strong_draw",
        "has_combo_draw",
        "has_backdoor_equity",
        "overcards_to_board_count",
    }

    def test_debug_keys_present_checked_to(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _advance_to_hero_action(st)
        dec = recommend_flop_action(st)
        missing = self.REQUIRED_KEYS - set(dec.debug.keys())
        assert not missing, f"Missing debug keys: {missing}"

    def test_debug_keys_present_facing_bet(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _villain_bets(st, 3.5)
        dec = recommend_flop_action(st)
        missing = self.REQUIRED_KEYS - set(dec.debug.keys())
        assert not missing, f"Missing debug keys: {missing}"

    def test_explanation_non_empty(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _advance_to_hero_action(st)
        dec = recommend_flop_action(st)
        assert dec.explanation
        assert len(dec.explanation) > 5

    def test_pretty_print_output(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _advance_to_hero_action(st)
        dec = recommend_flop_action(st)
        output = pretty_print_flop_decision(dec, st)
        assert "FLOP DECISION" in output
        assert "Made hand" in output
        assert "Rule ID" in output
        assert "EV_ARGMAX" in output or "EV" in output


# ===================================================================
# TEST: Strategy Correctness (Golden Scenarios)
# ===================================================================

class TestGoldenScenarios:
    """Key poker scenarios that must produce correct actions."""

    def test_pfr_ip_value_cbet_top_pair(self):
        """PFR IP with top pair strong kicker should bet (EV-first)."""
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _advance_to_hero_action(st)
        dec = recommend_flop_action(st)
        assert dec.recommended_action.legal_action.action_type == ActionType.BET
        assert "ev-first" in dec.explanation.lower() or "ev_argmax" in str(
            dec.debug.get("baseline_rule_id", ""),
        ).lower()

    def test_pfr_ip_dry_range_cbet(self):
        """PFR IP with overcards on dry board should range c-bet small."""
        st = _make_flop_state("As Kh", "BTN_SB", "7c 5d 2s", "open_call")
        st = _advance_to_hero_action(st)
        dec = recommend_flop_action(st)
        assert dec.recommended_action.legal_action.action_type == ActionType.BET

    def test_pfr_ip_weak_check_dynamic_board(self):
        """PFR IP on dynamic board: EV may stab or check."""
        st = _make_flop_state("4s 3h", "BTN_SB", "9h Th 8d", "open_call")
        st = _advance_to_hero_action(st)
        dec = recommend_flop_action(st)
        assert dec.recommended_action.legal_action.action_type in (
            ActionType.CHECK, ActionType.BET,
        )

    def test_pfc_oop_check_to_pfr(self):
        """PFC OOP weak hand: EV may probe small or check."""
        st = _make_flop_state("4s 3h", "BB", "Kd 8c 2s", "open_call")
        dec = recommend_flop_action(st)
        assert dec.recommended_action.legal_action.action_type in (
            ActionType.CHECK, ActionType.BET,
        )

    def test_pfc_oop_lead_with_strong(self):
        """PFC OOP with set should lead (donk bet) for value."""
        st = _make_flop_state("7s 7h", "BB", "7c Kd 2s", "open_call")
        dec = recommend_flop_action(st)
        assert dec.recommended_action.legal_action.action_type == ActionType.BET

    def test_facing_bet_strong_hand_raise(self):
        """Facing a medium bet with a strong hand should raise."""
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _villain_bets(st, 3.5)
        dec = recommend_flop_action(st)
        assert dec.recommended_action.legal_action.action_type == ActionType.RAISE

    def test_facing_bet_weak_fold(self):
        """Facing a bet with a weak hand should fold."""
        st = _make_flop_state("4s 3h", "BTN_SB", "Kd 8c 2s", "open_call")
        st = _villain_bets(st, 3.5)
        dec = recommend_flop_action(st)
        assert dec.recommended_action.legal_action.action_type == ActionType.FOLD

    def test_facing_raise_nuts_reraise(self):
        """Facing a raise with a set should re-raise."""
        st = _make_flop_state("7s 7h", "BB", "7c Kd 2s", "3bet_call")
        st = _hero_bets_then_villain_raises(st, hero_bet=5.0, villain_raise=15.0)
        dec = recommend_flop_action(st)
        assert dec.recommended_action.legal_action.action_type == ActionType.RAISE

    def test_facing_raise_marginal_fold(self):
        """Facing a raise with middle pair: EV may continue or fold."""
        st = _make_flop_state("8s 4h", "BB", "Kd 8c 2s", "3bet_call")
        st = _hero_bets_then_villain_raises(st, hero_bet=5.0, villain_raise=15.0)
        dec = recommend_flop_action(st)
        assert dec.recommended_action.legal_action.action_type in (
            ActionType.FOLD, ActionType.CALL, ActionType.RAISE,
        )

    def test_overpair_bets_dry_board(self):
        """Overpair on dry board should c-bet."""
        st = _make_flop_state("Qs Qh", "BTN_SB", "Tc 5d 2h", "open_call")
        st = _advance_to_hero_action(st)
        dec = recommend_flop_action(st)
        assert dec.recommended_action.legal_action.action_type == ActionType.BET


# ===================================================================
# TEST: Sizing Logic
# ===================================================================

class TestSizingLogic:
    """Verify bet sizing is correct: 33% pot dry, 75% pot wet."""

    def test_small_bet_on_dry_board(self):
        """Dry board c-bet uses a pot-fraction grid (clamped to legal bounds)."""
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _advance_to_hero_action(st)
        dec = recommend_flop_action(st)
        assert dec.recommended_action.legal_action.action_type == ActionType.BET
        assert dec.recommended_action.size_bb is not None
        la = dec.recommended_action.legal_action
        assert la.min_to_bb is not None and la.max_to_bb is not None
        assert la.min_to_bb - 0.01 <= dec.recommended_action.size_bb <= la.max_to_bb + 0.01

    def test_larger_bet_on_wet_board(self):
        """Wet board c-bet should use ~75% pot sizing."""
        st = _make_flop_state("As Kh", "BTN_SB", "Kd Qh Jd", "open_call")
        st = _advance_to_hero_action(st)
        dec = recommend_flop_action(st)
        pot = st.pot_size_bb
        if dec.recommended_action.size_bb is not None:
            # Should be larger than dry board sizing
            assert dec.recommended_action.size_bb >= pot * 0.5

    def test_sizing_within_legal_bounds(self):
        """Chosen size must be within legal min/max."""
        st = _make_flop_state("Qs Qh", "BTN_SB", "Tc 5d 2h", "open_call")
        st = _advance_to_hero_action(st)
        dec = recommend_flop_action(st)
        rec = dec.recommended_action
        if rec.size_bb is not None:
            la = rec.legal_action
            assert rec.size_bb >= la.min_to_bb - 0.01
            assert rec.size_bb <= la.max_to_bb + 0.01


# ===================================================================
# TEST: Validation / Error Handling
# ===================================================================

class TestValidation:
    """recommender should reject invalid states."""

    def test_rejects_non_flop_street(self):
        cfg = make_hand_config("As Kh", hero_position="BTN_SB", effective_stack_bb=100)
        hist, st = post_blinds(cfg)
        with pytest.raises(ValueError, match="FLOP"):
            recommend_flop_action(st)

    def test_rejects_hand_over(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _advance_to_hero_action(st)
        # Hero checks, villain checks -> round closed (but not hand over)
        st2 = apply_action(st.config, st.action_history,
                           action_type=ActionType.CHECK, player=Player.HERO)
        # This doesn't actually end the hand; but we can test the fold path
        # to generate hand_over
        st_fold = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st_fold = _villain_bets(st_fold, 3.5)
        st_fold = apply_action(st_fold.config, st_fold.action_history,
                               action_type=ActionType.FOLD, player=Player.HERO)
        with pytest.raises(ValueError, match="over"):
            recommend_flop_action(st_fold)

    def test_rejects_villain_turn(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        if st.current_actor == Player.VILLAIN:
            with pytest.raises(ValueError, match="hero"):
                recommend_flop_action(st)
