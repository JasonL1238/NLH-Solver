"""Tests for flop adaptive / exploit layer (Phase E-style)."""

from __future__ import annotations

import pytest

from poker_core.models import ActionType, Player
from poker_core.parser import make_hand_config, parse_board, parse_card
from poker_core.transitions import apply_action, deal_flop, post_blinds

from flop_adaptive import (
    FlopOpponentProfile,
    compare_flop_baseline_vs_equity_vs_adaptive,
    compute_range_adjustment_hints,
    recommend_adaptive_flop_action,
    record_flop_hand,
    simulate_flop_overfolder,
    simulate_flop_raise_heavy,
    simulate_flop_stab_heavy,
    simulate_flop_sticky_caller,
)
from flop_adaptive.opponent_model import (
    K_PRIOR_STRENGTH,
    STAT_PRIORS,
    confidence_weight,
    smooth_rate,
)


# ---------------------------------------------------------------------------
# Helpers
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
    elif preflop_line == "3bet_call":
        st = apply_action(cfg, hist, action_type=ActionType.RAISE, player=btn, amount_to_bb=2.5)
        hist = st.action_history
        st = apply_action(cfg, hist, action_type=ActionType.RAISE, player=bb, amount_to_bb=8.0)
        hist = st.action_history
        st = apply_action(cfg, hist, action_type=ActionType.CALL, player=btn, amount_to_bb=8.0)
        hist = st.action_history
    else:
        raise ValueError(preflop_line)

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


# ---------------------------------------------------------------------------
# A. Stat tracking
# ---------------------------------------------------------------------------

class TestFlopStatTracking:
    def test_record_open_check_check(self):
        """BTN open BB call: villain not PFR, villain checks first OOP."""
        cfg = make_hand_config("As Kh", hero_position="BTN_SB", effective_stack_bb=100.0)
        hist, _ = post_blinds(cfg)
        btn, bb = cfg.btn_player, cfg.bb_player
        st = apply_action(cfg, hist, action_type=ActionType.RAISE, player=btn, amount_to_bb=2.5)
        hist = st.action_history
        st = apply_action(cfg, hist, action_type=ActionType.CALL, player=bb, amount_to_bb=2.5)
        hist = st.action_history
        st = deal_flop(cfg, hist, tuple(parse_board("7c 4d 2s")))
        hist = st.action_history
        st = apply_action(cfg, hist, action_type=ActionType.CHECK, player=bb)
        hist = st.action_history
        st = apply_action(cfg, hist, action_type=ActionType.CHECK, player=btn)
        hist = st.action_history

        p = FlopOpponentProfile()
        record_flop_hand(p, cfg, hist)
        assert p.villain_flop_bet_when_checked_to_opportunities >= 1
        assert p.villain_pfr_cbet_opportunities == 0

    def test_record_3bet_villain_check(self):
        """Villain 3-bet pre: villain PFR, checks flop (check-back)."""
        st = _make_flop_state("As Kh", "BTN_SB", "7c 4d 2s", "3bet_call")
        cfg = st.config
        hist = st.action_history
        st = apply_action(cfg, hist, action_type=ActionType.CHECK, player=Player.VILLAIN)
        hist = st.action_history
        st = apply_action(cfg, hist, action_type=ActionType.CHECK, player=Player.HERO)
        hist = st.action_history

        p = FlopOpponentProfile()
        record_flop_hand(p, cfg, hist)
        assert p.villain_pfr_cbet_opportunities == 1
        assert p.villain_pfr_check_back_count == 1


# ---------------------------------------------------------------------------
# B. Smoothing
# ---------------------------------------------------------------------------

class TestSmoothing:
    def test_small_sample_near_prior(self):
        prior = STAT_PRIORS["fold_to_flop_cbet"]
        sm = smooth_rate(count=2, opp=5, prior=prior, k=K_PRIOR_STRENGTH)
        assert abs(sm - prior) < 0.15

    def test_large_sample_moves_toward_observed(self):
        prior = STAT_PRIORS["fold_to_flop_cbet"]
        sm = smooth_rate(count=80, opp=100, prior=prior, k=K_PRIOR_STRENGTH)
        assert sm > prior + 0.15


# ---------------------------------------------------------------------------
# C. Confidence
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_low_n_low_confidence(self):
        assert confidence_weight(3) < confidence_weight(40)

    def test_zero_opportunities(self):
        assert confidence_weight(0) == 0.0


# ---------------------------------------------------------------------------
# D–J. Adaptive behaviour, equity, determinism, safety
# ---------------------------------------------------------------------------

class TestAdaptiveRecommender:
    def test_determinism(self):
        st = _make_flop_state("4s 2h", "BTN_SB", "Kd 9c 7s", "open_call")
        st = _advance_to_hero(st)
        p = FlopOpponentProfile()
        d1 = recommend_adaptive_flop_action(st, p, samples=300, seed=7)
        d2 = recommend_adaptive_flop_action(st, p, samples=300, seed=7)
        assert d1.adapted_recommendation.legal_action.action_type == d2.adapted_recommendation.legal_action.action_type
        assert d1.debug["equity_estimate"] == d2.debug["equity_estimate"]

    def test_adapted_action_always_legal(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _villain_bets(st, 3.5)
        p = FlopOpponentProfile()
        simulate_flop_overfolder(p, 25)
        dec = recommend_adaptive_flop_action(st, p, samples=400, seed=1)
        legal_types = {a.action_type for a in dec.legal_actions}
        assert dec.adapted_recommendation.legal_action.action_type in legal_types

    def test_debug_has_required_keys(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _advance_to_hero(st)
        p = FlopOpponentProfile()
        dec = recommend_adaptive_flop_action(st, p, samples=300, seed=2)
        for key in (
            "action_context_label",
            "baseline_action",
            "equity_action",
            "final_action",
            "villain_flop_profile_summary",
            "key_smoothed_flop_stats",
            "confidence_summary",
            "range_adjustment_notes",
            "exploit_adjustment_notes",
        ):
            assert key in dec.debug, key

    def test_equity_layer_not_skipped(self):
        st = _make_flop_state("Ks Ts", "BB", "8s 5s 2h", "open_call")
        st = apply_action(st.config, st.action_history, action_type=ActionType.CHECK, player=Player.HERO)
        st = apply_action(st.config, st.action_history, action_type=ActionType.BET, player=Player.VILLAIN, amount_to_bb=3.5)
        p = FlopOpponentProfile()
        dec = recommend_adaptive_flop_action(st, p, samples=400, seed=3)
        assert dec.debug.get("equity_estimate") is not None
        assert dec.debug.get("pot_odds_threshold") is not None

    def test_overfolder_widens_weak_check_to_bet_dry(self):
        """Dry board weak hand as PFR IP: heavy overfolder read -> may BET vs CHECK."""
        st = _make_flop_state("7s 2h", "BTN_SB", "Kc 5d 2s", "open_call")
        st = _advance_to_hero(st)
        p = FlopOpponentProfile()
        simulate_flop_overfolder(p, 60)
        dec = recommend_adaptive_flop_action(st, p, samples=350, seed=11)
        hints = compute_range_adjustment_hints(st, p, dec.baseline_decision.debug)
        assert hints.cbet_frequency_scalar >= 1.0

    def test_compare_helper(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _advance_to_hero(st)
        c = compare_flop_baseline_vs_equity_vs_adaptive(st, FlopOpponentProfile(), samples=200, seed=0)
        assert "baseline_action" in c and "adapted_action" in c

    def test_sticky_profile_tightens_cbet_scalar(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _advance_to_hero(st)
        p = FlopOpponentProfile()
        simulate_flop_sticky_caller(p, 50)
        from flop_baseline.recommender import recommend_flop_action

        dec = recommend_flop_action(st)
        h = compute_range_adjustment_hints(st, p, dec.debug)
        assert h.cbet_frequency_scalar <= 1.0


class TestEquityPreservation:
    def test_no_call_when_equity_fold_clear(self):
        """Air vs large bet: equity fold should not become call from exploit."""
        st = _make_flop_state("4s 3h", "BTN_SB", "Kd 8c 2s", "open_call")
        st = _villain_bets(st, 5.0)
        p = FlopOpponentProfile()
        simulate_flop_overfolder(p, 80)
        dec = recommend_adaptive_flop_action(st, p, samples=500, seed=99)
        if dec.equity_aware_recommendation.legal_action.action_type == ActionType.FOLD:
            thr = dec.debug.get("pot_odds_threshold")
            eq = dec.debug.get("equity_estimate")
            if thr is not None and eq is not None and float(eq) < float(thr) - 0.06:
                assert dec.adapted_recommendation.legal_action.action_type == ActionType.FOLD


class TestHints:
    def test_unknown_profile_neutral_scalars(self):
        from flop_baseline.recommender import recommend_flop_action

        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _advance_to_hero(st)
        p = FlopOpponentProfile()
        dec = recommend_flop_action(st)
        h = compute_range_adjustment_hints(st, p, dec.debug)
        assert 0.9 <= h.cbet_frequency_scalar <= 1.1
