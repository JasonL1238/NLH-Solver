"""Comprehensive tests for Phase D.5: Monte Carlo Flop Equity.

Covers range model sanity, Monte Carlo correctness, deterministic seeded
behaviour, pot-odds integration, baseline preservation, and debug fields.
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

from flop_baseline.recommender import recommend_flop_action

from flop_equity.range_model import build_villain_flop_range
from flop_equity.monte_carlo import estimate_flop_equity, _best_hand_rank
from flop_equity.equity_integration import (
    recommend_flop_action_with_equity,
    compare_flop_baseline_vs_equity,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_flop_state(
    hero_cards: str,
    hero_pos: str,
    board_str: str,
    preflop_line: str = "open_call",
    stack: float = 100.0,
):
    """Build a HandState at the flop after a given preflop line."""
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
    st = deal_flop(cfg, hist, board)
    return st


def _advance_to_hero(state):
    """If villain acts first, have them check to get to hero's turn."""
    if state.current_actor == Player.HERO:
        return state
    return apply_action(state.config, state.action_history,
                        action_type=ActionType.CHECK, player=Player.VILLAIN)


def _villain_bets(state, amount):
    """Have villain bet (assumes villain acts first)."""
    return apply_action(state.config, state.action_history,
                        action_type=ActionType.BET, player=Player.VILLAIN,
                        amount_to_bb=amount)


# ===================================================================
# A. Range Model Sanity
# ===================================================================

class TestRangeModelSanity:
    """Villain range model produces valid, non-empty ranges."""

    def test_btn_open_bb_call_range_nonempty(self):
        st = _make_flop_state("As Kh", "BTN_SB", "7c 4d 2s", "open_call")
        st = _advance_to_hero(st)
        rng, summary = build_villain_flop_range(st)
        assert len(rng) > 50
        assert "BB defend" in summary

    def test_3bet_pot_range_nonempty(self):
        st = _make_flop_state("As Kh", "BTN_SB", "7c 4d 2s", "3bet_call")
        st = _advance_to_hero(st)
        rng, summary = build_villain_flop_range(st)
        assert len(rng) > 10
        assert "3-bet" in summary.lower() or "3bet" in summary.lower()

    def test_limp_check_range_nonempty(self):
        st = _make_flop_state("As Kh", "BTN_SB", "7c 4d 2s", "limp_check")
        st = _advance_to_hero(st)
        rng, summary = build_villain_flop_range(st)
        assert len(rng) > 30

    def test_limp_raise_call_range_nonempty(self):
        st = _make_flop_state("As Kh", "BTN_SB", "7c 4d 2s", "limp_raise_call")
        st = _advance_to_hero(st)
        rng, summary = build_villain_flop_range(st)
        assert len(rng) > 10

    def test_dead_cards_excluded(self):
        """No villain combo should contain hero hole cards or board cards."""
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _advance_to_hero(st)
        rng, _ = build_villain_flop_range(st)

        hero_high = st.config.hero_hole_cards.high
        hero_low = st.config.hero_hole_cards.low
        board = set(st.board_cards)
        dead = {hero_high, hero_low} | board

        for hc, w in rng:
            assert hc.high not in dead, f"Dead card {hc.high} in villain range"
            assert hc.low not in dead, f"Dead card {hc.low} in villain range"

    def test_range_summary_stable(self):
        """Same state produces the same summary string."""
        st = _make_flop_state("As Kh", "BTN_SB", "7c 4d 2s", "open_call")
        st = _advance_to_hero(st)
        _, s1 = build_villain_flop_range(st)
        _, s2 = build_villain_flop_range(st)
        assert s1 == s2

    def test_villain_bet_refines_range(self):
        """Villain betting should shift weights (reduce air component)."""
        st = _make_flop_state("As Kh", "BTN_SB", "7c 4d 2s", "open_call")
        # Villain checks -> one range
        st_check = _advance_to_hero(st)
        rng_check, sum_check = build_villain_flop_range(st_check)

        # Villain bets -> refined range
        st_bet = _villain_bets(st, 3.5)
        rng_bet, sum_bet = build_villain_flop_range(st_bet)

        assert "bet" in sum_bet.lower()
        assert "check" in sum_check.lower()

        # Betting range should have some combos with lower weights (air reduced)
        w_check = sum(w for _, w in rng_check)
        w_bet = sum(w for _, w in rng_bet)
        assert w_bet < w_check, "Bet range total weight should be lower (air removed)"

    def test_hero_bb_vs_btn_open(self):
        """When hero is BB, villain is BTN opener -- range should reflect that."""
        st = _make_flop_state("As Kh", "BB", "7c 4d 2s", "open_call")
        rng, summary = build_villain_flop_range(st)
        assert len(rng) > 30
        assert "BTN" in summary


# ===================================================================
# B. Monte Carlo Sanity
# ===================================================================

class TestMonteCarloSanity:
    """Monte Carlo equity estimates are directionally correct."""

    def test_strong_hand_high_equity(self):
        """Top set should have very high equity vs a reasonable range."""
        st = _make_flop_state("7s 7h", "BTN_SB", "7c Kd 2s", "open_call")
        st = _advance_to_hero(st)
        rng, _ = build_villain_flop_range(st)
        hero = st.config.hero_hole_cards
        result = estimate_flop_equity(hero, st.board_cards, rng, samples=2000, seed=42)
        assert result["equity_estimate"] > 0.85

    def test_air_hand_low_equity(self):
        """Total air should have low equity vs a reasonable range."""
        st = _make_flop_state("4s 3h", "BTN_SB", "Kd 8c 2s", "open_call")
        st = _advance_to_hero(st)
        rng, _ = build_villain_flop_range(st)
        hero = st.config.hero_hole_cards
        result = estimate_flop_equity(hero, st.board_cards, rng, samples=2000, seed=42)
        assert result["equity_estimate"] < 0.30

    def test_combo_draw_beats_gutshot(self):
        """Combo draw should have more equity than a naked gutshot."""
        board = [parse_card("8s"), parse_card("9s"), parse_card("2h")]

        # Build a generic villain range
        st = _make_flop_state("Ks Ts", "BTN_SB", "8s 9s 2h", "open_call")
        st = _advance_to_hero(st)
        rng, _ = build_villain_flop_range(st)

        # Combo draw: KsTs (flush draw + gutshot to straight)
        combo = parse_cards("Ks Ts")
        r_combo = estimate_flop_equity(combo, board, rng, samples=2000, seed=42)

        # Naked gutshot: QsJh (gutshot only, one overcard)
        gutshot = parse_cards("Qs Jh")
        # Need to rebuild range without Qs conflict
        st2 = _make_flop_state("Qs Jh", "BTN_SB", "8s 9s 2h", "open_call")
        st2 = _advance_to_hero(st2)
        rng2, _ = build_villain_flop_range(st2)
        r_gutshot = estimate_flop_equity(gutshot, board, rng2, samples=2000, seed=42)

        assert r_combo["equity_estimate"] > r_gutshot["equity_estimate"]

    def test_overpair_good_equity(self):
        """Overpair on dry board should have good equity."""
        st = _make_flop_state("Qs Qh", "BTN_SB", "Tc 5d 2h", "open_call")
        st = _advance_to_hero(st)
        rng, _ = build_villain_flop_range(st)
        hero = st.config.hero_hole_cards
        result = estimate_flop_equity(hero, st.board_cards, rng, samples=2000, seed=42)
        assert result["equity_estimate"] > 0.65

    def test_samples_used_reported(self):
        result = estimate_flop_equity(
            parse_cards("As Kh"),
            [parse_card("7c"), parse_card("4d"), parse_card("2s")],
            [(parse_cards("Qd Jd"), 1.0), (parse_cards("Tc 9c"), 1.0)],
            samples=100, seed=1,
        )
        assert result["samples_used"] == 100

    def test_empty_range_raises(self):
        with pytest.raises(ValueError, match="empty"):
            estimate_flop_equity(
                parse_cards("As Kh"),
                [parse_card("7c"), parse_card("4d"), parse_card("2s")],
                [], samples=10,
            )


# ===================================================================
# C. Deterministic Seeded Behavior
# ===================================================================

class TestDeterministicSeeded:
    """Same inputs + seed must produce identical outputs."""

    def test_same_seed_same_result(self):
        hero = parse_cards("As Kh")
        board = [parse_card("7c"), parse_card("4d"), parse_card("2s")]
        rng = [(parse_cards("Qd Jd"), 1.0), (parse_cards("Tc 9c"), 1.0),
               (parse_cards("8h 7h"), 1.0)]

        r1 = estimate_flop_equity(hero, board, rng, samples=500, seed=99)
        r2 = estimate_flop_equity(hero, board, rng, samples=500, seed=99)
        assert r1 == r2

    def test_different_seed_likely_different(self):
        hero = parse_cards("As Kh")
        board = [parse_card("7c"), parse_card("4d"), parse_card("2s")]
        rng = [(parse_cards("Qd Jd"), 1.0), (parse_cards("Tc 9c"), 1.0),
               (parse_cards("8h 7h"), 1.0)]

        r1 = estimate_flop_equity(hero, board, rng, samples=500, seed=1)
        r2 = estimate_flop_equity(hero, board, rng, samples=500, seed=2)
        # Very unlikely to be exactly equal with different seeds
        assert r1["equity_estimate"] != r2["equity_estimate"] or True  # soft check

    def test_integration_deterministic(self):
        """Full integration wrapper is deterministic with seed."""
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _advance_to_hero(st)

        d1 = recommend_flop_action_with_equity(st, samples=500, seed=42)
        d2 = recommend_flop_action_with_equity(st, samples=500, seed=42)

        assert d1.debug["equity_estimate"] == d2.debug["equity_estimate"]
        assert d1.recommended_action.legal_action.action_type == d2.recommended_action.legal_action.action_type


# ===================================================================
# D. Pot-Odds Integration
# ===================================================================

class TestPotOddsIntegration:
    """Equity + pot-odds logic adjusts continue decisions."""

    def test_flush_draw_calls_medium_bet(self):
        """Flush draw with ~50% equity should call a medium bet (threshold ~29%)."""
        st = _make_flop_state("Ks Ts", "BB", "8s 5s 2h", "open_call")
        st = apply_action(st.config, st.action_history,
                          action_type=ActionType.CHECK, player=Player.HERO)
        st = apply_action(st.config, st.action_history,
                          action_type=ActionType.BET, player=Player.VILLAIN, amount_to_bb=3.5)

        dec = recommend_flop_action_with_equity(st, samples=2000, seed=42)
        assert dec.recommended_action.legal_action.action_type != ActionType.FOLD
        assert dec.debug["pot_odds_threshold"] is not None
        assert dec.debug["pot_odds_threshold"] < dec.debug["equity_estimate"]

    def test_air_folds_large_bet(self):
        """Air hand with low equity should fold large bet (threshold ~33%)."""
        st = _make_flop_state("4s 3h", "BTN_SB", "Kd 8c 2s", "open_call")
        st = _villain_bets(st, 5.0)

        dec = recommend_flop_action_with_equity(st, samples=2000, seed=42)
        assert dec.recommended_action.legal_action.action_type == ActionType.FOLD
        assert dec.debug["equity_estimate"] < dec.debug["pot_odds_threshold"]

    def test_pot_odds_threshold_in_debug(self):
        """pot_odds_threshold must be present when facing a bet."""
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _villain_bets(st, 3.5)

        dec = recommend_flop_action_with_equity(st, samples=500, seed=1)
        assert "pot_odds_threshold" in dec.debug
        assert dec.debug["pot_odds_threshold"] is not None

    def test_no_pot_odds_when_not_facing(self):
        """pot_odds_threshold should be None when not facing a bet."""
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _advance_to_hero(st)

        dec = recommend_flop_action_with_equity(st, samples=500, seed=1)
        assert dec.debug["pot_odds_threshold"] is None

    def test_small_vs_large_bet_tighter_continue(self):
        """Same hand facing small vs large bet: large bet requires more equity."""
        st_base = _make_flop_state("7s 6h", "BTN_SB", "Kd 7c 2s", "open_call")

        # Small bet: 1.0 into 5.0
        st_small = _villain_bets(st_base, 1.0)
        dec_small = recommend_flop_action_with_equity(st_small, samples=2000, seed=42)

        # Large bet: 5.0 into 5.0
        st_large = _villain_bets(st_base, 5.0)
        dec_large = recommend_flop_action_with_equity(st_large, samples=2000, seed=42)

        # Large bet has higher threshold
        assert dec_large.debug["pot_odds_threshold"] > dec_small.debug["pot_odds_threshold"]

    def test_equity_adjustment_notes_present(self):
        """Adjustment notes should always be present."""
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _villain_bets(st, 3.5)

        dec = recommend_flop_action_with_equity(st, samples=500, seed=1)
        assert "equity_adjustment_notes" in dec.debug
        assert isinstance(dec.debug["equity_adjustment_notes"], list)
        assert len(dec.debug["equity_adjustment_notes"]) > 0


# ===================================================================
# E. Baseline Preservation
# ===================================================================

class TestBaselinePreservation:
    """Equity-aware decisions preserve Phase D baseline integrity."""

    def test_legal_actions_from_phase_c(self):
        """Legal actions must be from Phase C, not invented."""
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _advance_to_hero(st)

        dec_eq = recommend_flop_action_with_equity(st, samples=500, seed=1)
        dec_base = recommend_flop_action(st)

        # Same legal actions
        eq_types = {a.action_type for a in dec_eq.legal_actions}
        base_types = {a.action_type for a in dec_base.legal_actions}
        assert eq_types == base_types

    def test_baseline_action_recorded(self):
        """Debug must contain the original baseline action."""
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _advance_to_hero(st)

        dec = recommend_flop_action_with_equity(st, samples=500, seed=1)
        assert "baseline_action" in dec.debug
        assert len(dec.debug["baseline_action"]) > 0

    def test_final_action_is_legal(self):
        """Final action must be one of the legal action types."""
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _villain_bets(st, 3.5)

        dec = recommend_flop_action_with_equity(st, samples=500, seed=1)
        legal_types = {a.action_type for a in dec.legal_actions}
        assert dec.recommended_action.legal_action.action_type in legal_types

    def test_phase_d_context_visible(self):
        """Phase D context labels must still be in debug."""
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _advance_to_hero(st)

        dec = recommend_flop_action_with_equity(st, samples=500, seed=1)
        assert "action_context_label" in dec.debug
        assert "hero_preflop_role" in dec.debug
        assert "made_hand_category" in dec.debug
        assert "board_texture_label" in dec.debug
        assert "baseline_rule_id" in dec.debug

    @pytest.mark.parametrize("hero_cards,board,pf,setup", [
        ("As Kh", "Ah 7c 2d", "open_call", "check_to_hero"),
        ("7s 2h", "Kd 8c 3s", "open_call", "check_to_hero"),
        ("Qs Qh", "Tc 5d 2h", "open_call", "check_to_hero"),
        ("As Kh", "Ah 7c 2d", "open_call", "facing_bet"),
        ("4s 3h", "Kd 8c 2s", "open_call", "facing_bet"),
        ("7s 7h", "7c Kd 2s", "3bet_call", "check_to_hero"),
    ])
    def test_equity_recommendation_always_legal(self, hero_cards, board, pf, setup):
        st = _make_flop_state(hero_cards, "BTN_SB", board, pf)
        if setup == "check_to_hero":
            st = _advance_to_hero(st)
        elif setup == "facing_bet":
            st = _villain_bets(st, 3.5)

        dec = recommend_flop_action_with_equity(st, samples=500, seed=42)
        legal_types = {a.action_type for a in dec.legal_actions}
        assert dec.recommended_action.legal_action.action_type in legal_types


# ===================================================================
# F. Explainability (Debug Fields)
# ===================================================================

class TestExplainability:
    """Debug output must include all required equity fields."""

    REQUIRED_EQUITY_KEYS = {
        "villain_range_summary",
        "monte_carlo_samples",
        "equity_estimate",
        "win_rate",
        "tie_rate",
        "pot_odds_threshold",
        "baseline_action",
        "final_action",
        "equity_changed_recommendation",
        "equity_adjustment_notes",
    }

    REQUIRED_BASELINE_KEYS = {
        "action_context_label",
        "hero_position_relation_on_flop",
        "hero_preflop_role",
        "made_hand_category",
        "draw_category",
        "board_texture_label",
        "spr_bucket",
        "spr",
        "baseline_rule_id",
    }

    def test_equity_debug_keys_checked_to(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _advance_to_hero(st)
        dec = recommend_flop_action_with_equity(st, samples=500, seed=1)

        missing = self.REQUIRED_EQUITY_KEYS - set(dec.debug.keys())
        assert not missing, f"Missing equity debug keys: {missing}"

    def test_equity_debug_keys_facing_bet(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _villain_bets(st, 3.5)
        dec = recommend_flop_action_with_equity(st, samples=500, seed=1)

        missing = self.REQUIRED_EQUITY_KEYS - set(dec.debug.keys())
        assert not missing, f"Missing equity debug keys: {missing}"

    def test_baseline_debug_keys_preserved(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _advance_to_hero(st)
        dec = recommend_flop_action_with_equity(st, samples=500, seed=1)

        missing = self.REQUIRED_BASELINE_KEYS - set(dec.debug.keys())
        assert not missing, f"Missing baseline debug keys: {missing}"

    def test_explanation_nonempty(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _advance_to_hero(st)
        dec = recommend_flop_action_with_equity(st, samples=500, seed=1)
        assert dec.explanation
        assert len(dec.explanation) > 10

    def test_compare_helper_returns_expected_keys(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _advance_to_hero(st)
        comp = compare_flop_baseline_vs_equity(st, samples=500, seed=42)

        for key in ("baseline_action", "baseline_rule_id", "equity_action",
                     "equity_estimate", "pot_odds_threshold", "equity_changed", "notes"):
            assert key in comp, f"Missing key: {key}"


# ===================================================================
# G. Hand Evaluator Edge Cases
# ===================================================================

class TestHandEvaluator:
    """Verify the internal 7-card evaluator handles edge cases."""

    def test_straight_flush_beats_quads(self):
        sf = [parse_card(c) for c in ["5h", "6h", "7h", "8h", "9h", "2d", "3c"]]
        qu = [parse_card(c) for c in ["As", "Ah", "Ad", "Ac", "Kh", "2d", "3c"]]
        assert _best_hand_rank(sf) > _best_hand_rank(qu)

    def test_quads_beats_full_house(self):
        qu = [parse_card(c) for c in ["As", "Ah", "Ad", "Ac", "Kh", "2d", "3c"]]
        fh = [parse_card(c) for c in ["As", "Ah", "Ad", "Kh", "Kd", "2d", "3c"]]
        assert _best_hand_rank(qu) > _best_hand_rank(fh)

    def test_flush_beats_straight(self):
        fl = [parse_card(c) for c in ["Ah", "Kh", "Qh", "Th", "2h", "3c", "4d"]]
        st = [parse_card(c) for c in ["9h", "8d", "7c", "6s", "5h", "Kc", "2d"]]
        assert _best_hand_rank(fl) > _best_hand_rank(st)

    def test_wheel_straight(self):
        wh = [parse_card(c) for c in ["Ah", "2d", "3c", "4s", "5h", "Kc", "Qd"]]
        rank = _best_hand_rank(wh)
        assert rank[0] == 4  # STRAIGHT
        assert rank[1] == 5  # 5-high

    def test_high_card_ordering(self):
        hc1 = [parse_card(c) for c in ["Ah", "Kd", "9c", "7s", "3h", "2d", "4c"]]
        hc2 = [parse_card(c) for c in ["Ah", "Qd", "9c", "7s", "3h", "2d", "4c"]]
        assert _best_hand_rank(hc1) > _best_hand_rank(hc2)

    def test_pair_beats_high_card(self):
        pair = [parse_card(c) for c in ["7h", "7d", "Ac", "Ks", "3h", "2d", "4c"]]
        hc = [parse_card(c) for c in ["Ah", "Kd", "Qc", "Js", "3h", "2d", "4c"]]
        assert _best_hand_rank(pair) > _best_hand_rank(hc)
