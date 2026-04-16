"""Comprehensive tests for the adaptive preflop decision layer (Phase B)."""

from __future__ import annotations

import random

import pytest

from baseline_preflop.adaptive_recommender import (
    AdaptiveDecision,
    recommend_adaptive_preflop_action,
)
from baseline_preflop.models import ActionType
from baseline_preflop.opponent_model import (
    OpponentPreflopProfile,
    bucket_open_size,
    bucket_3bet_size,
    bucket_iso_size,
    confidence_weight,
    record_preflop_hand,
    simulate_bb_aggressive_3bet,
    simulate_bb_overfolds,
    simulate_btn_limp_heavy,
    simulate_btn_overopens,
    smooth_rate,
    PRIORS,
    K_PRIOR_STRENGTH,
    C_CONFIDENCE,
)
from baseline_preflop.parser import (
    bb_vs_limp_decision,
    bb_vs_open_decision,
    btn_vs_3bet_decision,
    unopened_btn_decision,
)
from baseline_preflop.recommender import recommend_preflop_action


# ===================================================================
# Helper
# ===================================================================

def _assert_legal(dec: AdaptiveDecision) -> None:
    if dec.legal_actions:
        assert dec.adapted_recommendation in dec.legal_actions


def _blinds(hero_pos: str):
    if hero_pos == "BTN_SB":
        return [
            {"player": "HERO", "action": "POST_BLIND", "amount": 0.5},
            {"player": "VILLAIN", "action": "POST_BLIND", "amount": 1.0},
        ]
    return [
        {"player": "VILLAIN", "action": "POST_BLIND", "amount": 0.5},
        {"player": "HERO", "action": "POST_BLIND", "amount": 1.0},
    ]


# ===================================================================
# A. Opponent stat tracking
# ===================================================================

class TestStatTracking:
    def test_bb_fold_to_steal(self):
        p = OpponentPreflopProfile()
        history = _blinds("BTN_SB") + [
            {"player": "HERO", "action": "RAISE", "amount": 2.5},
            {"player": "VILLAIN", "action": "FOLD"},
        ]
        record_preflop_hand(p, "BTN_SB", history)
        assert p.hands_observed == 1
        assert p.bb_vs_open_opportunities == 1
        assert p.bb_fold_to_steal_count == 1

    def test_btn_open_tracking(self):
        p = OpponentPreflopProfile()
        history = _blinds("BB") + [
            {"player": "VILLAIN", "action": "RAISE", "amount": 2.5},
            {"player": "HERO", "action": "FOLD"},
        ]
        record_preflop_hand(p, "BB", history)
        assert p.btn_open_opportunities == 1
        assert p.btn_open_count == 1

    def test_btn_limp_tracking(self):
        p = OpponentPreflopProfile()
        history = _blinds("BB") + [
            {"player": "VILLAIN", "action": "CALL", "amount": 1.0},
            {"player": "HERO", "action": "CHECK"},
        ]
        record_preflop_hand(p, "BB", history)
        assert p.btn_limp_opportunities == 1
        assert p.btn_limp_count == 1

    def test_bb_3bet_vs_open_tracking(self):
        p = OpponentPreflopProfile()
        history = _blinds("BTN_SB") + [
            {"player": "HERO", "action": "RAISE", "amount": 2.5},
            {"player": "VILLAIN", "action": "RAISE", "amount": 8.0},
            {"player": "HERO", "action": "FOLD"},
        ]
        record_preflop_hand(p, "BTN_SB", history)
        assert p.bb_vs_open_opportunities == 1
        assert p.bb_3bet_vs_open_count == 1

    def test_bb_call_vs_open_tracking(self):
        p = OpponentPreflopProfile()
        history = _blinds("BTN_SB") + [
            {"player": "HERO", "action": "RAISE", "amount": 2.5},
            {"player": "VILLAIN", "action": "CALL"},
        ]
        record_preflop_hand(p, "BTN_SB", history)
        assert p.bb_vs_open_opportunities == 1
        assert p.bb_call_vs_open_count == 1

    def test_btn_fold_to_3bet_tracking(self):
        p = OpponentPreflopProfile()
        history = _blinds("BB") + [
            {"player": "VILLAIN", "action": "RAISE", "amount": 2.5},
            {"player": "HERO", "action": "RAISE", "amount": 8.0},
            {"player": "VILLAIN", "action": "FOLD"},
        ]
        record_preflop_hand(p, "BB", history)
        assert p.btn_facing_3bet_opportunities == 1
        assert p.btn_fold_to_3bet_count == 1

    def test_btn_call_3bet_tracking(self):
        p = OpponentPreflopProfile()
        history = _blinds("BB") + [
            {"player": "VILLAIN", "action": "RAISE", "amount": 2.5},
            {"player": "HERO", "action": "RAISE", "amount": 8.0},
            {"player": "VILLAIN", "action": "CALL"},
        ]
        record_preflop_hand(p, "BB", history)
        assert p.btn_facing_3bet_opportunities == 1
        assert p.btn_call_3bet_count == 1

    def test_btn_4bet_tracking(self):
        p = OpponentPreflopProfile()
        history = _blinds("BB") + [
            {"player": "VILLAIN", "action": "RAISE", "amount": 2.5},
            {"player": "HERO", "action": "RAISE", "amount": 8.0},
            {"player": "VILLAIN", "action": "RAISE", "amount": 20.0},
            {"player": "HERO", "action": "FOLD"},
        ]
        record_preflop_hand(p, "BB", history)
        assert p.btn_4bet_count == 1

    def test_bb_raise_vs_limp_tracking(self):
        p = OpponentPreflopProfile()
        history = _blinds("BTN_SB") + [
            {"player": "HERO", "action": "CALL", "amount": 1.0},
            {"player": "VILLAIN", "action": "RAISE", "amount": 4.0},
            {"player": "HERO", "action": "FOLD"},
        ]
        record_preflop_hand(p, "BTN_SB", history)
        assert p.bb_vs_limp_opportunities == 1
        assert p.bb_raise_vs_limp_count == 1

    def test_bb_check_vs_limp_tracking(self):
        p = OpponentPreflopProfile()
        history = _blinds("BTN_SB") + [
            {"player": "HERO", "action": "CALL", "amount": 1.0},
            {"player": "VILLAIN", "action": "CHECK"},
        ]
        record_preflop_hand(p, "BTN_SB", history)
        assert p.bb_vs_limp_opportunities == 1
        assert p.bb_check_vs_limp_count == 1

    def test_btn_limp_then_fold_to_raise(self):
        p = OpponentPreflopProfile()
        history = _blinds("BB") + [
            {"player": "VILLAIN", "action": "CALL", "amount": 1.0},
            {"player": "HERO", "action": "RAISE", "amount": 3.5},
            {"player": "VILLAIN", "action": "FOLD"},
        ]
        record_preflop_hand(p, "BB", history)
        assert p.btn_limp_then_fold_to_raise_count == 1

    def test_multi_hand_accumulation(self):
        p = OpponentPreflopProfile()
        simulate_bb_overfolds(p, 5)
        simulate_bb_aggressive_3bet(p, 3)
        assert p.hands_observed == 8
        assert p.bb_fold_to_steal_count == 5
        assert p.bb_3bet_vs_open_count == 3
        assert p.bb_vs_open_opportunities == 8


# ===================================================================
# Size bucket tracking
# ===================================================================

class TestSizeBucketTracking:
    def test_open_size_buckets(self):
        assert bucket_open_size(2.0) == "SMALL"
        assert bucket_open_size(2.5) == "STANDARD"
        assert bucket_open_size(3.0) == "LARGE"

    def test_3bet_size_buckets(self):
        assert bucket_3bet_size(6.0, 2.5) == "SMALL"    # 2.4x
        assert bucket_3bet_size(8.0, 2.5) == "STANDARD"  # 3.2x
        assert bucket_3bet_size(12.0, 2.5) == "LARGE"    # 4.8x

    def test_iso_size_buckets(self):
        assert bucket_iso_size(2.5) == "SMALL"
        assert bucket_iso_size(3.5) == "STANDARD"
        assert bucket_iso_size(5.0) == "LARGE"

    def test_open_size_tracking_in_profile(self):
        p = OpponentPreflopProfile()
        for _ in range(10):
            record_preflop_hand(p, "BB", _blinds("BB") + [
                {"player": "VILLAIN", "action": "RAISE", "amount": 2.0},
                {"player": "HERO", "action": "FOLD"},
            ])
        assert p.btn_open_size_bucket_counts.get("SMALL", 0) == 10
        dist = p.dist_btn_open_size()
        assert dist["SMALL"] > 0.5

    def test_3bet_size_tracking_in_profile(self):
        p = OpponentPreflopProfile()
        for _ in range(10):
            record_preflop_hand(p, "BTN_SB", _blinds("BTN_SB") + [
                {"player": "HERO", "action": "RAISE", "amount": 2.5},
                {"player": "VILLAIN", "action": "RAISE", "amount": 12.0},
                {"player": "HERO", "action": "FOLD"},
            ])
        assert p.bb_3bet_size_bucket_counts.get("LARGE", 0) == 10
        dist = p.dist_bb_3bet_size()
        assert dist["LARGE"] > 0.3


# ===================================================================
# B. Smoothing behavior
# ===================================================================

class TestSmoothing:
    def test_tiny_sample_near_prior(self):
        p = OpponentPreflopProfile()
        simulate_bb_overfolds(p, 2)
        stat = p.stat_bb_fold_to_steal()
        assert stat.smoothed_rate < 0.80
        assert stat.confidence < 0.15

    def test_large_sample_near_observed(self):
        p = OpponentPreflopProfile()
        simulate_bb_overfolds(p, 40)
        stat = p.stat_bb_fold_to_steal()
        assert stat.smoothed_rate > 0.60
        assert stat.confidence > 0.60

    def test_smoothing_formula(self):
        prior = PRIORS["bb_fold_to_steal_rate"]
        s = smooth_rate(10, 10, prior, K_PRIOR_STRENGTH)
        expected = (K_PRIOR_STRENGTH * prior + 10 * 1.0) / (K_PRIOR_STRENGTH + 10)
        assert abs(s - expected) < 1e-9

    def test_zero_observations_returns_prior(self):
        s = smooth_rate(0, 0, 0.5, K_PRIOR_STRENGTH)
        assert abs(s - 0.5) < 1e-9


# ===================================================================
# C. Confidence weighting
# ===================================================================

class TestConfidence:
    def test_zero_opportunities(self):
        assert confidence_weight(0) == 0.0

    def test_confidence_increases_with_sample(self):
        c1 = confidence_weight(5)
        c2 = confidence_weight(20)
        c3 = confidence_weight(80)
        assert c1 < c2 < c3

    def test_confidence_formula(self):
        c = confidence_weight(20, C_CONFIDENCE)
        assert abs(c - 0.5) < 1e-9

    def test_confidence_surfaced_in_debug(self):
        p = OpponentPreflopProfile()
        simulate_bb_overfolds(p, 20)
        state = unopened_btn_decision("Qh 3c", 100)
        dec = recommend_adaptive_preflop_action(state, p, rng=random.Random(1))
        cs = dec.debug.get("confidence_summary", {})
        assert "bb_fold_to_steal_confidence" in cs
        assert cs["bb_fold_to_steal_confidence"] > 0.0


# ===================================================================
# D. Baseline vs adaptive: unknown profile stays near baseline
# ===================================================================

class TestUnknownProfile:
    def test_unknown_profile_stays_near_baseline(self):
        state = unopened_btn_decision("Qh 3c", 100)
        base = recommend_preflop_action(state)
        dec = recommend_adaptive_preflop_action(
            state, OpponentPreflopProfile(), rng=random.Random(42)
        )
        assert dec.adapted_recommendation.action_type == base.recommended_action.action_type
        assert dec.adaptation_changed is False

    def test_unknown_profile_archetypes(self):
        p = OpponentPreflopProfile()
        assert p.archetypes() == ["UNKNOWN"]


# ===================================================================
# D. Overfolder profile widens steals
# ===================================================================

class TestOverfolderExploit:
    def test_overfolder_adds_raise_for_trash(self):
        state = unopened_btn_decision("Qh 3c", 100)
        base = recommend_preflop_action(state)
        assert base.recommended_action.action_type == ActionType.FOLD

        p = OpponentPreflopProfile()
        simulate_bb_overfolds(p, 40)

        dec = recommend_adaptive_preflop_action(state, p, rng=random.Random(1))
        freqs = dec.action_frequencies
        assert abs(sum(freqs.values()) - 1.0) < 1e-9
        assert freqs.get("RAISE", 0.0) > 0.05
        assert freqs.get("FOLD", 0.0) < 0.95
        _assert_legal(dec)

    def test_overfolder_prefers_small_size(self):
        p = OpponentPreflopProfile()
        simulate_bb_overfolds(p, 50)
        state = unopened_btn_decision("Qh 3c", 100)
        dec = recommend_adaptive_preflop_action(state, p, rng=random.Random(1))
        raise_freqs = dec.debug.get("raise_size_frequencies", {})
        if raise_freqs:
            best = max(raise_freqs, key=raise_freqs.get)
            assert best <= 2.5


# ===================================================================
# D. Aggressive 3-bettor tightens marginal opens
# ===================================================================

class TestAggressive3BettorExploit:
    def test_tightens_marginal_opens(self):
        state = unopened_btn_decision("Td 8c", 100)
        base = recommend_preflop_action(state)
        assert base.recommended_action.action_type == ActionType.RAISE

        p = OpponentPreflopProfile()
        simulate_bb_aggressive_3bet(p, 40, threeb_size=9.0)

        dec = recommend_adaptive_preflop_action(state, p, rng=random.Random(2))
        freqs = dec.action_frequencies
        assert freqs.get("FOLD", 0.0) > 0.01
        assert freqs.get("RAISE", 0.0) < 1.0


# ===================================================================
# E. BB vs wide opener
# ===================================================================

class TestBBvsWideOpener:
    def test_bb_adds_3bet_vs_wide_opener(self):
        state = bb_vs_open_decision("Qs Jd", 2.5, 100)
        base = recommend_preflop_action(state)

        p = OpponentPreflopProfile()
        simulate_btn_overopens(p, 60)

        dec = recommend_adaptive_preflop_action(state, p, rng=random.Random(42))
        # Wide opener exploit should add some RAISE frequency from CALL
        freqs = dec.action_frequencies
        assert freqs.get("RAISE", 0.0) > 0.0
        _assert_legal(dec)

    def test_bb_still_calls_vs_wide_opener_weaker_hand(self):
        state = bb_vs_open_decision("8d 6c", 2.5, 100)
        p = OpponentPreflopProfile()
        simulate_btn_overopens(p, 60)
        dec = recommend_adaptive_preflop_action(state, p, rng=random.Random(42))
        _assert_legal(dec)


# ===================================================================
# F. BB vs limp-heavy BTN
# ===================================================================

class TestBBvsLimpHeavy:
    def test_punish_limp_heavy_btn(self):
        state = bb_vs_limp_decision("Kd 7c", 100)
        base = recommend_preflop_action(state)

        p = OpponentPreflopProfile()
        simulate_btn_limp_heavy(p, 50)

        dec = recommend_adaptive_preflop_action(state, p, rng=random.Random(42))
        freqs = dec.action_frequencies
        # K7o should get some RAISE frequency added vs limp-heavy opponent
        if base.recommended_action.action_type == ActionType.CHECK:
            assert freqs.get("RAISE", 0.0) > 0.0
        _assert_legal(dec)


# ===================================================================
# G. Preserve baseline current-hand raise-size filtering
# ===================================================================

class TestRaiseSizePreservation:
    def test_no_loosen_when_baseline_tightens(self):
        state = bb_vs_open_decision("7s 6s", 5.0, 100)
        base = recommend_preflop_action(state)
        assert base.recommended_action.action_type == ActionType.FOLD
        assert base.debug.get("defense_scalar") is not None

        p = OpponentPreflopProfile()
        simulate_bb_overfolds(p, 60)
        dec = recommend_adaptive_preflop_action(state, p, rng=random.Random(3))
        assert dec.action_frequencies.get("FOLD", 0.0) == 1.0
        assert dec.adapted_recommendation.action_type == ActionType.FOLD

    def test_filter_applied_flag_present(self):
        state = bb_vs_open_decision("7s 6s", 5.0, 100)
        p = OpponentPreflopProfile()
        simulate_bb_overfolds(p, 60)
        dec = recommend_adaptive_preflop_action(state, p, rng=random.Random(3))
        assert dec.debug.get("baseline_raise_size_filter_applied") is True
        assert "baseline_raise_size_note" in dec.debug

    def test_premium_survives_large_open(self):
        state = bb_vs_open_decision("Ah Ad", 5.0, 100)
        p = OpponentPreflopProfile()
        simulate_bb_overfolds(p, 60)
        dec = recommend_adaptive_preflop_action(state, p, rng=random.Random(1))
        assert dec.adapted_recommendation.action_type == ActionType.RAISE
        _assert_legal(dec)


# ===================================================================
# H. Sizing tendency modeling
# ===================================================================

class TestSizingTendencyModeling:
    def test_small_opener_distribution(self):
        p = OpponentPreflopProfile()
        for _ in range(15):
            record_preflop_hand(p, "BB", _blinds("BB") + [
                {"player": "VILLAIN", "action": "RAISE", "amount": 2.0},
                {"player": "HERO", "action": "FOLD"},
            ])
        dist = p.dist_btn_open_size()
        assert dist["SMALL"] > dist["STANDARD"]

    def test_large_opener_distribution(self):
        p = OpponentPreflopProfile()
        for _ in range(15):
            record_preflop_hand(p, "BB", _blinds("BB") + [
                {"player": "VILLAIN", "action": "RAISE", "amount": 3.5},
                {"player": "HERO", "action": "FOLD"},
            ])
        dist = p.dist_btn_open_size()
        assert dist["LARGE"] > dist["STANDARD"]

    def test_sizing_does_not_override_current_hand_filter(self):
        state = bb_vs_open_decision("7s 6s", 5.0, 100)
        base = recommend_preflop_action(state)
        assert base.recommended_action.action_type == ActionType.FOLD

        p = OpponentPreflopProfile()
        for _ in range(15):
            record_preflop_hand(p, "BB", _blinds("BB") + [
                {"player": "VILLAIN", "action": "RAISE", "amount": 2.0},
                {"player": "HERO", "action": "FOLD"},
            ])
        dec = recommend_adaptive_preflop_action(state, p, rng=random.Random(1))
        assert dec.adapted_recommendation.action_type == ActionType.FOLD


# ===================================================================
# I. Explainability / debug output
# ===================================================================

class TestExplainability:
    def test_all_required_debug_fields(self):
        p = OpponentPreflopProfile()
        simulate_bb_overfolds(p, 40)
        state = unopened_btn_decision("Qh 3c", 100)
        dec = recommend_adaptive_preflop_action(state, p, rng=random.Random(1))

        required_keys = [
            "baseline_action",
            "final_action",
            "adaptation_changed",
            "key_smoothed_stats",
            "confidence_summary",
            "range_adjustment_notes",
            "exploit_adjustment_notes",
            "baseline_raise_size_filter_applied",
            "baseline_raise_size_note",
            "villain_profile_summary",
            "action_frequencies",
        ]
        for key in required_keys:
            assert key in dec.debug, f"Missing debug key: {key}"

    def test_key_smoothed_stats_populated(self):
        p = OpponentPreflopProfile()
        simulate_bb_overfolds(p, 20)
        state = unopened_btn_decision("As Kd", 100)
        dec = recommend_adaptive_preflop_action(state, p, rng=random.Random(1))
        stats = dec.debug.get("key_smoothed_stats", {})
        assert "bb_fold_to_steal" in stats
        assert "bb_3bet_vs_open" in stats
        assert "btn_open_rate" in stats
        assert "btn_limp_rate" in stats

    def test_exploit_notes_when_exploit_active(self):
        p = OpponentPreflopProfile()
        simulate_bb_overfolds(p, 40)
        state = unopened_btn_decision("Qh 3c", 100)
        dec = recommend_adaptive_preflop_action(state, p, rng=random.Random(1))
        notes = dec.debug.get("exploit_adjustment_notes", [])
        assert len(notes) > 0

    def test_no_exploit_notes_unknown_profile(self):
        state = unopened_btn_decision("As Kd", 100)
        dec = recommend_adaptive_preflop_action(
            state, OpponentPreflopProfile(), rng=random.Random(1)
        )
        notes = dec.debug.get("exploit_adjustment_notes", [])
        assert len(notes) == 0


# ===================================================================
# J. Determinism
# ===================================================================

class TestDeterminism:
    def test_seeded_rng_reproducible(self):
        p = OpponentPreflopProfile()
        simulate_bb_overfolds(p, 50)
        state = unopened_btn_decision("Qh 3c", 100)

        d1 = recommend_adaptive_preflop_action(state, p, rng=random.Random(123))
        d2 = recommend_adaptive_preflop_action(state, p, rng=random.Random(123))
        assert d1.roll == d2.roll
        assert d1.adapted_recommendation == d2.adapted_recommendation
        assert d1.action_frequencies == d2.action_frequencies

    def test_same_profile_same_output(self):
        p = OpponentPreflopProfile()
        simulate_bb_overfolds(p, 30)
        state = unopened_btn_decision("Td 8c", 100)

        d1 = recommend_adaptive_preflop_action(state, p, rng=random.Random(999))
        d2 = recommend_adaptive_preflop_action(state, p, rng=random.Random(999))
        assert d1.debug["action_frequencies"] == d2.debug["action_frequencies"]
        assert d1.adapted_recommendation == d2.adapted_recommendation


# ===================================================================
# K. Safety: legal action enforcement
# ===================================================================

class TestSafety:
    @pytest.mark.parametrize("cards,stack", [
        ("Qh 3c", 100), ("As Kd", 100), ("7d 2c", 100),
        ("Td 9c", 50), ("5s 5d", 8),
    ])
    def test_adaptive_always_legal_btn(self, cards, stack):
        p = OpponentPreflopProfile()
        simulate_bb_overfolds(p, 40)
        state = unopened_btn_decision(cards, stack)
        dec = recommend_adaptive_preflop_action(state, p, rng=random.Random(42))
        _assert_legal(dec)

    @pytest.mark.parametrize("cards,open_size", [
        ("As Kd", 2.5), ("7s 6s", 2.0), ("Td 9c", 5.0),
    ])
    def test_adaptive_always_legal_bb(self, cards, open_size):
        p = OpponentPreflopProfile()
        simulate_btn_overopens(p, 60)
        state = bb_vs_open_decision(cards, open_size, 100)
        dec = recommend_adaptive_preflop_action(state, p, rng=random.Random(42))
        _assert_legal(dec)

    def test_baseline_always_preserved_alongside_final(self):
        p = OpponentPreflopProfile()
        simulate_bb_overfolds(p, 40)
        state = unopened_btn_decision("Qh 3c", 100)
        dec = recommend_adaptive_preflop_action(state, p, rng=random.Random(1))
        assert dec.baseline_recommendation is not None
        assert dec.adapted_recommendation is not None
        assert dec.debug.get("baseline_action") is not None
        assert dec.debug.get("final_action") is not None


# ===================================================================
# Archetype detection
# ===================================================================

class TestArchetypes:
    def test_overfolder_archetype(self):
        p = OpponentPreflopProfile()
        simulate_bb_overfolds(p, 50)
        labels = p.archetypes()
        assert "OVERFOLDER_TO_STEALS" in labels

    def test_aggressive_3bettor_archetype(self):
        p = OpponentPreflopProfile()
        simulate_bb_aggressive_3bet(p, 50)
        labels = p.archetypes()
        assert "AGGRESSIVE_3BETTER" in labels

    def test_limp_heavy_archetype(self):
        p = OpponentPreflopProfile()
        simulate_btn_limp_heavy(p, 50)
        labels = p.archetypes()
        assert "LIMP_HEAVY_BTN" in labels

    def test_wide_btn_archetype(self):
        p = OpponentPreflopProfile()
        simulate_btn_overopens(p, 60)
        labels = p.archetypes()
        assert "WIDE_BTN" in labels


# ===================================================================
# Archetype boundary tests
# ===================================================================

class TestArchetypeBoundaries:
    """Verify archetype thresholds trigger at the right sample sizes."""

    def test_below_min_confidence_returns_unknown(self):
        p = OpponentPreflopProfile()
        simulate_bb_overfolds(p, 8)
        stat = p.stat_bb_fold_to_steal()
        assert stat.confidence < 0.35
        assert p.archetypes() == ["UNKNOWN"]

    def test_at_min_confidence_overfolder_triggers(self):
        p = OpponentPreflopProfile()
        simulate_bb_overfolds(p, 11)
        stat = p.stat_bb_fold_to_steal()
        assert stat.confidence >= 0.35
        assert stat.smoothed_rate >= 0.50
        assert "OVERFOLDER_TO_STEALS" in p.archetypes()
        assert "TIGHT_BB" in p.archetypes()

    def test_sticky_defender_triggers(self):
        p = OpponentPreflopProfile()
        for _ in range(11):
            record_preflop_hand(p, "BTN_SB", _blinds("BTN_SB") + [
                {"player": "HERO", "action": "RAISE", "amount": 2.5},
                {"player": "VILLAIN", "action": "CALL"},
            ])
        stat = p.stat_bb_fold_to_steal()
        assert stat.confidence >= 0.35
        assert stat.smoothed_rate <= 0.28
        assert "STICKY_DEFENDER" in p.archetypes()
        assert "LOOSE_BB" in p.archetypes()

    def test_aggressive_3bettor_triggers(self):
        p = OpponentPreflopProfile()
        simulate_bb_aggressive_3bet(p, 11)
        stat = p.stat_bb_3bet_vs_open()
        assert stat.confidence >= 0.35
        assert stat.smoothed_rate >= 0.18
        assert "AGGRESSIVE_3BETTER" in p.archetypes()

    def test_passive_defender_triggers(self):
        p = OpponentPreflopProfile()
        simulate_bb_overfolds(p, 11)
        stat = p.stat_bb_3bet_vs_open()
        assert stat.smoothed_rate <= 0.07
        assert "PASSIVE_DEFENDER" in p.archetypes()

    def test_nitty_btn_triggers(self):
        p = OpponentPreflopProfile()
        for _ in range(11):
            record_preflop_hand(p, "BB", _blinds("BB") + [
                {"player": "VILLAIN", "action": "FOLD"},
            ])
        stat = p.stat_btn_open_rate()
        assert stat.confidence >= 0.35
        assert stat.smoothed_rate <= 0.60
        assert "NITTY_BTN" in p.archetypes()

    def test_small_opener_archetype(self):
        p = OpponentPreflopProfile()
        for _ in range(15):
            record_preflop_hand(p, "BB", _blinds("BB") + [
                {"player": "VILLAIN", "action": "RAISE", "amount": 2.0},
                {"player": "HERO", "action": "FOLD"},
            ])
        dist = p.dist_btn_open_size()
        assert dist["SMALL"] >= 0.45
        assert "SMALL_OPENER" in p.archetypes()

    def test_large_opener_archetype(self):
        p = OpponentPreflopProfile()
        for _ in range(15):
            record_preflop_hand(p, "BB", _blinds("BB") + [
                {"player": "VILLAIN", "action": "RAISE", "amount": 3.5},
                {"player": "HERO", "action": "FOLD"},
            ])
        dist = p.dist_btn_open_size()
        assert dist["LARGE"] >= 0.30
        assert "LARGE_OPENER" in p.archetypes()

    def test_sizing_archetype_needs_min_10_samples(self):
        p = OpponentPreflopProfile()
        for _ in range(9):
            record_preflop_hand(p, "BB", _blinds("BB") + [
                {"player": "VILLAIN", "action": "RAISE", "amount": 2.0},
                {"player": "HERO", "action": "FOLD"},
            ])
        assert "SMALL_OPENER" not in p.archetypes()

