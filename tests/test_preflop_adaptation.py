from __future__ import annotations

import random

import pytest

from baseline_preflop.adaptive_recommender import recommend_adaptive_preflop_action
from baseline_preflop.opponent_model import (
    OpponentPreflopProfile,
    record_preflop_hand,
    simulate_bb_aggressive_3bet,
    simulate_bb_overfolds,
)
from baseline_preflop.parser import bb_vs_open_decision, unopened_btn_decision
from baseline_preflop.recommender import recommend_preflop_action


def test_profile_tracking_bb_fold_to_steal():
    p = OpponentPreflopProfile()
    history = [
        {"player": "HERO", "action": "POST_BLIND", "amount": 0.5},
        {"player": "VILLAIN", "action": "POST_BLIND", "amount": 1.0},
        {"player": "HERO", "action": "RAISE", "amount": 2.5},
        {"player": "VILLAIN", "action": "FOLD"},
    ]
    record_preflop_hand(p, "BTN_SB", history)
    assert p.hands_observed == 1
    assert p.bb_vs_open_opportunities == 1
    assert p.bb_fold_to_steal_count == 1


def test_smoothing_small_sample_stays_near_prior():
    p = OpponentPreflopProfile()
    simulate_bb_overfolds(p, 1, open_size=2.5)  # 1/1 folds
    stat = p.stat_bb_fold_to_steal()
    # Should be far below 1.0 because of prior smoothing
    assert stat.smoothed_rate < 0.80
    assert stat.confidence < 0.10


def test_smoothing_large_sample_moves_toward_observed():
    p = OpponentPreflopProfile()
    simulate_bb_overfolds(p, 40, open_size=2.5)  # 40/40 folds
    stat = p.stat_bb_fold_to_steal()
    assert stat.smoothed_rate > 0.60
    assert stat.confidence > 0.60


def test_adaptive_overfolder_adds_raise_frequency_for_trash_btn_open():
    # Baseline folds trash like Q3o from BTN
    state = unopened_btn_decision("Qh 3c", 100)
    base = recommend_preflop_action(state)
    assert base.recommended_action.action_type.value == "FOLD"

    p = OpponentPreflopProfile()
    simulate_bb_overfolds(p, 40, open_size=2.5)

    dec = recommend_adaptive_preflop_action(state, p, rng=random.Random(1))
    freqs = dec.action_frequencies
    assert abs(sum(freqs.values()) - 1.0) < 1e-9
    # Overfolder should create some steal-raise probability for bottom hands
    assert freqs.get("RAISE", 0.0) > 0.05
    assert freqs.get("FOLD", 0.0) < 0.95
    # Final action must be legal
    assert dec.adapted_recommendation in dec.legal_actions


def test_adaptive_aggressive_3bettor_reduces_marginal_open_frequency():
    # Pick a marginal-ish open that baseline raises (T8o is in BTN open chart)
    state = unopened_btn_decision("Td 8c", 100)
    base = recommend_preflop_action(state)
    assert base.recommended_action.action_type.value == "RAISE"

    p = OpponentPreflopProfile()
    simulate_bb_aggressive_3bet(p, 40, open_size=2.5, threeb_size=9.0)

    dec = recommend_adaptive_preflop_action(state, p, rng=random.Random(2))
    freqs = dec.action_frequencies
    assert freqs.get("FOLD", 0.0) > 0.01
    assert freqs.get("RAISE", 0.0) < 1.0


def test_raise_size_preservation_no_loosen_when_baseline_tightens():
    # Facing an oversized open, baseline MDF filter should tighten (often folding 76s)
    state = bb_vs_open_decision("7s 6s", 5.0, 100)
    base = recommend_preflop_action(state)
    assert base.recommended_action.action_type.value == "FOLD"
    assert (base.debug or {}).get("defense_scalar") is not None

    # Even if villain is an overfolder historically, do NOT loosen beyond baseline fold
    p = OpponentPreflopProfile()
    simulate_bb_overfolds(p, 60, open_size=2.5)
    dec = recommend_adaptive_preflop_action(state, p, rng=random.Random(3))
    assert dec.action_frequencies.get("FOLD", 0.0) == 1.0
    assert dec.adapted_recommendation.action_type.value == "FOLD"


def test_sampling_is_reproducible_with_seeded_rng():
    p = OpponentPreflopProfile()
    simulate_bb_overfolds(p, 50, open_size=2.5)
    state = unopened_btn_decision("Qh 3c", 100)

    d1 = recommend_adaptive_preflop_action(state, p, rng=random.Random(123))
    d2 = recommend_adaptive_preflop_action(state, p, rng=random.Random(123))
    assert d1.roll == d2.roll
    assert d1.adapted_recommendation == d2.adapted_recommendation
    assert d1.action_frequencies == d2.action_frequencies

