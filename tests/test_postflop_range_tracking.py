"""Tests for particle-based postflop villain range (postflop_range package)."""

from __future__ import annotations

import random

import pytest

from poker_core.models import ActionType, HoleCards, Player, Street
from poker_core.parser import make_hand_config, parse_card
from poker_core.transitions import apply_action, deal_flop, deal_turn, post_blinds

from flop_equity.range_model import detect_preflop_line_key, expand_label_to_combos
from postflop_range.action_update import (
    AggressionSize,
    VillainActionContext,
    apply_compatibility_to_particles,
    compatibility_multiplier,
)
from postflop_range.board_update import classify_combo_on_board, filter_particles_for_new_dead
from postflop_range.debug import build_debug_dict, build_tracker_from_state, replay_through_state
from postflop_range.initial_range import build_weighted_combo_pool
from postflop_range.particles import CoarseBucket, Particle, RangeDegenerateError
from postflop_range.range_tracker import VillainParticleTracker
from postflop_range.resampling import effective_sample_size, multinomial_resample, normalize_weights


def _flop_state_open_call():
    cfg = make_hand_config("As Kd", hero_position="BTN_SB", effective_stack_bb=100.0)
    h, _ = post_blinds(cfg)
    s = apply_action(cfg, h, action_type=ActionType.RAISE, player=Player.HERO, amount_to_bb=2.5)
    s = apply_action(cfg, s.action_history, action_type=ActionType.CALL, player=Player.VILLAIN, amount_to_bb=2.5)
    flop = (parse_card("Ah"), parse_card("7c"), parse_card("2d"))
    return cfg, deal_flop(cfg, s.action_history, flop)


def _flop_state_limp_check():
    cfg = make_hand_config("As Kd", hero_position="BTN_SB", effective_stack_bb=100.0)
    h, _ = post_blinds(cfg)
    s = apply_action(cfg, h, action_type=ActionType.CALL, player=Player.HERO, amount_to_bb=1.0)
    s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK, player=Player.VILLAIN)
    flop = (parse_card("Kh"), parse_card("9c"), parse_card("3d"))
    return cfg, deal_flop(cfg, s.action_history, flop)


class TestInitialRange:
    def test_preflop_lines_differ(self) -> None:
        _, s1 = _flop_state_open_call()
        _, s2 = _flop_state_limp_check()
        assert detect_preflop_line_key(s1) != detect_preflop_line_key(s2)

    def test_pool_legal_dead_cards(self) -> None:
        _, s = _flop_state_open_call()
        pool, _ = build_weighted_combo_pool(s)
        dead = {s.config.hero_hole_cards.high, s.config.hero_hole_cards.low} | set(s.board_cards)
        for hc, _, _ in pool:
            assert hc.high not in dead and hc.low not in dead

    def test_expand_label_produces_combos(self) -> None:
        combos = expand_label_to_combos("AKs")
        assert len(combos) == 4


class TestBoardFilterAndReclassify:
    def test_turn_card_kills_conflicting_particle(self) -> None:
        c9s = parse_card("9s")
        p_alive = Particle(
            combo=HoleCards(high=parse_card("Ah"), low=parse_card("Ad")),
            weight=0.6,
            preflop_bucket_label="AA",
            current_bucket=CoarseBucket.NUTTED,
            made_category="x",
            draw_category="y",
            alive=True,
        )
        p_dead = Particle(
            combo=HoleCards(high=c9s, low=parse_card("9h")),
            weight=0.4,
            preflop_bucket_label="99",
            current_bucket=CoarseBucket.MEDIUM_MADE,
            made_category="x",
            draw_category="y",
            alive=True,
        )
        killed, _ = filter_particles_for_new_dead([p_alive, p_dead], {c9s})
        assert killed == 1
        assert p_dead.alive is False and p_dead.weight == 0.0
        assert p_alive.alive

    def test_reclassify_flop_vs_river(self) -> None:
        combo = HoleCards(high=parse_card("As"), low=parse_card("Ks"))
        flop = [parse_card("Ah"), parse_card("7c"), parse_card("2d")]
        b1, _, _ = classify_combo_on_board(combo, flop)
        river = flop + [parse_card("Ad"), parse_card("Kd")]
        b2, _, _ = classify_combo_on_board(combo, river)
        assert b1 != b2 or b2 == CoarseBucket.NUTTED


class TestActionReweight:
    def test_normalized_after_reweight(self) -> None:
        parts = [
            Particle(
                combo=HoleCards(high=parse_card("As"), low=parse_card("Ks")),
                weight=0.5,
                preflop_bucket_label="AKs",
                current_bucket=CoarseBucket.NUTTED,
                made_category="m",
                draw_category="d",
            ),
            Particle(
                combo=HoleCards(high=parse_card("2c"), low=parse_card("3d")),
                weight=0.5,
                preflop_bucket_label="x",
                current_bucket=CoarseBucket.AIR,
                made_category="m",
                draw_category="d",
            ),
        ]
        ctx = VillainActionContext(
            street=Street.FLOP,
            action_type=ActionType.BET,
            aggression_size=AggressionSize.LARGE,
            raises_this_street_before=0,
        )
        apply_compatibility_to_particles(parts, ctx)
        normalize_weights(parts)
        s = sum(p.weight for p in parts if p.alive)
        assert abs(s - 1.0) < 1e-6
        assert parts[0].weight > parts[1].weight

    def test_compatibility_multiplier_check(self) -> None:
        p = Particle(
            combo=HoleCards(high=parse_card("7h"), low=parse_card("6h")),
            weight=1.0,
            preflop_bucket_label="76s",
            current_bucket=CoarseBucket.AIR,
            made_category="m",
            draw_category="d",
        )
        ctx = VillainActionContext(
            street=Street.FLOP,
            action_type=ActionType.CHECK,
            aggression_size=AggressionSize.NONE,
            raises_this_street_before=0,
        )
        assert compatibility_multiplier(p, ctx) >= 1.0


class TestESSAndResample:
    def test_effective_sample_size_uniform(self) -> None:
        combos = [
            HoleCards(high=parse_card("As"), low=parse_card("Ah")),
            HoleCards(high=parse_card("Ks"), low=parse_card("Kh")),
            HoleCards(high=parse_card("Qs"), low=parse_card("Qh")),
            HoleCards(high=parse_card("Js"), low=parse_card("Jh")),
        ]
        parts = [
            Particle(
                combo=c,
                weight=0.25,
                preflop_bucket_label="x",
                current_bucket=CoarseBucket.NUTTED,
                made_category="m",
                draw_category="d",
            )
            for c in combos
        ]
        normalize_weights(parts)
        neff = effective_sample_size(parts)
        assert abs(neff - 4.0) < 1e-5

    def test_effective_sample_size_concentrated(self) -> None:
        parts = [
            Particle(
                combo=HoleCards(high=parse_card("As"), low=parse_card("Ah")),
                weight=0.99,
                preflop_bucket_label="AA",
                current_bucket=CoarseBucket.NUTTED,
                made_category="m",
                draw_category="d",
            ),
            Particle(
                combo=HoleCards(high=parse_card("2c"), low=parse_card("3d")),
                weight=0.01,
                preflop_bucket_label="x",
                current_bucket=CoarseBucket.AIR,
                made_category="m",
                draw_category="d",
            ),
        ]
        normalize_weights(parts)
        assert effective_sample_size(parts) < 2.0

    def test_resample_deterministic(self) -> None:
        ranks = ["9", "8", "7", "6", "5", "4"]
        weights = [0.1, 0.15, 0.15, 0.2, 0.2, 0.2]
        parts = [
            Particle(
                combo=HoleCards(high=parse_card(f"{r}s"), low=parse_card(f"{r}h")),
                weight=w,
                preflop_bucket_label="x",
                current_bucket=CoarseBucket.MEDIUM_MADE,
                made_category="m",
                draw_category="d",
            )
            for r, w in zip(ranks, weights)
        ]
        normalize_weights(parts)
        rng1 = random.Random(999)
        rng2 = random.Random(999)
        a = multinomial_resample(parts, 30, rng1)
        b = multinomial_resample(parts, 30, rng2)
        assert [p.combo_str for p in a] == [p.combo_str for p in b]
        s = sum(p.weight for p in a)
        assert abs(s - 1.0) < 1e-6


class TestTrackerContinuity:
    def test_sequential_updates_not_full_rebuild(self) -> None:
        _, s = _flop_state_open_call()
        tr = VillainParticleTracker.build_initial_from_state(s, n_particles=120, seed=5)
        w0 = sum(p.weight for p in tr.particles if p.alive)
        assert abs(w0 - 1.0) < 1e-5
        ctx = VillainActionContext(
            street=Street.FLOP,
            action_type=ActionType.BET,
            aggression_size=AggressionSize.LARGE,
            raises_this_street_before=0,
        )
        apply_compatibility_to_particles(tr.particles, ctx)
        normalize_weights(tr.particles)
        assert len(tr.particles) == 120

    def test_replay_through_state_smoke(self) -> None:
        cfg, s_flop = _flop_state_open_call()
        h = s_flop.action_history
        s_flop = apply_action(cfg, h, action_type=ActionType.CHECK, player=Player.VILLAIN)
        s_flop = apply_action(cfg, s_flop.action_history, action_type=ActionType.CHECK, player=Player.HERO)
        s_turn = deal_turn(cfg, s_flop.action_history, parse_card("Td"))
        tr1 = replay_through_state(s_turn, n_particles=80, seed=11)
        assert tr1.summarize()["particle_count"] >= 1
        dbg = build_debug_dict(tr1)
        assert "particle_count" in dbg
        assert "effective_sample_size" in dbg
        assert "bucket_weight_summary" in dbg
        assert "last_update_type" in dbg


class TestExplainability:
    def test_debug_dict_keys(self) -> None:
        _, s = _flop_state_open_call()
        tr = VillainParticleTracker.build_initial_from_state(s, n_particles=50, seed=0)
        d = build_debug_dict(tr)
        for k in (
            "particle_count",
            "effective_sample_size",
            "total_weight",
            "top_particles",
            "bucket_weight_summary",
            "last_update_type",
            "last_update_notes",
        ):
            assert k in d


class TestExportAndSample:
    def test_export_weights_sum_one(self) -> None:
        _, s = _flop_state_open_call()
        tr = VillainParticleTracker.build_initial_from_state(s, n_particles=60, seed=3)
        exp = tr.export_weighted_range_for_rollouts()
        assert abs(sum(w for _, w in exp) - 1.0) < 1e-5

    def test_sample_many(self) -> None:
        _, s = _flop_state_open_call()
        tr = VillainParticleTracker.build_initial_from_state(s, n_particles=40, seed=4)
        rng = random.Random(0)
        xs = tr.sample_many_weighted_villain_combos(200, rng)
        assert len(xs) == 200


class TestDegenerate:
    def test_export_raises_when_all_dead(self) -> None:
        p = Particle(
            combo=HoleCards(high=parse_card("As"), low=parse_card("Ah")),
            weight=0.0,
            preflop_bucket_label="AA",
            current_bucket=CoarseBucket.NUTTED,
            made_category="m",
            draw_category="d",
            alive=False,
        )
        tr = VillainParticleTracker(particles=[p], board_snapshot=[], n_particles_target=1)
        with pytest.raises(RangeDegenerateError):
            tr.export_weighted_range_for_rollouts()
