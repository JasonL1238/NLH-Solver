"""Tests for EV thin-raise filter, nuts tie-break, and flop EV debug fields."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from poker_core.models import ActionType, LegalAction

from flop_policy.config import EvPolicyConfig
from flop_policy.ev_recommender import recommend_flop_action_ev
from flop_policy.hero_value_tier import hero_flop_value_tier
from flop_policy.range_metrics import villain_range_nut_metrics
from flop_spot.models import FlopActionChoice

from postflop_policy.ev_core import (
    apply_thin_raise_filter,
    pick_best_ev_candidate,
)


def _load_flop_equity_helpers():
    path = Path(__file__).resolve().parent / "test_flop_equity.py"
    spec = importlib.util.spec_from_file_location("_flop_eq_helpers", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_h = _load_flop_equity_helpers()
_make_flop_state = _h._make_flop_state
_advance_to_hero = _h._advance_to_hero
_villain_bets = _h._villain_bets


def _la(typ: ActionType, **kwargs) -> LegalAction:
    return LegalAction(action_type=typ, **kwargs)


@pytest.fixture
def cfg() -> EvPolicyConfig:
    return EvPolicyConfig()


class TestThinRaiseFilter:
    def test_filtered_drops_raises_when_thin_other_tier(self, cfg):
        fold = _la(ActionType.FOLD)
        call = _la(ActionType.CALL)
        raise_a = _la(ActionType.RAISE, min_to_bb=6.0, max_to_bb=40.0)
        raw = [
            (FlopActionChoice(legal_action=fold), 0.0, "f"),
            (FlopActionChoice(legal_action=call), 0.5, "c"),
            (FlopActionChoice(legal_action=raise_a, size_bb=10.0), 0.4, "r"),
        ]
        out, branch = apply_thin_raise_filter(
            raw,
            facing=True,
            pot_odds_threshold=0.5,
            eq=0.2,
            cfg=cfg,
            hero_value_tier="OTHER",
            villain_nut_frac=1.0,
            p_fold_raise_median=0.0,
            has_pressure_draw=False,
        )
        assert branch == "filtered"
        assert not any(
            c[0].legal_action.action_type == ActionType.RAISE for c in out
        )

    def test_value_exempt_keeps_raises_nuts_near(self, cfg):
        fold = _la(ActionType.FOLD)
        raise_a = _la(ActionType.RAISE, min_to_bb=6.0, max_to_bb=40.0)
        raw = [
            (FlopActionChoice(legal_action=fold), 0.0, "f"),
            (FlopActionChoice(legal_action=raise_a, size_bb=10.0), 0.4, "r"),
        ]
        out, branch = apply_thin_raise_filter(
            raw,
            facing=True,
            pot_odds_threshold=0.5,
            eq=0.2,
            cfg=cfg,
            hero_value_tier="NUTS_NEAR",
            villain_nut_frac=1.0,
            p_fold_raise_median=0.0,
            has_pressure_draw=False,
        )
        assert branch == "value_exempt"
        assert any(c[0].legal_action.action_type == ActionType.RAISE for c in out)

    def test_pressure_exempt_all_gates(self, cfg):
        fold = _la(ActionType.FOLD)
        raise_a = _la(ActionType.RAISE, min_to_bb=6.0, max_to_bb=40.0)
        raw = [
            (FlopActionChoice(legal_action=fold), 0.0, "f"),
            (FlopActionChoice(legal_action=raise_a, size_bb=10.0), 0.4, "r"),
        ]
        out, branch = apply_thin_raise_filter(
            raw,
            facing=True,
            pot_odds_threshold=0.5,
            eq=0.35,
            cfg=cfg,
            hero_value_tier="OTHER",
            villain_nut_frac=0.1,
            p_fold_raise_median=0.4,
            has_pressure_draw=True,
        )
        assert branch == "pressure_exempt"
        assert any(c[0].legal_action.action_type == ActionType.RAISE for c in out)


class TestPickBestNutsTieBreak:
    def test_prefers_raise_over_call_when_tied_ev_facing_nuts(self):
        la_fold = _la(ActionType.FOLD)
        la_call = _la(ActionType.CALL)
        la_raise = _la(ActionType.RAISE, min_to_bb=8.0, max_to_bb=40.0)
        la_list = [la_fold, la_call, la_raise]
        cands = [
            (FlopActionChoice(legal_action=la_call), 1.0, "call"),
            (FlopActionChoice(legal_action=la_raise, size_bb=12.0), 1.0, "raise"),
        ]
        choice, _, _ = pick_best_ev_candidate(
            cands,
            la_list,
            facing=True,
            hero_value_tier="NUTS_NEAR",
        )
        assert choice.legal_action.action_type == ActionType.RAISE

    def test_default_tie_prefers_check(self):
        la_chk = _la(ActionType.CHECK)
        la_bet = _la(ActionType.BET, min_to_bb=1.0, max_to_bb=20.0)
        la_list = [la_chk, la_bet]
        cands = [
            (FlopActionChoice(legal_action=la_bet, size_bb=3.0), 0.5, "b"),
            (FlopActionChoice(legal_action=la_chk), 0.5, "c"),
        ]
        choice, _, _ = pick_best_ev_candidate(
            cands,
            la_list,
            facing=False,
            hero_value_tier="NUTS_NEAR",
        )
        assert choice.legal_action.action_type == ActionType.CHECK


class TestRangeMetricsAndTier:
    def test_villain_nut_metrics_normalized(self):
        from poker_core.parser import parse_board, parse_cards

        board = parse_board("As Ah Kd")
        r = [
            (parse_cards("Ks Kh"), 1.0),
            (parse_cards("7c 6d"), 2.0),
        ]
        frac, summary = villain_range_nut_metrics(board, r)
        assert 0.0 <= frac <= 1.0
        assert summary

    def test_hero_tier_set_on_paired_board(self):
        st = _make_flop_state("7s 7h", "BTN_SB", "7c Kd 2s", "open_call")
        st = _advance_to_hero(st)
        assert hero_flop_value_tier(st) == "NUTS_NEAR"


class TestFlopEvRecommenderDebug:
    def test_ev_debug_includes_raise_policy_fields(self):
        st = _make_flop_state("As Kh", "BTN_SB", "Ah 7c 2d", "open_call")
        st = _advance_to_hero(st)
        dec = recommend_flop_action_ev(st, profile=None, samples=200, seed=1)
        for key in (
            "hero_value_tier",
            "villain_nut_weight",
            "villain_nut_range_top",
            "thin_raise_policy_branch",
            "p_fold_raise_median",
            "hero_has_pressure_draw",
        ):
            assert key in dec.debug, f"missing {key}"

    def test_recommendation_legal_facing_bet(self):
        st = _make_flop_state("4s 3h", "BTN_SB", "Kd 8c 2s", "open_call")
        st = _villain_bets(st, 3.5)
        dec = recommend_flop_action_ev(st, profile=None, samples=300, seed=42)
        legal = {a.action_type for a in dec.legal_actions}
        assert dec.recommended_action.legal_action.action_type in legal
