"""Comprehensive tests for the Phase C full-hand state machine engine.

Test groups:
  A. Blind posting and preflop actor order
  B. Postflop actor order
  C. Street progression
  D. Board count validation
  E. Pot / contribution reconstruction
  F. Amount-to-call derivation
  G. Legal action generation
  H. Closure logic
  I. All-in handling
  J. Determinism
  K. Invalid sequences rejected
"""

from __future__ import annotations

import pytest

from poker_core.models import (
    Action,
    ActionType,
    Card,
    HandConfig,
    HoleCards,
    LegalAction,
    Player,
    Position,
    Street,
)
from poker_core.legal_actions import legal_actions
from poker_core.parser import (
    make_blinds,
    make_hand_config,
    parse_board,
    parse_card,
    parse_cards,
)
from poker_core.reconstruction import ReconstructionError, reconstruct_hand_state
from poker_core.transitions import (
    apply_action,
    deal_flop,
    deal_river,
    deal_turn,
    post_blinds,
)
from poker_core.validation import ValidationError, validate_hand


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(hero_pos: str = "BTN_SB", stack: float = 100.0) -> HandConfig:
    return make_hand_config("As Kd", hero_pos, stack)


def _blinds(cfg: HandConfig) -> list[Action]:
    return make_blinds(cfg)


def _flop_cards() -> tuple[Card, Card, Card]:
    return (parse_card("Ah"), parse_card("7c"), parse_card("2d"))


def _turn_card() -> Card:
    return parse_card("9s")


def _river_card() -> Card:
    return parse_card("4h")


def _action_types(actions: list[LegalAction]) -> set[ActionType]:
    return {a.action_type for a in actions}


# ===================================================================
# A. Blind posting and preflop actor order
# ===================================================================

class TestBlindPostingAndPreflopOrder:
    def test_btn_posts_sb_first(self):
        cfg = _cfg("BTN_SB")
        history, state = post_blinds(cfg)
        assert history[0].player == Player.HERO
        assert history[0].action_type == ActionType.POST_BLIND
        assert history[0].amount_to_bb == 0.5

    def test_bb_posts_second(self):
        cfg = _cfg("BTN_SB")
        history, state = post_blinds(cfg)
        assert history[1].player == Player.VILLAIN
        assert history[1].action_type == ActionType.POST_BLIND
        assert history[1].amount_to_bb == 1.0

    def test_btn_acts_first_preflop(self):
        cfg = _cfg("BTN_SB")
        _, state = post_blinds(cfg)
        assert state.current_street == Street.PREFLOP
        assert state.current_actor == Player.HERO  # BTN acts first

    def test_bb_hero_btn_acts_first(self):
        cfg = _cfg("BB")
        _, state = post_blinds(cfg)
        assert state.current_actor == Player.VILLAIN  # Villain is BTN

    def test_after_btn_raise_bb_acts(self):
        cfg = _cfg("BTN_SB")
        history, _ = post_blinds(cfg)
        state = apply_action(cfg, history,
                             action_type=ActionType.RAISE,
                             player=Player.HERO,
                             amount_to_bb=2.5)
        assert state.current_actor == Player.VILLAIN


# ===================================================================
# B. Postflop actor order
# ===================================================================

class TestPostflopActorOrder:
    def test_bb_acts_first_on_flop(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        # BTN raises, BB calls -> preflop closes
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        h = s.action_history
        s = apply_action(cfg, h, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=2.5)
        h = s.action_history
        # Deal flop
        s = deal_flop(cfg, h, _flop_cards())
        # BB (Villain when Hero is BTN) acts first
        assert s.current_actor == Player.VILLAIN
        assert s.current_street == Street.FLOP

    def test_btn_acts_second_on_flop(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=2.5)
        s = deal_flop(cfg, s.action_history, _flop_cards())
        # BB checks
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.VILLAIN)
        # Now BTN acts
        assert s.current_actor == Player.HERO

    def test_bb_acts_first_on_turn(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=2.5)
        s = deal_flop(cfg, s.action_history, _flop_cards())
        # Check-check on flop
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.VILLAIN)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.HERO)
        # Deal turn
        s = deal_turn(cfg, s.action_history, _turn_card())
        assert s.current_actor == Player.VILLAIN
        assert s.current_street == Street.TURN


# ===================================================================
# C. Street progression
# ===================================================================

class TestStreetProgression:
    def _play_to_flop(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=2.5)
        s = deal_flop(cfg, s.action_history, _flop_cards())
        return cfg, s

    def test_preflop_to_flop(self):
        _, s = self._play_to_flop()
        assert s.current_street == Street.FLOP
        assert len(s.board_cards) == 3

    def test_flop_to_turn(self):
        cfg, s = self._play_to_flop()
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.VILLAIN)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.HERO)
        s = deal_turn(cfg, s.action_history, _turn_card())
        assert s.current_street == Street.TURN
        assert len(s.board_cards) == 4

    def test_turn_to_river(self):
        cfg, s = self._play_to_flop()
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.VILLAIN)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.HERO)
        s = deal_turn(cfg, s.action_history, _turn_card())
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.VILLAIN)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.HERO)
        s = deal_river(cfg, s.action_history, _river_card())
        assert s.current_street == Street.RIVER
        assert len(s.board_cards) == 5

    def test_river_to_showdown(self):
        cfg, s = self._play_to_flop()
        # Check through flop
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.VILLAIN)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.HERO)
        # Deal turn, check through
        s = deal_turn(cfg, s.action_history, _turn_card())
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.VILLAIN)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.HERO)
        # Deal river, check through -> showdown
        s = deal_river(cfg, s.action_history, _river_card())
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.VILLAIN)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.HERO)
        assert s.current_street == Street.SHOWDOWN
        assert s.showdown_ready is True
        assert s.hand_over is True


# ===================================================================
# D. Board count validation
# ===================================================================

class TestBoardCountValidation:
    def _preflop_closed(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=2.5)
        return cfg, s

    def test_reject_4_card_flop(self):
        cfg, s = self._preflop_closed()
        four = (parse_card("Ah"), parse_card("7c"),
                parse_card("2d"), parse_card("9s"))
        with pytest.raises(ValidationError):
            deal_flop(cfg, s.action_history, four)

    def test_reject_2_card_flop(self):
        cfg, s = self._preflop_closed()
        two = (parse_card("Ah"), parse_card("7c"))
        with pytest.raises(ValidationError):
            deal_flop(cfg, s.action_history, two)

    def test_reject_2_card_turn(self):
        cfg, s = self._preflop_closed()
        s = deal_flop(cfg, s.action_history, _flop_cards())
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.VILLAIN)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.HERO)
        two = (parse_card("9s"), parse_card("4h"))
        with pytest.raises(ValidationError):
            # deal_turn expects a single Card, but even if we tried to hack it
            # the reconstruction would reject
            new_action = Action(action_type=ActionType.DEAL_TURN,
                                cards=two)
            validate_hand(cfg, list(s.action_history) + [new_action])

    def test_reject_deal_out_of_order(self):
        cfg, s = self._preflop_closed()
        # Try dealing turn before flop
        with pytest.raises(ValidationError):
            deal_turn(cfg, s.action_history, _turn_card())


# ===================================================================
# E. Pot / contribution reconstruction
# ===================================================================

class TestPotContribution:
    def test_after_blinds(self):
        cfg = _cfg("BTN_SB")
        _, s = post_blinds(cfg)
        assert abs(s.pot_size_bb - 1.5) < 1e-9
        assert abs(s.hero_contribution_bb - 0.5) < 1e-9
        assert abs(s.villain_contribution_bb - 1.0) < 1e-9

    def test_after_raise_call(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=2.5)
        assert abs(s.pot_size_bb - 5.0) < 1e-9
        assert abs(s.hero_contribution_bb - 2.5) < 1e-9
        assert abs(s.villain_contribution_bb - 2.5) < 1e-9

    def test_after_3bet_call(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        s = apply_action(cfg, s.action_history, action_type=ActionType.RAISE,
                         player=Player.VILLAIN, amount_to_bb=8.0)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.HERO, amount_to_bb=8.0)
        assert abs(s.pot_size_bb - 16.0) < 1e-9
        assert abs(s.hero_contribution_bb - 8.0) < 1e-9
        assert abs(s.villain_contribution_bb - 8.0) < 1e-9

    def test_pot_after_flop_bet_call(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=2.5)
        s = deal_flop(cfg, s.action_history, _flop_cards())
        # BB bets 3, BTN calls
        s = apply_action(cfg, s.action_history, action_type=ActionType.BET,
                         player=Player.VILLAIN, amount_to_bb=3.0)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.HERO, amount_to_bb=3.0)
        assert abs(s.pot_size_bb - 11.0) < 1e-9  # 5 + 3 + 3
        assert abs(s.hero_contribution_bb - 5.5) < 1e-9  # 2.5 + 3
        assert abs(s.villain_contribution_bb - 5.5) < 1e-9  # 2.5 + 3


# ===================================================================
# F. Amount-to-call derivation
# ===================================================================

class TestAmountToCall:
    def test_unopened_preflop_btn(self):
        cfg = _cfg("BTN_SB")
        _, s = post_blinds(cfg)
        # BTN to act, to_call = BB - SB = 0.5
        assert abs(s.current_bet_to_call_bb - 0.5) < 1e-9

    def test_facing_raise(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        # BB to call 2.5 - 1.0 = 1.5
        assert abs(s.current_bet_to_call_bb - 1.5) < 1e-9

    def test_after_3bet(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        s = apply_action(cfg, s.action_history, action_type=ActionType.RAISE,
                         player=Player.VILLAIN, amount_to_bb=8.0)
        # BTN to call 8.0 - 2.5 = 5.5
        assert abs(s.current_bet_to_call_bb - 5.5) < 1e-9

    def test_postflop_facing_bet(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=2.5)
        s = deal_flop(cfg, s.action_history, _flop_cards())
        s = apply_action(cfg, s.action_history, action_type=ActionType.BET,
                         player=Player.VILLAIN, amount_to_bb=3.0)
        # BTN to call 3.0 bb (street-level)
        assert abs(s.current_bet_to_call_bb - 3.0) < 1e-9

    def test_to_call_zero_after_check(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=2.5)
        s = deal_flop(cfg, s.action_history, _flop_cards())
        # BB checks, BTN to act with nothing to call
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.VILLAIN)
        assert abs(s.current_bet_to_call_bb) < 1e-9


# ===================================================================
# G. Legal action generation
# ===================================================================

class TestLegalActions:
    def test_unopened_preflop_btn(self):
        cfg = _cfg("BTN_SB")
        _, s = post_blinds(cfg)
        la = legal_actions(s)
        types = _action_types(la)
        assert ActionType.FOLD in types
        assert ActionType.CALL in types
        assert ActionType.RAISE in types

    def test_facing_limp_bb(self):
        cfg = _cfg("BB")
        h, _ = post_blinds(cfg)
        # BTN limps (calls)
        s = apply_action(cfg, h, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=1.0)
        la = legal_actions(s)
        types = _action_types(la)
        assert ActionType.CHECK in types
        # BB "raising" a limp is a BET in this engine (first voluntary
        # aggressive action on the street; blind posts are structural).
        assert ActionType.BET in types
        assert ActionType.FOLD not in types

    def test_facing_open_raise_bb(self):
        cfg = _cfg("BB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.VILLAIN, amount_to_bb=2.5)
        la = legal_actions(s)
        types = _action_types(la)
        assert ActionType.FOLD in types
        assert ActionType.CALL in types
        assert ActionType.RAISE in types

    def test_postflop_checked_to_actor(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=2.5)
        s = deal_flop(cfg, s.action_history, _flop_cards())
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.VILLAIN)
        la = legal_actions(s)
        types = _action_types(la)
        assert ActionType.CHECK in types
        assert ActionType.BET in types
        assert ActionType.FOLD not in types

    def test_facing_bet_postflop(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=2.5)
        s = deal_flop(cfg, s.action_history, _flop_cards())
        s = apply_action(cfg, s.action_history, action_type=ActionType.BET,
                         player=Player.VILLAIN, amount_to_bb=3.0)
        la = legal_actions(s)
        types = _action_types(la)
        assert ActionType.FOLD in types
        assert ActionType.CALL in types
        assert ActionType.RAISE in types

    def test_facing_raise_postflop(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=2.5)
        s = deal_flop(cfg, s.action_history, _flop_cards())
        s = apply_action(cfg, s.action_history, action_type=ActionType.BET,
                         player=Player.VILLAIN, amount_to_bb=3.0)
        s = apply_action(cfg, s.action_history, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=9.0)
        la = legal_actions(s)
        types = _action_types(la)
        assert ActionType.FOLD in types
        assert ActionType.CALL in types
        assert ActionType.RAISE in types

    def test_closed_action_no_legal(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=2.5)
        la = legal_actions(s)
        assert la == []

    def test_hand_over_no_legal(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.FOLD,
                         player=Player.HERO)
        la = legal_actions(s)
        assert la == []

    def test_raise_bounds_min_max(self):
        cfg = _cfg("BTN_SB")
        _, s = post_blinds(cfg)
        la = legal_actions(s)
        raises = [a for a in la if a.action_type == ActionType.RAISE]
        assert len(raises) == 1
        r = raises[0]
        assert r.min_to_bb is not None
        assert r.max_to_bb is not None
        # Min raise: BB posted 1.0, last full raise = 1.0, min raise-to = 1.0 + 1.0 = 2.0
        assert abs(r.min_to_bb - 2.0) < 1e-9
        # Max raise = effective stack
        assert abs(r.max_to_bb - 100.0) < 1e-9

    def test_bet_bounds_postflop(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=2.5)
        s = deal_flop(cfg, s.action_history, _flop_cards())
        # BB to act with CHECK or BET
        la = legal_actions(s)
        bets = [a for a in la if a.action_type == ActionType.BET]
        assert len(bets) == 1
        b = bets[0]
        # Min bet = 1 BB
        assert abs(b.min_to_bb - 1.0) < 1e-9
        # Max bet = remaining stack (100 - 2.5 = 97.5)
        assert abs(b.max_to_bb - 97.5) < 1e-9


# ===================================================================
# H. Closure logic
# ===================================================================

class TestClosureLogic:
    def test_check_check_closes(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        # BTN limps
        s = apply_action(cfg, h, action_type=ActionType.CALL,
                         player=Player.HERO, amount_to_bb=1.0)
        # BB checks -> preflop closes
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.VILLAIN)
        assert s.betting_round_closed is True

    def test_bet_call_closes(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=2.5)
        assert s.betting_round_closed is True

    def test_raise_call_closes(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        s = apply_action(cfg, s.action_history, action_type=ActionType.RAISE,
                         player=Player.VILLAIN, amount_to_bb=8.0)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.HERO, amount_to_bb=8.0)
        assert s.betting_round_closed is True

    def test_fold_ends_hand(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.FOLD,
                         player=Player.HERO)
        assert s.hand_over is True
        assert s.fold_winner == Player.VILLAIN

    def test_postflop_check_check_closes(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=2.5)
        s = deal_flop(cfg, s.action_history, _flop_cards())
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.VILLAIN)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.HERO)
        assert s.betting_round_closed is True

    def test_postflop_bet_call_closes(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=2.5)
        s = deal_flop(cfg, s.action_history, _flop_cards())
        s = apply_action(cfg, s.action_history, action_type=ActionType.BET,
                         player=Player.VILLAIN, amount_to_bb=3.0)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.HERO, amount_to_bb=3.0)
        assert s.betting_round_closed is True

    def test_postflop_raise_call_closes(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=2.5)
        s = deal_flop(cfg, s.action_history, _flop_cards())
        s = apply_action(cfg, s.action_history, action_type=ActionType.BET,
                         player=Player.VILLAIN, amount_to_bb=3.0)
        s = apply_action(cfg, s.action_history, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=9.0)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=9.0)
        assert s.betting_round_closed is True


# ===================================================================
# I. All-in handling
# ===================================================================

class TestAllInHandling:
    def test_preflop_all_in_jam(self):
        cfg = _cfg("BTN_SB", stack=20.0)
        h, _ = post_blinds(cfg)
        # BTN jams
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=20.0)
        assert s.hero_all_in is True
        # BB can still act
        assert s.current_actor == Player.VILLAIN
        la = legal_actions(s)
        types = _action_types(la)
        assert ActionType.FOLD in types
        assert ActionType.CALL in types

    def test_preflop_both_all_in(self):
        cfg = _cfg("BTN_SB", stack=20.0)
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=20.0)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=20.0)
        assert s.hero_all_in is True
        assert s.villain_all_in is True
        assert s.betting_round_closed is True
        # Not hand_over yet – need to deal remaining board
        assert s.awaiting_runout is True

    def test_all_in_no_further_betting(self):
        cfg = _cfg("BTN_SB", stack=20.0)
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=20.0)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=20.0)
        # Deal flop
        s = deal_flop(cfg, s.action_history, _flop_cards())
        la = legal_actions(s)
        assert la == []
        assert s.betting_round_closed is True

    def test_all_in_runout_to_showdown(self):
        cfg = _cfg("BTN_SB", stack=20.0)
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=20.0)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=20.0)
        s = deal_flop(cfg, s.action_history, _flop_cards())
        s = deal_turn(cfg, s.action_history, _turn_card())
        s = deal_river(cfg, s.action_history, _river_card())
        assert s.showdown_ready is True
        assert s.hand_over is True

    def test_flop_all_in(self):
        cfg = _cfg("BTN_SB", stack=30.0)
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=2.5)
        s = deal_flop(cfg, s.action_history, _flop_cards())
        # BB jams
        s = apply_action(cfg, s.action_history, action_type=ActionType.BET,
                         player=Player.VILLAIN, amount_to_bb=27.5)
        assert s.villain_all_in is True
        # BTN can call or fold
        la = legal_actions(s)
        types = _action_types(la)
        assert ActionType.FOLD in types
        assert ActionType.CALL in types
        assert ActionType.RAISE not in types

    def test_flop_both_all_in_awaiting_runout(self):
        cfg = _cfg("BTN_SB", stack=30.0)
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=2.5)
        s = deal_flop(cfg, s.action_history, _flop_cards())
        s = apply_action(cfg, s.action_history, action_type=ActionType.BET,
                         player=Player.VILLAIN, amount_to_bb=27.5)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.HERO, amount_to_bb=27.5)
        assert s.hero_all_in is True
        assert s.villain_all_in is True
        assert s.awaiting_runout is True
        assert s.hand_over is False


# ===================================================================
# J. Determinism
# ===================================================================

class TestDeterminism:
    def test_same_config_same_history_same_state(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s1 = apply_action(cfg, h, action_type=ActionType.RAISE,
                          player=Player.HERO, amount_to_bb=2.5)
        s2 = apply_action(cfg, h, action_type=ActionType.RAISE,
                          player=Player.HERO, amount_to_bb=2.5)
        assert s1.pot_size_bb == s2.pot_size_bb
        assert s1.hero_contribution_bb == s2.hero_contribution_bb
        assert s1.villain_contribution_bb == s2.villain_contribution_bb
        assert s1.current_bet_to_call_bb == s2.current_bet_to_call_bb
        assert s1.current_actor == s2.current_actor
        assert s1.betting_round_closed == s2.betting_round_closed
        assert s1.hand_over == s2.hand_over
        assert s1.current_street == s2.current_street

    def test_determinism_multi_street(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=2.5)
        s = deal_flop(cfg, s.action_history, _flop_cards())

        # Reconstruct again from scratch with same history
        s2 = validate_hand(cfg, s.action_history)
        assert s.pot_size_bb == s2.pot_size_bb
        assert s.hero_contribution_bb == s2.hero_contribution_bb
        assert s.current_actor == s2.current_actor
        assert s.current_street == s2.current_street
        assert len(s.board_cards) == len(s2.board_cards)


# ===================================================================
# K. Invalid sequences rejected
# ===================================================================

class TestInvalidSequences:
    def test_wrong_actor(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        # BB tries to act when it's BTN's turn
        with pytest.raises(ValidationError):
            apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.VILLAIN, amount_to_bb=2.5)

    def test_action_after_closure(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=2.5)
        # Now action is closed – any further action should fail
        with pytest.raises(ValidationError):
            apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.VILLAIN)

    def test_action_after_fold(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.FOLD,
                         player=Player.HERO)
        with pytest.raises(ValidationError):
            apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.VILLAIN)

    def test_check_when_facing_bet(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        with pytest.raises(ValidationError):
            apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.VILLAIN)

    def test_call_when_nothing_to_call(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.CALL,
                         player=Player.HERO, amount_to_bb=1.0)
        # BB has nothing extra to call (limped pot)
        with pytest.raises(ValidationError):
            apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=1.0)

    def test_fold_when_nothing_to_call(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.CALL,
                         player=Player.HERO, amount_to_bb=1.0)
        # BB cannot fold when there's nothing to call
        with pytest.raises(ValidationError):
            apply_action(cfg, s.action_history, action_type=ActionType.FOLD,
                         player=Player.VILLAIN)

    def test_raise_below_current_bet(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=3.0)
        with pytest.raises(ValidationError):
            apply_action(cfg, s.action_history, action_type=ActionType.RAISE,
                         player=Player.VILLAIN, amount_to_bb=2.0)

    def test_raise_exceeds_stack(self):
        cfg = _cfg("BTN_SB", stack=20.0)
        h, _ = post_blinds(cfg)
        with pytest.raises(ValidationError):
            apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=25.0)

    def test_bet_when_facing_bet(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=2.5)
        s = deal_flop(cfg, s.action_history, _flop_cards())
        s = apply_action(cfg, s.action_history, action_type=ActionType.BET,
                         player=Player.VILLAIN, amount_to_bb=3.0)
        # BTN can't BET – must RAISE
        with pytest.raises(ValidationError):
            apply_action(cfg, s.action_history, action_type=ActionType.BET,
                         player=Player.HERO, amount_to_bb=9.0)

    def test_illegal_board_count(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=2.5)
        with pytest.raises(ValidationError):
            deal_flop(cfg, s.action_history,
                      (parse_card("Ah"), parse_card("7c")))

    def test_negative_stack_rejected(self):
        with pytest.raises(ValidationError):
            cfg = make_hand_config("As Kd", "BTN_SB", -5.0)
            validate_hand(cfg, [])

    def test_wrong_blind_player(self):
        cfg = _cfg("BTN_SB")
        # Post SB with wrong player
        bad_history = [
            Action(action_type=ActionType.POST_BLIND,
                   player=Player.VILLAIN, amount_to_bb=0.5),
        ]
        with pytest.raises(ValidationError):
            validate_hand(cfg, bad_history)


# ===================================================================
# Full-hand integration test
# ===================================================================

class TestFullHandIntegration:
    def test_complete_hand_fold_on_river(self):
        """The manual example from the spec: complete hand ending in river fold."""
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)

        # Preflop: BTN raises to 2.5, BB calls
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=2.5)
        assert s.betting_round_closed is True
        assert abs(s.pot_size_bb - 5.0) < 1e-9

        # Flop: Ah 7c 2d – BB checks, BTN bets 1.5, BB calls
        s = deal_flop(cfg, s.action_history, _flop_cards())
        assert s.current_street == Street.FLOP
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.VILLAIN)
        s = apply_action(cfg, s.action_history, action_type=ActionType.BET,
                         player=Player.HERO, amount_to_bb=1.5)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=1.5)
        assert abs(s.pot_size_bb - 8.0) < 1e-9

        # Turn: 9s – BB checks, BTN checks
        s = deal_turn(cfg, s.action_history, _turn_card())
        assert s.current_street == Street.TURN
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.VILLAIN)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.HERO)
        assert abs(s.pot_size_bb - 8.0) < 1e-9

        # River: 4h – BB bets 5, BTN folds
        s = deal_river(cfg, s.action_history, _river_card())
        assert s.current_street == Street.RIVER
        s = apply_action(cfg, s.action_history, action_type=ActionType.BET,
                         player=Player.VILLAIN, amount_to_bb=5.0)
        s = apply_action(cfg, s.action_history, action_type=ActionType.FOLD,
                         player=Player.HERO)

        assert s.hand_over is True
        assert s.fold_winner == Player.VILLAIN
        assert abs(s.pot_size_bb - 13.0) < 1e-9

    def test_complete_hand_showdown(self):
        """Complete hand going to showdown via check-check on river."""
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)

        # Preflop
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=2.5)

        # Flop
        s = deal_flop(cfg, s.action_history, _flop_cards())
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.VILLAIN)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.HERO)

        # Turn
        s = deal_turn(cfg, s.action_history, _turn_card())
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.VILLAIN)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.HERO)

        # River
        s = deal_river(cfg, s.action_history, _river_card())
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.VILLAIN)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.HERO)

        assert s.showdown_ready is True
        assert s.hand_over is True
        assert s.current_street == Street.SHOWDOWN
        assert abs(s.pot_size_bb - 5.0) < 1e-9


# ===================================================================
# Audit: additional preflop transitions
# ===================================================================

class TestAuditPreflopTransitions:
    def test_bb_folds_facing_open(self):
        cfg = _cfg("BB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.VILLAIN, amount_to_bb=2.5)
        s = apply_action(cfg, s.action_history, action_type=ActionType.FOLD,
                         player=Player.HERO)
        assert s.hand_over is True
        assert s.fold_winner == Player.VILLAIN
        assert abs(s.pot_size_bb - 3.5) < 1e-9

    def test_limp_check_transitions_to_flop(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.CALL,
                         player=Player.HERO, amount_to_bb=1.0)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.VILLAIN)
        assert s.betting_round_closed is True
        assert abs(s.pot_size_bb - 2.0) < 1e-9
        # Deal flop, BB acts first
        s = deal_flop(cfg, s.action_history, _flop_cards())
        assert s.current_actor == Player.VILLAIN
        assert s.current_street == Street.FLOP

    def test_open_3bet_call(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        s = apply_action(cfg, s.action_history, action_type=ActionType.RAISE,
                         player=Player.VILLAIN, amount_to_bb=8.0)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.HERO, amount_to_bb=8.0)
        assert s.betting_round_closed is True
        assert abs(s.pot_size_bb - 16.0) < 1e-9
        assert abs(s.hero_contribution_bb - 8.0) < 1e-9
        assert abs(s.villain_contribution_bb - 8.0) < 1e-9


# ===================================================================
# Audit: additional postflop transitions
# ===================================================================

class TestAuditPostflopTransitions:
    def _to_flop(self, cfg=None):
        if cfg is None:
            cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=2.5)
        s = deal_flop(cfg, s.action_history, _flop_cards())
        return cfg, s

    def test_flop_bet_fold(self):
        cfg, s = self._to_flop()
        s = apply_action(cfg, s.action_history, action_type=ActionType.BET,
                         player=Player.VILLAIN, amount_to_bb=3.0)
        s = apply_action(cfg, s.action_history, action_type=ActionType.FOLD,
                         player=Player.HERO)
        assert s.hand_over is True
        assert s.fold_winner == Player.VILLAIN
        assert s.current_street == Street.FLOP
        assert abs(s.pot_size_bb - 8.0) < 1e-9

    def test_flop_bet_call_transitions_to_turn(self):
        cfg, s = self._to_flop()
        s = apply_action(cfg, s.action_history, action_type=ActionType.BET,
                         player=Player.VILLAIN, amount_to_bb=3.0)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.HERO, amount_to_bb=3.0)
        assert s.betting_round_closed is True
        s = deal_turn(cfg, s.action_history, _turn_card())
        assert s.current_street == Street.TURN
        assert s.current_actor == Player.VILLAIN

    def test_turn_bet_raise_call(self):
        cfg, s = self._to_flop()
        # Check-check flop
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.VILLAIN)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.HERO)
        s = deal_turn(cfg, s.action_history, _turn_card())
        # BB bets 4, BTN raises to 12, BB calls
        s = apply_action(cfg, s.action_history, action_type=ActionType.BET,
                         player=Player.VILLAIN, amount_to_bb=4.0)
        s = apply_action(cfg, s.action_history, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=12.0)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=12.0)
        assert s.betting_round_closed is True
        assert abs(s.pot_size_bb - 29.0) < 1e-9
        assert abs(s.hero_contribution_bb - 14.5) < 1e-9
        assert abs(s.villain_contribution_bb - 14.5) < 1e-9

    def test_river_bet_call_showdown(self):
        cfg, s = self._to_flop()
        # Check through flop
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.VILLAIN)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.HERO)
        # Check through turn
        s = deal_turn(cfg, s.action_history, _turn_card())
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.VILLAIN)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.HERO)
        # River: BB bets 5, BTN calls -> showdown
        s = deal_river(cfg, s.action_history, _river_card())
        s = apply_action(cfg, s.action_history, action_type=ActionType.BET,
                         player=Player.VILLAIN, amount_to_bb=5.0)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.HERO, amount_to_bb=5.0)
        assert s.showdown_ready is True
        assert s.hand_over is True
        assert abs(s.pot_size_bb - 15.0) < 1e-9

    def test_fold_on_turn(self):
        cfg, s = self._to_flop()
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.VILLAIN)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.HERO)
        s = deal_turn(cfg, s.action_history, _turn_card())
        s = apply_action(cfg, s.action_history, action_type=ActionType.BET,
                         player=Player.VILLAIN, amount_to_bb=4.0)
        s = apply_action(cfg, s.action_history, action_type=ActionType.FOLD,
                         player=Player.HERO)
        assert s.hand_over is True
        assert s.fold_winner == Player.VILLAIN
        assert s.current_street == Street.TURN


# ===================================================================
# Audit: min-raise bounds
# ===================================================================

class TestAuditMinRaiseBounds:
    def test_min_raise_after_3bet(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        s = apply_action(cfg, s.action_history, action_type=ActionType.RAISE,
                         player=Player.VILLAIN, amount_to_bb=8.0)
        la = legal_actions(s)
        raises = [a for a in la if a.action_type == ActionType.RAISE]
        assert len(raises) == 1
        # raise_increment = 8.0 - 2.5 = 5.5; min raise-to = 8.0 + 5.5 = 13.5
        assert abs(raises[0].min_to_bb - 13.5) < 1e-9
        assert abs(raises[0].max_to_bb - 100.0) < 1e-9

    def test_min_raise_after_flop_bet(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=2.5)
        s = deal_flop(cfg, s.action_history, _flop_cards())
        s = apply_action(cfg, s.action_history, action_type=ActionType.BET,
                         player=Player.VILLAIN, amount_to_bb=3.0)
        la = legal_actions(s)
        raises = [a for a in la if a.action_type == ActionType.RAISE]
        assert len(raises) == 1
        # After 3bb bet: min raise-to = 3.0 + 3.0 = 6.0
        assert abs(raises[0].min_to_bb - 6.0) < 1e-9
        # Max raise-to = remaining stack as street contrib: 0 + (100-2.5) = 97.5
        assert abs(raises[0].max_to_bb - 97.5) < 1e-9

    def test_min_raise_after_flop_raise(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=2.5)
        s = deal_flop(cfg, s.action_history, _flop_cards())
        s = apply_action(cfg, s.action_history, action_type=ActionType.BET,
                         player=Player.VILLAIN, amount_to_bb=3.0)
        s = apply_action(cfg, s.action_history, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=9.0)
        la = legal_actions(s)
        raises = [a for a in la if a.action_type == ActionType.RAISE]
        assert len(raises) == 1
        # raise_increment = 9.0 - 3.0 = 6.0; min raise-to = 9.0 + 6.0 = 15.0
        assert abs(raises[0].min_to_bb - 15.0) < 1e-9
        # BB max: street_contrib_villain(3.0) + remaining(100-2.5-3.0=94.5) ... wait
        # BB total contrib so far = 2.5(preflop) + 3.0(flop bet) = 5.5
        # BB remaining = 100 - 5.5 = 94.5
        # BB street contrib = 3.0. Max raise-to = 3.0 + 94.5 = 97.5
        assert abs(raises[0].max_to_bb - 97.5) < 1e-9

    def test_short_stack_jam_only(self):
        """With 1.5bb stack, BTN can only jam."""
        cfg = _cfg("BTN_SB", stack=1.5)
        _, s = post_blinds(cfg)
        la = legal_actions(s)
        raises = [a for a in la if a.action_type == ActionType.RAISE]
        assert len(raises) == 1
        # min and max collapse to 1.5 (jam)
        assert abs(raises[0].min_to_bb - 1.5) < 1e-9
        assert abs(raises[0].max_to_bb - 1.5) < 1e-9


# ===================================================================
# Audit: BB iso-raise as BET after limp
# ===================================================================

class TestAuditIsoRaise:
    def test_bb_iso_raise_as_bet(self):
        """BB 'raising' a limper is a BET in this engine."""
        cfg = _cfg("BB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=1.0)
        # BB bets to 4 (iso-raise in poker terms)
        s = apply_action(cfg, s.action_history, action_type=ActionType.BET,
                         player=Player.HERO, amount_to_bb=4.0)
        assert abs(s.pot_size_bb - 5.0) < 1e-9
        assert abs(s.hero_contribution_bb - 4.0) < 1e-9
        assert s.current_actor == Player.VILLAIN
        # BTN faces: fold/call/raise
        la = legal_actions(s)
        types = _action_types(la)
        assert ActionType.FOLD in types
        assert ActionType.CALL in types
        assert ActionType.RAISE in types
        calls = [a for a in la if a.action_type == ActionType.CALL]
        assert abs(calls[0].call_amount_bb - 3.0) < 1e-9


# ===================================================================
# Audit: CALL amount_to_bb validation
# ===================================================================

class TestAuditCallValidation:
    def test_correct_call_amount_accepted(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        # Correct: BB calls to 2.5 (street-level)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=2.5)
        assert abs(s.villain_contribution_bb - 2.5) < 1e-9

    def test_wrong_call_amount_rejected(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        with pytest.raises(ValidationError, match="CALL amount_to_bb"):
            apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=999.0)

    def test_call_without_amount_still_works(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        # No amount_to_bb: should auto-compute correctly
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN)
        assert abs(s.villain_contribution_bb - 2.5) < 1e-9


# ===================================================================
# Audit: config validation
# ===================================================================

class TestAuditConfigValidation:
    def test_stack_below_bb_rejected(self):
        with pytest.raises(ValidationError, match="at least"):
            cfg = make_hand_config("As Kd", "BTN_SB", 0.5)
            validate_hand(cfg, [])

    def test_stack_equal_bb_accepted(self):
        cfg = make_hand_config("As Kd", "BTN_SB", 1.0)
        h, s = post_blinds(cfg)
        assert s.villain_all_in is True


# ===================================================================
# Audit: additional invalid sequences
# ===================================================================

class TestAuditInvalidSequences:
    def test_bet_on_preflop_facing_blind(self):
        """BTN can't BET preflop because there's a forced bet to call."""
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        with pytest.raises(ValidationError):
            apply_action(cfg, h, action_type=ActionType.BET,
                         player=Player.HERO, amount_to_bb=3.0)

    def test_deal_before_betting_closed(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        # Try to deal flop before preflop betting is closed
        with pytest.raises(ValidationError):
            deal_flop(cfg, h, _flop_cards())

    def test_raise_below_min_raise(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        # Min raise-to is 2.0; try 1.5
        with pytest.raises(ValidationError):
            apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=1.5)

    def test_deal_river_before_turn(self):
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=2.5)
        s = deal_flop(cfg, s.action_history, _flop_cards())
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.VILLAIN)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.HERO)
        # Skip turn, try to deal river directly
        with pytest.raises(ValidationError):
            deal_river(cfg, s.action_history, _river_card())


# ===================================================================
# Audit: full-hand determinism
# ===================================================================

class TestAuditFullHandDeterminism:
    def test_full_hand_determinism(self):
        """Reconstruct a full hand twice and verify identical states."""
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=2.5)
        s = deal_flop(cfg, s.action_history, _flop_cards())
        s = apply_action(cfg, s.action_history, action_type=ActionType.BET,
                         player=Player.VILLAIN, amount_to_bb=3.0)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.HERO, amount_to_bb=3.0)
        s = deal_turn(cfg, s.action_history, _turn_card())
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.VILLAIN)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.HERO)
        s = deal_river(cfg, s.action_history, _river_card())
        s = apply_action(cfg, s.action_history, action_type=ActionType.BET,
                         player=Player.VILLAIN, amount_to_bb=5.0)
        s = apply_action(cfg, s.action_history, action_type=ActionType.FOLD,
                         player=Player.HERO)

        # Reconstruct from scratch with same history
        s2 = validate_hand(cfg, s.action_history)

        assert s.pot_size_bb == s2.pot_size_bb
        assert s.hero_contribution_bb == s2.hero_contribution_bb
        assert s.villain_contribution_bb == s2.villain_contribution_bb
        assert s.current_bet_to_call_bb == s2.current_bet_to_call_bb
        assert s.current_actor == s2.current_actor
        assert s.current_street == s2.current_street
        assert s.betting_round_closed == s2.betting_round_closed
        assert s.hand_over == s2.hand_over
        assert s.showdown_ready == s2.showdown_ready
        assert s.fold_winner == s2.fold_winner
        assert s.hero_all_in == s2.hero_all_in
        assert s.villain_all_in == s2.villain_all_in
        assert len(s.board_cards) == len(s2.board_cards)
        assert s.number_of_raises_this_street == s2.number_of_raises_this_street
        assert s.last_aggressor == s2.last_aggressor


# ===================================================================
# Audit: debug output
# ===================================================================

class TestAuditDebugOutput:
    def test_debug_format_contains_required_fields(self):
        from poker_core.debug import format_state
        cfg = _cfg("BTN_SB")
        h, _ = post_blinds(cfg)
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        output = format_state(s)
        assert "Street:" in output
        assert "Board:" in output
        assert "Pot:" in output
        assert "Hero contrib:" in output
        assert "Villain contrib:" in output
        assert "To call:" in output
        assert "Current actor:" in output
        assert "Last aggressor:" in output
        assert "Raises (street):" in output
        assert "Betting closed:" in output
        assert "All-in:" in output
        assert "Showdown ready:" in output
        assert "Hand over:" in output
        assert "Legal actions:" in output
        assert "Hero position:" in output
        assert "Hero cards:" in output
        assert "Eff. stack:" in output


# ===================================================================
# Audit: walkthrough scenario helpers
# ===================================================================

class TestAuditWalkthrough:
    def test_step_by_step_river_fold_walkthrough(self):
        """Verify state at each step of a full hand ending in river fold."""
        cfg = _cfg("BTN_SB")

        # Step 1: blinds
        h, s = post_blinds(cfg)
        assert s.current_street == Street.PREFLOP
        assert s.current_actor == Player.HERO
        assert abs(s.pot_size_bb - 1.5) < 1e-9
        assert abs(s.current_bet_to_call_bb - 0.5) < 1e-9

        # Step 2: BTN raises to 2.5
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=2.5)
        assert s.current_actor == Player.VILLAIN
        assert abs(s.current_bet_to_call_bb - 1.5) < 1e-9
        assert s.last_aggressor == Player.HERO
        assert s.number_of_raises_this_street == 1

        # Step 3: BB calls
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=2.5)
        assert s.betting_round_closed is True
        assert abs(s.pot_size_bb - 5.0) < 1e-9

        # Step 4: flop dealt
        s = deal_flop(cfg, s.action_history, _flop_cards())
        assert s.current_street == Street.FLOP
        assert s.current_actor == Player.VILLAIN
        assert abs(s.current_bet_to_call_bb) < 1e-9
        assert s.number_of_raises_this_street == 0
        assert s.last_aggressor is None

        # Step 5: BB checks
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.VILLAIN)
        assert s.current_actor == Player.HERO

        # Step 6: BTN bets 1.5
        s = apply_action(cfg, s.action_history, action_type=ActionType.BET,
                         player=Player.HERO, amount_to_bb=1.5)
        assert s.current_actor == Player.VILLAIN
        assert abs(s.current_bet_to_call_bb - 1.5) < 1e-9
        assert s.last_aggressor == Player.HERO

        # Step 7: BB calls
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN, amount_to_bb=1.5)
        assert s.betting_round_closed is True
        assert abs(s.pot_size_bb - 8.0) < 1e-9

        # Step 8: turn dealt
        s = deal_turn(cfg, s.action_history, _turn_card())
        assert s.current_street == Street.TURN
        assert len(s.board_cards) == 4

        # Step 9-10: check-check on turn
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.VILLAIN)
        s = apply_action(cfg, s.action_history, action_type=ActionType.CHECK,
                         player=Player.HERO)
        assert s.betting_round_closed is True
        assert abs(s.pot_size_bb - 8.0) < 1e-9

        # Step 11: river dealt
        s = deal_river(cfg, s.action_history, _river_card())
        assert s.current_street == Street.RIVER
        assert len(s.board_cards) == 5

        # Step 12: BB bets 5
        s = apply_action(cfg, s.action_history, action_type=ActionType.BET,
                         player=Player.VILLAIN, amount_to_bb=5.0)
        assert s.current_actor == Player.HERO
        assert abs(s.current_bet_to_call_bb - 5.0) < 1e-9

        # Step 13: BTN folds
        s = apply_action(cfg, s.action_history, action_type=ActionType.FOLD,
                         player=Player.HERO)
        assert s.hand_over is True
        assert s.fold_winner == Player.VILLAIN
        assert abs(s.pot_size_bb - 13.0) < 1e-9
        la = legal_actions(s)
        assert la == []

    def test_step_by_step_all_in_runout(self):
        """Verify state at each step of a preflop all-in through runout."""
        cfg = _cfg("BTN_SB", stack=20.0)
        h, s = post_blinds(cfg)

        # BTN jams
        s = apply_action(cfg, h, action_type=ActionType.RAISE,
                         player=Player.HERO, amount_to_bb=20.0)
        assert s.hero_all_in is True
        assert s.current_actor == Player.VILLAIN
        assert not s.betting_round_closed

        # BB calls
        s = apply_action(cfg, s.action_history, action_type=ActionType.CALL,
                         player=Player.VILLAIN)
        assert s.hero_all_in is True
        assert s.villain_all_in is True
        assert s.betting_round_closed is True
        assert s.awaiting_runout is True
        assert not s.hand_over
        assert abs(s.pot_size_bb - 40.0) < 1e-9

        # Deal flop - no betting possible
        s = deal_flop(cfg, s.action_history, _flop_cards())
        assert s.current_actor is None
        assert s.betting_round_closed is True
        assert legal_actions(s) == []
        assert s.awaiting_runout is True

        # Deal turn
        s = deal_turn(cfg, s.action_history, _turn_card())
        assert s.awaiting_runout is True
        assert not s.hand_over

        # Deal river -> showdown
        s = deal_river(cfg, s.action_history, _river_card())
        assert s.showdown_ready is True
        assert s.hand_over is True
        assert s.current_street == Street.SHOWDOWN
        assert abs(s.pot_size_bb - 40.0) < 1e-9
