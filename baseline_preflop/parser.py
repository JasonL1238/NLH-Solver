"""Helper functions to build PokerState from human-readable input.

All ``amount`` values for RAISE/BET mean **raise-to bb** (total contribution
after the action).  We compute ``amount_added_bb`` internally.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

from .classification import parse_cards
from .models import (
    ActionRecord, ActionType, Player, PokerState, Position, Street,
)
from .validation import derive_preflop_state, validate_preflop_state


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def make_preflop_state(
    hero_cards: str,
    hero_position: str,
    effective_stack_bb: float,
    action_history: List[Dict[str, Union[str, float]]],
    small_blind_bb: float = 0.5,
    big_blind_bb: float = 1.0,
) -> PokerState:
    """Build a fully-populated PokerState from compact input.

    ``action_history`` entries::

        {"player": "HERO"|"VILLAIN",
         "action": "POST_BLIND"|"FOLD"|"CHECK"|"CALL"|"RAISE"|"BET",
         "amount": <float>}

    For RAISE/BET, ``amount`` is **raise-to** (total contribution after action).
    For POST_BLIND, ``amount`` is the blind size posted.
    For CALL, ``amount`` is optional (auto-computed).
    For FOLD/CHECK, ``amount`` is ignored.
    """
    hc = parse_cards(hero_cards)
    h_pos = Position(hero_position) if isinstance(hero_position, str) else hero_position
    v_pos = Position.BB if h_pos == Position.BTN_SB else Position.BTN_SB

    records = _build_records(action_history, h_pos, effective_stack_bb, small_blind_bb, big_blind_bb)

    derived_fields = derive_preflop_state(
        records, h_pos, effective_stack_bb, small_blind_bb, big_blind_bb,
    )

    state = PokerState(
        hero_hole_cards=hc,
        hero_position=h_pos,
        villain_position=v_pos,
        current_street=Street.PREFLOP,
        action_history=records,
        effective_stack_bb=effective_stack_bb,
        small_blind_bb=small_blind_bb,
        big_blind_bb=big_blind_bb,
        pot_size_bb=derived_fields["pot_size_bb"],
        current_bet_to_call_bb=derived_fields["current_bet_to_call_bb"],
        hero_contribution_bb=derived_fields["hero_contribution_bb"],
        villain_contribution_bb=derived_fields["villain_contribution_bb"],
        current_actor=derived_fields["current_actor"],
        betting_round_closed=derived_fields["betting_round_closed"],
        last_aggressor=derived_fields["last_aggressor"],
        number_of_raises_this_street=derived_fields["number_of_raises_this_street"],
        hand_over=derived_fields["hand_over"],
        derived=derived_fields["derived"],
    )
    validate_preflop_state(state)
    return state


def _build_records(
    raw: List[Dict[str, Union[str, float]]],
    hero_position: Position,
    effective_stack_bb: float,
    small_blind_bb: float,
    big_blind_bb: float,
) -> List[ActionRecord]:
    """Convert raw dict-based action list to ActionRecord list."""
    records: List[ActionRecord] = []
    contributions: Dict[Player, float] = {Player.HERO: 0.0, Player.VILLAIN: 0.0}
    bet_level = 0.0

    for idx, entry in enumerate(raw):
        player = Player(entry["player"])
        action = ActionType(entry["action"])
        amount_raw = entry.get("amount", 0.0)

        if action == ActionType.POST_BLIND:
            added = float(amount_raw)
            contributions[player] += added
            bet_level = max(bet_level, contributions[player])
            records.append(ActionRecord(
                player=player,
                street=Street.PREFLOP,
                action_type=action,
                amount_added_bb=added,
                total_contribution_after_action_bb=contributions[player],
                sequence_index=idx,
            ))

        elif action == ActionType.FOLD:
            records.append(ActionRecord(
                player=player, street=Street.PREFLOP, action_type=action,
                amount_added_bb=0.0,
                total_contribution_after_action_bb=contributions[player],
                sequence_index=idx,
            ))

        elif action == ActionType.CHECK:
            records.append(ActionRecord(
                player=player, street=Street.PREFLOP, action_type=action,
                amount_added_bb=0.0,
                total_contribution_after_action_bb=contributions[player],
                sequence_index=idx,
            ))

        elif action == ActionType.CALL:
            needed = bet_level - contributions[player]
            needed = min(needed, effective_stack_bb - contributions[player])
            contributions[player] += needed
            records.append(ActionRecord(
                player=player, street=Street.PREFLOP, action_type=action,
                amount_added_bb=needed,
                total_contribution_after_action_bb=contributions[player],
                sequence_index=idx,
            ))

        elif action in (ActionType.RAISE, ActionType.BET):
            raise_to = float(amount_raw)
            added = raise_to - contributions[player]
            contributions[player] = raise_to
            bet_level = raise_to
            records.append(ActionRecord(
                player=player, street=Street.PREFLOP, action_type=action,
                amount_added_bb=added,
                total_contribution_after_action_bb=raise_to,
                sequence_index=idx,
            ))

        else:
            raise ValueError(f"Unknown action type: {action}")

    return records


# ---------------------------------------------------------------------------
# Shorthand helpers for common spots
# ---------------------------------------------------------------------------

def _blinds(hero_position: str, sb: float = 0.5, bb: float = 1.0):
    """Return standard blind postings for the given hero position."""
    if hero_position == "BTN_SB":
        return [
            {"player": "HERO", "action": "POST_BLIND", "amount": sb},
            {"player": "VILLAIN", "action": "POST_BLIND", "amount": bb},
        ]
    return [
        {"player": "VILLAIN", "action": "POST_BLIND", "amount": sb},
        {"player": "HERO", "action": "POST_BLIND", "amount": bb},
    ]


def unopened_btn_decision(hero_cards: str, stack_bb: float, sb: float = 0.5, bb: float = 1.0) -> PokerState:
    """BTN_SB acts first, no prior voluntary action."""
    return make_preflop_state(
        hero_cards=hero_cards,
        hero_position="BTN_SB",
        effective_stack_bb=stack_bb,
        action_history=_blinds("BTN_SB", sb, bb),
        small_blind_bb=sb,
        big_blind_bb=bb,
    )


def bb_vs_limp_decision(hero_cards: str, stack_bb: float, sb: float = 0.5, bb: float = 1.0) -> PokerState:
    """BB facing a BTN limp (call)."""
    history = _blinds("BB", sb, bb) + [
        {"player": "VILLAIN", "action": "CALL", "amount": bb},
    ]
    return make_preflop_state(
        hero_cards=hero_cards,
        hero_position="BB",
        effective_stack_bb=stack_bb,
        action_history=history,
        small_blind_bb=sb,
        big_blind_bb=bb,
    )


def bb_vs_open_decision(
    hero_cards: str,
    open_size_bb: float,
    stack_bb: float,
    sb: float = 0.5,
    bb: float = 1.0,
) -> PokerState:
    """BB facing a BTN open raise to ``open_size_bb``."""
    history = _blinds("BB", sb, bb) + [
        {"player": "VILLAIN", "action": "RAISE", "amount": open_size_bb},
    ]
    return make_preflop_state(
        hero_cards=hero_cards,
        hero_position="BB",
        effective_stack_bb=stack_bb,
        action_history=history,
        small_blind_bb=sb,
        big_blind_bb=bb,
    )


def btn_vs_iso_after_limp_decision(
    hero_cards: str,
    raise_size_bb: float,
    stack_bb: float,
    sb: float = 0.5,
    bb: float = 1.0,
) -> PokerState:
    """BTN limped, BB raised, now BTN decides."""
    history = _blinds("BTN_SB", sb, bb) + [
        {"player": "HERO", "action": "CALL", "amount": bb},
        {"player": "VILLAIN", "action": "RAISE", "amount": raise_size_bb},
    ]
    return make_preflop_state(
        hero_cards=hero_cards,
        hero_position="BTN_SB",
        effective_stack_bb=stack_bb,
        action_history=history,
        small_blind_bb=sb,
        big_blind_bb=bb,
    )


def btn_vs_3bet_decision(
    hero_cards: str,
    open_size_bb: float,
    threebet_size_bb: float,
    stack_bb: float,
    sb: float = 0.5,
    bb: float = 1.0,
) -> PokerState:
    """BTN opened, BB 3bet, now BTN decides."""
    history = _blinds("BTN_SB", sb, bb) + [
        {"player": "HERO", "action": "RAISE", "amount": open_size_bb},
        {"player": "VILLAIN", "action": "RAISE", "amount": threebet_size_bb},
    ]
    return make_preflop_state(
        hero_cards=hero_cards,
        hero_position="BTN_SB",
        effective_stack_bb=stack_bb,
        action_history=history,
        small_blind_bb=sb,
        big_blind_bb=bb,
    )


def bb_vs_4bet_decision(
    hero_cards: str,
    open_size_bb: float,
    threebet_size_bb: float,
    fourbet_size_bb: float,
    stack_bb: float,
    sb: float = 0.5,
    bb: float = 1.0,
) -> PokerState:
    """BB 3bet, BTN 4bet, now BB decides."""
    history = _blinds("BB", sb, bb) + [
        {"player": "VILLAIN", "action": "RAISE", "amount": open_size_bb},
        {"player": "HERO", "action": "RAISE", "amount": threebet_size_bb},
        {"player": "VILLAIN", "action": "RAISE", "amount": fourbet_size_bb},
    ]
    return make_preflop_state(
        hero_cards=hero_cards,
        hero_position="BB",
        effective_stack_bb=stack_bb,
        action_history=history,
        small_blind_bb=sb,
        big_blind_bb=bb,
    )
