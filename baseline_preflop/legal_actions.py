"""Exact legal-action generation for heads-up preflop."""

from __future__ import annotations

from typing import List

from .models import (
    ActionType, LegalActionOption, Player, PokerState, Street,
)


def legal_actions_for_hero(state: PokerState) -> List[LegalActionOption]:
    """Compute the set of legal actions for HERO in the current state.

    Returns an empty list when:
    - hand is over
    - betting round is closed
    - hero is not the current actor
    - hero is already all-in
    """
    if state.hand_over:
        return []
    if state.betting_round_closed:
        return []
    if state.current_actor != Player.HERO:
        return []
    if state.current_street != Street.PREFLOP:
        return []

    hero_remaining = state.effective_stack_bb - state.hero_contribution_bb
    if hero_remaining < 1e-9:
        return []

    to_call = state.current_bet_to_call_bb
    actions: List[LegalActionOption] = []

    # FOLD is always legal when facing a bet
    if to_call > 1e-9:
        actions.append(LegalActionOption(action_type=ActionType.FOLD))

    # CHECK is legal only when there is nothing to call
    if to_call < 1e-9:
        actions.append(LegalActionOption(action_type=ActionType.CHECK))

    # CALL
    if to_call > 1e-9:
        actual_call = min(to_call, hero_remaining)
        actions.append(LegalActionOption(
            action_type=ActionType.CALL,
            call_amount_bb=round(actual_call, 2),
        ))

    # RAISE options
    _add_raise_options(state, actions, hero_remaining, to_call)

    return actions


def _add_raise_options(
    state: PokerState,
    actions: List[LegalActionOption],
    hero_remaining: float,
    to_call: float,
) -> None:
    """Add concrete raise-to options to the action list."""
    current_bet = state.hero_contribution_bb + to_call
    # This is the bet level hero would match by calling

    # Determine min raise-to
    last_raise_increment = _last_raise_increment(state)
    min_raise_to = current_bet + max(last_raise_increment, state.big_blind_bb)
    max_raise_to = state.effective_stack_bb

    if max_raise_to < min_raise_to - 1e-9:
        # Can only jam for less than a full raise
        if hero_remaining > to_call + 1e-9:
            actions.append(LegalActionOption(
                action_type=ActionType.RAISE,
                raise_to_bb=round(max_raise_to, 2),
            ))
        return

    if hero_remaining <= to_call + 1e-9:
        return

    raise_sizes: List[float] = []

    # Always include min raise
    raise_sizes.append(min_raise_to)

    # Context-dependent standard sizes
    d = state.derived
    if d is not None:
        if d.unopened_pot:
            for s in [2.5, 3.0]:
                if min_raise_to - 1e-9 <= s <= max_raise_to + 1e-9:
                    raise_sizes.append(s)
        elif d.facing_limp:
            for s in [3.5, 4.5]:
                if min_raise_to - 1e-9 <= s <= max_raise_to + 1e-9:
                    raise_sizes.append(s)
        elif d.facing_open_raise:
            open_size = current_bet
            for mult in [3.0, 3.5]:
                s = round(open_size * mult, 2)
                if min_raise_to - 1e-9 <= s <= max_raise_to + 1e-9:
                    raise_sizes.append(s)
        elif d.facing_3bet:
            threeb_size = current_bet
            for mult in [2.3, 2.7]:
                s = round(threeb_size * mult, 2)
                if min_raise_to - 1e-9 <= s <= max_raise_to + 1e-9:
                    raise_sizes.append(s)
        elif d.facing_4bet:
            fourb_size = current_bet
            for mult in [2.2, 2.5]:
                s = round(fourb_size * mult, 2)
                if min_raise_to - 1e-9 <= s <= max_raise_to + 1e-9:
                    raise_sizes.append(s)

    # Always include jam
    raise_sizes.append(max_raise_to)

    # Deduplicate and sort
    seen = set()
    unique: List[float] = []
    for s in raise_sizes:
        rounded = round(s, 2)
        if rounded not in seen:
            seen.add(rounded)
            unique.append(rounded)
    unique.sort()

    for size in unique:
        actions.append(LegalActionOption(
            action_type=ActionType.RAISE,
            raise_to_bb=size,
        ))


def _last_raise_increment(state: PokerState) -> float:
    """Compute the size of the last raise increment in bb."""
    prev_bet_level = 0.0
    current_bet_level = 0.0

    for rec in state.action_history:
        if rec.action_type in (ActionType.RAISE, ActionType.BET):
            prev_bet_level = current_bet_level
            current_bet_level = rec.total_contribution_after_action_bb
        elif rec.action_type == ActionType.POST_BLIND:
            current_bet_level = max(current_bet_level, rec.total_contribution_after_action_bb)

    if current_bet_level > prev_bet_level:
        return current_bet_level - prev_bet_level
    return state.big_blind_bb
