"""Street-agnostic legal action generation.

Computes legal actions for the current actor based on the canonical
``HandState``.  Returns bounds-based options so strategy layers can choose
exact sizes within the legal range.
"""

from __future__ import annotations

from typing import List

from .models import (
    ActionType,
    HandState,
    LegalAction,
    Player,
)


def legal_actions(state: HandState) -> List[LegalAction]:
    """Return the list of legal actions for the current actor.

    Returns an empty list when:
    - the hand is over
    - the betting round is closed
    - no current actor
    - the current actor is all-in
    """
    if state.hand_over or state.showdown_ready:
        return []
    if state.betting_round_closed:
        return []
    if state.current_actor is None:
        return []

    actor = state.current_actor
    actor_is_all_in = (
        state.hero_all_in if actor == Player.HERO else state.villain_all_in
    )
    if actor_is_all_in:
        return []

    if actor == Player.HERO:
        actor_contrib_total = state.hero_contribution_bb
        actor_street_contrib = state.street_contrib_hero
    else:
        actor_contrib_total = state.villain_contribution_bb
        actor_street_contrib = state.street_contrib_villain

    cap = state.config.stack_cap_bb(actor)
    remaining = cap - actor_contrib_total
    to_call = state.current_bet_to_call_bb
    bb = state.config.big_blind_bb

    actions: List[LegalAction] = []

    if to_call < 1e-9:
        # -- Nothing to call: CHECK or BET --
        actions.append(LegalAction(action_type=ActionType.CHECK))

        if remaining > 1e-9:
            min_bet_to = min(bb, remaining)
            max_bet_to = remaining
            actions.append(LegalAction(
                action_type=ActionType.BET,
                min_to_bb=round(min_bet_to, 2),
                max_to_bb=round(max_bet_to, 2),
            ))
    else:
        # -- Facing a bet/raise: FOLD, CALL, optionally RAISE --
        actions.append(LegalAction(action_type=ActionType.FOLD))

        actual_call = min(to_call, remaining)
        actions.append(LegalAction(
            action_type=ActionType.CALL,
            call_amount_bb=round(actual_call, 2),
        ))

        # RAISE is legal if the actor has chips beyond calling
        chips_after_call = remaining - actual_call
        if chips_after_call > 1e-9:
            min_raise_increment = max(state.last_full_raise_size, bb)
            min_raise_to = state.street_bet_level + min_raise_increment

            # max raise-to = actor's street contribution after going all-in
            max_raise_to = actor_street_contrib + remaining

            if min_raise_to > max_raise_to + 1e-9:
                # Can only jam for less than a full min-raise
                min_raise_to = max_raise_to

            actions.append(LegalAction(
                action_type=ActionType.RAISE,
                min_to_bb=round(min_raise_to, 2),
                max_to_bb=round(max_raise_to, 2),
            ))

    return actions
