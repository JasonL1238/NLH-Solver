"""Map strategy outputs and ``LegalAction`` bounds to concrete ``apply_action`` calls."""

from __future__ import annotations

from typing import Optional, Tuple

from baseline_preflop.models import ActionType as PreflopActionType
from baseline_preflop.models import LegalActionOption
from flop_baseline.models import FlopActionChoice
from poker_core.models import ActionType, HandState, LegalAction, Player
from poker_core.transitions import apply_action


def choose_raise_or_bet_amount(legal: LegalAction) -> float:
    """Deterministic sizing: midpoint of legal ``min_to_bb`` / ``max_to_bb``."""
    lo = legal.min_to_bb
    hi = legal.max_to_bb
    if lo is None or hi is None:
        raise ValueError("BET/RAISE legal action missing min/max bounds")
    mid = (float(lo) + float(hi)) / 2.0
    return round(mid, 2)


def poker_legal_amount_for_call(state: HandState, legal_call: LegalAction) -> float:
    """Total street contribution after CALL (``amount_to_bb`` on ``Action``)."""
    if legal_call.action_type != ActionType.CALL:
        raise ValueError("Expected CALL legal action")
    if legal_call.call_amount_bb is None:
        raise ValueError("CALL legal action missing call_amount_bb")
    actor = state.current_actor
    if actor is None:
        raise ValueError("No current actor")
    if actor == Player.HERO:
        street = state.street_contrib_hero
    else:
        street = state.street_contrib_villain
    return round(street + float(legal_call.call_amount_bb), 2)


def apply_legal_action(
    config,
    history: list,
    state: HandState,
    *,
    legal: LegalAction,
    amount_to_bb: Optional[float],
) -> HandState:
    """Append one betting action for ``state.current_actor``."""
    actor = state.current_actor
    if actor is None:
        raise ValueError("Cannot apply: no current actor")
    at = legal.action_type
    if at in (ActionType.BET, ActionType.RAISE):
        if amount_to_bb is None:
            amount_to_bb = choose_raise_or_bet_amount(legal)
        return apply_action(
            config,
            history,
            action_type=at,
            player=actor,
            amount_to_bb=amount_to_bb,
        )
    if at == ActionType.CALL:
        if amount_to_bb is None:
            amount_to_bb = poker_legal_amount_for_call(state, legal)
        return apply_action(
            config,
            history,
            action_type=ActionType.CALL,
            player=actor,
            amount_to_bb=amount_to_bb,
        )
    if at == ActionType.FOLD:
        return apply_action(config, history, action_type=ActionType.FOLD, player=actor)
    if at == ActionType.CHECK:
        return apply_action(config, history, action_type=ActionType.CHECK, player=actor)
    raise ValueError(f"Unsupported legal action type: {at}")


def preflop_option_to_poker_apply(
    state: HandState,
    legal_list: list,
    chosen: LegalActionOption,
) -> Tuple[LegalAction, Optional[float]]:
    """Resolve baseline ``LegalActionOption`` to a concrete ``LegalAction`` + amount."""
    actor = state.current_actor
    if actor is None:
        raise ValueError("No current actor")
    if actor != Player.HERO:
        raise ValueError("Preflop engine apply expects HERO to act")

    ct = chosen.action_type
    if ct == PreflopActionType.FOLD:
        la = _first_match(legal_list, ActionType.FOLD)
        return la, None
    if ct == PreflopActionType.CHECK:
        la = _first_match(legal_list, ActionType.CHECK)
        return la, None
    if ct == PreflopActionType.CALL:
        la = _first_match(legal_list, ActionType.CALL)
        return la, poker_legal_amount_for_call(state, la)
    if ct == PreflopActionType.RAISE:
        target = chosen.raise_to_bb
        if target is None:
            raise ValueError("RAISE option missing raise_to_bb")
        la = _raise_legal_for_target(legal_list, float(target))
        lo = la.min_to_bb or 0.0
        hi = la.max_to_bb or lo
        tgt = max(lo, min(float(target), hi))
        return la, round(tgt, 2)
    raise ValueError(f"Unsupported preflop option: {chosen}")


def flop_choice_to_poker_apply(
    state: HandState,
    choice: FlopActionChoice,
) -> Tuple[LegalAction, Optional[float]]:
    """Map ``FlopActionChoice`` (already tied to a concrete ``LegalAction``) to apply args."""
    la = choice.legal_action
    at = la.action_type
    if at in (ActionType.BET, ActionType.RAISE):
        if choice.size_bb is None:
            return la, choose_raise_or_bet_amount(la)
        return la, round(float(choice.size_bb), 2)
    if at == ActionType.CALL:
        if choice.size_bb is not None:
            return la, round(float(choice.size_bb), 2)
        return la, poker_legal_amount_for_call(state, la)
    if at in (ActionType.FOLD, ActionType.CHECK):
        return la, None
    raise ValueError(f"Unsupported flop choice action: {at}")


def _first_match(legal_list: list, wanted: ActionType) -> LegalAction:
    for la in legal_list:
        if la.action_type == wanted:
            return la
    raise ValueError(f"No legal action matching {wanted} in {legal_list!r}")


def _raise_legal_for_target(legal_list: list, raise_to_bb: float) -> LegalAction:
    """Pick the ``RAISE`` ``LegalAction`` bucket that contains ``raise_to_bb``."""
    raises = [
        la
        for la in legal_list
        if la.action_type == ActionType.RAISE
        and la.min_to_bb is not None
        and la.max_to_bb is not None
        and float(la.min_to_bb) - 1e-6 <= raise_to_bb <= float(la.max_to_bb) + 1e-6
    ]
    if raises:
        return min(
            raises,
            key=lambda la: abs(
                ((float(la.min_to_bb) + float(la.max_to_bb)) / 2.0) - raise_to_bb
            ),
        )
    return _first_match(legal_list, ActionType.RAISE)
