"""Convert ``poker_core`` preflop ``Action`` rows into Phase A/B compact dicts."""

from __future__ import annotations

from typing import Dict, List, Tuple, Union

from poker_core.models import Action, ActionType, Player

RawDict = Dict[str, Union[str, float]]


class PreflopBridgeError(Exception):
    """History cannot be represented in baseline preflop compact format."""


def find_deal_flop_index(actions: List[Action]) -> int:
    for i, a in enumerate(actions):
        if a.action_type == ActionType.DEAL_FLOP:
            return i
    return len(actions)


def poker_actions_to_preflop_raw(actions: List[Action]) -> List[RawDict]:
    """Map preflop segment (through index before ``DEAL_FLOP``) to recorder dicts."""
    end = find_deal_flop_index(actions)
    slice_ = actions[:end]
    out: List[RawDict] = []
    for a in slice_:
        at = a.action_type
        if at == ActionType.POST_BLIND:
            if a.player is None or a.amount_to_bb is None:
                raise PreflopBridgeError("POST_BLIND missing player or amount")
            out.append(
                {
                    "player": a.player.value,
                    "action": at.value,
                    "amount": float(a.amount_to_bb),
                }
            )
        elif at == ActionType.FOLD:
            if a.player is None:
                raise PreflopBridgeError("FOLD missing player")
            out.append({"player": a.player.value, "action": at.value, "amount": 0.0})
        elif at == ActionType.CHECK:
            if a.player is None:
                raise PreflopBridgeError("CHECK missing player")
            out.append({"player": a.player.value, "action": at.value, "amount": 0.0})
        elif at == ActionType.CALL:
            if a.player is None:
                raise PreflopBridgeError("CALL missing player")
            d: RawDict = {"player": a.player.value, "action": at.value}
            if a.amount_to_bb is not None:
                d["amount"] = float(a.amount_to_bb)
            out.append(d)
        elif at in (ActionType.RAISE, ActionType.BET):
            if a.player is None or a.amount_to_bb is None:
                raise PreflopBridgeError(f"{at.value} missing player or amount_to_bb")
            # Phase A/B uses RAISE for preflop opens in compact history; normalize BET->RAISE.
            out.append(
                {
                    "player": a.player.value,
                    "action": ActionType.RAISE.value,
                    "amount": float(a.amount_to_bb),
                }
            )
        elif at in (
            ActionType.DEAL_TURN,
            ActionType.DEAL_RIVER,
            ActionType.DEAL_FLOP,
        ):
            raise PreflopBridgeError(f"Unexpected deal action in preflop slice: {at}")
        else:
            raise PreflopBridgeError(f"Unsupported action in preflop bridge: {at}")
    return out


def split_preflop_postflop(
    actions: List[Action],
) -> Tuple[List[Action], List[Action]]:
    """Return (preflop_actions_including_blinds, actions_from_deal_flop_onward)."""
    idx = find_deal_flop_index(actions)
    if idx < len(actions) and actions[idx].action_type == ActionType.DEAL_FLOP:
        return actions[:idx], actions[idx:]
    return actions, []
