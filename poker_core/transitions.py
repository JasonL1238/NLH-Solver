"""Safe action and deal application helpers.

Each function appends to the action history and returns a freshly
reconstructed + validated ``HandState``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .models import (
    Action,
    ActionType,
    Card,
    HandConfig,
    HandState,
    Player,
)
from .validation import validate_hand


def apply_action(
    config: HandConfig,
    action_history: List[Action],
    *,
    action_type: ActionType,
    player: Optional[Player] = None,
    amount_to_bb: Optional[float] = None,
) -> HandState:
    """Append a betting action and return the new validated state."""
    new_action = Action(
        action_type=action_type,
        player=player,
        amount_to_bb=amount_to_bb,
    )
    new_history = list(action_history) + [new_action]
    return validate_hand(config, new_history)


def post_blinds(config: HandConfig) -> Tuple[List[Action], HandState]:
    """Post both blinds and return (history, state)."""
    history: List[Action] = [
        Action(
            action_type=ActionType.POST_BLIND,
            player=config.btn_player,
            amount_to_bb=config.small_blind_bb,
        ),
        Action(
            action_type=ActionType.POST_BLIND,
            player=config.bb_player,
            amount_to_bb=config.big_blind_bb,
        ),
    ]
    state = validate_hand(config, history)
    return history, state


def deal_flop(
    config: HandConfig,
    action_history: List[Action],
    cards: Tuple[Card, Card, Card],
) -> HandState:
    """Append a DEAL_FLOP action and return the new validated state."""
    new_action = Action(
        action_type=ActionType.DEAL_FLOP,
        cards=cards,
    )
    new_history = list(action_history) + [new_action]
    return validate_hand(config, new_history)


def deal_turn(
    config: HandConfig,
    action_history: List[Action],
    card: Card,
) -> HandState:
    """Append a DEAL_TURN action and return the new validated state."""
    new_action = Action(
        action_type=ActionType.DEAL_TURN,
        cards=(card,),
    )
    new_history = list(action_history) + [new_action]
    return validate_hand(config, new_history)


def deal_river(
    config: HandConfig,
    action_history: List[Action],
    card: Card,
) -> HandState:
    """Append a DEAL_RIVER action and return the new validated state."""
    new_action = Action(
        action_type=ActionType.DEAL_RIVER,
        cards=(card,),
    )
    new_history = list(action_history) + [new_action]
    return validate_hand(config, new_history)
