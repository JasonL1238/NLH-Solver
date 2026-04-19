"""Convenience helpers for manually constructing hands and inspecting state.

Intended for interactive exploration, tests, and debugging.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

from .models import (
    Action,
    ActionType,
    Card,
    HandConfig,
    HandState,
    HoleCards,
    Player,
    Position,
)
from .legal_actions import legal_actions
from .reconstruction import reconstruct_hand_state
from .validation import validate_hand


# ---------------------------------------------------------------------------
# Card / board parsing
# ---------------------------------------------------------------------------

def parse_card(s: str) -> Card:
    """Parse a two-character card string like ``'As'``."""
    s = s.strip()
    if len(s) != 2:
        raise ValueError(f"Card string must be 2 chars, got {s!r}")
    return Card(rank=s[0], suit=s[1])


def parse_cards(s: str) -> HoleCards:
    """Parse two space-separated cards into ``HoleCards``."""
    parts = s.strip().split()
    if len(parts) != 2:
        raise ValueError(f"Expected 2 cards, got {len(parts)}: {s!r}")
    return HoleCards(high=parse_card(parts[0]), low=parse_card(parts[1]))


def parse_board(s: str) -> Tuple[Card, ...]:
    """Parse space-separated board cards."""
    if not s or not s.strip():
        return ()
    parts = s.strip().split()
    return tuple(parse_card(p) for p in parts)


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def make_hand_config(
    hero_cards: str,
    hero_position: str = "BTN_SB",
    effective_stack_bb: float = 100.0,
    small_blind_bb: float = 0.5,
    big_blind_bb: float = 1.0,
    villain_cards: Optional[str] = None,
    *,
    hero_starting_bb: Optional[float] = None,
    villain_starting_bb: Optional[float] = None,
) -> HandConfig:
    """Build a ``HandConfig`` from human-readable strings.

    If ``hero_starting_bb`` / ``villain_starting_bb`` are omitted, both start at
    ``effective_stack_bb`` (symmetric). Otherwise each side uses its own start;
    ``effective_stack_bb`` on the config is set to ``min(hero, villain)`` for depth.
    """
    hc = parse_cards(hero_cards)
    pos = Position(hero_position)
    vc = parse_cards(villain_cards) if villain_cards else None
    hs = float(effective_stack_bb) if hero_starting_bb is None else float(hero_starting_bb)
    vs = float(effective_stack_bb) if villain_starting_bb is None else float(villain_starting_bb)
    eff_depth = min(hs, vs)
    return HandConfig(
        hero_position=pos,
        hero_hole_cards=hc,
        effective_stack_bb=eff_depth,
        hero_starting_bb=hs,
        villain_starting_bb=vs,
        small_blind_bb=small_blind_bb,
        big_blind_bb=big_blind_bb,
        villain_hole_cards=vc,
    )


# ---------------------------------------------------------------------------
# Action history builder
# ---------------------------------------------------------------------------

def make_action_history(
    raw: List[Dict[str, Union[str, float, None]]],
) -> List[Action]:
    """Convert compact dicts into ``Action`` objects.

    Each dict may contain:
    - ``"player"``: ``"HERO"`` / ``"VILLAIN"`` / ``None`` (for deals)
    - ``"action"``: action type string (e.g. ``"RAISE"``, ``"DEAL_FLOP"``)
    - ``"amount"``: total contribution after action (for BET/RAISE/CALL/POST_BLIND)
    - ``"cards"``: board card string (e.g. ``"Ah 7c 2d"``) for deal actions
    """
    actions: List[Action] = []
    for entry in raw:
        at = ActionType(entry["action"])
        player_str = entry.get("player")
        player = Player(player_str) if player_str else None
        amount = entry.get("amount")
        amount_f = float(amount) if amount is not None else None

        cards_str = entry.get("cards")
        cards: Optional[Tuple[Card, ...]] = None
        if cards_str and isinstance(cards_str, str):
            cards = parse_board(cards_str)

        actions.append(Action(
            action_type=at,
            player=player,
            amount_to_bb=amount_f,
            cards=cards,
        ))
    return actions


# ---------------------------------------------------------------------------
# Shorthand helpers
# ---------------------------------------------------------------------------

def make_blinds(config: HandConfig) -> List[Action]:
    """Return the two POST_BLIND actions for the given config."""
    return [
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


def reconstruct_and_print(
    config: HandConfig,
    action_history: List[Action],
) -> HandState:
    """Reconstruct, attach legal actions, print debug summary, return state."""
    from .debug import format_state

    state = validate_hand(config, action_history)
    state.legal_actions_list = legal_actions(state)
    print(format_state(state))
    return state


def apply_action_and_print(
    config: HandConfig,
    action_history: List[Action],
    *,
    action_type: ActionType,
    player: Optional[Player] = None,
    amount_to_bb: Optional[float] = None,
    cards: Optional[Tuple[Card, ...]] = None,
) -> Tuple[List[Action], HandState]:
    """Append an action, reconstruct, print, return (new_history, state)."""
    from .debug import format_state

    new_action = Action(
        action_type=action_type,
        player=player,
        amount_to_bb=amount_to_bb,
        cards=cards,
    )
    new_history = list(action_history) + [new_action]
    state = validate_hand(config, new_history)
    state.legal_actions_list = legal_actions(state)
    print(format_state(state))
    return new_history, state
