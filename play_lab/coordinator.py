"""Session state and hand lifecycle helpers for the Play lab (no opponent profiles)."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List

from poker_core.models import Action, ActionType, HandState, Street


@dataclass
class ScenarioState:
    """In-memory lab session: RNG, label, and default stack only."""

    label: str
    rng_seed: int
    rng: random.Random
    effective_stack_bb: float
    hands_completed: int = 0


def hole_cards_spaced(hc) -> str:
    """``As Kd`` for baseline ``parse_cards``."""
    return f"{hc.high!s} {hc.low!s}"


def new_scenario(
    *,
    label: str = "default",
    rng_seed: int = 42,
    effective_stack_bb: float = 100.0,
) -> ScenarioState:
    return ScenarioState(
        label=label,
        rng_seed=rng_seed,
        rng=random.Random(rng_seed),
        effective_stack_bb=effective_stack_bb,
        hands_completed=0,
    )


def needs_flop_deal(state: HandState) -> bool:
    """True when preflop betting is closed, hand alive, flop not yet dealt."""
    if state.hand_over:
        return False
    if state.current_street != Street.PREFLOP:
        return False
    if not state.betting_round_closed:
        return False
    return True


def needs_turn_deal(state: HandState) -> bool:
    """True when flop betting is closed, board has 3 cards, turn not yet dealt."""
    if state.hand_over:
        return False
    if state.awaiting_runout:
        return False
    if state.current_street != Street.FLOP:
        return False
    if not state.betting_round_closed:
        return False
    return len(state.board_cards) == 3


def needs_river_deal(state: HandState) -> bool:
    """True when turn betting is closed, board has 4 cards, river not yet dealt."""
    if state.hand_over:
        return False
    if state.awaiting_runout:
        return False
    if state.current_street != Street.TURN:
        return False
    if not state.betting_round_closed:
        return False
    return len(state.board_cards) == 4


def is_lab_hand_terminal(state: HandState) -> bool:
    """Fold, showdown, or any terminal ``hand_over`` (canonical engine)."""
    return state.hand_over


def has_deal_flop(actions: List[Action]) -> bool:
    return any(a.action_type == ActionType.DEAL_FLOP for a in actions)


def note_hand_completed(scenario: ScenarioState) -> None:
    scenario.hands_completed += 1


class HandCoordinator:
    """Namespace for small hand helpers used by the Streamlit UI."""

    new_scenario = staticmethod(new_scenario)
    needs_flop_deal = staticmethod(needs_flop_deal)
    needs_turn_deal = staticmethod(needs_turn_deal)
    needs_river_deal = staticmethod(needs_river_deal)
    is_lab_hand_terminal = staticmethod(is_lab_hand_terminal)
    has_deal_flop = staticmethod(has_deal_flop)
    hole_cards_spaced = staticmethod(hole_cards_spaced)
    note_hand_completed = staticmethod(note_hand_completed)
