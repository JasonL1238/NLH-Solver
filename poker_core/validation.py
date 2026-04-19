"""High-level validation for full-hand state.

Delegates structural replay to ``reconstruction.reconstruct_hand_state`` and
adds declarative invariant checks on the resulting ``HandState``.
"""

from __future__ import annotations

from typing import List, Optional

from .models import (
    Action,
    Card,
    EXPECTED_BOARD_CARDS,
    HandConfig,
    HandState,
    Street,
)
from .reconstruction import ReconstructionError, reconstruct_hand_state


class ValidationError(Exception):
    """Raised when a hand state violates an invariant."""


def validate_hand(
    config: HandConfig,
    action_history: List[Action],
    *,
    board_cards_mirror: Optional[List[Card]] = None,
) -> HandState:
    """Reconstruct and validate a hand.

    Returns the validated ``HandState``.  Raises ``ValidationError`` on any
    structural or invariant failure.
    """
    # --- Config sanity ---
    if config.effective_stack_bb <= 0:
        raise ValidationError("Effective stack must be positive")
    if config.hero_starting_bb <= 0 or config.villain_starting_bb <= 0:
        raise ValidationError("Starting stacks must be positive")
    if config.small_blind_bb <= 0 or config.big_blind_bb <= 0:
        raise ValidationError("Blinds must be positive")
    if config.small_blind_bb > config.big_blind_bb:
        raise ValidationError("Small blind must not exceed big blind")
    if config.effective_stack_bb < config.big_blind_bb:
        raise ValidationError(
            f"Effective (min) stack {config.effective_stack_bb} must be at least "
            f"the big blind {config.big_blind_bb}")
    if config.hero_starting_bb < config.big_blind_bb:
        raise ValidationError(
            f"Hero starting stack {config.hero_starting_bb} must be at least big blind "
            f"{config.big_blind_bb}")
    if config.villain_starting_bb < config.big_blind_bb:
        raise ValidationError(
            f"Villain starting stack {config.villain_starting_bb} must be at least big blind "
            f"{config.big_blind_bb}")

    # --- Reconstruct (structural replay) ---
    try:
        state = reconstruct_hand_state(
            config, action_history, board_cards_mirror=board_cards_mirror,
        )
    except ReconstructionError as exc:
        raise ValidationError(str(exc)) from exc

    # --- Post-reconstruction invariants ---
    _validate_contributions(state)
    _validate_board_count(state)
    _validate_pot_consistency(state)

    return state


# ---------------------------------------------------------------------------
# Invariant checks
# ---------------------------------------------------------------------------

def _validate_contributions(state: HandState) -> None:
    cfg = state.config
    h_cap = cfg.hero_starting_bb
    v_cap = cfg.villain_starting_bb
    if state.hero_contribution_bb < -1e-9:
        raise ValidationError("Hero contribution is negative")
    if state.villain_contribution_bb < -1e-9:
        raise ValidationError("Villain contribution is negative")
    if state.hero_contribution_bb > h_cap + 1e-9:
        raise ValidationError(
            f"Hero contribution {state.hero_contribution_bb} exceeds "
            f"hero starting stack {h_cap}")
    if state.villain_contribution_bb > v_cap + 1e-9:
        raise ValidationError(
            f"Villain contribution {state.villain_contribution_bb} exceeds "
            f"villain starting stack {v_cap}")


def _validate_board_count(state: HandState) -> None:
    expected = EXPECTED_BOARD_CARDS.get(state.current_street, -1)
    if expected < 0:
        return  # HAND_OVER can happen on any street
    actual = len(state.board_cards)
    if actual != expected:
        raise ValidationError(
            f"Board has {actual} cards on street {state.current_street.value}, "
            f"expected {expected}")


def _validate_pot_consistency(state: HandState) -> None:
    expected = state.hero_contribution_bb + state.villain_contribution_bb
    if abs(state.pot_size_bb - expected) > 1e-9:
        raise ValidationError(
            f"Pot {state.pot_size_bb} != hero({state.hero_contribution_bb}) "
            f"+ villain({state.villain_contribution_bb})")
