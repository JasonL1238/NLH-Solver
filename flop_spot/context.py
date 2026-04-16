"""Flop decision-node context derivation.

Consumes canonical HandState from Phase C and derives position,
preflop role, SPR, bet-size bucket, and flop context label.
"""

from __future__ import annotations

from typing import Optional

from poker_core.models import (
    ActionType,
    HandState,
    Player,
    Position,
    Street,
)

from .models import (
    BetSizeBucket,
    FlopContext,
    FlopDerivedContext,
    SPRBucket,
)


# ---------------------------------------------------------------------------
# SPR thresholds
# ---------------------------------------------------------------------------

_SPR_LOW = 3.0
_SPR_HIGH = 8.0

# ---------------------------------------------------------------------------
# Bet-size bucket thresholds (as fraction of pot)
# ---------------------------------------------------------------------------

_BET_SMALL = 0.33
_BET_LARGE = 0.75


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def derive_flop_context(state: HandState) -> FlopDerivedContext:
    """Derive all flop-specific context from canonical Phase C state."""
    hero_is_ip = state.config.hero_position == Position.BTN_SB
    hero_is_pfr = _is_hero_preflop_aggressor(state)

    spr = _compute_spr(state)
    spr_bucket = _bucket_spr(spr)

    to_call = state.current_bet_to_call_bb
    bet_size_bucket = _bucket_bet_size(to_call, state.pot_size_bb)

    flop_ctx = _determine_flop_context(
        hero_is_ip, hero_is_pfr, to_call, bet_size_bucket,
        state.number_of_raises_this_street,
    )

    return FlopDerivedContext(
        flop_context=flop_ctx,
        hero_is_ip=hero_is_ip,
        hero_is_pfr=hero_is_pfr,
        bet_size_bucket=bet_size_bucket,
        spr_bucket=spr_bucket,
        spr=round(spr, 2),
    )


# ---------------------------------------------------------------------------
# Preflop aggressor detection
# ---------------------------------------------------------------------------

def _is_hero_preflop_aggressor(state: HandState) -> bool:
    """Scan action_history for the last RAISE during PREFLOP.

    If that player is HERO, hero is the preflop aggressor (PFR).
    If no preflop raise exists (limped pot), return False.
    """
    last_raiser: Optional[Player] = None
    for action in state.action_history:
        if action.action_type in (ActionType.DEAL_FLOP, ActionType.DEAL_TURN,
                                  ActionType.DEAL_RIVER):
            break
        if action.action_type == ActionType.RAISE and action.player is not None:
            last_raiser = action.player
    return last_raiser == Player.HERO


# ---------------------------------------------------------------------------
# SPR
# ---------------------------------------------------------------------------

def _compute_spr(state: HandState) -> float:
    """Stack-to-pot ratio: remaining effective stack / pot at start of flop."""
    pot = state.pot_size_bb
    if pot < 1e-9:
        return 999.0
    max_contrib = max(state.hero_contribution_bb, state.villain_contribution_bb)
    remaining = state.config.effective_stack_bb - max_contrib
    if remaining < 1e-9:
        return 0.0
    return remaining / pot


# ---------------------------------------------------------------------------
# Bet-size bucketing
# ---------------------------------------------------------------------------

def _bucket_bet_size(
    to_call: float,
    pot_total: float,
) -> Optional[BetSizeBucket]:
    """Bucket the bet/raise size hero is facing as a fraction of pot.

    Returns None when there is nothing to call.
    """
    if to_call < 1e-9:
        return None
    pot_before_bet = pot_total - to_call
    if pot_before_bet < 1e-9:
        return BetSizeBucket.LARGE
    frac = to_call / pot_before_bet
    if frac <= _BET_SMALL:
        return BetSizeBucket.SMALL
    if frac <= _BET_LARGE:
        return BetSizeBucket.MEDIUM
    return BetSizeBucket.LARGE


def _bucket_spr(spr: float) -> SPRBucket:
    if spr <= _SPR_LOW:
        return SPRBucket.LOW_SPR
    if spr <= _SPR_HIGH:
        return SPRBucket.MEDIUM_SPR
    return SPRBucket.HIGH_SPR


# ---------------------------------------------------------------------------
# Flop context label
# ---------------------------------------------------------------------------

def _determine_flop_context(
    hero_is_ip: bool,
    hero_is_pfr: bool,
    to_call: float,
    bet_bucket: Optional[BetSizeBucket],
    num_raises_this_street: int,
) -> FlopContext:
    """Map position + role + action-state to a FlopContext label."""
    facing_bet = to_call > 1e-9

    if facing_bet:
        if num_raises_this_street >= 2:
            return FlopContext.FACING_RAISE_AFTER_BETTING
        if bet_bucket == BetSizeBucket.SMALL:
            return FlopContext.FACING_SMALL_BET
        if bet_bucket == BetSizeBucket.LARGE:
            return FlopContext.FACING_LARGE_BET
        return FlopContext.FACING_MEDIUM_BET

    # Checked to hero or hero first to act
    if hero_is_pfr:
        if hero_is_ip:
            return FlopContext.PFR_IP_CHECKED_TO
        return FlopContext.PFR_OOP_FIRST_TO_ACT
    if hero_is_ip:
        return FlopContext.PFC_IP_CHECKED_TO
    return FlopContext.PFC_OOP_FIRST_TO_ACT
