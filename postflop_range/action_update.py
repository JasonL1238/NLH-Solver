"""Coarse bucket-based compatibility multipliers after villain actions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from poker_core.models import ActionType, Street

from .particles import CoarseBucket, Particle

# ---------------------------------------------------------------------------
# Explicit multiplier tables (inspectable; tune for exploit layer later)
# ---------------------------------------------------------------------------

# CHECK: keep medium/air plausible; do not polarize
_MULT_CHECK: Dict[CoarseBucket, float] = {
    CoarseBucket.NUTTED: 0.85,
    CoarseBucket.STRONG_MADE: 0.95,
    CoarseBucket.MEDIUM_MADE: 1.15,
    CoarseBucket.WEAK_SHOWDOWN: 1.1,
    CoarseBucket.STRONG_DRAW: 1.05,
    CoarseBucket.WEAK_DRAW: 1.1,
    CoarseBucket.AIR: 1.2,
}

# CALL: facing aggression — retain draws and some medium
_MULT_CALL: Dict[CoarseBucket, float] = {
    CoarseBucket.NUTTED: 1.25,
    CoarseBucket.STRONG_MADE: 1.2,
    CoarseBucket.MEDIUM_MADE: 1.1,
    CoarseBucket.WEAK_SHOWDOWN: 0.95,
    CoarseBucket.STRONG_DRAW: 1.15,
    CoarseBucket.WEAK_DRAW: 1.05,
    CoarseBucket.AIR: 0.75,
}

# Small aggression
_MULT_AGG_SMALL: Dict[CoarseBucket, float] = {
    CoarseBucket.NUTTED: 1.15,
    CoarseBucket.STRONG_MADE: 1.2,
    CoarseBucket.MEDIUM_MADE: 1.1,
    CoarseBucket.WEAK_SHOWDOWN: 0.85,
    CoarseBucket.STRONG_DRAW: 1.1,
    CoarseBucket.WEAK_DRAW: 0.95,
    CoarseBucket.AIR: 0.65,
}

_MULT_AGG_MED: Dict[CoarseBucket, float] = {
    CoarseBucket.NUTTED: 1.35,
    CoarseBucket.STRONG_MADE: 1.35,
    CoarseBucket.MEDIUM_MADE: 1.05,
    CoarseBucket.WEAK_SHOWDOWN: 0.65,
    CoarseBucket.STRONG_DRAW: 1.15,
    CoarseBucket.WEAK_DRAW: 0.8,
    CoarseBucket.AIR: 0.45,
}

_MULT_AGG_LARGE: Dict[CoarseBucket, float] = {
    CoarseBucket.NUTTED: 1.55,
    CoarseBucket.STRONG_MADE: 1.45,
    CoarseBucket.MEDIUM_MADE: 0.85,
    CoarseBucket.WEAK_SHOWDOWN: 0.45,
    CoarseBucket.STRONG_DRAW: 1.2,
    CoarseBucket.WEAK_DRAW: 0.55,
    CoarseBucket.AIR: 0.25,
}

# Raise / re-raise — more value-heavy
_MULT_AGG_RAISE: Dict[CoarseBucket, float] = {
    CoarseBucket.NUTTED: 1.6,
    CoarseBucket.STRONG_MADE: 1.5,
    CoarseBucket.MEDIUM_MADE: 0.75,
    CoarseBucket.WEAK_SHOWDOWN: 0.4,
    CoarseBucket.STRONG_DRAW: 1.0,
    CoarseBucket.WEAK_DRAW: 0.5,
    CoarseBucket.AIR: 0.2,
}


class AggressionSize(str, Enum):
    NONE = "NONE"
    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    LARGE = "LARGE"


@dataclass(frozen=True)
class VillainActionContext:
    """Context for one villain betting-line update."""

    street: Street
    action_type: ActionType
    aggression_size: AggressionSize
    raises_this_street_before: int
    profile: Optional[Any] = None


def size_bucket_from_fraction(frac: float) -> AggressionSize:
    """Pot-odds style fraction: to_call / (pot_before + to_call) approximates bet/pot."""
    if frac <= 0.22:
        return AggressionSize.SMALL
    if frac <= 0.55:
        return AggressionSize.MEDIUM
    return AggressionSize.LARGE


def infer_aggression_size(
    *,
    action_type: ActionType,
    pot_after_villain_bb: float,
    hero_to_call_bb: float,
    raises_this_street: int,
) -> AggressionSize:
    """Infer SMALL/MED/LARGE from bet size vs pot *before* villain's wager."""
    if action_type not in (ActionType.BET, ActionType.RAISE):
        return AggressionSize.NONE
    if raises_this_street >= 2:
        return AggressionSize.LARGE
    pot_before = max(pot_after_villain_bb - hero_to_call_bb, 1e-9)
    frac = hero_to_call_bb / pot_before
    return size_bucket_from_fraction(frac)


def compatibility_multiplier(particle: Particle, ctx: VillainActionContext) -> float:
    """Coarse P(update) proxy: bucket × action table (no exact combo likelihood)."""
    if not particle.alive:
        return 1.0
    b = particle.current_bucket
    at = ctx.action_type

    if at == ActionType.CHECK:
        return _MULT_CHECK.get(b, 1.0)
    if at == ActionType.CALL:
        return _MULT_CALL.get(b, 1.0)

    if at in (ActionType.BET, ActionType.RAISE):
        if ctx.aggression_size == AggressionSize.SMALL:
            tbl = _MULT_AGG_SMALL
        elif ctx.aggression_size == AggressionSize.MEDIUM:
            tbl = _MULT_AGG_MED
        elif ctx.aggression_size == AggressionSize.LARGE:
            tbl = _MULT_AGG_LARGE if at == ActionType.BET else _MULT_AGG_RAISE
        else:
            tbl = _MULT_AGG_SMALL
        if at == ActionType.RAISE and ctx.aggression_size == AggressionSize.LARGE:
            tbl = _MULT_AGG_RAISE
        return tbl.get(b, 1.0)

    return 1.0


def apply_compatibility_to_particles(
    particles: list[Particle],
    ctx: VillainActionContext,
) -> str:
    """Multiply each alive particle weight; returns a short note."""
    for p in particles:
        if not p.alive:
            continue
        m = compatibility_multiplier(p, ctx)
        p.weight *= m
    return (
        f"Reweighted after villain {ctx.action_type.value} "
        f"({ctx.aggression_size.value}) on {ctx.street.value}"
    )
