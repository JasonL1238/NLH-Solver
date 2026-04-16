"""Core particle types for villain range tracking (HU NLHE)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List

from poker_core.models import HoleCards


class CoarseBucket(str, Enum):
    """Coarse board-relative strength for compatibility reweighting."""

    NUTTED = "NUTTED"
    STRONG_MADE = "STRONG_MADE"
    MEDIUM_MADE = "MEDIUM_MADE"
    WEAK_SHOWDOWN = "WEAK_SHOWDOWN"
    STRONG_DRAW = "STRONG_DRAW"
    WEAK_DRAW = "WEAK_DRAW"
    AIR = "AIR"


class RangeDegenerateError(ValueError):
    """Raised when all particle weights are zero or no legal particles remain."""


@dataclass
class Particle:
    """One weighted villain combo hypothesis."""

    combo: HoleCards
    weight: float
    preflop_bucket_label: str
    current_bucket: CoarseBucket
    made_category: str
    draw_category: str
    alive: bool = True

    @property
    def combo_str(self) -> str:
        return f"{self.combo.high!s}{self.combo.low!s}"


def alive_particles(particles: List[Particle]) -> List[Particle]:
    return [p for p in particles if p.alive and p.weight > 1e-15]


def total_weight(particles: List[Particle]) -> float:
    return sum(p.weight for p in particles if p.alive)
