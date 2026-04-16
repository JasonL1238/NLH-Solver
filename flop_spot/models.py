"""Flop-specific enums, dataclasses, and structured output types.

All poker state comes from poker_core (Phase C).  This module only
defines flop-strategy-specific classification labels and output shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from poker_core.models import LegalAction


# ---------------------------------------------------------------------------
# Made-hand categories
# ---------------------------------------------------------------------------

class MadeHandCategory(Enum):
    NUTS_OR_NEAR_NUTS = "NUTS_OR_NEAR_NUTS"
    SET = "SET"
    TWO_PAIR = "TWO_PAIR"
    OVERPAIR = "OVERPAIR"
    TOP_PAIR_STRONG_KICKER = "TOP_PAIR_STRONG_KICKER"
    TOP_PAIR_MEDIUM_KICKER = "TOP_PAIR_MEDIUM_KICKER"
    TOP_PAIR_WEAK_KICKER = "TOP_PAIR_WEAK_KICKER"
    MIDDLE_PAIR = "MIDDLE_PAIR"
    THIRD_PAIR_OR_WORSE_PAIR = "THIRD_PAIR_OR_WORSE_PAIR"
    UNDERPAIR_TO_BOARD = "UNDERPAIR_TO_BOARD"
    ACE_HIGH = "ACE_HIGH"
    KING_HIGH_OR_WORSE_HIGH_CARD = "KING_HIGH_OR_WORSE_HIGH_CARD"


# ---------------------------------------------------------------------------
# Draw categories
# ---------------------------------------------------------------------------

class DrawCategory(Enum):
    MADE_STRAIGHT = "MADE_STRAIGHT"
    MADE_FLUSH = "MADE_FLUSH"
    OPEN_ENDED_STRAIGHT_DRAW = "OPEN_ENDED_STRAIGHT_DRAW"
    GUTSHOT = "GUTSHOT"
    FLUSH_DRAW = "FLUSH_DRAW"
    COMBO_DRAW = "COMBO_DRAW"
    BACKDOOR_FLUSH_DRAW = "BACKDOOR_FLUSH_DRAW"
    BACKDOOR_STRAIGHT_DRAW = "BACKDOOR_STRAIGHT_DRAW"
    NO_REAL_DRAW = "NO_REAL_DRAW"


# ---------------------------------------------------------------------------
# Board texture
# ---------------------------------------------------------------------------

class BoardTexture(Enum):
    DRY_HIGH_CARD = "DRY_HIGH_CARD"
    DRY_LOW_BOARD = "DRY_LOW_BOARD"
    PAIRED_BOARD = "PAIRED_BOARD"
    MONOTONE_BOARD = "MONOTONE_BOARD"
    TWO_TONE_BOARD = "TWO_TONE_BOARD"
    RAINBOW_CONNECTED = "RAINBOW_CONNECTED"
    LOW_CONNECTED = "LOW_CONNECTED"
    HIGH_CONNECTED = "HIGH_CONNECTED"
    DYNAMIC_DRAW_HEAVY = "DYNAMIC_DRAW_HEAVY"
    STATIC_BOARD = "STATIC_BOARD"


# ---------------------------------------------------------------------------
# Flop decision-node context
# ---------------------------------------------------------------------------

class FlopContext(Enum):
    PFR_IP_CHECKED_TO = "PFR_IP_CHECKED_TO"
    PFR_OOP_FIRST_TO_ACT = "PFR_OOP_FIRST_TO_ACT"
    PFC_IP_CHECKED_TO = "PFC_IP_CHECKED_TO"
    PFC_OOP_FIRST_TO_ACT = "PFC_OOP_FIRST_TO_ACT"
    FACING_SMALL_BET = "FACING_SMALL_BET"
    FACING_MEDIUM_BET = "FACING_MEDIUM_BET"
    FACING_LARGE_BET = "FACING_LARGE_BET"
    FACING_RAISE_AFTER_BETTING = "FACING_RAISE_AFTER_BETTING"


# ---------------------------------------------------------------------------
# Bet-size bucket  (fraction of pot)
# ---------------------------------------------------------------------------

class BetSizeBucket(Enum):
    SMALL = "SMALL"    # <= 33 %
    MEDIUM = "MEDIUM"  # > 33 % to <= 75 %
    LARGE = "LARGE"    # > 75 %


# ---------------------------------------------------------------------------
# Stack-to-pot ratio bucket
# ---------------------------------------------------------------------------

class SPRBucket(Enum):
    LOW_SPR = "LOW_SPR"      # <= 3
    MEDIUM_SPR = "MEDIUM_SPR"  # > 3 to <= 8
    HIGH_SPR = "HIGH_SPR"    # > 8


# ---------------------------------------------------------------------------
# Hand classification output
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HandClassification:
    made_hand: MadeHandCategory
    draw: DrawCategory
    has_pair: bool
    has_two_pair_plus: bool
    has_overpair: bool
    has_top_pair: bool
    has_showdown_value: bool
    has_strong_draw: bool
    has_combo_draw: bool
    has_backdoor_equity: bool
    overcards_to_board_count: int
    pair_rank_relative_to_board: Optional[str] = None


# ---------------------------------------------------------------------------
# Board feature output
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BoardFeatures:
    texture: BoardTexture
    is_paired: bool
    is_monotone: bool
    is_two_tone: bool
    is_rainbow: bool
    has_straight_heaviness: bool
    top_card_rank: int
    board_is_high_card_heavy: bool
    board_is_low_connected: bool
    board_is_dynamic: bool


# ---------------------------------------------------------------------------
# Derived flop context
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FlopDerivedContext:
    flop_context: FlopContext
    hero_is_ip: bool
    hero_is_pfr: bool
    bet_size_bucket: Optional[BetSizeBucket]
    spr_bucket: SPRBucket
    spr: float


# ---------------------------------------------------------------------------
# Recommended action wrapper (carries the chosen size)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FlopActionChoice:
    """Concrete action chosen by the recommender.

    For BET/RAISE ``size_bb`` is the chosen total street-contribution.
    For CHECK/FOLD/CALL ``size_bb`` is None.
    """
    legal_action: LegalAction
    size_bb: Optional[float] = None

    def __repr__(self) -> str:
        if self.size_bb is not None:
            return f"{self.legal_action.action_type.value}(to {self.size_bb}bb)"
        if self.legal_action.call_amount_bb is not None:
            return f"CALL({self.legal_action.call_amount_bb}bb)"
        return self.legal_action.action_type.value


# ---------------------------------------------------------------------------
# Decision output
# ---------------------------------------------------------------------------

@dataclass
class FlopDecision:
    legal_actions: List[LegalAction]
    recommended_action: FlopActionChoice
    explanation: str
    debug: Dict = field(default_factory=dict)
