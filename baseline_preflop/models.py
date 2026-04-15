"""Core enums, dataclasses, and canonical state objects for HU NLHE preflop."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Player(Enum):
    HERO = "HERO"
    VILLAIN = "VILLAIN"


class Position(Enum):
    BTN_SB = "BTN_SB"
    BB = "BB"


class Street(Enum):
    PREFLOP = "PREFLOP"
    FLOP = "FLOP"
    TURN = "TURN"
    RIVER = "RIVER"


class ActionType(Enum):
    POST_BLIND = "POST_BLIND"
    FOLD = "FOLD"
    CHECK = "CHECK"
    CALL = "CALL"
    BET = "BET"
    RAISE = "RAISE"


class StackDepthBucket(Enum):
    ULTRA_SHORT = "ULTRA_SHORT"   # 0-5bb
    VERY_SHORT = "VERY_SHORT"     # 6-10bb
    SHORT = "SHORT"               # 11-20bb
    MEDIUM = "MEDIUM"             # 21-40bb
    DEEP = "DEEP"                 # 41-90bb
    VERY_DEEP = "VERY_DEEP"       # 91bb+


class HandBucket(Enum):
    """Detailed hand classification bucket (A-N)."""
    A_PREMIUM_PAIRS = "A_PREMIUM_PAIRS"
    A_HIGH_PAIRS = "A_HIGH_PAIRS"
    A_MID_PAIRS = "A_MID_PAIRS"
    A_LOW_PAIRS = "A_LOW_PAIRS"
    A_MICRO_PAIRS = "A_MICRO_PAIRS"
    B_SUITED_BIG_BROADWAY = "B_SUITED_BIG_BROADWAY"
    C_OFFSUIT_BIG_BROADWAY = "C_OFFSUIT_BIG_BROADWAY"
    D_SUITED_AX_HIGH = "D_SUITED_AX_HIGH"
    D_SUITED_AX_LOW = "D_SUITED_AX_LOW"
    E_OFFSUIT_AX_HIGH = "E_OFFSUIT_AX_HIGH"
    E_OFFSUIT_AX_LOW = "E_OFFSUIT_AX_LOW"
    F_SUITED_KX_HIGH = "F_SUITED_KX_HIGH"
    F_SUITED_KX_LOW = "F_SUITED_KX_LOW"
    G_OFFSUIT_KX_HIGH = "G_OFFSUIT_KX_HIGH"
    G_OFFSUIT_KX_LOW = "G_OFFSUIT_KX_LOW"
    H_SUITED_QJ_HIGH = "H_SUITED_QJ_HIGH"
    H_SUITED_QJ_LOW = "H_SUITED_QJ_LOW"
    I_OFFSUIT_QJ_HIGH = "I_OFFSUIT_QJ_HIGH"
    I_OFFSUIT_QJ_LOW = "I_OFFSUIT_QJ_LOW"
    J_SUITED_CONNECTORS = "J_SUITED_CONNECTORS"
    K_SUITED_ONE_GAPPERS = "K_SUITED_ONE_GAPPERS"
    L_SUITED_TWO_GAPPERS_TRASH = "L_SUITED_TWO_GAPPERS_TRASH"
    M_OFFSUIT_CONNECTORS = "M_OFFSUIT_CONNECTORS"
    N_WEAK_OFFSUIT_TRASH = "N_WEAK_OFFSUIT_TRASH"


# ---------------------------------------------------------------------------
# Card primitives
# ---------------------------------------------------------------------------

RANKS = "23456789TJQKA"
SUITS = "cdhs"

RANK_VALUES = {r: i for i, r in enumerate(RANKS, 2)}


@dataclass(frozen=True)
class Card:
    rank: str  # single char from RANKS
    suit: str  # single char from SUITS

    def __post_init__(self):
        if self.rank not in RANKS:
            raise ValueError(f"Invalid rank: {self.rank!r}")
        if self.suit not in SUITS:
            raise ValueError(f"Invalid suit: {self.suit!r}")

    @property
    def rank_value(self) -> int:
        return RANK_VALUES[self.rank]

    def __repr__(self) -> str:
        return f"{self.rank}{self.suit}"


@dataclass(frozen=True)
class HoleCards:
    high: Card
    low: Card

    def __post_init__(self):
        if self.high == self.low:
            raise ValueError("Hole cards must be two distinct cards")
        if self.high.rank_value < self.low.rank_value:
            h, l = self.low, self.high
            object.__setattr__(self, "high", h)
            object.__setattr__(self, "low", l)

    def __repr__(self) -> str:
        return f"{self.high}{self.low}"


# ---------------------------------------------------------------------------
# Action record
# ---------------------------------------------------------------------------

@dataclass
class ActionRecord:
    player: Player
    street: Street
    action_type: ActionType
    amount_added_bb: float
    total_contribution_after_action_bb: float
    sequence_index: int


# ---------------------------------------------------------------------------
# Legal action option
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LegalActionOption:
    action_type: ActionType
    raise_to_bb: Optional[float] = None
    call_amount_bb: Optional[float] = None

    def __repr__(self) -> str:
        if self.action_type == ActionType.RAISE and self.raise_to_bb is not None:
            return f"RAISE(to {self.raise_to_bb}bb)"
        if self.action_type == ActionType.CALL and self.call_amount_bb is not None:
            return f"CALL({self.call_amount_bb}bb)"
        return self.action_type.value


# ---------------------------------------------------------------------------
# Hand features
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HandFeatures:
    rank_high: int
    rank_low: int
    is_pair: bool
    is_suited: bool
    gap_size: int
    is_connector: bool
    is_one_gapper: bool
    is_wheel_ace: bool
    broadway_count: int
    high_card_points_simple: float
    hand_class_label: str
    hand_bucket: HandBucket


# ---------------------------------------------------------------------------
# Derived preflop state
# ---------------------------------------------------------------------------

@dataclass
class DerivedState:
    hero_is_first_to_act_preflop: bool
    hero_is_in_position_postflop_future_flag: bool
    unopened_pot: bool
    facing_limp: bool
    facing_open_raise: bool
    facing_3bet: bool
    facing_4bet: bool
    facing_all_in: bool
    hero_already_acted_this_round: bool
    villain_already_acted_this_round: bool
    raise_size_in_bb: Optional[float]
    raise_size_as_multiple_of_bb: Optional[float]
    raise_size_as_multiple_of_previous_bet: Optional[float]
    stack_to_open_ratio: Optional[float]
    stack_to_3bet_ratio: Optional[float]
    stack_depth_bucket: StackDepthBucket


# ---------------------------------------------------------------------------
# Poker state (canonical)
# ---------------------------------------------------------------------------

@dataclass
class PokerState:
    # Direct fields
    hero_hole_cards: HoleCards
    hero_position: Position
    villain_position: Position
    current_street: Street
    action_history: List[ActionRecord]
    effective_stack_bb: float
    small_blind_bb: float
    big_blind_bb: float
    pot_size_bb: float
    current_bet_to_call_bb: float
    hero_contribution_bb: float
    villain_contribution_bb: float
    current_actor: Optional[Player]
    betting_round_closed: bool
    last_aggressor: Optional[Player]
    number_of_raises_this_street: int
    hand_over: bool = False

    # Derived (populated by validation/derivation layer)
    derived: Optional[DerivedState] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Decision output
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    legal_actions: List[LegalActionOption]
    recommended_action: LegalActionOption
    explanation: str
    debug: dict = field(default_factory=dict)
