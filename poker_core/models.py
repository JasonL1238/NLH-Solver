"""Core enums, dataclasses, and canonical state objects for the full-hand
HU NLHE engine.

Action amount semantics mirror Phase A/B: ``amount_to_bb`` for BET/RAISE/CALL
means **total contribution after the action** on the current street.  The
reconstruction layer computes incremental chip movements internally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


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
    PRE_HAND = "PRE_HAND"
    PREFLOP = "PREFLOP"
    FLOP = "FLOP"
    TURN = "TURN"
    RIVER = "RIVER"
    SHOWDOWN = "SHOWDOWN"
    HAND_OVER = "HAND_OVER"


class ActionType(Enum):
    POST_BLIND = "POST_BLIND"
    FOLD = "FOLD"
    CHECK = "CHECK"
    CALL = "CALL"
    BET = "BET"
    RAISE = "RAISE"
    DEAL_FLOP = "DEAL_FLOP"
    DEAL_TURN = "DEAL_TURN"
    DEAL_RIVER = "DEAL_RIVER"


DEAL_ACTIONS = frozenset({ActionType.DEAL_FLOP, ActionType.DEAL_TURN, ActionType.DEAL_RIVER})

BETTING_ACTIONS = frozenset({
    ActionType.FOLD, ActionType.CHECK, ActionType.CALL,
    ActionType.BET, ActionType.RAISE,
})

EXPECTED_BOARD_CARDS = {
    Street.PRE_HAND: 0,
    Street.PREFLOP: 0,
    Street.FLOP: 3,
    Street.TURN: 4,
    Street.RIVER: 5,
    Street.SHOWDOWN: 5,
    Street.HAND_OVER: -1,  # variable – fold can happen on any street
}

STREET_ORDER: List[Street] = [
    Street.PRE_HAND,
    Street.PREFLOP,
    Street.FLOP,
    Street.TURN,
    Street.RIVER,
    Street.SHOWDOWN,
    Street.HAND_OVER,
]

BETTING_STREETS = frozenset({Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER})


# ---------------------------------------------------------------------------
# Card primitives
# ---------------------------------------------------------------------------

RANKS = "23456789TJQKA"
SUITS = "cdhs"

RANK_VALUES = {r: i for i, r in enumerate(RANKS, 2)}


@dataclass(frozen=True)
class Card:
    rank: str
    suit: str

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
# Hand configuration (immutable per-hand setup)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HandConfig:
    hero_position: Position
    hero_hole_cards: HoleCards
    effective_stack_bb: float
    """Minimum starting stack (bb); used for HU depth / preflop charts."""
    hero_starting_bb: float
    villain_starting_bb: float
    small_blind_bb: float = 0.5
    big_blind_bb: float = 1.0
    villain_hole_cards: Optional[HoleCards] = None

    def stack_cap_bb(self, player: Player) -> float:
        """Maximum total contribution this player can make this hand (starting stack)."""
        return self.hero_starting_bb if player == Player.HERO else self.villain_starting_bb

    @property
    def villain_position(self) -> Position:
        return Position.BB if self.hero_position == Position.BTN_SB else Position.BTN_SB

    def player_for_position(self, pos: Position) -> Player:
        if pos == self.hero_position:
            return Player.HERO
        return Player.VILLAIN

    def position_for_player(self, player: Player) -> Position:
        if player == Player.HERO:
            return self.hero_position
        return self.villain_position

    @property
    def btn_player(self) -> Player:
        return self.player_for_position(Position.BTN_SB)

    @property
    def bb_player(self) -> Player:
        return self.player_for_position(Position.BB)


# ---------------------------------------------------------------------------
# Action record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Action:
    """A single action in the hand history.

    - ``player`` is ``None`` for DEAL_* actions.
    - ``amount_to_bb`` is the **total contribution after the action** for
      BET/RAISE/CALL/POST_BLIND.  ``None`` for FOLD/CHECK/DEAL_*.
    - ``cards`` carries dealt board cards for DEAL_* actions.
    """
    action_type: ActionType
    player: Optional[Player] = None
    amount_to_bb: Optional[float] = None
    cards: Optional[Tuple[Card, ...]] = None

    def __repr__(self) -> str:
        parts = [self.action_type.value]
        if self.player is not None:
            parts.insert(0, self.player.value)
        if self.amount_to_bb is not None:
            parts.append(f"to={self.amount_to_bb}")
        if self.cards:
            parts.append("cards=" + " ".join(repr(c) for c in self.cards))
        return f"Action({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Legal action option
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LegalAction:
    """Describes a single legal action available to the current actor.

    For BET/RAISE the bounds ``min_to_bb`` / ``max_to_bb`` describe the
    legal sizing range (total contribution after the action on this street).
    Strategy layers choose a specific size within those bounds.
    """
    action_type: ActionType
    min_to_bb: Optional[float] = None
    max_to_bb: Optional[float] = None
    call_amount_bb: Optional[float] = None

    def __repr__(self) -> str:
        if self.action_type in (ActionType.BET, ActionType.RAISE):
            return f"{self.action_type.value}({self.min_to_bb}-{self.max_to_bb})"
        if self.action_type == ActionType.CALL and self.call_amount_bb is not None:
            return f"CALL({self.call_amount_bb}bb)"
        return self.action_type.value


# ---------------------------------------------------------------------------
# Canonical hand state (derived by reconstruction)
# ---------------------------------------------------------------------------

@dataclass
class HandState:
    """Fully derived canonical state for a hand in progress or completed."""

    config: HandConfig
    action_history: List[Action]

    # Street / board
    current_street: Street
    board_cards: List[Card] = field(default_factory=list)

    # Pot / contributions (cumulative across all streets)
    pot_size_bb: float = 0.0
    hero_contribution_bb: float = 0.0
    villain_contribution_bb: float = 0.0

    # Current-street betting state
    current_bet_to_call_bb: float = 0.0
    current_actor: Optional[Player] = None
    last_aggressor: Optional[Player] = None
    number_of_raises_this_street: int = 0
    last_full_raise_size: float = 0.0

    # Per-street tracking (reset each street by reconstruction)
    street_contrib_hero: float = 0.0
    street_contrib_villain: float = 0.0
    street_bet_level: float = 0.0

    # Closure / terminal
    betting_round_closed: bool = False
    hand_over: bool = False
    showdown_ready: bool = False
    fold_winner: Optional[Player] = None
    awaiting_runout: bool = False

    # All-in tracking
    hero_all_in: bool = False
    villain_all_in: bool = False

    # Cached legal actions (populated by legal_actions module)
    legal_actions_list: Optional[List[LegalAction]] = field(default=None, repr=False)
