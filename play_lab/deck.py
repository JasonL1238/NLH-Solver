"""Deck helpers for lab dealing (RNG flop / turn / river excluding known cards)."""

from __future__ import annotations

import random
from typing import List, Optional, Set, Tuple

from poker_core.models import Card, HoleCards
from poker_core.parser import parse_card


def _all_cards() -> List[Card]:
    from poker_core.models import RANKS, SUITS

    return [Card(rank=r, suit=s) for r in RANKS for s in SUITS]


def used_cards(
    hero: HoleCards,
    villain: Optional[HoleCards],
) -> List[Card]:
    out = [hero.high, hero.low]
    if villain is not None:
        out.extend([villain.high, villain.low])
    return out


def deal_random_hole_cards(rng: random.Random, blocked: Set[Card]) -> HoleCards:
    """Sample two hole cards not in ``blocked``."""
    pool = [c for c in _all_cards() if c not in blocked]
    pair = rng.sample(pool, 2)
    return HoleCards(high=pair[0], low=pair[1])


def deal_random_hands(rng: random.Random) -> Tuple[HoleCards, HoleCards]:
    """Return ``(hero_hole_cards, villain_hole_cards)`` for engine vs human."""
    hero = deal_random_hole_cards(rng, set())
    blocked = {hero.high, hero.low}
    villain = deal_random_hole_cards(rng, blocked)
    return hero, villain


def draw_flop_cards(
    rng: random.Random,
    hero: HoleCards,
    villain: Optional[HoleCards],
) -> Tuple[Card, Card, Card]:
    """Return three distinct random flop cards not in either hole."""
    blocked = set(used_cards(hero, villain))
    pool = [c for c in _all_cards() if c not in blocked]
    if len(pool) < 3:
        raise ValueError("Not enough cards left to deal a flop")
    three = rng.sample(pool, 3)
    return three[0], three[1], three[2]


def draw_street_card(
    rng: random.Random,
    blocked: Set[Card],
) -> Card:
    """Draw one random card not in ``blocked`` (turn or river)."""
    pool = [c for c in _all_cards() if c not in blocked]
    if not pool:
        raise ValueError("No cards left to deal")
    return rng.choice(pool)


def blocked_for_runout(hero: HoleCards, villain: Optional[HoleCards], board: List[Card]) -> Set[Card]:
    """All cards already assigned (holes + board)."""
    out = set(used_cards(hero, villain))
    out.update(board)
    return out


def validate_flop_input(text: str) -> tuple[bool, str]:
    """Return ``(True, "")`` if ``text`` has three whitespace-separated tokens.

    Used by the Play Lab UI to avoid crashing Streamlit on empty or partial input.
    """
    stripped = (text or "").strip()
    if not stripped:
        return False, (
            "Flop cards field is empty. Enter three cards like `Ah 7c 2d`, "
            "or click **Fill random flop** first."
        )
    parts = stripped.split()
    if len(parts) != 3:
        return False, (
            f"Need exactly **3** card tokens (spaces between); you have **{len(parts)}**. "
            "Example: `Ah 7c 2d`."
        )
    return True, ""


def validate_single_card_input(text: str) -> tuple[bool, str]:
    """Single card token like ``Ks`` for turn/river fields."""
    stripped = (text or "").strip()
    if not stripped:
        return False, "Enter one card (e.g. `Ks`) or use **Fill random**."
    parts = stripped.split()
    if len(parts) != 1:
        return False, f"Need exactly **1** card token; got {len(parts)}."
    return True, ""


def parse_flop_triple(text: str) -> Tuple[Card, Card, Card]:
    """Parse ``'Ah 7c 2d'`` into three ``Card`` objects."""
    parts = text.strip().split()
    if len(parts) != 3:
        raise ValueError(f"Expected 3 card tokens, got {len(parts)}: {text!r}")
    return parse_card(parts[0]), parse_card(parts[1]), parse_card(parts[2])
