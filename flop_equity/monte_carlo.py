"""Seeded Monte Carlo equity estimation on the flop.

Evaluates hero equity vs a weighted villain range by sampling villain
hands and runouts (turn + river), then comparing best 5-card hands at
showdown.  Includes a self-contained 7-card evaluator.
"""

from __future__ import annotations

import random
from collections import Counter
from itertools import combinations
from typing import Dict, List, Optional, Tuple

from poker_core.models import Card, HoleCards, RANKS, SUITS, Street


# ===================================================================
# 7-card hand evaluator (self-contained, no external dependencies)
# ===================================================================

# Hand categories ranked 0 (worst) to 8 (best).
_HIGH_CARD = 0
_ONE_PAIR = 1
_TWO_PAIR = 2
_THREE_OF_A_KIND = 3
_STRAIGHT = 4
_FLUSH = 5
_FULL_HOUSE = 6
_FOUR_OF_A_KIND = 7
_STRAIGHT_FLUSH = 8


def best_hand_rank_seven_cards(cards: List[Card]) -> Tuple[int, ...]:
    """Public API: best 5-of-7 rank tuple (higher tuple wins at showdown).

    Raises ``ValueError`` if ``len(cards) != 7``.
    """
    if len(cards) != 7:
        raise ValueError(f"Expected exactly 7 cards, got {len(cards)}")
    return _best_hand_rank(cards)


def best_hand_rank_hole_board(hole: HoleCards, board: List[Card]) -> Tuple[int, ...]:
    """Best 5-card rank from exactly two hole cards plus ``board`` (3–5 cards).

    Uses best-of when more than five cards are available (flop/turn/river).
    """
    cards = [hole.high, hole.low] + list(board)
    n = len(cards)
    if n < 5:
        raise ValueError(f"Need at least 5 cards (2 hole + 3+ board), got {n}")
    if n == 5:
        return _eval_five(cards)
    return _best_hand_rank(cards)


def _best_hand_rank(cards: List[Card]) -> Tuple[int, ...]:
    """Evaluate the best 5-card hand from 7 cards.

    Returns a comparable tuple: (category, *kickers) where higher is
    better at every position.
    """
    best: Optional[Tuple[int, ...]] = None
    for five in combinations(cards, 5):
        rank = _eval_five(list(five))
        if best is None or rank > best:
            best = rank
    return best  # type: ignore[return-value]


def _eval_five(cards: List[Card]) -> Tuple[int, ...]:
    """Evaluate exactly 5 cards and return a comparable rank tuple."""
    ranks = sorted([c.rank_value for c in cards], reverse=True)
    suits = [c.suit for c in cards]
    is_flush = len(set(suits)) == 1

    unique_sorted = sorted(set(ranks), reverse=True)
    is_straight, straight_high = _check_straight(unique_sorted)

    counts = Counter(ranks)
    groups = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)

    if is_straight and is_flush:
        return (_STRAIGHT_FLUSH, straight_high)

    if groups[0][1] == 4:
        quad_r = groups[0][0]
        kicker = max(r for r in ranks if r != quad_r)
        return (_FOUR_OF_A_KIND, quad_r, kicker)

    if groups[0][1] == 3 and groups[1][1] >= 2:
        return (_FULL_HOUSE, groups[0][0], groups[1][0])

    if is_flush:
        return (_FLUSH,) + tuple(ranks)

    if is_straight:
        return (_STRAIGHT, straight_high)

    if groups[0][1] == 3:
        trip_r = groups[0][0]
        kickers = sorted([r for r in ranks if r != trip_r], reverse=True)
        return (_THREE_OF_A_KIND, trip_r) + tuple(kickers[:2])

    if groups[0][1] == 2 and groups[1][1] == 2:
        high_pair = max(groups[0][0], groups[1][0])
        low_pair = min(groups[0][0], groups[1][0])
        kicker = max(r for r in ranks if r != high_pair and r != low_pair)
        return (_TWO_PAIR, high_pair, low_pair, kicker)

    if groups[0][1] == 2:
        pair_r = groups[0][0]
        kickers = sorted([r for r in ranks if r != pair_r], reverse=True)
        return (_ONE_PAIR, pair_r) + tuple(kickers[:3])

    return (_HIGH_CARD,) + tuple(ranks)


def _check_straight(unique_sorted_desc: List[int]) -> Tuple[bool, int]:
    """Check if 5 unique ranks form a straight. Returns (is_straight, high_card).

    Handles wheel (A-2-3-4-5).
    """
    if len(unique_sorted_desc) != 5:
        return False, 0

    # Normal straight: consecutive descending
    if unique_sorted_desc[0] - unique_sorted_desc[4] == 4:
        return True, unique_sorted_desc[0]

    # Wheel: A-5-4-3-2
    if unique_sorted_desc == [14, 5, 4, 3, 2]:
        return True, 5  # 5-high straight

    return False, 0


# ===================================================================
# Monte Carlo simulation
# ===================================================================

_ALL_CARDS_SET = frozenset(Card(rank=r, suit=s) for r in RANKS for s in SUITS)


def estimate_flop_equity(
    hero: HoleCards,
    board: List[Card],
    villain_range: List[Tuple[HoleCards, float]],
    samples: int = 5000,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Run Monte Carlo equity estimation for hero vs villain range on the flop.

    Args:
        hero: Hero's hole cards.
        board: Flop board cards (exactly 3).
        villain_range: List of (HoleCards, weight) tuples.
        samples: Number of simulation iterations.
        seed: RNG seed for deterministic results.

    Returns:
        Dict with keys: equity_estimate, win_rate, tie_rate, samples_used.
    """
    if len(board) != 3:
        raise ValueError(f"Expected 3 board cards for flop, got {len(board)}")
    if not villain_range:
        raise ValueError("Villain range is empty")

    rng = random.Random(seed)

    dead_hero = frozenset({hero.high, hero.low})
    dead_board = frozenset(board)
    base_dead = dead_hero | dead_board

    # Pre-filter villain range to remove dead-card conflicts
    valid_range = [
        (hc, w) for hc, w in villain_range
        if hc.high not in base_dead and hc.low not in base_dead
    ]
    if not valid_range:
        raise ValueError("No valid villain combos after dead-card filtering")

    # Build cumulative weight array for weighted sampling
    total_weight = sum(w for _, w in valid_range)
    cum_weights = []
    running = 0.0
    for _, w in valid_range:
        running += w
        cum_weights.append(running)

    wins = 0
    ties = 0
    used = 0

    remaining_deck_base = list(_ALL_CARDS_SET - base_dead)

    for _ in range(samples):
        # Sample villain hand (weighted)
        r = rng.random() * total_weight
        idx = _bisect_left(cum_weights, r)
        if idx >= len(valid_range):
            idx = len(valid_range) - 1
        villain_hand = valid_range[idx][0]

        # Remove villain cards from deck
        villain_dead = {villain_hand.high, villain_hand.low}
        deck = [c for c in remaining_deck_base if c not in villain_dead]

        if len(deck) < 2:
            continue

        # Sample turn + river
        runout = rng.sample(deck, 2)
        full_board = list(board) + runout

        hero_cards = [hero.high, hero.low] + full_board
        villain_cards = [villain_hand.high, villain_hand.low] + full_board

        hero_rank = _best_hand_rank(hero_cards)
        villain_rank = _best_hand_rank(villain_cards)

        if hero_rank > villain_rank:
            wins += 1
        elif hero_rank == villain_rank:
            ties += 1

        used += 1

    if used == 0:
        return {
            "equity_estimate": 0.0,
            "win_rate": 0.0,
            "tie_rate": 0.0,
            "samples_used": 0,
        }

    equity = (wins + 0.5 * ties) / used
    return {
        "equity_estimate": round(equity, 4),
        "win_rate": round(wins / used, 4),
        "tie_rate": round(ties / used, 4),
        "samples_used": used,
    }


def estimate_showdown_equity(
    hero: HoleCards,
    board: List[Card],
    villain_range: List[Tuple[HoleCards, float]],
    *,
    street: Street,
    samples: int = 5000,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Hero equity vs a weighted villain range by street (chip showdown).

    - **FLOP** (``len(board)==3``): Monte Carlo turn+river (same as
      ``estimate_flop_equity``).
    - **TURN** (``len(board)==4``): Monte Carlo over one river card.
    - **RIVER** (``len(board)==5``): exact weighted equity vs range (no RNG).

    Returns dict keys: ``equity_estimate``, ``win_rate``, ``tie_rate``,
    ``samples_used`` (MC iterations on flop/turn; villain combo count on
    river).
    """
    n_board = len(board)
    if street == Street.FLOP:
        if n_board != 3:
            raise ValueError(f"FLOP equity expects 3 board cards, got {n_board}")
        return estimate_flop_equity(hero, board, villain_range, samples=samples, seed=seed)

    if street == Street.TURN:
        if n_board != 4:
            raise ValueError(f"TURN equity expects 4 board cards, got {n_board}")
        return _estimate_turn_showdown_equity(hero, board, villain_range, samples, seed)

    if street == Street.RIVER:
        if n_board != 5:
            raise ValueError(f"RIVER equity expects 5 board cards, got {n_board}")
        return _estimate_river_showdown_equity_exact(hero, board, villain_range)

    raise ValueError(f"Unsupported street for showdown equity: {street!r}")


def _estimate_turn_showdown_equity(
    hero: HoleCards,
    board: List[Card],
    villain_range: List[Tuple[HoleCards, float]],
    samples: int,
    seed: Optional[int],
) -> Dict[str, float]:
    if not villain_range:
        raise ValueError("Villain range is empty")

    rng = random.Random(seed)
    dead_hero = frozenset({hero.high, hero.low})
    dead_board = frozenset(board)
    base_dead = dead_hero | dead_board

    valid_range = [
        (hc, w) for hc, w in villain_range
        if hc.high not in base_dead and hc.low not in base_dead
    ]
    if not valid_range:
        raise ValueError("No valid villain combos after dead-card filtering")

    total_weight = sum(w for _, w in valid_range)
    cum_weights: List[float] = []
    running = 0.0
    for _, w in valid_range:
        running += w
        cum_weights.append(running)

    wins = 0
    ties = 0
    used = 0
    remaining_deck_base = list(_ALL_CARDS_SET - base_dead)

    for _ in range(samples):
        r = rng.random() * total_weight
        idx = _bisect_left(cum_weights, r)
        if idx >= len(valid_range):
            idx = len(valid_range) - 1
        villain_hand = valid_range[idx][0]
        villain_dead = {villain_hand.high, villain_hand.low}
        deck = [c for c in remaining_deck_base if c not in villain_dead]
        if len(deck) < 1:
            continue
        river = rng.choice(deck)
        full_board = list(board) + [river]

        hero_cards = [hero.high, hero.low] + full_board
        villain_cards = [villain_hand.high, villain_hand.low] + full_board

        hero_rank = _best_hand_rank(hero_cards)
        villain_rank = _best_hand_rank(villain_cards)

        if hero_rank > villain_rank:
            wins += 1
        elif hero_rank == villain_rank:
            ties += 1
        used += 1

    if used == 0:
        return {
            "equity_estimate": 0.0,
            "win_rate": 0.0,
            "tie_rate": 0.0,
            "samples_used": 0,
        }

    equity = (wins + 0.5 * ties) / used
    return {
        "equity_estimate": round(equity, 4),
        "win_rate": round(wins / used, 4),
        "tie_rate": round(ties / used, 4),
        "samples_used": used,
    }


def _estimate_river_showdown_equity_exact(
    hero: HoleCards,
    board: List[Card],
    villain_range: List[Tuple[HoleCards, float]],
) -> Dict[str, float]:
    if not villain_range:
        raise ValueError("Villain range is empty")

    dead_hero = frozenset({hero.high, hero.low})
    dead_board = frozenset(board)
    base_dead = dead_hero | dead_board

    valid_range = [
        (hc, w) for hc, w in villain_range
        if hc.high not in base_dead and hc.low not in base_dead
        and w > 1e-12
    ]
    if not valid_range:
        raise ValueError("No valid villain combos after dead-card filtering")

    hero_cards = [hero.high, hero.low] + list(board)
    hero_rank = _best_hand_rank(hero_cards)

    wins_w = 0.0
    ties_w = 0.0
    total_w = 0.0

    for villain_hand, w in valid_range:
        villain_cards = [villain_hand.high, villain_hand.low] + list(board)
        villain_rank = _best_hand_rank(villain_cards)
        total_w += w
        if hero_rank > villain_rank:
            wins_w += w
        elif hero_rank == villain_rank:
            ties_w += w

    if total_w < 1e-15:
        return {
            "equity_estimate": 0.0,
            "win_rate": 0.0,
            "tie_rate": 0.0,
            "samples_used": 0,
        }

    equity = (wins_w + 0.5 * ties_w) / total_w
    return {
        "equity_estimate": round(equity, 4),
        "win_rate": round(wins_w / total_w, 4),
        "tie_rate": round(ties_w / total_w, 4),
        "samples_used": len(valid_range),
    }


def _bisect_left(a: List[float], x: float) -> int:
    """Simple bisect_left without importing bisect."""
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo
