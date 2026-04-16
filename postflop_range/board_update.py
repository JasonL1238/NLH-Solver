"""Board card filtering and particle reclassification (no full range rebuild)."""

from __future__ import annotations

from collections import Counter
from typing import List, Set, Tuple

from poker_core.models import Card, HoleCards

from flop_equity.monte_carlo import best_hand_rank_hole_board

# Category ints match flop_equity.monte_carlo._eval_five ordering.
_CAT_HIGH = 0
_CAT_PAIR = 1
_CAT_TWO_PAIR = 2
_CAT_TRIPS = 3
_CAT_STRAIGHT = 4
_CAT_FLUSH = 5
_CAT_FULL = 6
_CAT_QUADS = 7
_CAT_SF = 8
from flop_spot.classification import classify_hand
from flop_spot.models import DrawCategory, MadeHandCategory

from .particles import CoarseBucket, Particle


def _rank_tuple_to_coarse(rank: Tuple[int, ...]) -> CoarseBucket:
    """Map best 5-card rank tuple (monte_carlo categories) to coarse bucket."""
    cat = rank[0]
    if cat == _CAT_SF or cat == _CAT_QUADS or cat == _CAT_FULL:
        return CoarseBucket.NUTTED
    if cat == _CAT_FLUSH or cat == _CAT_STRAIGHT:
        return CoarseBucket.STRONG_MADE
    if cat == _CAT_TRIPS or cat == _CAT_TWO_PAIR:
        return CoarseBucket.MEDIUM_MADE
    if cat == _CAT_PAIR:
        pr = rank[1]
        if pr >= 12:
            return CoarseBucket.MEDIUM_MADE
        if pr >= 9:
            return CoarseBucket.WEAK_SHOWDOWN
        return CoarseBucket.WEAK_SHOWDOWN
    return CoarseBucket.AIR


def _flop_draw_bucket(draw: DrawCategory) -> CoarseBucket:
    if draw in (
        DrawCategory.OPEN_ENDED_STRAIGHT_DRAW,
        DrawCategory.COMBO_DRAW,
        DrawCategory.FLUSH_DRAW,
    ):
        return CoarseBucket.STRONG_DRAW
    if draw in (
        DrawCategory.GUTSHOT,
        DrawCategory.BACKDOOR_FLUSH_DRAW,
        DrawCategory.BACKDOOR_STRAIGHT_DRAW,
    ):
        return CoarseBucket.WEAK_DRAW
    return CoarseBucket.AIR


def _flop_made_bucket(made: MadeHandCategory) -> CoarseBucket:
    if made in (MadeHandCategory.NUTS_OR_NEAR_NUTS, MadeHandCategory.SET):
        return CoarseBucket.NUTTED
    if made in (
        MadeHandCategory.TWO_PAIR,
        MadeHandCategory.OVERPAIR,
        MadeHandCategory.TOP_PAIR_STRONG_KICKER,
    ):
        return CoarseBucket.STRONG_MADE
    if made in (
        MadeHandCategory.TOP_PAIR_MEDIUM_KICKER,
        MadeHandCategory.TOP_PAIR_WEAK_KICKER,
        MadeHandCategory.MIDDLE_PAIR,
    ):
        return CoarseBucket.MEDIUM_MADE
    if made in (
        MadeHandCategory.THIRD_PAIR_OR_WORSE_PAIR,
        MadeHandCategory.UNDERPAIR_TO_BOARD,
    ):
        return CoarseBucket.WEAK_SHOWDOWN
    return CoarseBucket.AIR


def _combine_flop_made_draw(made_b: CoarseBucket, draw_b: CoarseBucket) -> CoarseBucket:
    """Prefer stronger of made-only vs draw-only on flop."""
    order = [
        CoarseBucket.AIR,
        CoarseBucket.WEAK_DRAW,
        CoarseBucket.STRONG_DRAW,
        CoarseBucket.WEAK_SHOWDOWN,
        CoarseBucket.MEDIUM_MADE,
        CoarseBucket.STRONG_MADE,
        CoarseBucket.NUTTED,
    ]
    return max(made_b, draw_b, key=lambda b: order.index(b))


def classify_combo_on_board(
    combo: HoleCards,
    board: List[Card],
) -> Tuple[CoarseBucket, str, str]:
    """Return (coarse_bucket, made_label, draw_label) for villain combo vs board."""
    nb = len(board)
    if nb == 3:
        h = classify_hand(combo, board)
        made_b = _flop_made_bucket(h.made_hand)
        draw_b = _flop_draw_bucket(h.draw)
        coarse = _combine_flop_made_draw(made_b, draw_b)
        return coarse, h.made_hand.value, h.draw.value

    rank = best_hand_rank_hole_board(combo, board)
    coarse = _rank_tuple_to_coarse(rank)
    made_label = f"rank_cat_{rank[0]}"
    draw_label = "NO_REAL_DRAW" if nb == 5 else _turn_river_draw_label(combo, board)
    if nb == 4 and draw_label not in ("NO_REAL_DRAW",):
        # upgrade weak made with strong draw
        if coarse == CoarseBucket.AIR and draw_label in ("FLUSH_DRAW", "OESD", "COMBO_DRAW"):
            coarse = CoarseBucket.STRONG_DRAW
        elif coarse in (CoarseBucket.AIR, CoarseBucket.WEAK_SHOWDOWN) and draw_label in (
            "FLUSH_DRAW",
            "OESD",
        ):
            coarse = CoarseBucket.STRONG_DRAW
        elif coarse == CoarseBucket.AIR and draw_label == "GUTSHOT":
            coarse = CoarseBucket.WEAK_DRAW
    return coarse, made_label, draw_label


def _turn_river_draw_label(combo: HoleCards, board: List[Card]) -> str:
    """Minimal draw labels on 4-card board (turn); river has no draws."""
    if len(board) != 4:
        return "NO_REAL_DRAW"
    cards = [combo.high, combo.low] + list(board)
    suits = [c.suit for c in cards]
    mx = max(Counter(suits).values())
    if mx >= 5:
        return "MADE_FLUSH"
    if mx == 4:
        return "FLUSH_DRAW"
    ranks = sorted({c.rank_value for c in cards}, reverse=True)
    if _has_oesd_window(ranks):
        return "OESD"
    if _has_gutshot_window(ranks):
        return "GUTSHOT"
    return "NO_REAL_DRAW"


def _has_oesd_window(ranks: List[int]) -> bool:
    r = sorted(set(ranks))
    if len(r) < 4:
        return False
    for i in range(len(r) - 3):
        window = r[i : i + 4]
        if window[3] - window[0] == 3:
            return True
    if 14 in r and {2, 3, 4, 5}.issubset(set(r)):
        return True
    return False


def _has_gutshot_window(ranks: List[int]) -> bool:
    r = sorted(set(ranks))
    if len(r) < 4:
        return False
    for i in range(len(r) - 3):
        window = r[i : i + 4]
        if window[3] - window[0] == 4 and window[3] - window[2] > 1:
            return True
    return False


def filter_particles_for_new_dead(
    particles: List[Particle],
    new_dead: Set[Card],
) -> Tuple[int, str]:
    """Zero particles whose combo intersects ``new_dead``. Returns (killed_count, note)."""
    killed = 0
    for p in particles:
        if not p.alive:
            continue
        if p.combo.high in new_dead or p.combo.low in new_dead:
            p.alive = False
            p.weight = 0.0
            killed += 1
    return killed, f"Filtered {killed} particles conflicting with new board/dead cards."


def reclassify_all_particles(particles: List[Particle], board: List[Card]) -> str:
    """Recompute coarse bucket and labels for all alive particles."""
    for p in particles:
        if not p.alive:
            continue
        coarse, made_l, draw_l = classify_combo_on_board(p.combo, board)
        p.current_bucket = coarse
        p.made_category = made_l
        p.draw_category = draw_l
    return f"Reclassified {sum(1 for p in particles if p.alive)} particles on {len(board)}-card board."
