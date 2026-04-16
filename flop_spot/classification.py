"""Flop hand-strength and board-texture classification.

Works on the 5-card set (2 hero hole cards + 3 board cards).
No equity simulation -- pure category-based classification.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Optional, Tuple

from poker_core.models import Card, HoleCards

from .models import (
    BoardFeatures,
    BoardTexture,
    DrawCategory,
    HandClassification,
    MadeHandCategory,
)


# ---------------------------------------------------------------------------
# Rank helpers
# ---------------------------------------------------------------------------

_RANK_VAL = {r: i for i, r in enumerate("23456789TJQKA", 2)}


def _rv(card: Card) -> int:
    return card.rank_value


# ---------------------------------------------------------------------------
# Board texture classification
# ---------------------------------------------------------------------------

def classify_board(board: List[Card]) -> BoardFeatures:
    """Classify a 3-card flop board."""
    if len(board) != 3:
        raise ValueError(f"Flop board must have exactly 3 cards, got {len(board)}")

    ranks = sorted([_rv(c) for c in board], reverse=True)
    suits = [c.suit for c in board]
    suit_counts = Counter(suits)

    top_rank = ranks[0]
    is_paired = len(set(ranks)) < 3
    is_monotone = max(suit_counts.values()) == 3
    is_two_tone = max(suit_counts.values()) == 2
    is_rainbow = max(suit_counts.values()) == 1

    gap1 = ranks[0] - ranks[1]
    gap2 = ranks[1] - ranks[2]
    max_gap = max(gap1, gap2)
    total_span = ranks[0] - ranks[2]

    # Straight heaviness: cards within a 5-rank window allowing straight
    has_straight_heaviness = total_span <= 4 and not is_paired

    board_is_high_card_heavy = top_rank >= 10  # T+
    all_low = all(r <= 8 for r in ranks)
    board_is_low_connected = all_low and max_gap <= 2

    connected = max_gap <= 1 and not is_paired
    semi_connected = max_gap <= 2 and not is_paired

    board_is_dynamic = (is_two_tone or is_monotone) and semi_connected

    # Texture label by priority
    texture = _determine_texture(
        is_paired, is_monotone, is_two_tone, is_rainbow,
        connected, semi_connected, has_straight_heaviness,
        board_is_high_card_heavy, all_low, board_is_dynamic, top_rank,
    )

    return BoardFeatures(
        texture=texture,
        is_paired=is_paired,
        is_monotone=is_monotone,
        is_two_tone=is_two_tone,
        is_rainbow=is_rainbow,
        has_straight_heaviness=has_straight_heaviness,
        top_card_rank=top_rank,
        board_is_high_card_heavy=board_is_high_card_heavy,
        board_is_low_connected=board_is_low_connected,
        board_is_dynamic=board_is_dynamic,
    )


def _determine_texture(
    is_paired: bool,
    is_monotone: bool,
    is_two_tone: bool,
    is_rainbow: bool,
    connected: bool,
    semi_connected: bool,
    has_straight_heaviness: bool,
    high_card_heavy: bool,
    all_low: bool,
    dynamic: bool,
    top_rank: int,
) -> BoardTexture:
    if is_paired:
        return BoardTexture.PAIRED_BOARD
    if is_monotone:
        return BoardTexture.MONOTONE_BOARD
    if dynamic:
        return BoardTexture.DYNAMIC_DRAW_HEAVY
    if is_two_tone and connected:
        if all_low:
            return BoardTexture.LOW_CONNECTED
        if high_card_heavy:
            return BoardTexture.HIGH_CONNECTED
        return BoardTexture.TWO_TONE_BOARD
    if is_two_tone and semi_connected:
        return BoardTexture.TWO_TONE_BOARD
    if is_rainbow and connected:
        if all_low:
            return BoardTexture.LOW_CONNECTED
        return BoardTexture.RAINBOW_CONNECTED
    if is_rainbow and semi_connected:
        if all_low:
            return BoardTexture.LOW_CONNECTED
        return BoardTexture.RAINBOW_CONNECTED
    if is_rainbow and not semi_connected:
        if high_card_heavy:
            return BoardTexture.DRY_HIGH_CARD
        if all_low:
            return BoardTexture.DRY_LOW_BOARD
        return BoardTexture.STATIC_BOARD
    if is_two_tone:
        if high_card_heavy:
            return BoardTexture.DRY_HIGH_CARD
        return BoardTexture.TWO_TONE_BOARD
    if is_rainbow:
        if high_card_heavy:
            return BoardTexture.DRY_HIGH_CARD
        if all_low:
            return BoardTexture.DRY_LOW_BOARD
        return BoardTexture.STATIC_BOARD
    return BoardTexture.STATIC_BOARD


# ---------------------------------------------------------------------------
# Hand classification
# ---------------------------------------------------------------------------

def classify_hand(
    hero_cards: HoleCards,
    board: List[Card],
) -> HandClassification:
    """Classify hero's made hand and draw potential on a 3-card flop."""
    if len(board) != 3:
        raise ValueError(f"Flop board must have exactly 3 cards, got {len(board)}")

    h1, h2 = hero_cards.high, hero_cards.low
    board_ranks = sorted([_rv(c) for c in board], reverse=True)
    all_cards = [h1, h2] + list(board)
    all_ranks = sorted([_rv(c) for c in all_cards], reverse=True)

    made_hand, pair_rel = _classify_made_hand(h1, h2, board, board_ranks)
    draw, has_fd, has_oesd, has_gs, has_bd_flush, has_bd_straight = _classify_draw(
        h1, h2, board, board_ranks, all_cards,
    )

    # Upgrade to combo draw if flush draw + straight draw
    if has_fd and (has_oesd or has_gs):
        draw = DrawCategory.COMBO_DRAW

    has_pair = made_hand not in (
        MadeHandCategory.ACE_HIGH,
        MadeHandCategory.KING_HIGH_OR_WORSE_HIGH_CARD,
    )
    has_two_pair_plus = made_hand in (
        MadeHandCategory.TWO_PAIR,
        MadeHandCategory.SET,
        MadeHandCategory.NUTS_OR_NEAR_NUTS,
    )
    has_overpair = made_hand == MadeHandCategory.OVERPAIR
    has_top_pair = made_hand in (
        MadeHandCategory.TOP_PAIR_STRONG_KICKER,
        MadeHandCategory.TOP_PAIR_MEDIUM_KICKER,
        MadeHandCategory.TOP_PAIR_WEAK_KICKER,
    )
    has_showdown_value = has_pair or made_hand == MadeHandCategory.ACE_HIGH
    has_strong_draw = draw in (
        DrawCategory.FLUSH_DRAW,
        DrawCategory.OPEN_ENDED_STRAIGHT_DRAW,
        DrawCategory.COMBO_DRAW,
        DrawCategory.MADE_FLUSH,
        DrawCategory.MADE_STRAIGHT,
    )
    has_combo = draw == DrawCategory.COMBO_DRAW
    has_backdoor = has_bd_flush or has_bd_straight

    overcards = sum(
        1 for c in (h1, h2) if _rv(c) > board_ranks[0]
    )

    return HandClassification(
        made_hand=made_hand,
        draw=draw,
        has_pair=has_pair,
        has_two_pair_plus=has_two_pair_plus,
        has_overpair=has_overpair,
        has_top_pair=has_top_pair,
        has_showdown_value=has_showdown_value,
        has_strong_draw=has_strong_draw,
        has_combo_draw=has_combo,
        has_backdoor_equity=has_backdoor,
        overcards_to_board_count=overcards,
        pair_rank_relative_to_board=pair_rel,
    )


# ---------------------------------------------------------------------------
# Made-hand logic
# ---------------------------------------------------------------------------

def _classify_made_hand(
    h1: Card, h2: Card,
    board: List[Card], board_ranks: List[int],
) -> Tuple[MadeHandCategory, Optional[str]]:
    """Return (MadeHandCategory, pair_relationship_label)."""
    hr1, hr2 = _rv(h1), _rv(h2)
    all_ranks = sorted([hr1, hr2] + board_ranks, reverse=True)
    all_cards = [h1, h2] + list(board)

    # --- Made flush ---
    suit_counts = Counter(c.suit for c in all_cards)
    for suit, cnt in suit_counts.items():
        if cnt >= 5:
            return MadeHandCategory.NUTS_OR_NEAR_NUTS, None

    # --- Made straight ---
    if _has_straight(all_ranks):
        return MadeHandCategory.NUTS_OR_NEAR_NUTS, None

    # --- Trips / set ---
    rank_counts = Counter(all_ranks)
    for rank, cnt in rank_counts.items():
        if cnt >= 3:
            board_rank_set = set(board_ranks)
            if hr1 == hr2 == rank and rank in board_rank_set:
                return MadeHandCategory.SET, "SET"
            return MadeHandCategory.NUTS_OR_NEAR_NUTS, "TRIPS"

    # --- Two pair ---
    pairs = [r for r, c in rank_counts.items() if c == 2]
    if len(pairs) >= 2:
        # At least one pair must use a hero card
        hero_in_pair = any(r in {hr1, hr2} for r in pairs)
        if hero_in_pair:
            return MadeHandCategory.TWO_PAIR, "TWO_PAIR"

    # --- One pair ---
    hero_pair = hr1 == hr2
    if hero_pair:
        if hr1 > board_ranks[0]:
            return MadeHandCategory.OVERPAIR, "OVERPAIR"
        if hr1 < board_ranks[-1]:
            return MadeHandCategory.UNDERPAIR_TO_BOARD, "UNDERPAIR"
        return MadeHandCategory.MIDDLE_PAIR, "POCKET_BETWEEN_BOARD"

    # Hero card paired with board
    for hero_rank in (hr1, hr2):
        if hero_rank in board_ranks:
            kicker = hr1 if hero_rank == hr2 else hr2
            kicker_v = _rv(Card(rank=kicker.rank, suit=kicker.suit)) if isinstance(kicker, Card) else kicker
            if isinstance(kicker, Card):
                kicker_v = _rv(kicker)
            return _classify_top_mid_bottom_pair(
                hero_rank, kicker_v, board_ranks,
            )

    # --- No pair: high card ---
    if hr1 == 14:
        return MadeHandCategory.ACE_HIGH, None
    return MadeHandCategory.KING_HIGH_OR_WORSE_HIGH_CARD, None


def _classify_top_mid_bottom_pair(
    pair_rank: int,
    kicker_rank: int,
    board_ranks: List[int],
) -> Tuple[MadeHandCategory, str]:
    """Classify a single pair made with the board."""
    if pair_rank == board_ranks[0]:
        if kicker_rank >= 12:  # Q+
            return MadeHandCategory.TOP_PAIR_STRONG_KICKER, "TOP_PAIR"
        if kicker_rank >= 9:   # 9-J
            return MadeHandCategory.TOP_PAIR_MEDIUM_KICKER, "TOP_PAIR"
        return MadeHandCategory.TOP_PAIR_WEAK_KICKER, "TOP_PAIR"
    if pair_rank == board_ranks[1]:
        return MadeHandCategory.MIDDLE_PAIR, "MIDDLE_PAIR"
    return MadeHandCategory.THIRD_PAIR_OR_WORSE_PAIR, "BOTTOM_PAIR"


# ---------------------------------------------------------------------------
# Straight detection
# ---------------------------------------------------------------------------

def _has_straight(ranks: List[int]) -> bool:
    """Check if any 5 of the given ranks form a straight.

    Works for 5 cards (flop: 2 hero + 3 board).
    Also handles wheel (A-2-3-4-5).
    """
    unique = sorted(set(ranks), reverse=True)
    # Add low-ace for wheel check
    if 14 in unique:
        unique.append(1)

    for i in range(len(unique) - 4):
        if unique[i] - unique[i + 4] == 4:
            return True
    return False


# ---------------------------------------------------------------------------
# Draw classification
# ---------------------------------------------------------------------------

def _classify_draw(
    h1: Card, h2: Card,
    board: List[Card], board_ranks: List[int],
    all_cards: List[Card],
) -> Tuple[DrawCategory, bool, bool, bool, bool, bool]:
    """Return (DrawCategory, has_fd, has_oesd, has_gutshot, has_bd_flush, has_bd_straight)."""
    has_fd = _has_flush_draw(h1, h2, board)
    has_oesd = _has_oesd(h1, h2, board)
    has_gs = _has_gutshot(h1, h2, board) if not has_oesd else False
    has_bd_flush = _has_backdoor_flush_draw(h1, h2, board)
    has_bd_straight = _has_backdoor_straight_draw(h1, h2, board)

    # Combo draw: flush draw + any straight draw
    if has_fd and (has_oesd or has_gs):
        return DrawCategory.COMBO_DRAW, has_fd, has_oesd, has_gs, has_bd_flush, has_bd_straight

    if has_fd:
        return DrawCategory.FLUSH_DRAW, has_fd, has_oesd, has_gs, has_bd_flush, has_bd_straight
    if has_oesd:
        return DrawCategory.OPEN_ENDED_STRAIGHT_DRAW, has_fd, has_oesd, has_gs, has_bd_flush, has_bd_straight
    if has_gs:
        return DrawCategory.GUTSHOT, has_fd, has_oesd, has_gs, has_bd_flush, has_bd_straight
    if has_bd_flush:
        return DrawCategory.BACKDOOR_FLUSH_DRAW, has_fd, has_oesd, has_gs, has_bd_flush, has_bd_straight
    if has_bd_straight:
        return DrawCategory.BACKDOOR_STRAIGHT_DRAW, has_fd, has_oesd, has_gs, has_bd_flush, has_bd_straight
    return DrawCategory.NO_REAL_DRAW, False, False, False, False, False


def _has_flush_draw(h1: Card, h2: Card, board: List[Card]) -> bool:
    """4 cards of one suit among hero + board (need 1 more for flush)."""
    suit_counts = Counter(c.suit for c in [h1, h2] + list(board))
    hero_suits = {h1.suit, h2.suit}
    for suit, cnt in suit_counts.items():
        if cnt == 4 and suit in hero_suits:
            return True
    return False


def _has_oesd(h1: Card, h2: Card, board: List[Card]) -> bool:
    """Open-ended straight draw: 4 consecutive ranks with both ends open.

    Hero must contribute at least one card to the 4-card run.
    """
    all_ranks = sorted(set(_rv(c) for c in [h1, h2] + list(board)))
    hero_ranks = {_rv(h1), _rv(h2)}

    if 14 in all_ranks:
        all_ranks_with_ace_low = sorted(set(all_ranks + [1]))
    else:
        all_ranks_with_ace_low = all_ranks

    for i in range(len(all_ranks_with_ace_low) - 3):
        window = all_ranks_with_ace_low[i:i + 4]
        if window[-1] - window[0] == 3 and len(set(window)) == 4:
            low_end = window[0] - 1
            high_end = window[-1] + 1
            both_open = (1 <= low_end <= 14) and (1 <= high_end <= 14)
            # Exclude cases capped by ace on high end (only wheel wraps)
            if window[-1] == 14:
                both_open = False
            if window[0] == 1:
                both_open = False
            if both_open:
                hero_contributes = any(r in hero_ranks for r in window) or (
                    1 in window and 14 in hero_ranks
                )
                if hero_contributes:
                    return True
    return False


def _has_gutshot(h1: Card, h2: Card, board: List[Card]) -> bool:
    """Gutshot: 4 ranks within a span of 5 (one gap) with hero contributing."""
    all_ranks = sorted(set(_rv(c) for c in [h1, h2] + list(board)))
    hero_ranks = {_rv(h1), _rv(h2)}

    if 14 in all_ranks:
        extended = sorted(set(all_ranks + [1]))
    else:
        extended = all_ranks

    for target_high in range(5, 15):
        target_low = target_high - 4
        window_ranks = [r for r in extended if target_low <= r <= target_high]
        unique_in_window = len(set(window_ranks))
        if unique_in_window == 4:
            hero_contributes = any(r in hero_ranks for r in window_ranks) or (
                1 in window_ranks and 14 in hero_ranks
            )
            if hero_contributes:
                return True

    # Wheel gutshot: A-2-3-4-5 with ace as low
    wheel_ranks = {1, 2, 3, 4, 5}
    present = set(extended) & wheel_ranks
    if len(present) == 4 and (14 in hero_ranks or any(r in hero_ranks for r in present if r != 1)):
        return True

    return False


def _has_backdoor_flush_draw(h1: Card, h2: Card, board: List[Card]) -> bool:
    """Exactly 3 cards of one suit (hero must contribute at least one)."""
    suit_counts = Counter(c.suit for c in [h1, h2] + list(board))
    hero_suits = {h1.suit, h2.suit}
    for suit, cnt in suit_counts.items():
        if cnt == 3 and suit in hero_suits:
            return True
    return False


def _has_backdoor_straight_draw(h1: Card, h2: Card, board: List[Card]) -> bool:
    """Runner-runner straight potential: 3 cards within a 5-rank window,
    hero contributing at least one."""
    all_ranks = sorted(set(_rv(c) for c in [h1, h2] + list(board)))
    hero_ranks = {_rv(h1), _rv(h2)}

    if 14 in all_ranks:
        extended = sorted(set(all_ranks + [1]))
    else:
        extended = all_ranks

    for target_high in range(5, 15):
        target_low = target_high - 4
        window_ranks = [r for r in extended if target_low <= r <= target_high]
        if len(set(window_ranks)) >= 3:
            if any(r in hero_ranks for r in window_ranks) or (
                1 in window_ranks and 14 in hero_ranks
            ):
                return True
    return False
