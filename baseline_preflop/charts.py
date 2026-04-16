"""Preflop hand charts for heads-up no-limit Hold'em.

Each context has a RAISE set and optionally a CALL set.
Hands not in either set default to FOLD (or CHECK when no bet to face).
Charts are defined for deep stacks (DEEP / VERY_DEEP).
Shorter stacks use tightening filters or explicit jam tables.
"""

from __future__ import annotations

from .models import StackDepthBucket

# -------------------------------------------------------------------
# All 169 unique preflop hand labels
# -------------------------------------------------------------------

_RANK_CHARS = "AKQJT98765432"


def _generate_all_hands() -> frozenset:
    hands: set[str] = set()
    ranks = list(_RANK_CHARS)
    for i, r1 in enumerate(ranks):
        for j, r2 in enumerate(ranks):
            if i < j:
                hands.add(f"{r1}{r2}s")
                hands.add(f"{r1}{r2}o")
            elif i == j:
                hands.add(f"{r1}{r2}")
    return frozenset(hands)


ALL_HANDS = _generate_all_hands()  # 169 hands


# ===================================================================
# Hand strength score (used only for stack-depth tightening of calls)
# ===================================================================

_RV = {r: i for i, r in enumerate("23456789TJQKA", 2)}


def hand_strength(label: str) -> int:
    """Approximate numeric strength for chart tightening. Higher = stronger."""
    if len(label) == 2:
        return 40 + _RV[label[0]] * 4
    high = _RV[label[0]]
    low = _RV[label[1]]
    suited = label[2] == "s"
    gap = high - low - 1
    score = high * 3 + low
    if suited:
        score += 8
    if gap == 0:
        score += 3
    elif gap == 1:
        score += 1
    return score


# ===================================================================
# BTN open: raise or fold (deep stacks, no limping)
# ~80% open frequency
# ===================================================================

_BTN_OPEN_FOLD = frozenset({
    "Q5o", "Q4o", "Q3o", "Q2o",
    "J6o", "J5o", "J4o", "J3o", "J2o",
    "T6o", "T5o", "T4o", "T3o", "T2o",
    "96o", "95o", "94o", "93o", "92o",
    "86o", "85o", "84o", "83o", "82o",
    "75o", "74o", "73o", "72o",
    "64o", "63o", "62o",
    "53o", "52o",
    "43o", "42o",
    "32o",
})

BTN_OPEN_RAISE = ALL_HANDS - _BTN_OPEN_FOLD


# ===================================================================
# BB vs limp: raise or check (no fold option)
# ===================================================================

BB_VS_LIMP_RAISE = frozenset({
    # Premium + high pairs
    "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77",
    # Aces
    "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s",
    "A5s", "A4s", "A3s", "A2s",
    "AKo", "AQo", "AJo", "ATo", "A9o",
    # Kings
    "KQs", "KJs", "KTs", "K9s",
    "KQo", "KJo", "KTo",
    # Queens
    "QJs", "QTs", "Q9s",
    "QJo", "QTo",
    # Jacks
    "JTs", "J9s",
    "JTo",
    # Suited connectors
    "T9s", "98s", "87s", "76s", "65s", "54s",
})


# ===================================================================
# BB vs open raise: 3bet / call / fold (deep stacks)
# ===================================================================

BB_VS_OPEN_RAISE = frozenset({
    # Value 3bets
    "AA", "KK", "QQ", "JJ",
    "AKs", "AKo", "AQs", "AQo", "AJs",
    "KQs",
    # Bluff 3bets (ace blockers)
    "A5s", "A4s", "A3s", "A2s",
})

_BB_VS_OPEN_FOLD = frozenset({
    # Weak offsuit queens
    "Q5o", "Q4o", "Q3o", "Q2o",
    # Weak offsuit jacks
    "J7o", "J6o", "J5o", "J4o", "J3o", "J2o",
    # Weak offsuit tens
    "T7o", "T6o", "T5o", "T4o", "T3o", "T2o",
    # Weak offsuit nines
    "97o", "96o", "95o", "94o", "93o", "92o",
    # Weak offsuit eights
    "86o", "85o", "84o", "83o", "82o",
    # Weak offsuit sevens
    "75o", "74o", "73o", "72o",
    # Weak offsuit sixes and below
    "64o", "63o", "62o",
    "53o", "52o",
    "43o", "42o",
    "32o",
    # Bottom suited trash
    "93s", "92s", "83s", "82s", "73s", "72s",
    "62s", "52s", "42s", "32s",
    # More weak offsuit
    "K3o", "K2o",
    "Q6o",
})

BB_VS_OPEN_CALL = ALL_HANDS - BB_VS_OPEN_RAISE - _BB_VS_OPEN_FOLD


# ===================================================================
# BTN vs iso (BTN limped, BB raised): re-raise / call / fold
# ===================================================================

BTN_VS_ISO_RAISE = frozenset({
    "AA", "KK", "QQ", "JJ", "TT",
    "AKs", "AKo", "AQs", "AQo", "AJs",
})

BTN_VS_ISO_CALL = frozenset({
    "99", "88", "77",
    "ATs", "A9s", "AJo", "ATo",
    "KQs", "KJs", "KTs", "KQo", "KJo",
    "QJs", "QTs",
    "JTs", "J9s",
    "T9s", "98s", "87s", "76s", "65s",
})


# ===================================================================
# BTN vs 3bet: 4bet / call / fold (deep stacks)
# ===================================================================

BTN_VS_3BET_RAISE = frozenset({
    "AA", "KK", "QQ",
    "AKs", "AKo",
    "A5s",  # bluff 4bet
})

BTN_VS_3BET_CALL = frozenset({
    "JJ", "TT", "99",
    "AQs", "AQo", "AJs", "ATs",
    "KQs", "KJs",
    "QJs", "JTs",
    "T9s", "98s", "87s", "76s", "65s", "54s",
})


# ===================================================================
# BB vs 4bet: 5bet (jam) / call / fold
# ===================================================================

BB_VS_4BET_RAISE = frozenset({
    "AA", "KK",
})

BB_VS_4BET_CALL = frozenset({
    "QQ", "JJ",
    "AKs", "AKo", "AQs",
})


# ===================================================================
# Short-stack jam tables
# ===================================================================

# ULTRA_SHORT (0-5bb): jam wide (~65% of hands)
ULTRA_SHORT_JAM = frozenset({
    # All pairs
    "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
    # All suited aces
    "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s",
    "A5s", "A4s", "A3s", "A2s",
    # All offsuit aces
    "AKo", "AQo", "AJo", "ATo", "A9o", "A8o", "A7o", "A6o",
    "A5o", "A4o", "A3o", "A2o",
    # Kings
    "KQs", "KJs", "KTs", "K9s", "K8s", "K7s", "K6s", "K5s", "K4s", "K3s", "K2s",
    "KQo", "KJo", "KTo", "K9o", "K8o", "K7o", "K6o", "K5o",
    # Queens
    "QJs", "QTs", "Q9s", "Q8s", "Q7s", "Q6s",
    "QJo", "QTo", "Q9o",
    # Jacks
    "JTs", "J9s", "J8s", "J7s",
    "JTo", "J9o", "J8o",
    # Tens
    "T9s", "T8s", "T7s",
    "T9o",
    # Connectors / semi-connectors
    "98s", "97s", "96s",
    "87s", "86s", "85s",
    "76s", "75s",
    "65s", "64s",
    "54s", "53s", "43s",
    # Offsuit connectors
    "98o", "87o", "76o", "65o",
})

# VERY_SHORT (6-10bb): tighter jam range (~40%)
VERY_SHORT_JAM = frozenset({
    # Pairs
    "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
    # Suited aces
    "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s",
    "A5s", "A4s", "A3s", "A2s",
    # Offsuit aces
    "AKo", "AQo", "AJo", "ATo", "A9o", "A8o", "A7o", "A6o", "A5o",
    # Kings
    "KQs", "KJs", "KTs", "K9s", "K8s", "K7s",
    "KQo", "KJo", "KTo",
    # Queens
    "QJs", "QTs", "Q9s",
    "QJo",
    # Jacks
    "JTs", "J9s",
    "JTo",
    # Connectors
    "T9s", "98s", "87s", "76s", "65s", "54s",
})

# Tighter jam range when facing a raise at short stacks
VERY_SHORT_VS_RAISE_JAM = frozenset({
    "AA", "KK", "QQ", "JJ", "TT",
    "AKs", "AKo", "AQs", "AQo", "AJs", "ATs",
    "A9s", "A8s", "A7s", "A6s", "A5s",
    "KQs", "KJs",
    "99", "88",
})


# ===================================================================
# Chart lookup
# ===================================================================

# Minimum hand_strength to stay in a CALL set at tighter stack depths
_CALL_MIN_STRENGTH = {
    StackDepthBucket.MEDIUM: 28,
    StackDepthBucket.SHORT: 36,
}

# When facing a raise at SHORT stacks, speculative calls need much more strength
_CALL_MIN_STRENGTH_VS_RAISE = {
    StackDepthBucket.MEDIUM: 34,
    StackDepthBucket.SHORT: 48,
}


def get_chart_action(
    label: str,
    context: str,
    stack_bucket: StackDepthBucket,
    facing_raise: bool = False,
) -> str:
    """Look up the baseline chart action for a hand.

    Returns one of: "RAISE", "CALL", "CHECK", "FOLD".
    """
    # --- Short-stack jam/fold ---
    if stack_bucket == StackDepthBucket.ULTRA_SHORT:
        return "RAISE" if label in ULTRA_SHORT_JAM else "FOLD"

    if stack_bucket == StackDepthBucket.VERY_SHORT:
        if facing_raise:
            return "RAISE" if label in VERY_SHORT_VS_RAISE_JAM else "FOLD"
        return "RAISE" if label in VERY_SHORT_JAM else "FOLD"

    # --- Standard chart lookup ---
    chart = _CHARTS.get(context)
    if chart is None:
        return "FOLD"

    raise_set = chart["raise"]
    call_set = chart.get("call")
    is_limp_context = (context == "BB_VS_LIMP")

    if label in raise_set:
        return "RAISE"

    if call_set is not None and label in call_set:
        thresholds = _CALL_MIN_STRENGTH_VS_RAISE if facing_raise else _CALL_MIN_STRENGTH
        min_str = thresholds.get(stack_bucket, 0)
        if min_str > 0 and hand_strength(label) < min_str:
            if is_limp_context:
                return "CHECK"
            return "FOLD"
        return "CALL"

    if is_limp_context:
        return "CHECK"
    return "FOLD"


_CHARTS = {
    "BTN_OPEN": {"raise": BTN_OPEN_RAISE},
    "BB_VS_LIMP": {"raise": BB_VS_LIMP_RAISE},
    "BB_VS_OPEN": {"raise": BB_VS_OPEN_RAISE, "call": BB_VS_OPEN_CALL},
    "BTN_VS_ISO": {"raise": BTN_VS_ISO_RAISE, "call": BTN_VS_ISO_CALL},
    "BTN_VS_3BET": {"raise": BTN_VS_3BET_RAISE, "call": BTN_VS_3BET_CALL},
    "BB_VS_4BET": {"raise": BB_VS_4BET_RAISE, "call": BB_VS_4BET_CALL},
}


def get_chart_sets(context: str) -> tuple[frozenset[str], frozenset[str] | None]:
    """Return (raise_set, call_set) for a chart context.

    If context is unknown, returns (empty_set, None).
    """
    chart = _CHARTS.get(context)
    if chart is None:
        return frozenset(), None
    return chart["raise"], chart.get("call")
