"""Hand parsing, feature extraction, and detailed hand classification."""

from __future__ import annotations

from .models import (
    Card, HoleCards, HandFeatures, HandBucket,
    RANKS, SUITS, RANK_VALUES,
)


# ---------------------------------------------------------------------------
# Card / hole-card parsing
# ---------------------------------------------------------------------------

def parse_card(token: str) -> Card:
    """Parse a single card token like 'As' or 'Td'."""
    token = token.strip()
    if len(token) != 2:
        raise ValueError(f"Card token must be exactly 2 chars, got {token!r}")
    return Card(rank=token[0], suit=token[1])


def parse_cards(text: str) -> HoleCards:
    """Parse two hole cards from a string like 'As Kd'."""
    parts = text.strip().split()
    if len(parts) != 2:
        raise ValueError(f"Expected exactly 2 card tokens, got {len(parts)}: {text!r}")
    c1 = parse_card(parts[0])
    c2 = parse_card(parts[1])
    return HoleCards(high=c1, low=c2)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

_BROADWAY_RANKS = frozenset("TJQKA")

# Simple high-card points: A=4, K=3, Q=2, J=1
_HCP = {"A": 4.0, "K": 3.0, "Q": 2.0, "J": 1.0, "T": 0.5}


def hand_features(hc: HoleCards) -> HandFeatures:
    """Compute all required hand features and classify into bucket."""
    rh = hc.high.rank_value
    rl = hc.low.rank_value
    is_pair = rh == rl
    is_suited = hc.high.suit == hc.low.suit
    gap = rh - rl - 1 if not is_pair else 0
    is_connector = (gap == 0 and not is_pair)
    is_one_gapper = (gap == 1 and not is_pair)
    is_wheel_ace = (hc.high.rank == "A" and hc.low.rank_value <= 5 and not is_pair)
    bc = sum(1 for c in (hc.high, hc.low) if c.rank in _BROADWAY_RANKS)
    hcp = _HCP.get(hc.high.rank, 0.0) + _HCP.get(hc.low.rank, 0.0)

    label = _hand_class_label(hc, is_pair, is_suited)
    bucket = _classify_bucket(hc, is_pair, is_suited, is_connector, is_one_gapper, gap, rh, rl)

    return HandFeatures(
        rank_high=rh,
        rank_low=rl,
        is_pair=is_pair,
        is_suited=is_suited,
        gap_size=gap,
        is_connector=is_connector,
        is_one_gapper=is_one_gapper,
        is_wheel_ace=is_wheel_ace,
        broadway_count=bc,
        high_card_points_simple=hcp,
        hand_class_label=label,
        hand_bucket=bucket,
    )


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def _hand_class_label(hc: HoleCards, is_pair: bool, is_suited: bool) -> str:
    if is_pair:
        return f"{hc.high.rank}{hc.low.rank}"
    suffix = "s" if is_suited else "o"
    return f"{hc.high.rank}{hc.low.rank}{suffix}"


# ---------------------------------------------------------------------------
# Bucket classification (A-N)
# ---------------------------------------------------------------------------

_SUITED_BIG_BROADWAY = {
    "AKs", "AQs", "AJs", "ATs",
    "KQs", "KJs", "KTs",
    "QJs", "QTs",
    "JTs",
}

_OFFSUIT_BIG_BROADWAY = {
    "AKo", "AQo", "AJo", "ATo",
    "KQo", "KJo",
    "QJo",
}

_SUITED_CONNECTORS = {"T9s", "98s", "87s", "76s", "65s", "54s"}

_SUITED_ONE_GAPPERS = {"J9s", "T8s", "97s", "86s", "75s", "64s", "53s"}

_SUITED_TWO_GAPPERS_TRASH = {
    "T7s", "96s", "85s", "74s", "63s", "52s", "43s", "42s", "32s",
}

_OFFSUIT_CONNECTORS_SEMI = {
    "T9o", "98o", "87o", "76o", "65o",
    "J9o", "T8o", "97o",
}


def _classify_bucket(
    hc: HoleCards,
    is_pair: bool,
    is_suited: bool,
    is_connector: bool,
    is_one_gapper: bool,
    gap: int,
    rh: int,
    rl: int,
) -> HandBucket:
    label = _hand_class_label(hc, is_pair, is_suited)

    # --- A: Pocket pairs ---
    if is_pair:
        if rh >= 12:  # QQ+
            return HandBucket.A_PREMIUM_PAIRS
        if rh >= 10:  # JJ-TT
            return HandBucket.A_HIGH_PAIRS
        if rh >= 7:   # 99-77
            return HandBucket.A_MID_PAIRS
        if rh >= 4:   # 66-44
            return HandBucket.A_LOW_PAIRS
        return HandBucket.A_MICRO_PAIRS  # 33-22

    # --- B: Suited big broadway ---
    if label in _SUITED_BIG_BROADWAY:
        return HandBucket.B_SUITED_BIG_BROADWAY

    # --- C: Offsuit big broadway ---
    if label in _OFFSUIT_BIG_BROADWAY:
        return HandBucket.C_OFFSUIT_BIG_BROADWAY

    # --- D: Suited Ax ---
    if is_suited and hc.high.rank == "A" and rl >= 5:
        return HandBucket.D_SUITED_AX_HIGH  # A9s-A5s
    if is_suited and hc.high.rank == "A" and rl < 5:
        return HandBucket.D_SUITED_AX_LOW   # A4s-A2s

    # --- E: Offsuit Ax ---
    if not is_suited and hc.high.rank == "A" and rl >= 5:
        return HandBucket.E_OFFSUIT_AX_HIGH  # A9o-A5o
    if not is_suited and hc.high.rank == "A" and rl < 5:
        return HandBucket.E_OFFSUIT_AX_LOW   # A4o-A2o

    # --- F: Suited kings ---
    if is_suited and hc.high.rank == "K":
        if rl >= 6:
            return HandBucket.F_SUITED_KX_HIGH  # K9s-K6s
        return HandBucket.F_SUITED_KX_LOW       # K5s-K2s

    # --- G: Offsuit kings ---
    if not is_suited and hc.high.rank == "K":
        if rl >= 8:
            return HandBucket.G_OFFSUIT_KX_HIGH  # KTo-K8o
        return HandBucket.G_OFFSUIT_KX_LOW       # K7o-K2o

    # --- H: Suited queens / jacks ---
    if is_suited and hc.high.rank == "Q":
        if rl >= 6:
            return HandBucket.H_SUITED_QJ_HIGH
        return HandBucket.H_SUITED_QJ_LOW
    if is_suited and hc.high.rank == "J":
        if rl >= 7:
            return HandBucket.H_SUITED_QJ_HIGH
        return HandBucket.H_SUITED_QJ_LOW

    # --- I: Offsuit queens / jacks ---
    if not is_suited and hc.high.rank == "Q":
        if rl >= 8:
            return HandBucket.I_OFFSUIT_QJ_HIGH
        return HandBucket.I_OFFSUIT_QJ_LOW
    if not is_suited and hc.high.rank == "J":
        if rl >= 8:
            return HandBucket.I_OFFSUIT_QJ_HIGH
        return HandBucket.I_OFFSUIT_QJ_LOW

    # --- J: Suited connectors ---
    if label in _SUITED_CONNECTORS:
        return HandBucket.J_SUITED_CONNECTORS

    # --- K: Suited one-gappers ---
    if label in _SUITED_ONE_GAPPERS:
        return HandBucket.K_SUITED_ONE_GAPPERS

    # --- L: Suited two-gappers / wheel-ish suited trash ---
    if label in _SUITED_TWO_GAPPERS_TRASH:
        return HandBucket.L_SUITED_TWO_GAPPERS_TRASH
    # Catch remaining suited hands not yet classified
    if is_suited:
        return HandBucket.L_SUITED_TWO_GAPPERS_TRASH

    # --- M: Offsuit connectors / semi-connectors ---
    if label in _OFFSUIT_CONNECTORS_SEMI:
        return HandBucket.M_OFFSUIT_CONNECTORS

    # --- N: Everything else (weak offsuit trash) ---
    return HandBucket.N_WEAK_OFFSUIT_TRASH
