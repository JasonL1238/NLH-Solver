"""Simple explicit villain range model for flop spots.

Derives an initial range from the preflop action line, then optionally
refines weights based on the villain's flop action (bet vs check).

Range output is a list of ``(HoleCards, weight)`` tuples expanded to
concrete 2-card combos with dead-card filtering applied.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Tuple

from poker_core.models import (
    ActionType,
    Card,
    HandState,
    HoleCards,
    Player,
    Position,
    RANKS,
    SUITS,
)


# ---------------------------------------------------------------------------
# Hand-class definitions (rank-pair strings -> weight)
# ---------------------------------------------------------------------------

# Format: "AKs" means suited, "AKo" means offsuit, "AA" means pair.
# Weights are baseline frequencies (0.0–1.0).

_PREMIUM_PAIRS: Dict[str, float] = {
    "AA": 1.0, "KK": 1.0, "QQ": 1.0, "JJ": 1.0,
}

_MEDIUM_PAIRS: Dict[str, float] = {
    "TT": 1.0, "99": 1.0, "88": 1.0, "77": 1.0, "66": 1.0,
}

_SMALL_PAIRS: Dict[str, float] = {
    "55": 1.0, "44": 1.0, "33": 1.0, "22": 1.0,
}

_BROADWAYS_SUITED: Dict[str, float] = {
    "AKs": 1.0, "AQs": 1.0, "AJs": 1.0, "ATs": 1.0,
    "KQs": 1.0, "KJs": 1.0, "KTs": 1.0,
    "QJs": 1.0, "QTs": 1.0,
    "JTs": 1.0,
}

_BROADWAYS_OFFSUIT: Dict[str, float] = {
    "AKo": 1.0, "AQo": 1.0, "AJo": 1.0, "ATo": 0.8,
    "KQo": 1.0, "KJo": 0.8, "KTo": 0.5,
    "QJo": 0.8, "QTo": 0.5,
    "JTo": 0.5,
}

_SUITED_ACES_LOW: Dict[str, float] = {
    "A9s": 0.8, "A8s": 0.7, "A7s": 0.7, "A6s": 0.6,
    "A5s": 0.8, "A4s": 0.7, "A3s": 0.6, "A2s": 0.6,
}

_SUITED_CONNECTORS: Dict[str, float] = {
    "T9s": 1.0, "98s": 1.0, "87s": 1.0, "76s": 1.0,
    "65s": 0.8, "54s": 0.7,
}

_SUITED_GAPPERS: Dict[str, float] = {
    "T8s": 0.5, "97s": 0.5, "86s": 0.5, "75s": 0.5, "64s": 0.4,
}

_BB_DEFEND_WEAK: Dict[str, float] = {
    "K9o": 0.4, "K8o": 0.3, "Q9o": 0.4, "J9o": 0.4,
    "T9o": 0.5, "98o": 0.4, "87o": 0.3, "76o": 0.3,
    "K9s": 0.7, "Q9s": 0.6, "J9s": 0.6, "T8s": 0.5,
    "97s": 0.5, "86s": 0.4, "75s": 0.4,
    "K8s": 0.4, "K7s": 0.3, "K6s": 0.3,
    "Q8s": 0.3, "J8s": 0.3,
    "53s": 0.3, "43s": 0.3,
}


# ---------------------------------------------------------------------------
# Preflop line categories
# ---------------------------------------------------------------------------

class _PreflopLine:
    BTN_OPEN_BB_CALL = "BTN_OPEN_BB_CALL"
    BTN_OPEN_BB_3BET_BTN_CALL = "BTN_OPEN_BB_3BET_BTN_CALL"
    BTN_LIMP_BB_CHECK = "BTN_LIMP_BB_CHECK"
    BTN_LIMP_BB_RAISE_BTN_CALL = "BTN_LIMP_BB_RAISE_BTN_CALL"
    UNKNOWN = "UNKNOWN"


def _detect_preflop_line(state: HandState) -> Tuple[str, Player]:
    """Parse action_history to determine the preflop line and who villain is.

    Returns (line_label, villain_player).
    """
    villain = (Player.VILLAIN)
    preflop_actions = []
    for a in state.action_history:
        if a.action_type == ActionType.DEAL_FLOP:
            break
        if a.action_type == ActionType.POST_BLIND:
            continue
        preflop_actions.append(a)

    btn_player = state.config.btn_player
    bb_player = state.config.bb_player

    types = [(a.action_type, a.player) for a in preflop_actions]

    # BTN open (RAISE) -> BB call
    if (len(types) == 2
            and types[0] == (ActionType.RAISE, btn_player)
            and types[1] == (ActionType.CALL, bb_player)):
        return _PreflopLine.BTN_OPEN_BB_CALL, villain

    # BTN open -> BB 3-bet -> BTN call
    if (len(types) == 3
            and types[0] == (ActionType.RAISE, btn_player)
            and types[1] == (ActionType.RAISE, bb_player)
            and types[2] == (ActionType.CALL, btn_player)):
        return _PreflopLine.BTN_OPEN_BB_3BET_BTN_CALL, villain

    # BTN limp (CALL) -> BB check
    if (len(types) == 2
            and types[0] == (ActionType.CALL, btn_player)
            and types[1] == (ActionType.CHECK, bb_player)):
        return _PreflopLine.BTN_LIMP_BB_CHECK, villain

    # BTN limp -> BB iso-raise (BET since to_call=0) -> BTN call
    if (len(types) == 3
            and types[0] == (ActionType.CALL, btn_player)
            and types[1] == (ActionType.BET, bb_player)
            and types[2] == (ActionType.CALL, btn_player)):
        return _PreflopLine.BTN_LIMP_BB_RAISE_BTN_CALL, villain

    return _PreflopLine.UNKNOWN, villain


# ---------------------------------------------------------------------------
# Range composition per line
# ---------------------------------------------------------------------------

def _range_for_line(
    line: str,
    villain: Player,
    config_villain_pos: Position,
) -> Tuple[Dict[str, float], str]:
    """Return (hand_label -> weight dict, summary string)."""

    if line == _PreflopLine.BTN_OPEN_BB_CALL:
        if config_villain_pos == Position.BB:
            # Villain is BB defending vs BTN open: wide defend range
            r: Dict[str, float] = {}
            r.update(_PREMIUM_PAIRS)
            r.update(_MEDIUM_PAIRS)
            r.update(_SMALL_PAIRS)
            r.update(_BROADWAYS_SUITED)
            r.update(_BROADWAYS_OFFSUIT)
            r.update(_SUITED_ACES_LOW)
            r.update(_SUITED_CONNECTORS)
            r.update(_SUITED_GAPPERS)
            r.update(_BB_DEFEND_WEAK)
            return r, "BB defend vs BTN open (wide)"
        else:
            # Villain is BTN who opened: wide open range
            r = {}
            r.update(_PREMIUM_PAIRS)
            r.update(_MEDIUM_PAIRS)
            r.update(_SMALL_PAIRS)
            r.update(_BROADWAYS_SUITED)
            r.update(_BROADWAYS_OFFSUIT)
            r.update(_SUITED_ACES_LOW)
            r.update(_SUITED_CONNECTORS)
            r.update(_SUITED_GAPPERS)
            return r, "BTN open range (wide)"

    if line == _PreflopLine.BTN_OPEN_BB_3BET_BTN_CALL:
        if config_villain_pos == Position.BTN_SB:
            # Villain is BTN who called a 3-bet: strong/medium range
            r = {}
            r.update(_PREMIUM_PAIRS)
            r.update(_MEDIUM_PAIRS)
            r.update(_BROADWAYS_SUITED)
            for k in ("AKo", "AQo", "KQo"):
                r[k] = 1.0
            r.update(_SUITED_CONNECTORS)
            return r, "BTN call vs BB 3-bet (medium-strong)"
        else:
            # Villain is BB who 3-bet: polarized range
            r = {}
            r.update(_PREMIUM_PAIRS)
            for k in ("AKs", "AQs", "AKo"):
                r[k] = 1.0
            # Bluff component
            for k in ("A5s", "A4s", "A3s", "A2s"):
                r[k] = 0.6
            for k in ("76s", "65s", "54s"):
                r[k] = 0.4
            return r, "BB 3-bet range (polarized)"

    if line == _PreflopLine.BTN_LIMP_BB_CHECK:
        if config_villain_pos == Position.BTN_SB:
            # Villain is BTN who limped: weak/trapping range
            r = {}
            r.update({k: 0.3 for k in _PREMIUM_PAIRS})  # traps
            r.update(_SMALL_PAIRS)
            r.update(_SUITED_CONNECTORS)
            r.update(_SUITED_GAPPERS)
            r.update(_SUITED_ACES_LOW)
            r.update({k: 0.5 for k in _BROADWAYS_OFFSUIT})
            return r, "BTN limp range (weak/traps)"
        else:
            # Villain is BB who checked behind limp: everything
            r = {}
            r.update(_PREMIUM_PAIRS)
            r.update(_MEDIUM_PAIRS)
            r.update(_SMALL_PAIRS)
            r.update(_BROADWAYS_SUITED)
            r.update(_BROADWAYS_OFFSUIT)
            r.update(_SUITED_ACES_LOW)
            r.update(_SUITED_CONNECTORS)
            r.update(_SUITED_GAPPERS)
            r.update(_BB_DEFEND_WEAK)
            return r, "BB check behind limp (wide)"

    if line == _PreflopLine.BTN_LIMP_BB_RAISE_BTN_CALL:
        if config_villain_pos == Position.BTN_SB:
            # Villain is BTN who called the iso-raise: medium range
            r = {}
            r.update(_MEDIUM_PAIRS)
            r.update(_SMALL_PAIRS)
            r.update(_SUITED_CONNECTORS)
            r.update(_SUITED_ACES_LOW)
            for k in ("AKo", "AQo", "KQo"):
                r[k] = 0.8
            return r, "BTN call vs BB iso-raise (medium)"
        else:
            # Villain is BB who iso-raised a limp: value-heavy
            r = {}
            r.update(_PREMIUM_PAIRS)
            r.update(_MEDIUM_PAIRS)
            r.update(_BROADWAYS_SUITED)
            for k in ("AKo", "AQo", "AJo", "KQo"):
                r[k] = 1.0
            return r, "BB iso-raise vs limp (value-heavy)"

    # Fallback: generic medium range
    r = {}
    r.update(_PREMIUM_PAIRS)
    r.update(_MEDIUM_PAIRS)
    r.update(_SMALL_PAIRS)
    r.update(_BROADWAYS_SUITED)
    r.update(_BROADWAYS_OFFSUIT)
    r.update(_SUITED_ACES_LOW)
    r.update(_SUITED_CONNECTORS)
    return r, "Unknown preflop line (generic range)"


# ---------------------------------------------------------------------------
# Flop action refinement
# ---------------------------------------------------------------------------

def _detect_villain_flop_action(state: HandState) -> Optional[str]:
    """Return 'bet', 'check', or None based on villain's flop action so far."""
    saw_flop = False
    for a in state.action_history:
        if a.action_type == ActionType.DEAL_FLOP:
            saw_flop = True
            continue
        if not saw_flop:
            continue
        if a.player == Player.VILLAIN:
            if a.action_type == ActionType.BET:
                return "bet"
            if a.action_type == ActionType.RAISE:
                return "bet"
            if a.action_type == ActionType.CHECK:
                return "check"
    return None


_VALUE_DRAW_LABELS = frozenset(
    list(_PREMIUM_PAIRS) + list(_MEDIUM_PAIRS)
    + list(_BROADWAYS_SUITED) + ["AKo", "AQo", "KQo"]
    + list(_SUITED_CONNECTORS) + list(_SUITED_GAPPERS)
)

_AIR_LABELS = frozenset(
    list(_BB_DEFEND_WEAK)
    + ["K8o", "K7o", "Q8o", "J8o", "T8o", "97o", "86o", "75o"]
)


def _refine_for_flop_action(
    weights: Dict[str, float],
    flop_action: Optional[str],
) -> Dict[str, float]:
    """Shift weights based on villain's flop action."""
    if flop_action is None:
        return weights

    refined = dict(weights)

    if flop_action == "bet":
        for label in refined:
            if label in _AIR_LABELS:
                refined[label] = refined[label] * 0.4
    elif flop_action == "check":
        for label in refined:
            if label in _PREMIUM_PAIRS:
                refined[label] = refined[label] * 0.5

    return refined


# ---------------------------------------------------------------------------
# Combo expansion (label -> concrete HoleCards)
# ---------------------------------------------------------------------------

_RANK_CHAR_ORDER = list(RANKS)  # '2' through 'A'
_ALL_CARDS = [Card(rank=r, suit=s) for r in RANKS for s in SUITS]


def expand_label_to_combos(label: str) -> List[HoleCards]:
    """Expand a hand label like 'AKs', 'AKo', or 'AA' to concrete combos."""
    if len(label) == 2:
        # Pair: e.g. "AA"
        r = label[0]
        cards_of_rank = [c for c in _ALL_CARDS if c.rank == r]
        return [HoleCards(high=a, low=b) for a, b in combinations(cards_of_rank, 2)]

    r1, r2, suitedness = label[0], label[1], label[2]

    cards_r1 = [c for c in _ALL_CARDS if c.rank == r1]
    cards_r2 = [c for c in _ALL_CARDS if c.rank == r2]

    combos = []
    for c1 in cards_r1:
        for c2 in cards_r2:
            if c1 == c2:
                continue
            if suitedness == "s" and c1.suit != c2.suit:
                continue
            if suitedness == "o" and c1.suit == c2.suit:
                continue
            combos.append(HoleCards(high=c1, low=c2))

    # Deduplicate (HoleCards auto-sorts high/low)
    seen = set()
    unique = []
    for hc in combos:
        key = (hc.high, hc.low)
        if key not in seen:
            seen.add(key)
            unique.append(hc)
    return unique


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_preflop_line_key(state: HandState) -> str:
    """Return the detected preflop line id (same labels as internal ``_PreflopLine``)."""
    line, _ = _detect_preflop_line(state)
    return line


def prior_label_weights(state: HandState) -> Tuple[Dict[str, float], str]:
    """Coarse label→weight prior for villain from preflop line only (no flop refinement).

    Used by particle-based postflop range initialization.
    """
    line, villain = _detect_preflop_line(state)
    villain_pos = state.config.position_for_player(Player.VILLAIN)
    return _range_for_line(line, villain, villain_pos)


def dead_cards_frozen(state: HandState) -> frozenset:
    """Hero holes plus all board cards (canonical dead set)."""
    return _dead_cards(state)


def build_villain_flop_range(
    state: HandState,
) -> Tuple[List[Tuple[HoleCards, float]], str]:
    """Build a villain range estimate for the current flop state.

    Returns (list of (HoleCards, weight) combos, summary string).
    Dead cards (hero hole cards + board) are excluded.
    """
    line, villain = _detect_preflop_line(state)
    villain_pos = state.config.position_for_player(Player.VILLAIN)
    label_weights, summary = _range_for_line(line, villain, villain_pos)

    flop_action = _detect_villain_flop_action(state)
    label_weights = _refine_for_flop_action(label_weights, flop_action)

    if flop_action == "bet":
        summary += ", bet on flop (weighted toward value/draws)"
    elif flop_action == "check":
        summary += ", checked flop (weighted toward medium/give-up)"

    dead = _dead_cards(state)
    expanded = _expand_and_filter(label_weights, dead)

    return expanded, summary


def villain_flop_range_debug_lines(state: HandState) -> List[str]:
    """Human-readable steps for how :func:`build_villain_flop_range` narrows weights.

    Intended for UI / traces (not a second source of truth — mirrors the builder).
    """
    try:
        line, villain = _detect_preflop_line(state)
        villain_pos = state.config.position_for_player(Player.VILLAIN)
        label_weights, summary = _range_for_line(line, villain, villain_pos)
        flop_action = _detect_villain_flop_action(state)
        refined = _refine_for_flop_action(dict(label_weights), flop_action)
        dead = _dead_cards(state)
        expanded = _expand_and_filter(refined, dead)
        tw = sum(w for _hc, w in expanded)
        n_labels = sum(1 for w in refined.values() if w > 1e-9)
        lines = [
            f"Detected **preflop line**: `{line}` (labels are for **VILLAIN** holding).",
            f"**Starting prior:** {summary} — **{n_labels}** non-empty hand-class buckets.",
        ]
        if flop_action == "bet":
            lines.append(
                "**Flop signal:** villain **bet** → downweight air labels (×0.4) before expansion."
            )
        elif flop_action == "check":
            lines.append(
                "**Flop signal:** villain **check** → downweight premium pairs (×0.5) before expansion."
            )
        else:
            lines.append(
                "**Flop signal:** no separate villain bet/check line yet "
                "(e.g. first to act on flop or only blind money so far)."
            )
        lines.append(
            f"**Blockers:** remove hero hole cards + **{len(state.board_cards)}** board cards, "
            f"then expand labels to **{len(expanded)}** alive weighted combos."
        )
        lines.append(f"**Alive weight sum** (relative, not normalized to 1): **{tw:.3f}**.")
        return lines
    except (AttributeError, TypeError):
        return [
            "_Villain range narrowing needs a full engine `HandState` "
            "(`action_history`, `config`, board)._"
        ]


def _dead_cards(state: HandState) -> frozenset:
    hero = state.config.hero_hole_cards
    return frozenset({hero.high, hero.low} | set(state.board_cards))


def _expand_and_filter(
    label_weights: Dict[str, float],
    dead: frozenset,
) -> List[Tuple[HoleCards, float]]:
    """Expand labels to combos, filter dead cards, attach weights."""
    result = []
    for label, weight in label_weights.items():
        if weight < 1e-9:
            continue
        combos = expand_label_to_combos(label)
        for hc in combos:
            if hc.high in dead or hc.low in dead:
                continue
            result.append((hc, weight))
    return result
