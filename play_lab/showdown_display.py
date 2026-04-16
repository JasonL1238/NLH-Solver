"""Heads-up showdown display helpers (Play Lab UI; display-only)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List

from flop_equity.monte_carlo import best_hand_rank_seven_cards
from poker_core.models import Card, HoleCards

from play_lab.ui_helpers import board_cards_colored_html, hole_cards_colored_html


class ShowdownWinner(Enum):
    HERO = "HERO"
    VILLAIN = "VILLAIN"
    SPLIT = "SPLIT"


_CATEGORY_NAMES = (
    "High card",
    "One pair",
    "Two pair",
    "Three of a kind",
    "Straight",
    "Flush",
    "Full house",
    "Four of a kind",
    "Straight flush",
)


@dataclass(frozen=True)
class ShowdownResult:
    winner: ShowdownWinner
    hero_rank: tuple
    villain_rank: tuple
    hero_hand_label: str
    villain_hand_label: str

    def summary_for_villain_perspective(self) -> str:
        if self.winner == ShowdownWinner.SPLIT:
            return "Chop — same best five-card hand."
        if self.winner == ShowdownWinner.VILLAIN:
            return "You win the pot at showdown."
        return "Engine (HERO) wins the pot at showdown."


def _label_from_rank(rank: tuple) -> str:
    if not rank:
        return "?"
    cat = int(rank[0])
    name = _CATEGORY_NAMES[cat] if 0 <= cat < len(_CATEGORY_NAMES) else f"Cat{cat}"
    return f"{name} ({', '.join(str(x) for x in rank)})"


def hu_showdown_result(
    hero: HoleCards,
    villain: HoleCards,
    board: List[Card],
) -> Optional[ShowdownResult]:
    """Compare best 5-of-7 for each player; return ``None`` if board is not five cards."""
    if len(board) != 5:
        return None
    hero_cards = [hero.high, hero.low, *board]
    vill_cards = [villain.high, villain.low, *board]
    hr = best_hand_rank_seven_cards(hero_cards)
    vr = best_hand_rank_seven_cards(vill_cards)
    if hr > vr:
        win = ShowdownWinner.HERO
    elif vr > hr:
        win = ShowdownWinner.VILLAIN
    else:
        win = ShowdownWinner.SPLIT
    return ShowdownResult(
        winner=win,
        hero_rank=hr,
        villain_rank=vr,
        hero_hand_label=_label_from_rank(hr),
        villain_hand_label=_label_from_rank(vr),
    )


def hole_cards_spaced(hc: HoleCards) -> str:
    return f"{hc.high!s} {hc.low!s}"


def format_showdown_block(
    *,
    hero: HoleCards,
    villain: HoleCards,
    board: List[Card],
    hero_player_label: str = "HERO (engine)",
    villain_player_label: str = "VILLAIN (you)",
) -> str:
    """Markdown-ish text block for Streamlit."""
    res = hu_showdown_result(hero, villain, board)
    if res is None:
        return "_Showdown needs a 5-card board._"
    lines = [
        f"**Board:** {' '.join(f'{c!s}' for c in board)}",
        f"- **{hero_player_label}:** {hole_cards_spaced(hero)} → _{res.hero_hand_label}_",
        f"- **{villain_player_label}:** {hole_cards_spaced(villain)} → _{res.villain_hand_label}_",
        "",
        f"**Result:** {res.summary_for_villain_perspective()}",
    ]
    return "\n".join(lines)


def format_showdown_block_html(
    *,
    hero: HoleCards,
    villain: HoleCards,
    board: List[Card],
    hero_player_label: str = "HERO (engine)",
    villain_player_label: str = "VILLAIN (you)",
) -> str:
    """HTML fragment for Streamlit ``unsafe_allow_html`` (colored cards)."""
    res = hu_showdown_result(hero, villain, board)
    if res is None:
        return "<p><em>Showdown needs a 5-card board.</em></p>"
    brd = board_cards_colored_html(board)
    h_h = hole_cards_colored_html(hero, sep=" ")
    v_h = hole_cards_colored_html(villain, sep=" ")
    return (
        "<p><strong>Board:</strong> "
        f"{brd}</p><ul>"
        f"<li><strong>{hero_player_label}:</strong> {h_h} → <em>{res.hero_hand_label}</em></li>"
        f"<li><strong>{villain_player_label}:</strong> {v_h} → <em>{res.villain_hand_label}</em></li>"
        "</ul>"
        f"<p><strong>Result:</strong> {res.summary_for_villain_perspective()}</p>"
    )
