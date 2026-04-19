"""Pure helpers for Streamlit play lab (testable without running Streamlit)."""

from __future__ import annotations

from typing import Optional, Sequence

from poker_core.models import Card, HandConfig, HoleCards

from play_lab.coordinator import hole_cards_spaced

# Unicode suits; hearts/diamonds red, clubs/spades black (standard US deck).
_SUIT_SYMBOL = {"h": "\u2665", "d": "\u2666", "c": "\u2663", "s": "\u2660"}
_RED_SUITS = frozenset({"h", "d"})


def card_colored_html(card: Card) -> str:
    """Single card as HTML span (red ♥/♦, black ♣/♠)."""
    suit = card.suit.lower()
    sym = _SUIT_SYMBOL.get(suit, card.suit)
    color = "#c62828" if suit in _RED_SUITS else "#212121"
    return (
        f'<span style="color:{color};font-weight:600;font-family:ui-monospace,'
        f'monospace;white-space:nowrap">{card.rank}{sym}</span>'
    )


def hole_cards_colored_html(hc: HoleCards, *, sep: str = "&nbsp;") -> str:
    """Two hole cards with suit colors."""
    return sep.join([card_colored_html(hc.high), card_colored_html(hc.low)])


def board_cards_colored_html(cards: Sequence[Card]) -> str:
    """Board cards joined with spaces."""
    if not cards:
        return "—"
    return " ".join(card_colored_html(c) for c in cards)


def format_engine_villain_banner(cfg: HandConfig, *, hide_hero_holes: bool) -> str:
    """One-line hole summary for the table; mask engine cards when ``hide_hero_holes``."""
    vhc = cfg.villain_hole_cards
    villain_s = hole_cards_spaced(vhc) if vhc else "?"
    if hide_hero_holes:
        hero_s = "(hidden)"
    else:
        hero_s = hole_cards_spaced(cfg.hero_hole_cards)
    return f"Engine cards: **{hero_s}** | Yours: **{villain_s}**"


def format_engine_villain_banner_html(cfg: HandConfig, *, hide_hero_holes: bool) -> str:
    """Same as ``format_engine_villain_banner`` but with colored suits (use ``unsafe_allow_html``)."""
    vhc = cfg.villain_hole_cards
    villain_h = hole_cards_colored_html(vhc) if vhc else "?"
    if hide_hero_holes:
        hero_h = "(hidden)"
    else:
        hero_h = hole_cards_colored_html(cfg.hero_hole_cards)
    return (
        "<p><strong>Engine cards:</strong> "
        f"{hero_h} &nbsp;|&nbsp; <strong>Yours:</strong> {villain_h}</p>"
    )


def last_hand_settings_complete(settings: Optional[dict]) -> bool:
    """True if we can start a one-click next hand from saved settings."""
    if not settings:
        return False
    need = ("hero_position", "stack", "randomize")
    if not all(k in settings for k in need):
        return False
    if settings.get("randomize"):
        return True
    return "hero_txt" in settings and "vil_txt" in settings


def hero_villain_stacks_from_last_settings(settings: dict) -> tuple[float, float]:
    """Return ``(hero_start_bb, villain_start_bb)`` from saved last-hand dict."""
    base = float(settings["stack"])
    h = float(settings.get("hero_stack_bb", base))
    v = float(settings.get("villain_stack_bb", base))
    return h, v
