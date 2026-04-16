"""Showdown rank + HU winner (Play Lab)."""

from __future__ import annotations

from poker_core.models import HoleCards
from poker_core.parser import parse_card

from flop_equity.monte_carlo import best_hand_rank_seven_cards
from play_lab.showdown_display import ShowdownWinner, hu_showdown_result


def test_best_hand_rank_seven_requires_seven() -> None:
    c = [parse_card("As"), parse_card("Ks")]
    try:
        best_hand_rank_seven_cards(c)
    except ValueError as e:
        assert "7" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_hu_showdown_hero_wins_with_better_kicker() -> None:
    hero = HoleCards(parse_card("Ah"), parse_card("Kd"))
    vill = HoleCards(parse_card("As"), parse_card("Qc"))
    board = [parse_card("2d"), parse_card("3h"), parse_card("4c"), parse_card("9s"), parse_card("Td")]
    r = hu_showdown_result(hero, vill, board)
    assert r is not None
    assert r.winner == ShowdownWinner.HERO
    assert r.hero_rank > r.villain_rank


def test_hu_showdown_split_on_identical_strength() -> None:
    hero = HoleCards(parse_card("Ah"), parse_card("Kd"))
    vill = HoleCards(parse_card("As"), parse_card("Kc"))
    board = [parse_card("2d"), parse_card("3h"), parse_card("4c"), parse_card("9s"), parse_card("Td")]
    r = hu_showdown_result(hero, vill, board)
    assert r is not None
    assert r.winner == ShowdownWinner.SPLIT
