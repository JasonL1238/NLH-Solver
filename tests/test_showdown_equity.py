"""Unit tests for ``estimate_showdown_equity`` (turn MC, river exact)."""

from __future__ import annotations

import pytest

from flop_equity.monte_carlo import estimate_flop_equity, estimate_showdown_equity
from poker_core.models import Card, HoleCards, Street


def _hc(s: str) -> HoleCards:
    """Two cards like ``AsKh``."""
    assert len(s) == 4
    return HoleCards(
        high=Card(rank=s[0], suit=s[1].lower()),
        low=Card(rank=s[2], suit=s[3].lower()),
    )


def _board(*sp: str) -> list[Card]:
    return [Card(rank=p[0], suit=p[1].lower()) for p in sp]


def test_river_exact_equity_vs_single_combo() -> None:
    """Hero nut flush vs one pair on a fixed river board (no shared Broadway)."""
    hero = _hc("AcJc")
    board = _board("Kc", "Qc", "9c", "2d", "3s")
    villain = _hc("4h4d")
    out = estimate_showdown_equity(
        hero, board, [(villain, 1.0)], street=Street.RIVER, samples=1, seed=0,
    )
    assert out["equity_estimate"] == 1.0
    assert out["win_rate"] == 1.0
    assert out["tie_rate"] == 0.0
    assert out["samples_used"] == 1


def test_river_exact_split_pot() -> None:
    """Identical best five-card hands → weighted tie equity 0.5."""
    hero = _hc("AsKs")
    villain = _hc("AdKh")
    board = _board("Qh", "Jc", "Td", "2c", "3s")
    out = estimate_showdown_equity(
        hero, board, [(villain, 1.0)], street=Street.RIVER,
    )
    assert out["equity_estimate"] == 0.5
    assert out["tie_rate"] == 1.0
    assert out["samples_used"] == 1


def test_flop_branch_matches_estimate_flop_equity() -> None:
    hero = _hc("AsKh")
    board = _board("Qd", "Jc", "Td")
    rng = [(_hc("2c3d"), 1.0)]
    a = estimate_flop_equity(hero, board, rng, samples=200, seed=7)
    b = estimate_showdown_equity(
        hero, board, rng, street=Street.FLOP, samples=200, seed=7,
    )
    assert a == b


def test_turn_equity_deterministic_with_seed() -> None:
    hero = _hc("AsAh")
    board = _board("Ad", "Kh", "Qc", "2d")
    rng = [(_hc("KcKd"), 0.5), (_hc("2h3h"), 0.5)]
    a = estimate_showdown_equity(
        hero, board, rng, street=Street.TURN, samples=500, seed=123,
    )
    b = estimate_showdown_equity(
        hero, board, rng, street=Street.TURN, samples=500, seed=123,
    )
    assert a == b
    assert a["samples_used"] == 500


def test_showdown_equity_rejects_wrong_board_length() -> None:
    hero = _hc("AsAh")
    board = _board("Ad", "Kh", "Qc")
    with pytest.raises(ValueError, match="TURN"):
        estimate_showdown_equity(
            hero, board, [(_hc("2c3d"), 1.0)], street=Street.TURN,
        )
