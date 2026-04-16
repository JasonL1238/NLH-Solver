"""Tests for ``play_lab.ui_helpers``."""

from __future__ import annotations

from poker_core.parser import make_hand_config, parse_card

from play_lab.ui_helpers import (
    card_colored_html,
    format_engine_villain_banner,
    format_engine_villain_banner_html,
    last_hand_settings_complete,
)


def test_banner_shows_hero_when_not_hidden() -> None:
    cfg = make_hand_config("As Kd", "BTN_SB", 100.0, villain_cards="Qc Jd")
    s = format_engine_villain_banner(cfg, hide_hero_holes=False)
    assert "As" in s and "Kd" in s
    assert "Qc" in s and "Jd" in s
    assert "(hidden)" not in s


def test_banner_hides_hero() -> None:
    cfg = make_hand_config("As Kd", "BTN_SB", 100.0, villain_cards="Qc Jd")
    s = format_engine_villain_banner(cfg, hide_hero_holes=True)
    assert "(hidden)" in s
    assert "As" not in s


def test_card_colored_html_red_for_hearts() -> None:
    h = parse_card("Ah")
    html = card_colored_html(h)
    assert "#c62828" in html or "c62828" in html
    assert "\u2665" in html  # ♥


def test_card_colored_html_black_for_spades() -> None:
    s = parse_card("As")
    html = card_colored_html(s)
    assert "#212121" in html
    assert "\u2660" in html  # ♠


def test_banner_html_shows_colored_suits() -> None:
    cfg = make_hand_config("As Kd", "BTN_SB", 100.0, villain_cards="Qc Jd")
    html = format_engine_villain_banner_html(cfg, hide_hero_holes=False)
    assert "\u2660" in html  # spade black
    assert "\u2666" in html  # diamond red
    assert "\u2663" in html  # club


def test_last_hand_settings_complete() -> None:
    assert not last_hand_settings_complete(None)
    assert not last_hand_settings_complete({"hero_position": "BTN_SB"})
    assert last_hand_settings_complete(
        {"hero_position": "BTN_SB", "stack": 100.0, "randomize": True}
    )
    assert not last_hand_settings_complete(
        {"hero_position": "BTN_SB", "stack": 100.0, "randomize": False}
    )
    assert last_hand_settings_complete(
        {
            "hero_position": "BTN_SB",
            "stack": 100.0,
            "randomize": False,
            "hero_txt": "As Kd",
            "vil_txt": "Qc Jd",
        }
    )
