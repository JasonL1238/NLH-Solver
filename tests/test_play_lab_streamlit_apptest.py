"""Programmatic smoke tests for the Streamlit Play Lab (``AppTest``).

Manual QA still covers full widget flows; these tests catch import/runtime
errors on ``main()`` and a minimal in-app navigation path.
"""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


def _button_labels(at: AppTest) -> list[str]:
    return [str(b.label) for b in at.button]


def test_play_lab_streamlit_idle_and_post_blinds_apptest() -> None:
    at = AppTest.from_file("play_lab/streamlit_app.py", default_timeout=120)
    at.run(timeout=120)
    assert len(at.exception) == 0
    assert any("Play lab" in t.value for t in at.title)

    post = next(b for b in at.button if str(b.label) == "Post blinds & start")
    post.click().run(timeout=120)
    assert len(at.exception) == 0
    assert "Apply engine preflop action" in _button_labels(at)

    apply = next(b for b in at.button if "Apply engine preflop" in str(b.label))
    apply.click().run(timeout=120)
    assert len(at.exception) == 0


def test_play_lab_sidebar_logic_flags_rerun() -> None:
    at = AppTest.from_file("play_lab/streamlit_app.py", default_timeout=120)
    at.run(timeout=120)
    at.session_state["play_lab_show_engine_logic"] = True
    at.session_state["play_lab_hide_hero_holes"] = True
    at.run(timeout=120)
    assert len(at.exception) == 0
