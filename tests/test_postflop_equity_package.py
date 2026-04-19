"""Package init must not eagerly import integration (avoids ev_recommender cycles)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_postflop_equity_lazy_integration_subprocess() -> None:
    code = r"""
import sys
for k in list(sys.modules):
    if k == "postflop_policy.ev_recommender" or k.startswith("postflop_equity"):
        del sys.modules[k]
import postflop_equity
assert "postflop_equity.integration" not in sys.modules
_ = postflop_equity.recommend_turn_action_with_equity
assert "postflop_equity.integration" in sys.modules
print("ok")
"""
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(root),
    )
    assert r.returncode == 0, r.stderr + r.stdout
    assert "ok" in r.stdout
