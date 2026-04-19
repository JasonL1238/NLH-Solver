"""Compact decision-flow text + Mermaid for Play Lab (flop EV debug)."""

from __future__ import annotations

from typing import Any, Dict, List


def _short(s: Any, max_len: int = 40) -> str:
    t = str(s if s is not None else "—").replace('"', "'").replace("\n", " ").strip()
    return t[:max_len] + ("…" if len(t) > max_len else "")


def flop_decision_flow_bullets(dbg: Dict[str, Any]) -> List[str]:
    """Ordered steps for the EV-first flop pipeline (plain text)."""
    lines = [
        "1. **Spot** — classify flop context (roles, SPR, facing bet bucket).",
        "2. **Hand buckets** — hero made-hand + draw tags vs the board.",
        (
            "3. **Villain range** — preflop line → label prior → flop-line reweight "
            "→ expand combos, drop blockers (hero + board)."
        ),
        (
            f"4. **Monte Carlo equity** — hero vs that weighted range "
            f"(eq ≈ {_short(dbg.get('equity_estimate'))})."
        ),
    ]
    facing = dbg.get("pot_odds_threshold")
    if facing is not None:
        lines.append(
            f"5. **Pot-odds / thin-raise gate** — threshold {_short(facing)}; "
            f"branch `{_short(dbg.get('thin_raise_policy_branch'))}`."
        )
    else:
        lines.append(
            f"5. **Thin-raise gate** — branch `{_short(dbg.get('thin_raise_policy_branch'))}` "
            "(no facing bet / no call price)."
        )
    lines.append(
        "6. **EV grid** — discrete check/call/fold + bet/raise sizes; simple villain response model."
    )
    lines.append(
        f"7. **Argmax** — pick `{_short(dbg.get('final_action') or dbg.get('recommended_action'))}` "
        f"(best EV ≈ {_short(dbg.get('ev_best'))} bb)."
    )
    return lines


def _mermaid_safe(s: str) -> str:
    for ch in "[](){}|&<>#":
        s = s.replace(ch, " ")
    return " ".join(s.split())[:48]


def flop_decision_flow_mermaid(dbg: Dict[str, Any]) -> str:
    """Mermaid flowchart (keep labels short for Streamlit renderer)."""
    eq = _mermaid_safe(_short(dbg.get("equity_estimate"), 12))
    thr = _mermaid_safe(_short(dbg.get("pot_odds_threshold"), 10))
    thin = _mermaid_safe(_short(dbg.get("thin_raise_policy_branch"), 24))
    pick = _mermaid_safe(_short(dbg.get("final_action") or dbg.get("recommended_action"), 28))
    best = _mermaid_safe(_short(dbg.get("ev_best"), 10))
    return (
        "flowchart TD\n"
        "  A[Preflop line] --> B[Flop action reweight]\n"
        "  B --> C[Expand minus blockers]\n"
        "  C --> D[MC equity]\n"
        f"  D --> E[eq {eq}]\n"
        f"  E --> F[pot thr {thr}]\n"
        f"  F --> G[thin gate {thin}]\n"
        "  G --> H[EV candidates]\n"
        f"  H --> I[Pick {pick}]\n"
        f"  I --> J[EV {best} bb]\n"
    )
