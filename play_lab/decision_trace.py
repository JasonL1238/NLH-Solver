"""Structured decision traces for Play Lab (preflop + flop).

``TraceStep`` / ``TraceMetric`` support optional ``glossary_key`` for per-metric help text
in ``play_lab.trace_glossary.GLOSSARY``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from baseline_preflop.models import Decision, DerivedState, PokerState
from flop_equity.range_model import villain_flop_range_debug_lines
from poker_core.models import HandState


@dataclass
class TraceMetric:
    label: str
    value: str
    glossary_key: Optional[str] = None


@dataclass
class TraceStep:
    title: str
    intro_md: str = ""
    metrics: List[TraceMetric] = field(default_factory=list)
    footer_md: str = ""


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4g}" if abs(v) < 1000 else f"{v:.4f}"
    return str(v)


def _derived_metrics(d: DerivedState) -> List[TraceMetric]:
    return [
        TraceMetric("Stack depth bucket", _fmt(getattr(d.stack_depth_bucket, "value", d.stack_depth_bucket)), "stack_depth_bucket"),
        TraceMetric("Unopened pot", _fmt(d.unopened_pot)),
        TraceMetric("Facing limp", _fmt(d.facing_limp)),
        TraceMetric("Facing open raise", _fmt(d.facing_open_raise)),
        TraceMetric("Facing 3-bet", _fmt(d.facing_3bet)),
        TraceMetric("Facing 4-bet", _fmt(d.facing_4bet)),
        TraceMetric("Facing all-in", _fmt(d.facing_all_in)),
        TraceMetric("Last raise / chip add (bb)", _fmt(d.raise_size_in_bb), "raise_size_in_bb"),
        TraceMetric("Raise as × BB", _fmt(d.raise_size_as_multiple_of_bb)),
        TraceMetric("Raise as × previous bet", _fmt(d.raise_size_as_multiple_of_previous_bet)),
        TraceMetric("Stack / open ratio", _fmt(d.stack_to_open_ratio)),
        TraceMetric("Stack / 3-bet ratio", _fmt(d.stack_to_3bet_ratio)),
    ]


def preflop_trace_steps(dec: Decision, state: Optional[PokerState] = None) -> List[TraceStep]:
    dbg = dict(dec.debug or {})
    rid = str(dbg.get("baseline_rule_id", ""))

    if rid == "NO_ACTION" or not dec.legal_actions:
        return [
            TraceStep(
                title="No recommendation",
                metrics=[
                    TraceMetric("Reason", _fmt(dbg.get("explanation", dec.explanation))),
                    TraceMetric("Hand class", _fmt(dbg.get("hand_class_label")), "hand_class_label"),
                    TraceMetric("Stack bucket", _fmt(dbg.get("stack_depth_bucket")), "stack_depth_bucket"),
                    TraceMetric("Context label", _fmt(dbg.get("action_context_label"))),
                ],
            )
        ]

    steps: List[TraceStep] = []

    if state is not None:
        d = state.derived
        snap = [
            TraceMetric("Effective stack (bb)", _fmt(state.effective_stack_bb), "effective_stack_bb"),
            TraceMetric("Pot (bb)", _fmt(state.pot_size_bb), "pot_bb"),
            TraceMetric("To call (bb)", _fmt(state.current_bet_to_call_bb), "to_call_bb"),
            TraceMetric("Hero position", _fmt(state.hero_position.value), "hero_position"),
        ]
        steps.append(TraceStep(title="Table snapshot", metrics=snap))
        if d is not None:
            intro = (
                "These flags are **recomputed from ordered action history** (canonical state), "
                "not typed in manually. Expand **Spot shape** for how to read them together."
            )
            dm = [TraceMetric("Spot shape", "how the line maps to charts", "derived_facing")] + _derived_metrics(d)
            steps.append(TraceStep(title="Derived spot (from action history)", intro_md=intro, metrics=dm))

    steps.append(
        TraceStep(
            title="Hand classification",
            metrics=[
                TraceMetric("Chart label (e.g. AKs)", _fmt(dbg.get("hand_class_label")), "hand_class_label"),
                TraceMetric("Fine bucket", _fmt(dbg.get("hand_bucket")), "hand_bucket"),
            ],
        )
    )

    steps.append(
        TraceStep(
            title="Chart lookup",
            metrics=[
                TraceMetric("Chart context key", _fmt(dbg.get("chart_context")), "chart_context"),
                TraceMetric("Human context", _fmt(dbg.get("action_context_label"))),
                TraceMetric("Raw chart action", _fmt(dbg.get("chart_action_raw")), "chart_action_raw"),
                TraceMetric("After MDF filter", _fmt(dbg.get("chart_action_filtered")), "chart_action_filtered"),
                TraceMetric("MDF rule id", _fmt(dbg.get("mdf_rule")), "mdf_rule"),
            ],
        )
    )

    std, act, dsc = dbg.get("standard_mdf"), dbg.get("actual_mdf"), dbg.get("defense_scalar")
    mdf_intro = (
        "Baseline charts assume **standard** open/3-bet/4-bet sizes per context. "
        "When you face a raise, we compare **standard MDF** (from assumed sizing) to "
        "**actual MDF** (from the real last raise size), then form a **defense scalar** "
        "to tighten or widen marginal combos.\n\n"
        f"- **standard_mdf:** `{std}`\n"
        f"- **actual_mdf:** `{act}`\n"
        f"- **defense_scalar:** `{dsc}`"
    )
    steps.append(
        TraceStep(
            title="MDF vs sizing (math)",
            intro_md=mdf_intro,
            metrics=[
                TraceMetric("standard_mdf", _fmt(std), "standard_mdf"),
                TraceMetric("actual_mdf", _fmt(act), "actual_mdf"),
                TraceMetric("defense_scalar", _fmt(dsc), "defense_scalar"),
            ],
        )
    )

    rs = dbg.get("raise_size_bucket")
    sizing_intro = (
        "If the chart says **RAISE**, we pick a legal raise-to from the grid:\n\n"
        "- **Ultra / very short** — jam (largest legal raise).\n"
        "- **Unopened** — closest to **2.5 bb** open.\n"
        "- **vs limp** — closest to **3.5 bb** iso.\n"
        "- **vs open** — closest to **3.0 ×** villain’s raise-to.\n"
        "- **vs 3-bet** — closest to **2.5 ×** villain’s raise-to.\n"
        "- **vs 4-bet** — jam."
    )
    steps.append(
        TraceStep(
            title="Raise-to selection (when chart = RAISE)",
            intro_md=sizing_intro,
            metrics=[TraceMetric("Size bucket on pick", _fmt(rs), "raise_size_bucket")],
        )
    )

    legal = dbg.get("legal_actions")
    if isinstance(legal, list) and legal:
        legal_txt = "\n".join(f"{i + 1}. `{x}`" for i, x in enumerate(legal))
    else:
        legal_txt = "_—_"

    footer = (
        "**One-line summary:** "
        + str(dbg.get("explanation", dec.explanation))
        + "\n\n**Legal actions (engine grid):**\n"
        + legal_txt
    )
    steps.append(
        TraceStep(
            title="Final choice",
            metrics=[
                TraceMetric("Baseline rule id", _fmt(dbg.get("baseline_rule_id")), "baseline_rule_id"),
                TraceMetric("Recommended", _fmt(dbg.get("recommended_action")), "recommended_action"),
            ],
            footer_md=footer,
        )
    )

    return steps


def flop_trace_steps(state: HandState, flop_dec: Any) -> List[TraceStep]:
    dbg = dict(getattr(flop_dec, "debug", None) or {})
    cfg = state.config
    pot = float(state.pot_size_bb)
    to_call = float(state.current_bet_to_call_bb)
    facing = to_call > 1e-9
    thr = dbg.get("pot_odds_threshold")

    board_s = " ".join(f"{c!s}" for c in state.board_cards) or "—"
    snap_metrics: List[TraceMetric] = [
        TraceMetric("Board", board_s, "flop_board"),
        TraceMetric("Pot (bb)", _fmt(round(pot, 3)), "pot_bb"),
        TraceMetric("To call (bb)", _fmt(round(to_call, 3)), "to_call_bb"),
        TraceMetric("Effective stack (bb)", _fmt(cfg.effective_stack_bb), "effective_stack_bb"),
        TraceMetric("Facing a bet", _fmt(facing), "facing_bet"),
    ]
    intro_snap = ""
    if facing and thr is not None:
        intro_snap = (
            "**Price to continue:** `to_call / (pot + to_call)` = "
            f"**{_fmt(thr)}** — expand **Pot odds (call)** for meaning.\n\n"
        )
        snap_metrics.append(TraceMetric("Pot odds (call)", _fmt(thr), "pot_odds_threshold"))

    range_intro = "\n\n".join(villain_flop_range_debug_lines(state))
    steps: List[TraceStep] = [
        TraceStep(title="Table snapshot", intro_md=intro_snap, metrics=snap_metrics),
        TraceStep(
            title="Spot & roles (flop context)",
            metrics=[
                TraceMetric("Flop context enum", _fmt(dbg.get("action_context_label")), "flop_context"),
                TraceMetric("Hero IP/OOP", _fmt(dbg.get("hero_position_relation_on_flop")), "hero_ip_oop"),
                TraceMetric("Preflop role", _fmt(dbg.get("hero_preflop_role")), "hero_preflop_role"),
                TraceMetric("SPR bucket", _fmt(dbg.get("spr_bucket")), "spr_bucket"),
                TraceMetric("SPR (numeric)", _fmt(dbg.get("spr")), "spr"),
                TraceMetric("Facing bet size bucket", _fmt(dbg.get("flop_bet_size_bucket")), "flop_bet_size_bucket"),
            ],
        ),
        TraceStep(
            title="Villain range — how it was narrowed",
            intro_md=range_intro,
            metrics=[
                TraceMetric(
                    "Same line as MC summary",
                    _fmt(dbg.get("villain_range_summary")),
                    "villain_range_summary",
                ),
            ],
        ),
        TraceStep(
            title="Board & hand classification",
            metrics=[
                TraceMetric("Board texture", _fmt(dbg.get("board_texture_label")), "board_texture"),
                TraceMetric("Made hand", _fmt(dbg.get("made_hand_category")), "made_hand_category"),
                TraceMetric("Draw", _fmt(dbg.get("draw_category")), "draw_category"),
                TraceMetric("Showdown value", _fmt(dbg.get("has_showdown_value"))),
                TraceMetric("Strong draw", _fmt(dbg.get("has_strong_draw"))),
                TraceMetric("Combo draw", _fmt(dbg.get("has_combo_draw"))),
                TraceMetric("Backdoor equity", _fmt(dbg.get("has_backdoor_equity"))),
                TraceMetric("Overcards to board", _fmt(dbg.get("overcards_to_board_count"))),
            ],
        ),
    ]

    eq_intro = (
        "Equity is **P(hero wins or splits)** vs the weighted villain range on this flop, "
        "estimated by Monte Carlo runouts (random remaining cards)."
    )
    steps.append(
        TraceStep(
            title="Range + Monte Carlo equity",
            intro_md=eq_intro,
            metrics=[
                TraceMetric("Villain range (summary)", _fmt(dbg.get("villain_range_summary")), "villain_range_summary"),
                TraceMetric("Monte Carlo samples", _fmt(dbg.get("samples_used") or dbg.get("monte_carlo_samples")), "monte_carlo_samples"),
                TraceMetric("Equity estimate", _fmt(dbg.get("equity_estimate")), "equity_estimate"),
                TraceMetric("Win rate", _fmt(dbg.get("win_rate")), "win_rate"),
                TraceMetric("Tie rate", _fmt(dbg.get("tie_rate")), "tie_rate"),
            ],
        )
    )

    raw_cands = dbg.get("ev_candidates") or []
    cands: List[Dict[str, Any]] = [dict(x) for x in raw_cands] if raw_cands else []
    cands.sort(key=lambda r: float(r.get("ev", -1e18)), reverse=True)
    lines: List[str] = []
    for i, r in enumerate(cands[:24], start=1):
        evv = r.get("ev", "?")
        note = str(r.get("note", "")).replace("\n", " ")
        lines.append(f"{i}. `{r.get('repr', '?')}` — EV **{evv}** bb — _{note}_")
    if len(cands) > 24:
        lines.append(f"_… {len(cands) - 24} more candidates omitted_")

    ev_intro = (
        "Policy: **EV-first** — build candidates (check/call/fold + discrete bet/raise sizes), "
        "estimate **expected chips (bb)** for each using equity + a simple villain response model, "
        "apply thin-raise / pot-odds heuristics where configured, then **argmax EV**."
    )
    ev_footer = "**Candidate list (sorted by EV, best first):**\n" + ("\n".join(lines) if lines else "_—_")
    steps.append(
        TraceStep(
            title="EV grid (step-through)",
            intro_md=ev_intro,
            metrics=[
                TraceMetric("Best EV (bb)", _fmt(dbg.get("ev_best")), "ev_best"),
                TraceMetric("Policy rule", _fmt(dbg.get("baseline_rule_id") or dbg.get("policy_rule_id")), "policy_rule_id"),
                TraceMetric("Config version", _fmt(dbg.get("street_policy_version")), "street_policy_version"),
            ],
            footer_md=ev_footer,
        )
    )

    notes = dbg.get("equity_adjustment_notes")
    note_txt = ""
    if isinstance(notes, list) and notes:
        note_txt = "\n".join(f"- {n}" for n in notes)
    elif notes:
        note_txt = str(notes)
    final_footer = ""
    if note_txt:
        final_footer = "**Pipeline notes:**\n" + note_txt
    steps.append(
        TraceStep(
            title="Final choice",
            metrics=[
                TraceMetric("Chosen action", _fmt(dbg.get("final_action") or dbg.get("recommended_action")), "recommended_action"),
                TraceMetric("Explanation", _fmt(dbg.get("explanation", getattr(flop_dec, "explanation", "")))),
            ],
            footer_md=final_footer,
        )
    )

    return steps


def postflop_ev_trace_steps(
    state: HandState, ev_dec: Any, *, street_label: str
) -> List[TraceStep]:
    """Same trace as ``flop_trace_steps`` but tags the first step with ``street_label`` (TURN/RIVER)."""
    steps = flop_trace_steps(state, ev_dec)
    if not steps:
        return steps
    s0 = steps[0]
    steps[0] = TraceStep(
        title=f"{s0.title} ({street_label})",
        intro_md=s0.intro_md,
        metrics=list(s0.metrics),
        footer_md=s0.footer_md,
    )
    return steps


# --- Legacy flat format for tests / callers that expect (title, markdown) tuples ---


def _flatten_step(step: TraceStep) -> str:
    parts: List[str] = []
    if step.intro_md:
        parts.append(step.intro_md)
    for m in step.metrics:
        parts.append(f"- **{m.label}:** `{m.value}`")
    if step.footer_md:
        parts.append(step.footer_md)
    return "\n\n".join(parts) if parts else ""


def preflop_trace_sections(dec: Decision, state: Optional[PokerState] = None) -> List[tuple[str, str]]:
    return [(s.title, _flatten_step(s)) for s in preflop_trace_steps(dec, state)]


def flop_trace_sections(state: HandState, flop_dec: Any) -> List[tuple[str, str]]:
    return [(s.title, _flatten_step(s)) for s in flop_trace_steps(state, flop_dec)]
