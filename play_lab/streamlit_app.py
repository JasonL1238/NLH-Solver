"""Streamlit HU lab: you are VILLAIN vs engine (HERO), preflop through showdown.

Run from repository root::

    streamlit run play_lab/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

# Streamlit sets sys.path[0] to this script's directory; ensure repo root is importable.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st

from baseline_preflop.parser import make_preflop_state
from baseline_preflop.recommender import recommend_preflop_action
from flop_equity.equity_integration import recommend_flop_action_with_equity
from poker_core.legal_actions import legal_actions
from poker_core.models import Action, ActionType, HandConfig, HandState, Player, Street
from poker_core.parser import make_hand_config, parse_card
from poker_core.transitions import apply_action, deal_flop, deal_river, deal_turn, post_blinds
from poker_core.validation import ValidationError as PokerValidationError
from poker_core.validation import validate_hand

from play_lab.coordinator import (
    HandCoordinator,
    ScenarioState,
    hole_cards_spaced,
    is_lab_hand_terminal,
    needs_flop_deal,
    needs_river_deal,
    needs_turn_deal,
)
from play_lab.decision_trace import (
    TraceStep,
    flop_trace_steps,
    postflop_ev_trace_steps,
    preflop_trace_steps,
)
from play_lab.trace_glossary import GLOSSARY
from play_lab.deck import (
    blocked_for_runout,
    deal_random_hands,
    draw_flop_cards,
    draw_street_card,
    parse_flop_triple,
    validate_flop_input,
    validate_single_card_input,
)
from play_lab.runout import auto_runout_board_if_needed
from play_lab.showdown_display import format_showdown_block_html
from play_lab.engine_apply import (
    apply_legal_action,
    choose_raise_or_bet_amount,
    flop_choice_to_poker_apply,
    preflop_option_to_poker_apply,
)
from play_lab.preflop_bridge import PreflopBridgeError, poker_actions_to_preflop_raw
from play_lab.ui_helpers import (
    board_cards_colored_html,
    format_engine_villain_banner_html,
    hole_cards_colored_html,
    last_hand_settings_complete,
)

# Written by handlers; consumed at start of _init_session before widgets bind the key.
_PENDING_NEXT_STACK_BB = "_play_lab_pending_next_stack_bb"


def _request_next_stack_bb(value: float) -> None:
    """Defer updating ``play_lab_next_stack_bb`` until next run (Streamlit forbids mutating widget keys mid-run)."""
    st.session_state[_PENDING_NEXT_STACK_BB] = float(value)


def _init_session() -> None:
    if "play_lab_scenario" not in st.session_state:
        st.session_state.play_lab_scenario = HandCoordinator.new_scenario()
    pend = st.session_state.pop(_PENDING_NEXT_STACK_BB, None)
    if pend is not None:
        st.session_state.play_lab_next_stack_bb = float(pend)
    elif "play_lab_next_stack_bb" not in st.session_state:
        st.session_state.play_lab_next_stack_bb = float(
            st.session_state.play_lab_scenario.effective_stack_bb
        )
    if "play_lab_hand_cfg" not in st.session_state:
        st.session_state.play_lab_hand_cfg = None
    if "play_lab_hand_hist" not in st.session_state:
        st.session_state.play_lab_hand_hist = None
    if "play_lab_flop_input" not in st.session_state:
        st.session_state.play_lab_flop_input = "Ah 7c 2d"
    if "play_lab_turn_input" not in st.session_state:
        st.session_state.play_lab_turn_input = "9s"
    if "play_lab_river_input" not in st.session_state:
        st.session_state.play_lab_river_input = "Td"
    if "play_lab_flop_samples" not in st.session_state:
        st.session_state.play_lab_flop_samples = 2500
    if "play_lab_hide_hero_holes" not in st.session_state:
        st.session_state.play_lab_hide_hero_holes = False
    if "play_lab_show_engine_logic" not in st.session_state:
        st.session_state.play_lab_show_engine_logic = False


def _scenario() -> ScenarioState:
    return st.session_state.play_lab_scenario


def _rebuild_state(cfg: HandConfig, hist: List[Action]) -> HandState:
    s = validate_hand(cfg, hist)
    s.legal_actions_list = legal_actions(s)
    return s


def _clear_hand() -> None:
    st.session_state.play_lab_hand_cfg = None
    st.session_state.play_lab_hand_hist = None
    for k in list(st.session_state.keys()):
        sk = str(k)
        if sk.startswith("play_lab_flop_ad_") or sk.startswith("play_lab_flop_eq_") or sk.startswith(
            "play_lab_postflop_eq_"
        ):
            del st.session_state[k]


def _post_blinds_and_persist(
    sc: ScenarioState,
    *,
    pos: str,
    stack: float,
    randomize: bool,
    hero_txt: str,
    vil_txt: str,
) -> None:
    """Build config, post blinds, persist last-hand settings and effective stack."""
    rng = sc.rng
    if randomize:
        hero_h, vil_h = deal_random_hands(rng)
        h_str = hole_cards_spaced(hero_h)
        v_str = hole_cards_spaced(vil_h)
    else:
        h_str = hero_txt.strip()
        v_str = vil_txt.strip()
    cfg = make_hand_config(
        h_str,
        hero_position=pos,
        effective_stack_bb=float(stack),
        villain_cards=v_str,
    )
    hist, _state = post_blinds(cfg)
    st.session_state.play_lab_hand_cfg = cfg
    st.session_state.play_lab_hand_hist = hist
    st.session_state.play_lab_scenario.effective_stack_bb = float(stack)
    settings: Dict[str, Any] = {
        "hero_position": pos,
        "stack": float(stack),
        "randomize": randomize,
    }
    if not randomize:
        settings["hero_txt"] = h_str
        settings["vil_txt"] = v_str
    st.session_state.play_lab_last_hand_settings = settings
    _request_next_stack_bb(float(stack))
    st.rerun()


def _finalize_and_start_next(
    cfg: HandConfig,
    hist: List[Action],
    *,
    reason: str,
) -> None:
    """End this hand, then immediately deal the next in the same session (``reason`` is for logs only)."""
    _ = reason
    sc = _scenario()
    HandCoordinator.note_hand_completed(sc)
    _clear_hand()
    last = st.session_state.get("play_lab_last_hand_settings")
    stack = float(st.session_state.play_lab_next_stack_bb)
    if last_hand_settings_complete(last) and last is not None:
        _post_blinds_and_persist(
            sc,
            pos=str(last["hero_position"]),
            stack=stack,
            randomize=bool(last["randomize"]),
            hero_txt=str(last.get("hero_txt", "")),
            vil_txt=str(last.get("vil_txt", "")),
        )
        return
    st.success("Hand ended. No saved ‘last hand’ preset yet — start the next hand from the form below.")
    st.rerun()


def _render_idle_panel() -> None:
    """Between hands: same session vs new session."""
    sc = _scenario()
    st.info(
        f"**Same session** — `{sc.label}`, RNG seed `{sc.rng_seed}`, "
        f"**{sc.hands_completed}** hand(s) finished. Use the form or **Start next hand** to keep playing. "
        "Use **New session** in the sidebar to reset the RNG counter and default stack."
    )
    last: Optional[Dict[str, Any]] = st.session_state.get("play_lab_last_hand_settings")
    if last_hand_settings_complete(last) and last is not None:
        if st.button("Start next hand (same settings as last hand)", key="play_lab_next_same"):
            _post_blinds_and_persist(
                sc,
                pos=str(last["hero_position"]),
                stack=float(st.session_state.play_lab_next_stack_bb),
                randomize=bool(last["randomize"]),
                hero_txt=str(last.get("hero_txt", "")),
                vil_txt=str(last.get("vil_txt", "")),
            )
    elif last is not None:
        st.caption("Use **Post blinds & start** below — last hand used settings that need the form (e.g. manual cards).")


def _render_scenario_sidebar() -> None:
    sc = _scenario()
    st.sidebar.header("Session")
    st.sidebar.metric("Default / last stack (bb)", f"{sc.effective_stack_bb:.1f}")
    st.sidebar.caption(f"Label: **{sc.label}** · RNG seed `{sc.rng_seed}` · hands finished: **{sc.hands_completed}**")
    st.sidebar.checkbox(
        "Hide engine hole cards (in-hand display only)",
        key="play_lab_hide_hero_holes",
    )
    st.sidebar.checkbox(
        "Show engine decision logic (steps + metric glossary)",
        key="play_lab_show_engine_logic",
        help="Step-by-step trace; expand each metric for what it means.",
    )
    st.sidebar.number_input(
        "Next hand stack (bb)",
        min_value=1.0,
        value=float(st.session_state.play_lab_next_stack_bb),
        step=1.0,
        key="play_lab_next_stack_bb",
        help="Applied when you start the next hand (one-click or form).",
    )
    with st.sidebar.form("new_session"):
        lab = st.text_input("Session label", value="lab")
        seed = st.number_input("RNG seed", value=int(sc.rng_seed), step=1)
        stack = st.number_input("Default stack (bb)", value=float(sc.effective_stack_bb), min_value=1.0)
        if st.form_submit_button("New session (reset counter & stack default)"):
            st.session_state.play_lab_scenario = HandCoordinator.new_scenario(
                label=lab or "lab",
                rng_seed=int(seed),
                effective_stack_bb=float(stack),
            )
            _clear_hand()
            _request_next_stack_bb(float(stack))
            st.session_state.pop("play_lab_last_hand_settings", None)
            st.rerun()

    if st.sidebar.button("Clear current hand only"):
        _clear_hand()
        st.rerun()


def _render_new_hand() -> None:
    sc = _scenario()
    defaults: Dict[str, Any] = st.session_state.get("play_lab_last_hand_settings") or {}
    pos_default = str(defaults.get("hero_position", "BTN_SB"))
    pos_idx = 0 if pos_default == "BTN_SB" else 1
    stack_default = float(defaults.get("stack", sc.effective_stack_bb))
    rand_default = bool(defaults.get("randomize", True))
    hero_default = str(defaults.get("hero_txt", "As Kd"))
    vil_default = str(defaults.get("vil_txt", "Qc Jd"))

    st.subheader("New hand")
    hide = bool(st.session_state.get("play_lab_hide_hero_holes", False))
    if hide:
        st.caption(
            "Hidden mode hides the engine’s hole cards **during a hand** only. "
            "Manual card fields below still show what you type (for setup). "
            "Preflop/flop debug JSON can still imply strength (hand labels)."
        )
    col1, col2 = st.columns(2)
    with col1:
        pos = st.selectbox("Engine (HERO) position", ["BTN_SB", "BB"], index=pos_idx)
        stack = st.number_input("Stack (bb)", value=stack_default, min_value=1.0)
        randomize = st.checkbox("Random hero + villain hole cards", value=rand_default)
    with col2:
        hero_txt = st.text_input("Engine hole cards", value=hero_default, disabled=randomize)
        vil_txt = st.text_input("Your hole cards", value=vil_default, disabled=randomize)

    hand_busy = st.session_state.play_lab_hand_hist is not None
    if st.button("Post blinds & start", disabled=hand_busy):
        _post_blinds_and_persist(
            sc,
            pos=pos,
            stack=float(stack),
            randomize=randomize,
            hero_txt=hero_txt,
            vil_txt=vil_txt,
        )


def _render_trace_steps(steps: List[TraceStep]) -> None:
    """Numbered walkthrough; metrics with a glossary key get an expander for definitions."""
    with st.expander("How this decision was computed (step-by-step)", expanded=False):
        for i, step in enumerate(steps, start=1):
            st.markdown(f"##### Step {i} — {step.title}")
            if step.intro_md:
                st.markdown(step.intro_md)
            for m in step.metrics:
                if m.glossary_key:
                    title = f"{m.label}: `{m.value}`"
                    body = GLOSSARY.get(m.glossary_key, "_No glossary entry for this metric yet._")
                    with st.expander(title):
                        st.markdown(body)
                else:
                    st.markdown(f"- **{m.label}:** `{m.value}`")
            if step.footer_md:
                st.markdown(step.footer_md)


def _render_deal_flop_section(cfg: HandConfig, hist: List[Action], state: HandState) -> None:
    if not needs_flop_deal(state):
        return
    st.subheader("Deal flop")
    if st.button("Fill random flop (excluding hole cards)"):
        trip = draw_flop_cards(
            _scenario().rng,
            cfg.hero_hole_cards,
            cfg.villain_hole_cards,
        )
        st.session_state.play_lab_flop_input = f"{trip[0]!s} {trip[1]!s} {trip[2]!s}"
        st.rerun()
    flop_s = st.text_input("Flop cards", key="play_lab_flop_input")
    st.caption("Three tokens separated by spaces, e.g. `Ah 7c 2d`. Field must not be empty when you press **Deal flop**.")
    if st.button("Deal flop"):
        ok, err = validate_flop_input(flop_s)
        if not ok:
            st.error(err)
            return
        try:
            trip = parse_flop_triple(flop_s)
        except ValueError as exc:
            st.error(f"Invalid flop card(s): {exc}")
            return
        new_state = deal_flop(cfg, hist, trip)
        st.session_state.play_lab_hand_hist = new_state.action_history
        st.rerun()


def _render_deal_turn_section(cfg: HandConfig, hist: List[Action], state: HandState) -> None:
    if not needs_turn_deal(state):
        return
    st.subheader("Deal turn")
    sc = _scenario()
    if st.button("Fill random turn (excluding holes + board)", key="play_lab_fill_turn"):
        blocked = blocked_for_runout(
            cfg.hero_hole_cards, cfg.villain_hole_cards, list(state.board_cards)
        )
        c = draw_street_card(sc.rng, blocked)
        st.session_state.play_lab_turn_input = f"{c!s}"
        st.rerun()
    turn_s = st.text_input("Turn card", key="play_lab_turn_input")
    st.caption("One token, e.g. `9s`.")
    if st.button("Deal turn", key="play_lab_deal_turn"):
        ok, err = validate_single_card_input(turn_s)
        if not ok:
            st.error(err)
            return
        try:
            card = parse_card(turn_s.strip().split()[0])
        except ValueError as exc:
            st.error(f"Invalid card: {exc}")
            return
        blocked = blocked_for_runout(
            cfg.hero_hole_cards, cfg.villain_hole_cards, list(state.board_cards)
        )
        if card in blocked:
            st.error("That card is already used (board or hole cards).")
            return
        ns = deal_turn(cfg, hist, card)
        st.session_state.play_lab_hand_hist = ns.action_history
        st.rerun()


def _render_deal_river_section(cfg: HandConfig, hist: List[Action], state: HandState) -> None:
    if not needs_river_deal(state):
        return
    st.subheader("Deal river")
    sc = _scenario()
    if st.button("Fill random river (excluding holes + board)", key="play_lab_fill_river"):
        blocked = blocked_for_runout(
            cfg.hero_hole_cards, cfg.villain_hole_cards, list(state.board_cards)
        )
        c = draw_street_card(sc.rng, blocked)
        st.session_state.play_lab_river_input = f"{c!s}"
        st.rerun()
    river_s = st.text_input("River card", key="play_lab_river_input")
    st.caption("One token, e.g. `Td`.")
    if st.button("Deal river", key="play_lab_deal_river"):
        ok, err = validate_single_card_input(river_s)
        if not ok:
            st.error(err)
            return
        try:
            card = parse_card(river_s.strip().split()[0])
        except ValueError as exc:
            st.error(f"Invalid card: {exc}")
            return
        blocked = blocked_for_runout(
            cfg.hero_hole_cards, cfg.villain_hole_cards, list(state.board_cards)
        )
        if card in blocked:
            st.error("That card is already used (board or hole cards).")
            return
        ns = deal_river(cfg, hist, card)
        st.session_state.play_lab_hand_hist = ns.action_history
        st.rerun()


def _render_villain_panel(cfg: HandConfig, hist: List[Action], state: HandState) -> None:
    st.subheader("Your action (VILLAIN)")
    if state.hand_over:
        st.info("Hand is over.")
        return
    if state.current_actor != Player.VILLAIN:
        st.info("Waiting for engine (HERO).")
        return
    la_list = state.legal_actions_list or legal_actions(state)
    for i, la in enumerate(la_list):
        key = f"v_{state.current_street.value}_{len(hist)}_{i}"
        if la.action_type in (ActionType.BET, ActionType.RAISE):
            lo = float(la.min_to_bb or 0)
            hi = float(la.max_to_bb or lo)
            default = choose_raise_or_bet_amount(la)
            amt = st.number_input(
                f"{la.action_type.value} sizing (bb)",
                min_value=lo,
                max_value=hi,
                value=default,
                key=key + "_amt",
            )
            if st.button(f"Play {la.action_type.value}", key=key):
                new_s = apply_legal_action(cfg, hist, state, legal=la, amount_to_bb=float(amt))
                st.session_state.play_lab_hand_hist = new_s.action_history
                st.rerun()
        else:
            if st.button(f"Play {la.action_type.value}", key=key):
                new_s = apply_legal_action(cfg, hist, state, legal=la, amount_to_bb=None)
                st.session_state.play_lab_hand_hist = new_s.action_history
                st.rerun()


def _render_hero_engine(cfg: HandConfig, hist: List[Action], state: HandState) -> None:
    st.subheader("Engine (HERO)")
    if state.hand_over:
        st.caption("Hand is over — no engine action.")
        return
    if state.current_actor != Player.HERO:
        st.caption("HERO is not up to act.")
        return
    sc = _scenario()
    la_list = state.legal_actions_list or legal_actions(state)
    show_logic = bool(st.session_state.get("play_lab_show_engine_logic", False))

    if state.current_street == Street.PREFLOP:
        try:
            raw = poker_actions_to_preflop_raw(hist)
        except PreflopBridgeError as exc:
            st.error(f"Preflop bridge error: {exc}")
            return
        try:
            pstate = make_preflop_state(
                hole_cards_spaced(cfg.hero_hole_cards),
                cfg.hero_position.value,
                cfg.effective_stack_bb,
                raw,
                small_blind_bb=cfg.small_blind_bb,
                big_blind_bb=cfg.big_blind_bb,
            )
        except Exception as exc:
            st.error(f"make_preflop_state failed: {exc}")
            return
        dec = recommend_preflop_action(pstate)
        st.write("**Engine (baseline):**", repr(dec.recommended_action))
        st.caption(dec.explanation)
        if show_logic:
            _render_trace_steps(preflop_trace_steps(dec, pstate))
            with st.expander("Preflop raw debug (JSON)"):
                st.json({k: v for k, v in (dec.debug or {}).items() if k != "derived"})

        if st.button("Apply engine preflop action"):
            la, amt = preflop_option_to_poker_apply(state, la_list, dec.recommended_action)
            new_s = apply_legal_action(cfg, hist, state, legal=la, amount_to_bb=amt)
            st.session_state.play_lab_hand_hist = new_s.action_history
            st.rerun()

    elif state.current_street == Street.FLOP:
        st.number_input(
            "Flop MC samples",
            min_value=500,
            max_value=20000,
            value=int(st.session_state.play_lab_flop_samples),
            step=500,
            key="play_lab_flop_samples",
        )
        samples = int(st.session_state.play_lab_flop_samples)
        seed = sc.rng_seed + len(hist)
        cache_key = f"play_lab_flop_eq_{len(hist)}_{samples}_{seed}"
        do_refresh = st.button("Refresh flop recommendation")
        if cache_key not in st.session_state or do_refresh:
            with st.spinner("Flop equity (Monte Carlo)…"):
                st.session_state[cache_key] = recommend_flop_action_with_equity(
                    state, samples=samples, seed=seed,
                )
        fdec = st.session_state[cache_key]
        st.write("**Engine (equity-aware):**", repr(fdec.recommended_action))
        st.caption(fdec.explanation)
        if show_logic:
            _render_trace_steps(flop_trace_steps(state, fdec))
            with st.expander("Flop raw debug (JSON)"):
                st.json({k: fdec.debug.get(k) for k in list((fdec.debug or {}).keys())[:80]})

        if st.button("Apply engine flop action"):
            la, amt = flop_choice_to_poker_apply(state, fdec.recommended_action)
            new_s = apply_legal_action(cfg, hist, state, legal=la, amount_to_bb=amt)
            st.session_state.play_lab_hand_hist = new_s.action_history
            st.rerun()

    elif state.current_street in (Street.TURN, Street.RIVER):
        st.number_input(
            "Postflop MC samples (turn; ignored on river exact)",
            min_value=500,
            max_value=20000,
            value=int(st.session_state.play_lab_flop_samples),
            step=500,
            key="play_lab_flop_samples",
        )
        samples = int(st.session_state.play_lab_flop_samples)
        seed = sc.rng_seed + len(hist)
        st_label = state.current_street.value
        cache_key = f"play_lab_postflop_eq_{st_label}_{len(hist)}_{samples}_{seed}"
        do_refresh = st.button(f"Refresh {st_label.lower()} recommendation", key=f"play_lab_refresh_{st_label}")
        if cache_key not in st.session_state or do_refresh:
            with st.spinner(f"{st_label} equity / EV…"):
                # Lazy import avoids package cycle (postflop_equity ↔ postflop_policy).
                from postflop_policy.ev_recommender import (
                    recommend_river_action_ev,
                    recommend_turn_action_ev,
                )

                if state.current_street == Street.TURN:
                    st.session_state[cache_key] = recommend_turn_action_ev(
                        state, samples=samples, seed=seed,
                    )
                else:
                    st.session_state[cache_key] = recommend_river_action_ev(
                        state, samples=samples, seed=seed,
                    )
        ev_dec = st.session_state[cache_key]
        st.write("**Engine (EV):**", repr(ev_dec.recommended_action))
        st.caption(ev_dec.explanation)
        if show_logic:
            _render_trace_steps(postflop_ev_trace_steps(state, ev_dec, street_label=st_label))
            with st.expander(f"{st_label} raw debug (JSON)"):
                st.json({k: ev_dec.debug.get(k) for k in list((ev_dec.debug or {}).keys())[:80]})

        if st.button(f"Apply engine {st_label.lower()} action", key=f"play_lab_apply_{st_label}"):
            la, amt = flop_choice_to_poker_apply(state, ev_dec.recommended_action)
            new_s = apply_legal_action(cfg, hist, state, legal=la, amount_to_bb=amt)
            st.session_state.play_lab_hand_hist = new_s.action_history
            st.rerun()
    else:
        st.caption(f"No engine picker for street `{state.current_street.value}`.")


def _end_hand(cfg: HandConfig, hist: List[Action], reason: str) -> None:
    _ = cfg, hist, reason
    sc = _scenario()
    HandCoordinator.note_hand_completed(sc)
    st.success("Hand ended. Start the next hand below (same session), or **New session** in the sidebar.")
    _clear_hand()
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="NLH Play Lab", layout="wide")
    _init_session()
    st.title("Play lab — you are VILLAIN vs engine (HERO)")
    st.caption(
        "Canonical state: **poker_core**. Engine: **baseline_preflop** + **flop_equity** (flop) + "
        "**postflop_policy** EV (turn/river). All-in runout deals turn/river automatically. "
        "**End & next** keeps the same session; **New session** resets label, seed, and hand counter. "
        "Optional **engine decision logic** (sidebar) adds a step-by-step trace with per-metric definitions. "
        "Hide-engine mode still allows peek; traces may show hand labels."
    )

    _render_scenario_sidebar()

    cfg = st.session_state.play_lab_hand_cfg
    hist = st.session_state.play_lab_hand_hist
    if cfg is None or hist is None:
        _render_idle_panel()
        _render_new_hand()
        return

    try:
        state = _rebuild_state(cfg, hist)
    except PokerValidationError as exc:
        st.error(f"Invalid hand state: {exc}")
        if st.button("Discard broken hand"):
            _clear_hand()
            st.rerun()
        return

    new_hist = auto_runout_board_if_needed(cfg, hist, _scenario().rng)
    if len(new_hist) > len(hist):
        st.session_state.play_lab_hand_hist = new_hist
        st.rerun()
    hist = st.session_state.play_lab_hand_hist
    state = _rebuild_state(cfg, hist)

    eff = float(cfg.effective_stack_bb)
    hero_rem = max(0.0, eff - float(state.hero_contribution_bb))
    vil_rem = max(0.0, eff - float(state.villain_contribution_bb))

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Street", state.current_street.value)
        st.metric("Actor", state.current_actor.value if state.current_actor else "—")
    with c2:
        st.metric("Pot (bb)", f"{state.pot_size_bb:.2f}")
        st.metric("To call (bb)", f"{state.current_bet_to_call_bb:.2f}")
    with c3:
        st.metric("Effective stack (bb)", f"{eff:.1f}")
        st.caption("Symmetric HU cap used by the engine (both start here).")
    with c4:
        st.metric("Engine stack left (bb)", f"{hero_rem:.2f}")
        st.caption(f"In pot: {state.hero_contribution_bb:.2f} bb")
    with c5:
        st.metric("Your stack left (bb)", f"{vil_rem:.2f}")
        st.caption(f"In pot: {state.villain_contribution_bb:.2f} bb")

    st.markdown(
        f"<p><strong>Board:</strong> {board_cards_colored_html(state.board_cards)}</p>",
        unsafe_allow_html=True,
    )

    hide = bool(st.session_state.get("play_lab_hide_hero_holes", False))
    st.markdown(format_engine_villain_banner_html(cfg, hide_hero_holes=hide), unsafe_allow_html=True)
    if hide:
        with st.expander("Peek engine cards (debug)"):
            st.markdown(hole_cards_colored_html(cfg.hero_hole_cards), unsafe_allow_html=True)

    if state.hand_over and state.fold_winner is not None:
        who = "You" if state.fold_winner == Player.VILLAIN else "Engine (HERO)"
        st.success(f"**{who}** wins the pot — opponent folded.")
    elif (
        state.hand_over
        and len(state.board_cards) == 5
        and cfg.villain_hole_cards is not None
        and state.fold_winner is None
    ):
        st.subheader("Showdown")
        st.markdown(
            format_showdown_block_html(
                hero=cfg.hero_hole_cards,
                villain=cfg.villain_hole_cards,
                board=list(state.board_cards),
            ),
            unsafe_allow_html=True,
        )

    _render_deal_flop_section(cfg, hist, state)
    _render_deal_turn_section(cfg, hist, state)
    _render_deal_river_section(cfg, hist, state)

    col_l, col_r = st.columns(2)
    with col_l:
        _render_villain_panel(cfg, hist, state)
    with col_r:
        _render_hero_engine(cfg, hist, state)

    st.divider()
    st.subheader("End hand")
    if state.hand_over:
        st.info("Hand is over (fold or showdown). End hand to play again.")
    elif is_lab_hand_terminal(state):
        st.info("Hand is terminal.")

    last = st.session_state.get("play_lab_last_hand_settings")
    can_chain = bool(last_hand_settings_complete(last) and last is not None)
    c_end1, c_end2, c_end3 = st.columns(3)
    with c_end1:
        if st.button("End hand & return to setup", key="play_lab_finalize_only"):
            if state.hand_over:
                _end_hand(cfg, hist, "hand_over")
            else:
                _end_hand(cfg, hist, "user_finalize")
    with c_end2:
        if can_chain and state.hand_over:
            if st.button("End hand & start next", key="play_lab_finalize_next"):
                _finalize_and_start_next(cfg, hist, reason="hand_over")
        elif can_chain:
            st.caption("Mid-line: **End & start next** discards this line, or end only.")
    with c_end3:
        if can_chain and st.button("End hand & start next (abandon line)", key="play_lab_abort_next"):
            _finalize_and_start_next(cfg, hist, reason="user_abort")

    if st.button("End hand only (stay on setup screen)", key="play_lab_abort_only"):
        _end_hand(cfg, hist, "user_abort")

    with st.expander("Raw action history"):
        st.code("\n".join(repr(a) for a in hist), language="text")


if __name__ == "__main__":
    main()
