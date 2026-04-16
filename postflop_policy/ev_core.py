"""Street-agnostic EV helpers and candidate enumeration for postflop policy."""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

from statistics import median

from poker_core.models import ActionType, HandState, LegalAction

from flop_policy.config import EvPolicyConfig
from flop_spot.models import FlopActionChoice


def linspace_sizes(lo: float, hi: float, n: int) -> List[float]:
    if hi < lo + 1e-9:
        return [round(lo, 2)]
    if n <= 1:
        return [round((lo + hi) / 2.0, 2)]
    step = (hi - lo) / (n - 1)
    return [round(lo + i * step, 2) for i in range(n)]


def pot_fraction_bet_targets(
    pot_bb: float,
    hero_street_bb: float,
    lo: float,
    hi: float,
) -> List[float]:
    fracs = (0.25, 0.33, 0.4, 0.5, 0.66, 0.75, 1.0, 1.25)
    out: List[float] = []
    for f in fracs:
        tgt = round(hero_street_bb + f * pot_bb, 2)
        tgt = max(lo, min(hi, tgt))
        out.append(tgt)
    return sorted(set(out))


def ev_call_fold_pot(eq: float, pot_bb: float, to_call_bb: float) -> float:
    return eq * (pot_bb + to_call_bb) - to_call_bb


def ev_aggression_line(
    eq: float,
    pot_before_bb: float,
    delta_bb: float,
    p_fold: float,
    p_call: float,
    p_raise: float,
    *,
    reraise_eq_multiplier: float = 0.82,
) -> float:
    """Shallow EV for bet/raise: fold wins pot; call uses ``eq``; re-raise branch uses
    ``eq * reraise_eq_multiplier`` (villain reraise stub).
    """
    ev_fold = p_fold * pot_before_bb
    pot_if_called = pot_before_bb + 2.0 * delta_bb
    ev_call = p_call * (eq * pot_if_called - delta_bb)
    eq_r = max(0.0, min(1.0, eq * reraise_eq_multiplier))
    ev_raise = p_raise * (eq_r * pot_if_called - delta_bb)
    return ev_fold + ev_call + ev_raise


BetResponseFn = Callable[[Any, Any, float], Tuple[float, float, float]]
CheckStabFn = Callable[[Any, Any], float]


def flop_raise_pot_fractions(
    la_list: List[LegalAction],
    state: HandState,
    cfg: EvPolicyConfig,
) -> List[float]:
    """Pot-fraction (delta/pot) for each discrete raise size on the current grid."""
    pot = float(state.pot_size_bb)
    raise_la = next((a for a in la_list if a.action_type == ActionType.RAISE), None)
    if raise_la is None or raise_la.min_to_bb is None or raise_la.max_to_bb is None:
        return []
    lo_r, hi_r = float(raise_la.min_to_bb), float(raise_la.max_to_bb)
    cap_r = float(state.street_bet_level) + 2.5 * max(pot, 1e-9)
    hi_r = min(hi_r, max(lo_r, cap_r))
    sizes = linspace_sizes(lo_r, hi_r, cfg.grid_points)
    fracs: List[float] = []
    for s in sizes:
        delta = s - float(state.street_contrib_hero)
        if delta < 1e-9:
            continue
        fracs.append(delta / max(pot, 1e-9))
    return fracs


def median_p_fold_over_raise_grid(
    *,
    la_list: List[LegalAction],
    state: HandState,
    cfg: EvPolicyConfig,
    ctx: Any,
    profile: Any,
    bet_response: BetResponseFn,
) -> float:
    """``p_fold`` from ``bet_response`` at the **median** raise pot-fraction (documented)."""
    fracs = flop_raise_pot_fractions(la_list, state, cfg)
    if not fracs:
        return 0.0
    med_frac = float(median(fracs))
    pf, _, _ = bet_response(ctx, profile, med_frac)
    return float(pf)


def build_ev_candidates(
    *,
    la_list: List[LegalAction],
    state: HandState,
    eq: float,
    cfg: EvPolicyConfig,
    ctx: Any,
    profile: Any,
    bet_response: BetResponseFn,
    check_stab_prob: CheckStabFn,
    hero_value_tier: str = "OTHER",
) -> List[Tuple[FlopActionChoice, float, str]]:
    """Enumerate CHECK/CALL/FOLD and discrete BET/RAISE sizes with EV stubs."""
    pot = float(state.pot_size_bb)
    to_call = float(state.current_bet_to_call_bb)
    facing = to_call > 1e-9
    bb = float(state.config.big_blind_bb)
    stub_bet = min(bb, pot * 0.33)

    mult_default = float(cfg.reraised_eq_discount_default)
    mult_raise = (
        float(cfg.reraised_eq_discount_nuts)
        if hero_value_tier == "NUTS_NEAR"
        else mult_default
    )

    out: List[Tuple[FlopActionChoice, float, str]] = []

    fold_la = next((a for a in la_list if a.action_type == ActionType.FOLD), None)
    if fold_la:
        out.append((FlopActionChoice(legal_action=fold_la), 0.0, "FOLD: ref EV=0"))

    chk = next((a for a in la_list if a.action_type == ActionType.CHECK), None)
    if chk:
        p_stab = check_stab_prob(ctx, profile)
        ev_chk = (1.0 - p_stab) * 0.0 + p_stab * (eq * (pot + 2.0 * stub_bet) - stub_bet)
        out.append((FlopActionChoice(legal_action=chk), ev_chk, "CHECK: stub vs stab"))

    call_la = next((a for a in la_list if a.action_type == ActionType.CALL), None)
    if call_la and facing:
        ev_c = ev_call_fold_pot(eq, pot, to_call)
        out.append((FlopActionChoice(legal_action=call_la), ev_c, "CALL: eq*(P+C)-C"))

    bet_la = next((a for a in la_list if a.action_type == ActionType.BET), None)
    if bet_la and bet_la.min_to_bb is not None and bet_la.max_to_bb is not None:
        lo_b, hi_b = float(bet_la.min_to_bb), float(bet_la.max_to_bb)
        if not facing:
            sizes = pot_fraction_bet_targets(
                pot, float(state.street_contrib_hero), lo_b, hi_b,
            )
        else:
            cap_b = float(state.street_bet_level) + 2.5 * max(pot, 1e-9)
            hi_b = min(hi_b, max(lo_b, cap_b))
            sizes = linspace_sizes(lo_b, hi_b, cfg.grid_points)
        for s in sizes:
            delta = s - float(state.street_contrib_hero)
            if delta < 1e-9:
                continue
            frac = delta / max(pot, 1e-9)
            pf, pc, pr = bet_response(ctx, profile, frac)
            ev_b = ev_aggression_line(
                eq, pot, delta, pf, pc, pr,
                reraise_eq_multiplier=mult_default,
            )
            out.append((
                FlopActionChoice(legal_action=bet_la, size_bb=s),
                ev_b,
                f"BET to {s}: EV_line fold={pf:.2f}",
            ))

    raise_la = next((a for a in la_list if a.action_type == ActionType.RAISE), None)
    if raise_la and raise_la.min_to_bb is not None and raise_la.max_to_bb is not None:
        lo_r, hi_r = float(raise_la.min_to_bb), float(raise_la.max_to_bb)
        cap_r = float(state.street_bet_level) + 2.5 * max(pot, 1e-9)
        hi_r = min(hi_r, max(lo_r, cap_r))
        sizes = linspace_sizes(lo_r, hi_r, cfg.grid_points)
        for s in sizes:
            delta = s - float(state.street_contrib_hero)
            if delta < 1e-9:
                continue
            frac = delta / max(pot, 1e-9)
            pf, pc, pr = bet_response(ctx, profile, frac)
            ev_r = ev_aggression_line(
                eq, pot, delta, pf, pc, pr,
                reraise_eq_multiplier=mult_raise,
            )
            out.append((
                FlopActionChoice(legal_action=raise_la, size_bb=s),
                ev_r,
                f"RAISE to {s}",
            ))

    return out


def apply_thin_raise_filter(
    candidates: List[Tuple[FlopActionChoice, float, str]],
    *,
    facing: bool,
    pot_odds_threshold: Optional[float],
    eq: float,
    cfg: EvPolicyConfig,
    hero_value_tier: str = "OTHER",
    villain_nut_frac: float = 1.0,
    p_fold_raise_median: float = 0.0,
    has_pressure_draw: bool = False,
) -> Tuple[List[Tuple[FlopActionChoice, float, str]], str]:
    """Drop thin raises unless value (nuts) or pressure gates pass.

    Returns (filtered_candidates, branch_id) where branch_id is one of
    ``not_facing``, ``no_threshold``, ``eq_above_cap``, ``value_exempt``,
    ``pressure_exempt``, ``filtered``.
    """
    if not facing or pot_odds_threshold is None:
        return candidates, "not_facing"
    min_bar = float(cfg.thin_raise_min_bar)
    cap_thr = max(float(pot_odds_threshold), min_bar)
    if eq + 1e-9 >= cap_thr:
        return candidates, "eq_above_cap"

    if hero_value_tier == "NUTS_NEAR":
        return candidates, "value_exempt"

    if (
        eq + 1e-9 >= float(cfg.eq_pressure_min)
        and villain_nut_frac <= float(cfg.villain_nut_frac_max) + 1e-9
        and p_fold_raise_median + 1e-9 >= float(cfg.p_fold_pressure_min)
        and has_pressure_draw
    ):
        return candidates, "pressure_exempt"

    return [
        c for c in candidates
        if c[0].legal_action.action_type != ActionType.RAISE
    ], "filtered"


def pick_best_ev_candidate(
    candidates: List[Tuple[FlopActionChoice, float, str]],
    la_list: List[LegalAction],
    *,
    facing: bool = False,
    hero_value_tier: str = "OTHER",
) -> Tuple[FlopActionChoice, float, str]:
    if not candidates:
        return FlopActionChoice(legal_action=la_list[0]), 0.0, "FALLBACK_FIRST_LEGAL"
    best = max(candidates, key=lambda t: t[1])
    best_ties = [c for c in candidates if abs(c[1] - best[1]) < 1e-6]

    def _tie_key_default(t: Tuple[FlopActionChoice, float, str]) -> Tuple[int, float]:
        ch, _, _ = t
        at = ch.legal_action.action_type
        pref = 0 if at == ActionType.CHECK else 1
        sz = ch.size_bb or 0.0
        return (pref, sz)

    def _tie_key_nuts_facing(t: Tuple[FlopActionChoice, float, str]) -> Tuple[int, float]:
        ch, _, _ = t
        at = ch.legal_action.action_type
        order = {
            ActionType.RAISE: 0,
            ActionType.CALL: 1,
            ActionType.FOLD: 2,
            ActionType.CHECK: 3,
            ActionType.BET: 4,
        }
        pref = order.get(at, 9)
        sz = ch.size_bb or 0.0
        return (pref, -sz)

    use_nuts = facing and hero_value_tier == "NUTS_NEAR"
    key = _tie_key_nuts_facing if use_nuts else _tie_key_default
    choice, ev, note = min(best_ties, key=key)
    return choice, ev, note
