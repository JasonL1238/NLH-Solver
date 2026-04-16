"""Flop-only opponent profile: raw counts, smoothing, confidence, archetypes.

Tracks villain (and hero response) tendencies from canonical Phase C action
history.  Stats are compartmentalized for flop nodes only — no preflop or
turn/river leakage.

Recording uses ``validate_hand`` prefixes to derive pot and street state
accurately before each flop action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from poker_core.models import (
    Action,
    ActionType,
    HandConfig,
    Player,
    Street,
)
from poker_core.validation import validate_hand

from flop_spot.classification import classify_board

# ---------------------------------------------------------------------------
# Priors + smoothing (mirror baseline_preflop/opponent_model spirit)
# ---------------------------------------------------------------------------

STAT_PRIORS: Dict[str, float] = {
    "villain_cbet_when_pfr": 0.62,
    "fold_to_flop_cbet": 0.42,
    "fold_to_flop_stab": 0.38,
    "villain_bet_when_checked_to": 0.45,
    "villain_flop_raise_freq": 0.10,
    "hero_raise_vs_cbet": 0.09,
}

K_PRIOR_STRENGTH = 12.0
C_CONFIDENCE = 20.0
ARCHETYPE_MIN_CONF = 0.35

# Match flop_spot/context.py bet buckets (fraction of pot before bet)
_BET_SMALL_FRAC = 0.33
_BET_LARGE_FRAC = 0.75


# ---------------------------------------------------------------------------
# Stat helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SmoothedStat:
    name: str
    count: int
    opportunities: int
    prior: float
    k: float
    c: float
    raw_rate: Optional[float]
    smoothed_rate: float
    confidence: float


def _safe_rate(count: int, opp: int) -> Optional[float]:
    if opp <= 0:
        return None
    return count / opp


def smooth_rate(count: int, opp: int, prior: float, k: float = K_PRIOR_STRENGTH) -> float:
    observed = _safe_rate(count, opp)
    if observed is None:
        observed = prior
    return (k * prior + opp * observed) / (k + opp)


def confidence_weight(opp: int, c: float = C_CONFIDENCE) -> float:
    if opp <= 0:
        return 0.0
    return opp / (opp + c)


def _bucket_bet_frac(to_call: float, pot_total: float) -> str:
    """SMALL / MEDIUM / LARGE using pot before the bet (pot_total - to_call)."""
    if to_call < 1e-9:
        return "UNKNOWN"
    pot_before = pot_total - to_call
    if pot_before < 1e-9:
        return "LARGE"
    frac = to_call / pot_before
    if frac <= _BET_SMALL_FRAC:
        return "SMALL"
    if frac <= _BET_LARGE_FRAC:
        return "MEDIUM"
    return "LARGE"


# ---------------------------------------------------------------------------
# Profile (mutable counters)
# ---------------------------------------------------------------------------


@dataclass
class FlopOpponentProfile:
    """Raw flop tendency counts for the villain (and hero responses).

    All *_opportunities / *_count pairs follow: opportunity increments when
    the spot arises; count increments on the observed outcome.
    """

    # A. Villain as PFR on flop (continuation)
    villain_pfr_cbet_opportunities: int = 0
    villain_pfr_cbet_count: int = 0
    villain_pfr_check_back_opportunities: int = 0
    villain_pfr_check_back_count: int = 0

    # B. Hero facing villain flop c-bet (villain was PFR and bet flop first)
    hero_fold_to_flop_cbet_opportunities: int = 0
    hero_fold_to_flop_cbet_count: int = 0
    hero_call_flop_cbet_count: int = 0
    hero_raise_vs_flop_cbet_count: int = 0

    # C. Hero facing stab (villain NOT PFR, bets flop after checks / weak line)
    hero_fold_to_flop_stab_opportunities: int = 0
    hero_fold_to_flop_stab_count: int = 0
    hero_call_flop_stab_count: int = 0
    hero_raise_vs_flop_stab_count: int = 0

    # D. Villain aggression when checked to on flop
    villain_flop_bet_when_checked_to_opportunities: int = 0
    villain_flop_bet_when_checked_to_count: int = 0
    villain_flop_raise_count: int = 0
    villain_flop_check_raise_count: int = 0

    # E. Sizing (villain flop bets / raises)
    villain_flop_small_bet_count: int = 0
    villain_flop_medium_bet_count: int = 0
    villain_flop_large_bet_count: int = 0
    villain_flop_small_raise_count: int = 0
    villain_flop_large_raise_count: int = 0

    # F. Board splits (optional)
    villain_cbet_dry_board_count: int = 0
    villain_cbet_dynamic_board_count: int = 0
    hero_fold_to_cbet_dry_opportunities: int = 0
    hero_fold_to_cbet_dry_count: int = 0
    hero_fold_to_cbet_dynamic_opportunities: int = 0
    hero_fold_to_cbet_dynamic_count: int = 0

    def smoothed_fold_to_flop_cbet(self) -> SmoothedStat:
        return self._stat(
            "fold_to_flop_cbet",
            self.hero_fold_to_flop_cbet_count,
            self.hero_fold_to_flop_cbet_opportunities,
            STAT_PRIORS["fold_to_flop_cbet"],
        )

    def smoothed_villain_cbet_when_pfr(self) -> SmoothedStat:
        return self._stat(
            "villain_cbet_when_pfr",
            self.villain_pfr_cbet_count,
            self.villain_pfr_cbet_opportunities,
            STAT_PRIORS["villain_cbet_when_pfr"],
        )

    def smoothed_fold_to_flop_stab(self) -> SmoothedStat:
        return self._stat(
            "fold_to_flop_stab",
            self.hero_fold_to_flop_stab_count,
            self.hero_fold_to_flop_stab_opportunities,
            STAT_PRIORS["fold_to_flop_stab"],
        )

    def smoothed_villain_bet_when_checked_to(self) -> SmoothedStat:
        return self._stat(
            "villain_bet_when_checked_to",
            self.villain_flop_bet_when_checked_to_count,
            self.villain_flop_bet_when_checked_to_opportunities,
            STAT_PRIORS["villain_bet_when_checked_to"],
        )

    def smoothed_hero_raise_vs_cbet(self) -> SmoothedStat:
        return self._stat(
            "hero_raise_vs_cbet",
            self.hero_raise_vs_flop_cbet_count,
            self.hero_fold_to_flop_cbet_opportunities,
            STAT_PRIORS["hero_raise_vs_cbet"],
        )

    def _stat(self, name: str, count: int, opp: int, prior: float) -> SmoothedStat:
        raw = _safe_rate(count, opp)
        sm = smooth_rate(count, opp, prior)
        conf = confidence_weight(opp)
        return SmoothedStat(
            name=name,
            count=count,
            opportunities=opp,
            prior=prior,
            k=K_PRIOR_STRENGTH,
            c=C_CONFIDENCE,
            raw_rate=raw,
            smoothed_rate=sm,
            confidence=conf,
        )


def _last_preflop_raiser(actions: List[Action], flop_idx: int) -> Optional[Player]:
    last: Optional[Player] = None
    for i in range(flop_idx):
        a = actions[i]
        if a.action_type == ActionType.RAISE and a.player is not None:
            last = a.player
    return last


def _find_deal_flop_index(actions: List[Action]) -> int:
    for i, a in enumerate(actions):
        if a.action_type == ActionType.DEAL_FLOP:
            return i
    raise ValueError("action_history must include DEAL_FLOP")


def _is_dynamic_board(board) -> bool:
    from flop_spot.models import BoardTexture

    bf = classify_board(list(board))
    return bf.texture in (
        BoardTexture.DYNAMIC_DRAW_HEAVY,
        BoardTexture.TWO_TONE_BOARD,
        BoardTexture.MONOTONE_BOARD,
        BoardTexture.RAINBOW_CONNECTED,
        BoardTexture.LOW_CONNECTED,
        BoardTexture.HIGH_CONNECTED,
    )


def record_flop_hand(profile: FlopOpponentProfile, config: HandConfig, action_history: List[Action]) -> None:
    """Update ``profile`` from a completed flop street in ``action_history``.

    Requires a valid full history through at least the end of flop betting
    (``DEAL_TURN`` may be present, or the hand may end on the flop).  Raises
    ``ValidationError`` if the history does not validate.
    """
    validate_hand(config, action_history)
    flop_idx = _find_deal_flop_index(action_history)
    board = list(action_history[flop_idx].cards or ())
    if len(board) != 3:
        raise ValueError("DEAL_FLOP must carry three board cards")

    villain_pfr = _last_preflop_raiser(action_history, flop_idx) == Player.VILLAIN
    dynamic_board = _is_dynamic_board(board)

    seen_villain_pfr_flop_decision = False
    seen_villain_caller_flop_decision = False
    opening_flop_bettor: Optional[Player] = None

    i = flop_idx + 1
    while i < len(action_history):
        a = action_history[i]
        if a.action_type == ActionType.DEAL_TURN:
            break

        st_before = validate_hand(config, action_history[:i])
        if st_before.hand_over or st_before.current_street != Street.FLOP:
            break

        if a.player is None or a.action_type not in (
            ActionType.FOLD, ActionType.CHECK, ActionType.CALL,
            ActionType.BET, ActionType.RAISE,
        ):
            i += 1
            continue

        actor = a.player
        at = a.action_type

        # --- Villain PFR: first flop decision (cbet or check back) ---
        if (
            villain_pfr
            and actor == Player.VILLAIN
            and st_before.current_actor == Player.VILLAIN
            and st_before.number_of_raises_this_street == 0
            and st_before.current_bet_to_call_bb < 1e-9
            and not seen_villain_pfr_flop_decision
        ):
            seen_villain_pfr_flop_decision = True
            profile.villain_pfr_cbet_opportunities += 1
            if at == ActionType.BET:
                profile.villain_pfr_cbet_count += 1
                opening_flop_bettor = Player.VILLAIN
                if dynamic_board:
                    profile.villain_cbet_dynamic_board_count += 1
                else:
                    profile.villain_cbet_dry_board_count += 1
            elif at == ActionType.CHECK:
                profile.villain_pfr_check_back_opportunities += 1
                profile.villain_pfr_check_back_count += 1

        # --- Villain as preflop caller: first flop spot (stab when checked to) ---
        if (
            not villain_pfr
            and actor == Player.VILLAIN
            and st_before.current_actor == Player.VILLAIN
            and st_before.number_of_raises_this_street == 0
            and st_before.current_bet_to_call_bb < 1e-9
            and not seen_villain_caller_flop_decision
        ):
            seen_villain_caller_flop_decision = True
            profile.villain_flop_bet_when_checked_to_opportunities += 1
            if at == ActionType.BET:
                profile.villain_flop_bet_when_checked_to_count += 1
                opening_flop_bettor = Player.VILLAIN

        if actor == Player.VILLAIN and at == ActionType.BET:
            _record_villain_bet_sizing(profile, st_before, a)
            if opening_flop_bettor is None:
                opening_flop_bettor = Player.VILLAIN
        elif actor == Player.HERO and at == ActionType.BET and opening_flop_bettor is None:
            opening_flop_bettor = Player.HERO

        if actor == Player.VILLAIN and at == ActionType.RAISE:
            profile.villain_flop_raise_count += 1
            if st_before.current_bet_to_call_bb > 1e-9:
                profile.villain_flop_check_raise_count += 1
            st_after = validate_hand(config, action_history[: i + 1])
            _record_villain_raise_sizing(profile, st_before, st_after, a)

        # --- Hero response to villain's first flop bet (c-bet vs stab) ---
        if (
            actor == Player.HERO
            and st_before.current_actor == Player.HERO
            and st_before.current_bet_to_call_bb > 1e-9
        ):
            is_cbet_line = opening_flop_bettor == Player.VILLAIN and villain_pfr
            is_stab_line = opening_flop_bettor == Player.VILLAIN and not villain_pfr
            if is_cbet_line:
                profile.hero_fold_to_flop_cbet_opportunities += 1
                if dynamic_board:
                    profile.hero_fold_to_cbet_dynamic_opportunities += 1
                else:
                    profile.hero_fold_to_cbet_dry_opportunities += 1
                if at == ActionType.FOLD:
                    profile.hero_fold_to_flop_cbet_count += 1
                    if dynamic_board:
                        profile.hero_fold_to_cbet_dynamic_count += 1
                    else:
                        profile.hero_fold_to_cbet_dry_count += 1
                elif at == ActionType.CALL:
                    profile.hero_call_flop_cbet_count += 1
                elif at == ActionType.RAISE:
                    profile.hero_raise_vs_flop_cbet_count += 1
            elif is_stab_line:
                profile.hero_fold_to_flop_stab_opportunities += 1
                if at == ActionType.FOLD:
                    profile.hero_fold_to_flop_stab_count += 1
                elif at == ActionType.CALL:
                    profile.hero_call_flop_stab_count += 1
                elif at == ActionType.RAISE:
                    profile.hero_raise_vs_flop_stab_count += 1

        i += 1


def _record_villain_bet_sizing(profile: FlopOpponentProfile, st_before, action: Action) -> None:
    if action.player != Player.VILLAIN or action.amount_to_bb is None:
        return
    inc = action.amount_to_bb - st_before.street_contrib_villain
    if inc < 1e-9:
        return
    pot_before = st_before.pot_size_bb
    if pot_before < 1e-9:
        profile.villain_flop_large_bet_count += 1
        return
    frac = inc / pot_before
    if frac <= _BET_SMALL_FRAC:
        profile.villain_flop_small_bet_count += 1
    elif frac <= _BET_LARGE_FRAC:
        profile.villain_flop_medium_bet_count += 1
    else:
        profile.villain_flop_large_bet_count += 1


def _record_villain_raise_sizing(
    profile: FlopOpponentProfile,
    st_before,
    st_after,
    action: Action,
) -> None:
    if action.player != Player.VILLAIN or action.amount_to_bb is None:
        return
    added = st_after.pot_size_bb - st_before.pot_size_bb
    pot_before = st_before.pot_size_bb
    if pot_before < 1e-9:
        profile.villain_flop_large_raise_count += 1
        return
    frac = added / pot_before
    if frac <= _BET_SMALL_FRAC:
        profile.villain_flop_small_raise_count += 1
    else:
        profile.villain_flop_large_raise_count += 1


def flop_archetypes(profile: FlopOpponentProfile) -> List[str]:
    """Derive readable labels from smoothed stats (flop-only)."""
    labels: List[str] = []
    ft = profile.smoothed_fold_to_flop_cbet()
    cb = profile.smoothed_villain_cbet_when_pfr()
    st = profile.smoothed_fold_to_flop_stab()
    bt = profile.smoothed_villain_bet_when_checked_to()
    rz = profile.smoothed_hero_raise_vs_cbet()

    if ft.confidence >= ARCHETYPE_MIN_CONF and ft.smoothed_rate > STAT_PRIORS["fold_to_flop_cbet"] + 0.12:
        labels.append("FLOP_OVERFOLDER_VS_CBET")
    if ft.confidence >= ARCHETYPE_MIN_CONF and ft.smoothed_rate < STAT_PRIORS["fold_to_flop_cbet"] - 0.12:
        labels.append("FLOP_STICKY_CALLER")

    if cb.confidence >= ARCHETYPE_MIN_CONF and cb.smoothed_rate > STAT_PRIORS["villain_cbet_when_pfr"] + 0.15:
        labels.append("AUTO_CBETS_TOO_MUCH")
    if cb.confidence >= ARCHETYPE_MIN_CONF and cb.smoothed_rate < STAT_PRIORS["villain_cbet_when_pfr"] - 0.15:
        labels.append("PASSIVE_ON_FLOP")

    if rz.confidence >= ARCHETYPE_MIN_CONF and rz.smoothed_rate > STAT_PRIORS["hero_raise_vs_cbet"] + 0.06:
        labels.append("FLOP_RAISES_LIGHT")
    if rz.confidence >= ARCHETYPE_MIN_CONF and rz.smoothed_rate < STAT_PRIORS["hero_raise_vs_cbet"] - 0.04:
        labels.append("FLOP_RAISES_VALUE_HEAVY")

    if bt.confidence >= ARCHETYPE_MIN_CONF and bt.smoothed_rate > STAT_PRIORS["villain_bet_when_checked_to"] + 0.15:
        labels.append("FLOP_STABS_TOO_MUCH")

    total_bets = (
        profile.villain_flop_small_bet_count
        + profile.villain_flop_medium_bet_count
        + profile.villain_flop_large_bet_count
    )
    if total_bets >= 8:
        lr = profile.villain_flop_large_bet_count / total_bets
        sr = profile.villain_flop_small_bet_count / total_bets
        if lr > 0.45:
            labels.append("LARGE_SIZING_FLOP_AGGRESSOR")
        if sr > 0.45:
            labels.append("SMALL_SIZING_FLOP_AGGRESSOR")

    if st.confidence >= ARCHETYPE_MIN_CONF and st.smoothed_rate > STAT_PRIORS["fold_to_flop_stab"] + 0.12:
        labels.append("FLOP_OVERFOLDER_VS_STAB")

    if not labels:
        labels.append("UNKNOWN_FLOP")
    return labels


def villain_flop_profile_summary(profile: FlopOpponentProfile) -> str:
    return ",".join(flop_archetypes(profile))


def key_smoothed_flop_stats_dict(profile: FlopOpponentProfile) -> Dict[str, float]:
    return {
        "fold_to_flop_cbet": round(profile.smoothed_fold_to_flop_cbet().smoothed_rate, 4),
        "villain_cbet_when_pfr": round(profile.smoothed_villain_cbet_when_pfr().smoothed_rate, 4),
        "fold_to_flop_stab": round(profile.smoothed_fold_to_flop_stab().smoothed_rate, 4),
        "villain_bet_when_checked_to": round(
            profile.smoothed_villain_bet_when_checked_to().smoothed_rate, 4
        ),
        "hero_raise_vs_cbet": round(profile.smoothed_hero_raise_vs_cbet().smoothed_rate, 4),
    }


def confidence_summary_dict(profile: FlopOpponentProfile) -> Dict[str, float]:
    return {
        "fold_to_flop_cbet_confidence": round(profile.smoothed_fold_to_flop_cbet().confidence, 4),
        "villain_cbet_confidence": round(profile.smoothed_villain_cbet_when_pfr().confidence, 4),
        "fold_to_flop_stab_confidence": round(profile.smoothed_fold_to_flop_stab().confidence, 4),
        "bet_when_checked_to_confidence": round(
            profile.smoothed_villain_bet_when_checked_to().confidence, 4
        ),
        "raise_vs_cbet_confidence": round(profile.smoothed_hero_raise_vs_cbet().confidence, 4),
    }
