"""Opponent preflop modeling (Phase B).

This module is preflop-only and heads-up-only. It tracks opponent tendencies
using raw counts/opportunities and exposes smoothed estimates + confidence.

Input format for updates is the same compact dict list used by
`baseline_preflop.parser.make_preflop_state`:

    {"player": "HERO"|"VILLAIN",
     "action": "POST_BLIND"|"FOLD"|"CHECK"|"CALL"|"RAISE"|"BET",
     "amount": <float>}

For RAISE/BET, `amount` is raise-to (total contribution after action).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from .models import ActionType, Player, Position


# ---------------------------------------------------------------------------
# Priors + smoothing constants
# ---------------------------------------------------------------------------

PRIORS = {
    # Rates
    "btn_open_rate": 0.80,
    "btn_limp_rate": 0.06,
    "bb_fold_to_steal_rate": 0.38,
    "bb_3bet_vs_open_rate": 0.12,
    "btn_fold_to_3bet_rate": 0.55,
    "bb_raise_vs_limp_rate": 0.35,
    # Size bucket priors (distributions)
    "btn_open_size_buckets": {"SMALL": 0.15, "STANDARD": 0.75, "LARGE": 0.10},
    "bb_3bet_size_buckets": {"SMALL": 0.15, "STANDARD": 0.70, "LARGE": 0.15},
    "bb_iso_size_buckets": {"SMALL": 0.20, "STANDARD": 0.65, "LARGE": 0.15},
}

# Prior strength for rates: smoothed = (k*prior + n*observed) / (k+n)
K_PRIOR_STRENGTH = 12.0

# Confidence: conf = n / (n + C)
C_CONFIDENCE = 20.0

# Minimum confidence to emit non-UNKNOWN archetypes
ARCHETYPE_MIN_CONF = 0.35


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


def _normalize(dist: Dict[str, float]) -> Dict[str, float]:
    total = sum(dist.values())
    if total <= 0:
        return {k: 0.0 for k in dist}
    return {k: v / total for k, v in dist.items()}


def smooth_bucket_distribution(
    counts: Dict[str, int],
    prior: Dict[str, float],
    k: float = K_PRIOR_STRENGTH,
) -> Dict[str, float]:
    """Dirichlet-like smoothing using (k * prior_bucket + count_bucket) / (k + n)."""
    n = sum(counts.values())
    out: Dict[str, float] = {}
    for bucket, p in prior.items():
        out[bucket] = (k * p + counts.get(bucket, 0)) / (k + n)
    return _normalize(out)


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------

def bucket_open_size(raise_to_bb: float) -> str:
    if raise_to_bb < 2.25:
        return "SMALL"
    if raise_to_bb <= 2.75:
        return "STANDARD"
    return "LARGE"


def bucket_iso_size(raise_to_bb: float) -> str:
    if raise_to_bb < 3.0:
        return "SMALL"
    if raise_to_bb <= 4.5:
        return "STANDARD"
    return "LARGE"


def bucket_3bet_size(threebet_to_bb: float, open_to_bb: float) -> str:
    if open_to_bb <= 0:
        return "STANDARD"
    mult = threebet_to_bb / open_to_bb
    if mult < 3.0:
        return "SMALL"
    if mult <= 4.0:
        return "STANDARD"
    return "LARGE"


# ---------------------------------------------------------------------------
# Profile model
# ---------------------------------------------------------------------------

@dataclass
class OpponentPreflopProfile:
    # General
    hands_observed: int = 0

    # BTN_SB behavior (when villain is BTN_SB)
    btn_open_opportunities: int = 0
    btn_open_count: int = 0
    btn_limp_opportunities: int = 0
    btn_limp_count: int = 0

    # BB behavior vs open (when villain is BB facing a BTN open)
    bb_vs_open_opportunities: int = 0
    bb_fold_to_steal_count: int = 0
    bb_call_vs_open_count: int = 0
    bb_3bet_vs_open_count: int = 0

    # BTN vs 3bet behavior (when villain is BTN facing a 3bet)
    btn_facing_3bet_opportunities: int = 0
    btn_fold_to_3bet_count: int = 0
    btn_call_3bet_count: int = 0
    btn_4bet_count: int = 0

    # BB vs limp (when villain is BB facing BTN limp)
    bb_vs_limp_opportunities: int = 0
    bb_check_vs_limp_count: int = 0
    bb_raise_vs_limp_count: int = 0

    # Limp then facing iso (when villain BTN limps and faces raise)
    btn_limp_then_fold_to_raise_count: int = 0
    btn_limp_then_call_raise_count: int = 0
    btn_limp_then_reraise_count: int = 0

    # Sizing tendencies (bucket counts by spot)
    btn_open_size_bucket_counts: Dict[str, int] = field(default_factory=dict)
    bb_3bet_size_bucket_counts: Dict[str, int] = field(default_factory=dict)
    bb_raise_vs_limp_size_bucket_counts: Dict[str, int] = field(default_factory=dict)

    def _stat(self, name: str, count: int, opp: int, prior: float) -> SmoothedStat:
        return SmoothedStat(
            name=name,
            count=count,
            opportunities=opp,
            prior=prior,
            k=K_PRIOR_STRENGTH,
            c=C_CONFIDENCE,
            raw_rate=_safe_rate(count, opp),
            smoothed_rate=smooth_rate(count, opp, prior, K_PRIOR_STRENGTH),
            confidence=confidence_weight(opp, C_CONFIDENCE),
        )

    # --- Smoothed stat accessors ---
    def stat_btn_open_rate(self) -> SmoothedStat:
        return self._stat(
            "btn_open_rate",
            self.btn_open_count,
            self.btn_open_opportunities,
            PRIORS["btn_open_rate"],
        )

    def stat_btn_limp_rate(self) -> SmoothedStat:
        return self._stat(
            "btn_limp_rate",
            self.btn_limp_count,
            self.btn_limp_opportunities,
            PRIORS["btn_limp_rate"],
        )

    def stat_bb_fold_to_steal(self) -> SmoothedStat:
        return self._stat(
            "bb_fold_to_steal_rate",
            self.bb_fold_to_steal_count,
            self.bb_vs_open_opportunities,
            PRIORS["bb_fold_to_steal_rate"],
        )

    def stat_bb_3bet_vs_open(self) -> SmoothedStat:
        return self._stat(
            "bb_3bet_vs_open_rate",
            self.bb_3bet_vs_open_count,
            self.bb_vs_open_opportunities,
            PRIORS["bb_3bet_vs_open_rate"],
        )

    def stat_btn_fold_to_3bet(self) -> SmoothedStat:
        return self._stat(
            "btn_fold_to_3bet_rate",
            self.btn_fold_to_3bet_count,
            self.btn_facing_3bet_opportunities,
            PRIORS["btn_fold_to_3bet_rate"],
        )

    def stat_bb_raise_vs_limp(self) -> SmoothedStat:
        return self._stat(
            "bb_raise_vs_limp_rate",
            self.bb_raise_vs_limp_count,
            self.bb_vs_limp_opportunities,
            PRIORS["bb_raise_vs_limp_rate"],
        )

    # --- Sizing distribution accessors ---
    def dist_btn_open_size(self) -> Dict[str, float]:
        return smooth_bucket_distribution(
            self.btn_open_size_bucket_counts,
            PRIORS["btn_open_size_buckets"],
            k=K_PRIOR_STRENGTH,
        )

    def dist_bb_3bet_size(self) -> Dict[str, float]:
        return smooth_bucket_distribution(
            self.bb_3bet_size_bucket_counts,
            PRIORS["bb_3bet_size_buckets"],
            k=K_PRIOR_STRENGTH,
        )

    def dist_bb_iso_size(self) -> Dict[str, float]:
        return smooth_bucket_distribution(
            self.bb_raise_vs_limp_size_bucket_counts,
            PRIORS["bb_iso_size_buckets"],
            k=K_PRIOR_STRENGTH,
        )

    def archetypes(self) -> List[str]:
        """Return readable archetype labels derived from smoothed stats."""
        labels: List[str] = []

        bb_fold = self.stat_bb_fold_to_steal()
        bb_3b = self.stat_bb_3bet_vs_open()
        btn_open = self.stat_btn_open_rate()
        btn_limp = self.stat_btn_limp_rate()

        if max(bb_fold.confidence, bb_3b.confidence, btn_open.confidence) < ARCHETYPE_MIN_CONF:
            return ["UNKNOWN"]

        # BB tendencies
        if bb_fold.confidence >= ARCHETYPE_MIN_CONF:
            if bb_fold.smoothed_rate >= 0.50:
                labels.append("OVERFOLDER_TO_STEALS")
                labels.append("TIGHT_BB")
            elif bb_fold.smoothed_rate <= 0.28:
                labels.append("STICKY_DEFENDER")
                labels.append("LOOSE_BB")

        if bb_3b.confidence >= ARCHETYPE_MIN_CONF:
            if bb_3b.smoothed_rate >= 0.18:
                labels.append("AGGRESSIVE_3BETTER")
            elif bb_3b.smoothed_rate <= 0.07:
                labels.append("PASSIVE_DEFENDER")

        # BTN tendencies
        if btn_open.confidence >= ARCHETYPE_MIN_CONF:
            if btn_open.smoothed_rate >= 0.88:
                labels.append("WIDE_BTN")
            elif btn_open.smoothed_rate <= 0.60:
                labels.append("NITTY_BTN")

        if btn_limp.confidence >= ARCHETYPE_MIN_CONF and btn_limp.smoothed_rate >= 0.20:
            labels.append("LIMP_HEAVY_BTN")

        # Sizing tendencies (distribution-based)
        open_dist = self.dist_btn_open_size()
        if sum(self.btn_open_size_bucket_counts.values()) >= 10:
            if open_dist.get("SMALL", 0.0) >= 0.45:
                labels.append("SMALL_OPENER")
            if open_dist.get("LARGE", 0.0) >= 0.30:
                labels.append("LARGE_OPENER")

        threeb_dist = self.dist_bb_3bet_size()
        if sum(self.bb_3bet_size_bucket_counts.values()) >= 10:
            if threeb_dist.get("SMALL", 0.0) >= 0.45:
                labels.append("SMALL_3BETTER")
            if threeb_dist.get("LARGE", 0.0) >= 0.30:
                labels.append("LARGE_3BETTER")

        return labels or ["UNKNOWN"]


# ---------------------------------------------------------------------------
# Parsing + recording
# ---------------------------------------------------------------------------

RawAction = Dict[str, Union[str, float]]


def _pos(v: Union[str, Position]) -> Position:
    return v if isinstance(v, Position) else Position(v)


def _btn_player(hero_position: Position) -> Player:
    return Player.HERO if hero_position == Position.BTN_SB else Player.VILLAIN


def _bb_player(hero_position: Position) -> Player:
    return Player.VILLAIN if hero_position == Position.BTN_SB else Player.HERO


def _voluntary_actions(actions: List[RawAction]) -> List[RawAction]:
    out: List[RawAction] = []
    for a in actions:
        at = ActionType(a["action"])
        if at == ActionType.POST_BLIND:
            continue
        out.append(a)
    return out


def record_preflop_hand(
    profile: OpponentPreflopProfile,
    hero_position: Union[str, Position],
    action_history: List[RawAction],
    sb: float = 0.5,
    bb: float = 1.0,
    effective_stack_bb: Optional[float] = None,
) -> None:
    """Update profile using a completed preflop hand history (HU only)."""
    _ = effective_stack_bb  # reserved for future use
    hero_pos = _pos(hero_position)
    btn = _btn_player(hero_pos)
    bbp = _bb_player(hero_pos)

    profile.hands_observed += 1

    vol = _voluntary_actions(action_history)
    if not vol:
        # hand ended preflop without voluntary action (unlikely) -> nothing to count
        return

    # Identify the first voluntary action (should be BTN)
    first = vol[0]
    first_player = Player(first["player"])
    first_type = ActionType(first["action"])

    # --- Villain as BTN: open/limp opportunities ---
    if btn == Player.VILLAIN and first_player == Player.VILLAIN:
        profile.btn_open_opportunities += 1
        profile.btn_limp_opportunities += 1
        if first_type == ActionType.RAISE:
            profile.btn_open_count += 1
            raise_to = float(first.get("amount", 0.0))
            bucket = bucket_open_size(raise_to)
            profile.btn_open_size_bucket_counts[bucket] = profile.btn_open_size_bucket_counts.get(bucket, 0) + 1
        elif first_type == ActionType.CALL:
            # BTN limp (complete)
            profile.btn_limp_count += 1
        # If BTN folds preflop first to act, ignore (non-standard)

    # --- Villain as BB: vs limp ---
    # BTN limp is represented as CALL to match big blind amount (1bb)
    if bbp == Player.VILLAIN:
        # If BTN limped and then BB acts
        if first_player == btn and first_type == ActionType.CALL:
            profile.bb_vs_limp_opportunities += 1
            if len(vol) >= 2:
                resp = vol[1]
                resp_player = Player(resp["player"])
                resp_type = ActionType(resp["action"])
                if resp_player == Player.VILLAIN:
                    if resp_type == ActionType.CHECK:
                        profile.bb_check_vs_limp_count += 1
                    elif resp_type == ActionType.RAISE:
                        profile.bb_raise_vs_limp_count += 1
                        iso_to = float(resp.get("amount", 0.0))
                        bucket = bucket_iso_size(iso_to)
                        profile.bb_raise_vs_limp_size_bucket_counts[bucket] = profile.bb_raise_vs_limp_size_bucket_counts.get(bucket, 0) + 1

        # --- Villain as BB: vs open raise ---
        if first_player == btn and first_type == ActionType.RAISE:
            profile.bb_vs_open_opportunities += 1
            open_to = float(first.get("amount", 0.0))
            if len(vol) >= 2:
                resp = vol[1]
                resp_player = Player(resp["player"])
                resp_type = ActionType(resp["action"])
                if resp_player == Player.VILLAIN:
                    if resp_type == ActionType.FOLD:
                        profile.bb_fold_to_steal_count += 1
                    elif resp_type == ActionType.CALL:
                        profile.bb_call_vs_open_count += 1
                    elif resp_type == ActionType.RAISE:
                        profile.bb_3bet_vs_open_count += 1
                        threeb_to = float(resp.get("amount", 0.0))
                        bucket = bucket_3bet_size(threeb_to, open_to)
                        profile.bb_3bet_size_bucket_counts[bucket] = profile.bb_3bet_size_bucket_counts.get(bucket, 0) + 1

    # --- Villain as BTN: facing 3bet after opening ---
    if btn == Player.VILLAIN and first_player == Player.VILLAIN and first_type == ActionType.RAISE:
        # Pattern: VILLAIN opens, HERO 3bets, VILLAIN responds
        if len(vol) >= 3:
            second = vol[1]
            third = vol[2]
            if Player(second["player"]) == bbp and ActionType(second["action"]) == ActionType.RAISE and Player(third["player"]) == Player.VILLAIN:
                profile.btn_facing_3bet_opportunities += 1
                a3 = ActionType(third["action"])
                if a3 == ActionType.FOLD:
                    profile.btn_fold_to_3bet_count += 1
                elif a3 == ActionType.CALL:
                    profile.btn_call_3bet_count += 1
                elif a3 == ActionType.RAISE:
                    profile.btn_4bet_count += 1

    # --- Villain as BTN: limp then facing raise (iso), then response ---
    if btn == Player.VILLAIN and first_player == Player.VILLAIN and first_type == ActionType.CALL:
        # Pattern: VILLAIN limps, HERO raises, VILLAIN responds
        if len(vol) >= 3:
            second = vol[1]
            third = vol[2]
            if Player(second["player"]) == bbp and ActionType(second["action"]) == ActionType.RAISE and Player(third["player"]) == Player.VILLAIN:
                a3 = ActionType(third["action"])
                if a3 == ActionType.FOLD:
                    profile.btn_limp_then_fold_to_raise_count += 1
                elif a3 == ActionType.CALL:
                    profile.btn_limp_then_call_raise_count += 1
                elif a3 == ActionType.RAISE:
                    profile.btn_limp_then_reraise_count += 1


# ---------------------------------------------------------------------------
# Manual simulation helpers (generate action_history dicts)
# ---------------------------------------------------------------------------

def _blinds(hero_position: str, sb: float = 0.5, bb: float = 1.0) -> List[RawAction]:
    if hero_position == "BTN_SB":
        return [
            {"player": "HERO", "action": "POST_BLIND", "amount": sb},
            {"player": "VILLAIN", "action": "POST_BLIND", "amount": bb},
        ]
    return [
        {"player": "VILLAIN", "action": "POST_BLIND", "amount": sb},
        {"player": "HERO", "action": "POST_BLIND", "amount": bb},
    ]


def simulate_bb_overfolds(profile: OpponentPreflopProfile, n: int, open_size: float = 2.5) -> None:
    """Simulate villain as BB folding too much vs BTN opens (hero is BTN)."""
    for _ in range(n):
        history = _blinds("BTN_SB") + [
            {"player": "HERO", "action": "RAISE", "amount": open_size},
            {"player": "VILLAIN", "action": "FOLD"},
        ]
        record_preflop_hand(profile, "BTN_SB", history)


def simulate_bb_aggressive_3bet(profile: OpponentPreflopProfile, n: int, open_size: float = 2.5, threeb_size: float = 8.0) -> None:
    """Simulate villain as BB 3betting often vs BTN opens (hero is BTN)."""
    for _ in range(n):
        history = _blinds("BTN_SB") + [
            {"player": "HERO", "action": "RAISE", "amount": open_size},
            {"player": "VILLAIN", "action": "RAISE", "amount": threeb_size},
            {"player": "HERO", "action": "FOLD"},
        ]
        record_preflop_hand(profile, "BTN_SB", history)


def simulate_btn_overopens(profile: OpponentPreflopProfile, n: int, open_size: float = 2.5) -> None:
    """Simulate villain as BTN opening very wide (hero is BB)."""
    for _ in range(n):
        history = _blinds("BB") + [
            {"player": "VILLAIN", "action": "RAISE", "amount": open_size},
            {"player": "HERO", "action": "FOLD"},
        ]
        record_preflop_hand(profile, "BB", history)


def simulate_btn_limp_heavy(profile: OpponentPreflopProfile, n: int) -> None:
    """Simulate villain as BTN limping a lot (hero is BB)."""
    for _ in range(n):
        history = _blinds("BB") + [
            {"player": "VILLAIN", "action": "CALL", "amount": 1.0},
            {"player": "HERO", "action": "CHECK"},
        ]
        record_preflop_hand(profile, "BB", history)


def compare_baseline_vs_adaptive(state, profile, adaptive_fn, rng_seed: int = 123) -> Tuple[object, object]:
    """Run baseline and adaptive recommenders and return both outputs."""
    import random

    baseline = __import__("baseline_preflop.recommender", fromlist=["recommend_preflop_action"]).recommend_preflop_action(state)
    adaptive = adaptive_fn(state, profile, rng=random.Random(rng_seed))
    return baseline, adaptive

