#!/usr/bin/env python3
"""Stateful session simulator (Phase B).

Runs many sequential preflop hands against a single persisting opponent profile.
This is a manual demo script (not a pytest test file).

Run from repo root:
    python tests/simulate_session.py --vs nit --hands 50 --seed 123
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import os
import sys

# Ensure repo root is on sys.path when running as `python tests/simulate_session.py`
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from baseline_preflop.adaptive_recommender import recommend_adaptive_preflop_action
from baseline_preflop.opponent_model import OpponentPreflopProfile, PRIORS, record_preflop_hand


RawAction = Dict[str, object]


@dataclass(frozen=True)
class VillainLeakConfig:
    name: str
    fold_to_steal: float          # when facing hero BTN open
    threebet_vs_open: float       # when facing hero BTN open (conditional on not folding)
    btn_open_rate: float          # when villain is BTN
    btn_open_size: float          # open size used by villain when opening


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


def _deal_hand_cycle(i: int) -> str:
    # Alternate positions each hand: even -> HERO BTN, odd -> HERO BB
    return "BTN_SB" if i % 2 == 0 else "BB"


def _deal_hero_cards(i: int, hero_position: str) -> str:
    # Use a small rotation of marginal hands to make adaptation visible.
    btn_hands = ["Qh 3c", "Td 8c", "7s 6s", "Jh 8h", "8s 7s", "72o".replace("o", "c")]  # last one is placeholder
    # The last placeholder is corrected below to keep valid card format.
    btn_hands[-1] = "7d 2c"
    bb_hands = ["7s 6s", "Kh Qc", "Td 9d", "Qd 9c", "Jh 8h"]
    if hero_position == "BTN_SB":
        return btn_hands[i % len(btn_hands)]
    return bb_hands[i % len(bb_hands)]


def _format_profile(profile: OpponentPreflopProfile) -> str:
    tags = profile.archetypes()
    bb_fold = profile.stat_bb_fold_to_steal()
    bb_3b = profile.stat_bb_3bet_vs_open()
    return (
        f"{','.join(tags)} | bb_fold={bb_fold.smoothed_rate:.2f} (c{bb_fold.confidence:.2f})"
        f" bb_3bet={bb_3b.smoothed_rate:.2f} (c{bb_3b.confidence:.2f})"
    )


def _simulate_villain_vs_hero_open(rng: random.Random, cfg: VillainLeakConfig) -> str:
    # returns: FOLD | CALL | 3BET
    if rng.random() < cfg.fold_to_steal:
        return "FOLD"
    # conditional 3bet
    if rng.random() < cfg.threebet_vs_open:
        return "3BET"
    return "CALL"


def _simulate_villain_btn_action(rng: random.Random, cfg: VillainLeakConfig) -> str:
    return "OPEN" if rng.random() < cfg.btn_open_rate else "LIMP"


def _summary_line(
    hand_idx: int,
    hero_position: str,
    hero_cards: str,
    spot: str,
    profile: OpponentPreflopProfile,
    dec,
    villain_action: str,
) -> str:
    freqs = dec.action_frequencies
    short_freqs = {k: round(v, 2) for k, v in freqs.items() if v > 0.001}
    return (
        f"Hand {hand_idx:>3} | {spot:<10} | Hero {hero_position:<6} {hero_cards:<6} "
        f"| Prof: {profile.archetypes()[0]:<22} | Baseline: {dec.debug.get('baseline_action','?'):<18} "
        f"| Freqs: {short_freqs!s:<22} | roll={dec.roll:.2f} | Hero: {dec.adapted_recommendation!r:<16} "
        f"| Villain: {villain_action}"
    )


class SessionRunner:
    def __init__(self, cfg: VillainLeakConfig, num_hands: int, seed: int = 123, stack_bb: float = 100.0):
        self.cfg = cfg
        self.num_hands = num_hands
        self.rng = random.Random(seed)
        self.stack_bb = stack_bb
        self.profile = OpponentPreflopProfile()

    def run(self) -> None:
        print(f"=== Session vs {self.cfg.name} | hands={self.num_hands} ===")
        print("(Profile persists across hands; updated from action_history.)")
        print("-")

        for i in range(1, self.num_hands + 1):
            hero_pos = _deal_hand_cycle(i)
            hero_cards = _deal_hero_cards(i, hero_pos)

            if hero_pos == "BTN_SB":
                # Hero decision first (unopened)
                state = __import__("baseline_preflop.parser", fromlist=["unopened_btn_decision"]).unopened_btn_decision(hero_cards, self.stack_bb)
                dec = recommend_adaptive_preflop_action(state, self.profile, rng=self.rng)

                # Build actual action_history
                hist: List[RawAction] = _blinds("BTN_SB")
                spot = "BTN_open"

                if dec.adapted_recommendation.action_type.value == "RAISE":
                    raise_to = dec.adapted_recommendation.raise_to_bb or 2.5
                    hist.append({"player": "HERO", "action": "RAISE", "amount": float(raise_to)})
                    vresp = _simulate_villain_vs_hero_open(self.rng, self.cfg)
                    if vresp == "FOLD":
                        hist.append({"player": "VILLAIN", "action": "FOLD"})
                    elif vresp == "CALL":
                        hist.append({"player": "VILLAIN", "action": "CALL"})
                    else:
                        threeb_to = float(max(raise_to * 3.5, 7.5))
                        hist.append({"player": "VILLAIN", "action": "RAISE", "amount": threeb_to})
                        # Hero response: keep it simple and fold in the simulator (still updates profile)
                        hist.append({"player": "HERO", "action": "FOLD"})
                else:
                    # fold/call/check shouldn't happen in unopened BTN spot in Phase A, but handle anyway
                    hist.append({"player": "HERO", "action": dec.adapted_recommendation.action_type.value})
                    vresp = "-"
                record_preflop_hand(self.profile, hero_pos, hist)
                print(_summary_line(i, hero_pos, hero_cards, spot, self.profile, dec, vresp))

            else:
                # Hero is BB; villain acts first with hidden strategy
                spot = "BB_vs_open"
                hist = _blinds("BB")

                vbtn = _simulate_villain_btn_action(self.rng, self.cfg)
                if vbtn == "OPEN":
                    open_to = float(self.cfg.btn_open_size)
                    hist.append({"player": "VILLAIN", "action": "RAISE", "amount": open_to})
                    state = __import__("baseline_preflop.parser", fromlist=["bb_vs_open_decision"]).bb_vs_open_decision(hero_cards, open_to, self.stack_bb)
                    dec = recommend_adaptive_preflop_action(state, self.profile, rng=self.rng)
                    # Apply hero action to history (simplified)
                    if dec.adapted_recommendation.action_type.value == "FOLD":
                        hist.append({"player": "HERO", "action": "FOLD"})
                    elif dec.adapted_recommendation.action_type.value == "CALL":
                        hist.append({"player": "HERO", "action": "CALL"})
                    else:
                        rto = dec.adapted_recommendation.raise_to_bb or float(open_to * 3.0)
                        hist.append({"player": "HERO", "action": "RAISE", "amount": float(rto)})
                        # villain folds (simplification)
                        hist.append({"player": "VILLAIN", "action": "FOLD"})
                    record_preflop_hand(self.profile, hero_pos, hist)
                    print(_summary_line(i, hero_pos, hero_cards, "BB_vs_open", self.profile, dec, f"OPEN({open_to})"))
                else:
                    # Villain limps; hero responds
                    hist.append({"player": "VILLAIN", "action": "CALL", "amount": 1.0})
                    state = __import__("baseline_preflop.parser", fromlist=["bb_vs_limp_decision"]).bb_vs_limp_decision(hero_cards, self.stack_bb)
                    dec = recommend_adaptive_preflop_action(state, self.profile, rng=self.rng)
                    if dec.adapted_recommendation.action_type.value == "CHECK":
                        hist.append({"player": "HERO", "action": "CHECK"})
                    else:
                        rto = dec.adapted_recommendation.raise_to_bb or 3.5
                        hist.append({"player": "HERO", "action": "RAISE", "amount": float(rto)})
                        hist.append({"player": "VILLAIN", "action": "FOLD"})
                    record_preflop_hand(self.profile, hero_pos, hist)
                    print(_summary_line(i, hero_pos, hero_cards, "BB_vs_limp", self.profile, dec, "LIMP"))

            if i % 10 == 0:
                print(f"---- @hand {i} | {_format_profile(self.profile)} ----")


def run_session_vs_nit(num_hands: int = 50, seed: int = 123) -> None:
    cfg = VillainLeakConfig(
        name="nit",
        fold_to_steal=0.85,
        threebet_vs_open=0.05,
        btn_open_rate=0.55,
        btn_open_size=2.5,
    )
    SessionRunner(cfg, num_hands=num_hands, seed=seed).run()


def run_session_vs_maniac(num_hands: int = 50, seed: int = 123) -> None:
    cfg = VillainLeakConfig(
        name="maniac",
        fold_to_steal=0.20,
        threebet_vs_open=0.25,
        btn_open_rate=0.90,
        btn_open_size=2.75,
    )
    SessionRunner(cfg, num_hands=num_hands, seed=seed).run()


def run_session_vs_gto(num_hands: int = 50, seed: int = 123) -> None:
    cfg = VillainLeakConfig(
        name="gto-ish",
        fold_to_steal=float(PRIORS["bb_fold_to_steal_rate"]),
        threebet_vs_open=float(PRIORS["bb_3bet_vs_open_rate"]),
        btn_open_rate=float(PRIORS["btn_open_rate"]),
        btn_open_size=2.5,
    )
    SessionRunner(cfg, num_hands=num_hands, seed=seed).run()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vs", choices=["nit", "maniac", "gto"], default="nit")
    ap.add_argument("--hands", type=int, default=50)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    if args.vs == "nit":
        run_session_vs_nit(args.hands, args.seed)
    elif args.vs == "maniac":
        run_session_vs_maniac(args.hands, args.seed)
    else:
        run_session_vs_gto(args.hands, args.seed)


if __name__ == "__main__":
    main()

