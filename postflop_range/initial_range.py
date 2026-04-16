"""Initial weighted particle set from preflop line (coarse prior, no flop action)."""

from __future__ import annotations

import random
from typing import List, Tuple

from poker_core.models import HandState, HoleCards

from flop_equity.range_model import (
    dead_cards_frozen,
    expand_label_to_combos,
    prior_label_weights,
)

from .board_update import reclassify_all_particles
from .particles import CoarseBucket, Particle


def build_weighted_combo_pool(state: HandState) -> Tuple[List[Tuple[HoleCards, float, str]], str]:
    """Expand prior label weights to (combo, weight, label) with dead-card filter."""
    label_weights, summary = prior_label_weights(state)
    dead = dead_cards_frozen(state)
    pool: List[Tuple[HoleCards, float, str]] = []
    for label, w in label_weights.items():
        if w < 1e-12:
            continue
        for hc in expand_label_to_combos(label):
            if hc.high in dead or hc.low in dead:
                continue
            pool.append((hc, float(w), label))
    return pool, summary


def weighted_sample_without_replacement(
    pool: List[Tuple[HoleCards, float, str]],
    n: int,
    rng: random.Random,
) -> List[Tuple[HoleCards, float, str]]:
    """Take up to ``n`` distinct combos with probability ∝ weight."""
    if not pool or n <= 0:
        return []
    if len(pool) <= n:
        return list(pool)
    remaining = list(pool)
    out: List[Tuple[HoleCards, float, str]] = []
    for _ in range(n):
        if not remaining:
            break
        weights = [t[1] for t in remaining]
        tot = sum(weights)
        if tot < 1e-15:
            break
        r = rng.random() * tot
        cum = 0.0
        pick_i = 0
        for i, w in enumerate(weights):
            cum += w
            if r <= cum:
                pick_i = i
                break
        out.append(remaining.pop(pick_i))
    return out


def build_initial_particles(
    state: HandState,
    *,
    n_particles: int,
    seed: int,
) -> Tuple[List[Particle], str]:
    """Create ``n_particles`` weighted particles at flop (3-card board required)."""
    if len(state.board_cards) != 3:
        raise ValueError("Initial particles require a 3-card flop in state.board_cards")
    pool, summary = build_weighted_combo_pool(state)
    if not pool:
        raise ValueError("Empty combo pool after dead-card filtering")
    rng = random.Random(seed)
    picked = weighted_sample_without_replacement(pool, n_particles, rng)
    tw = sum(w for _, w, _ in picked)
    particles: List[Particle] = []
    board = list(state.board_cards)
    for hc, w, label in picked:
        w_norm = w / tw if tw > 1e-15 else 1.0 / len(picked)
        p = Particle(
            combo=hc,
            weight=w_norm,
            preflop_bucket_label=label,
            current_bucket=CoarseBucket.AIR,
            made_category="",
            draw_category="",
            alive=True,
        )
        particles.append(p)
    reclassify_all_particles(particles, board)
    note = (
        f"Initialized {len(particles)} particles from preflop prior; "
        f"prior_summary={summary!r}"
    )
    return particles, note
