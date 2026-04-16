"""Normalization, ESS, and multinomial resampling."""

from __future__ import annotations

import random
from typing import List

from .particles import Particle


def normalize_weights(particles: List[Particle]) -> float:
    """Normalize alive particle weights to sum to 1. Returns total before norm."""
    alive = [p for p in particles if p.alive]
    tot = sum(p.weight for p in alive)
    if tot < 1e-15:
        for p in particles:
            p.weight = 0.0
        return 0.0
    for p in particles:
        if p.alive:
            p.weight /= tot
    return tot


def effective_sample_size(particles: List[Particle]) -> float:
    """N_eff = 1 / sum(w_i^2) for normalized weights among alive particles."""
    alive = [p for p in particles if p.alive and p.weight > 1e-15]
    if not alive:
        return 0.0
    s = sum(p.weight * p.weight for p in alive)
    if s < 1e-15:
        return 0.0
    return 1.0 / s


def multinomial_resample(
    particles: List[Particle],
    n_target: int,
    rng: random.Random,
) -> List[Particle]:
    """Resample ``n_target`` particles with replacement ∝ weight; equal new weights."""
    alive = [p for p in particles if p.alive and p.weight > 1e-15]
    if not alive or n_target <= 0:
        return []
    weights = [p.weight for p in alive]
    tot = sum(weights)
    if tot < 1e-15:
        return []
    wn = [w / tot for w in weights]
    idxs = rng.choices(range(len(alive)), weights=wn, k=n_target)
    out: List[Particle] = []
    for idx in idxs:
        src = alive[idx]
        out.append(
            Particle(
                combo=src.combo,
                weight=1.0 / n_target,
                preflop_bucket_label=src.preflop_bucket_label,
                current_bucket=src.current_bucket,
                made_category=src.made_category,
                draw_category=src.draw_category,
                alive=True,
            ),
        )
    eq = 1.0 / len(out)
    for p in out:
        p.weight = eq
    return out
