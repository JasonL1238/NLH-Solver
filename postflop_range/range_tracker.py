"""Orchestrator: incremental particle range over streets (canonical HandState).

Architecture
------------
- **Initial:** ``build_initial_from_state`` samples weighted combos from
  ``flop_equity.range_model.prior_label_weights`` (preflop line only), filters
  dead cards, normalizes, classifies on the flop.
- **Board:** ``apply_board_from_state`` removes particles conflicting with new
  public cards, reclassifies survivors (flop ``classify_hand``; turn/river
  ``best_hand_rank_hole_board`` + coarse buckets). Does not rebuild from preflop.
- **Actions:** ``apply_villain_action`` multiplies weights by coarse bucket ×
  action tables in ``action_update``; then normalizes.
- **Resample:** ``maybe_resample`` if ESS ``< resample_min_eff_frac * n_alive``;
  multinomial resample with deterministic RNG offset.
- **Rollouts:** ``export_weighted_range_for_rollouts`` and
  ``sample_weighted_villain_combo`` feed ``estimate_showdown_equity`` / MC.

Assumptions: HU only; coarse compatibility (no exact combo action likelihood);
turn/river strength from best 5-of-(2+board); exploit may later tune tables via
``profile`` on ``VillainActionContext``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from poker_core.models import Action, ActionType, Card, HandState, HoleCards, Player, Street

from .action_update import (
    AggressionSize,
    VillainActionContext,
    apply_compatibility_to_particles,
    infer_aggression_size,
)
from .board_update import filter_particles_for_new_dead, reclassify_all_particles
from .initial_range import build_initial_particles
from .particles import Particle, RangeDegenerateError, alive_particles, total_weight
from .resampling import effective_sample_size, multinomial_resample, normalize_weights


@dataclass
class TrackerConfig:
    """Tuning for resampling."""

    n_particles: int = 400
    resample_min_eff_frac: float = 0.25
    resample_rng_seed: int = 0


@dataclass
class VillainParticleTracker:
    """Live evolving villain range as weighted particles."""

    particles: List[Particle] = field(default_factory=list)
    board_snapshot: List[Card] = field(default_factory=list)
    street: Street = Street.FLOP
    n_particles_target: int = 400
    last_update_type: str = "NONE"
    last_update_notes: str = ""
    prior_summary: str = ""
    config: TrackerConfig = field(default_factory=TrackerConfig)
    _rng_counter: int = 0

    @classmethod
    def build_initial_from_state(
        cls,
        state: HandState,
        *,
        n_particles: int = 400,
        seed: int = 42,
        config: Optional[TrackerConfig] = None,
    ) -> VillainParticleTracker:
        if len(state.board_cards) != 3:
            raise ValueError("build_initial_from_state requires exactly 3 board cards")
        parts, note = build_initial_particles(state, n_particles=n_particles, seed=seed)
        cfg = config or TrackerConfig(n_particles=n_particles)
        tr = cls(
            particles=parts,
            board_snapshot=list(state.board_cards),
            street=Street.FLOP,
            n_particles_target=n_particles,
            last_update_type="INIT",
            last_update_notes=note,
            prior_summary=note,
            config=cfg,
        )
        tr._post_update_normalize("INIT", note)
        return tr

    def _post_update_normalize(self, kind: str, notes: str) -> None:
        self.last_update_type = kind
        self.last_update_notes = notes
        tw = normalize_weights(self.particles)
        if tw < 1e-15:
            raise RangeDegenerateError("All particle weights zero after update")

    def apply_board_from_state(self, state: HandState) -> None:
        """Incremental board: kill combos conflicting with new cards; reclassify."""
        new_board = list(state.board_cards)
        old = set(self.board_snapshot)
        new_set = set(new_board)
        added = new_set - old
        if not added:
            self.board_snapshot = new_board
            self.street = state.current_street
            return
        killed, n1 = filter_particles_for_new_dead(self.particles, added)
        self.board_snapshot = new_board
        self.street = state.current_street
        n2 = reclassify_all_particles(self.particles, self.board_snapshot)
        self._post_update_normalize("BOARD", f"{n1} {n2}")
        self.maybe_resample()

    def apply_villain_action(
        self,
        state_after: HandState,
        *,
        action: Action,
    ) -> None:
        """Reweight after a villain betting action (state is after the action)."""
        if action.player != Player.VILLAIN:
            return
        if action.action_type not in (
            ActionType.CHECK,
            ActionType.CALL,
            ActionType.BET,
            ActionType.RAISE,
        ):
            return
        pot = state_after.pot_size_bb
        to_call = state_after.current_bet_to_call_bb
        if action.action_type in (ActionType.BET, ActionType.RAISE):
            agg = infer_aggression_size(
                action_type=action.action_type,
                pot_after_villain_bb=pot,
                hero_to_call_bb=to_call,
                raises_this_street=state_after.number_of_raises_this_street,
            )
            if state_after.number_of_raises_this_street >= 2:
                agg = AggressionSize.LARGE
        else:
            agg = AggressionSize.NONE
        ctx = VillainActionContext(
            street=state_after.current_street,
            action_type=action.action_type,
            aggression_size=agg,
            raises_this_street_before=max(0, state_after.number_of_raises_this_street - 1),
        )
        note = apply_compatibility_to_particles(self.particles, ctx)
        self._post_update_normalize("ACTION", note)
        self.maybe_resample()

    def maybe_resample(self) -> None:
        """Resample if ESS too low (deterministic sub-seed)."""
        alive = alive_particles(self.particles)
        if len(alive) < 2:
            return
        n_eff = effective_sample_size(self.particles)
        thresh = self.config.resample_min_eff_frac * len(alive)
        if n_eff >= thresh:
            return
        self._rng_counter += 1
        rng = random.Random(self.config.resample_rng_seed + self._rng_counter)
        new_parts = multinomial_resample(
            self.particles,
            min(self.n_particles_target, len(alive)),
            rng,
        )
        if not new_parts:
            return
        self.particles = new_parts
        reclassify_all_particles(self.particles, self.board_snapshot)
        normalize_weights(self.particles)
        self.last_update_type = "RESAMPLE"
        self.last_update_notes = (
            f"ESS below threshold; resampled to {len(new_parts)} particles"
        )

    def export_weighted_range_for_rollouts(self) -> List[Tuple[HoleCards, float]]:
        """Shape expected by ``estimate_showdown_equity`` / flop MC."""
        out: List[Tuple[HoleCards, float]] = []
        for p in alive_particles(self.particles):
            out.append((p.combo, float(p.weight)))
        if not out:
            raise RangeDegenerateError("No alive particles for rollouts")
        return out

    def sample_weighted_villain_combo(self, rng: random.Random) -> HoleCards:
        """One combo draw ∝ particle weights (dead-safe by construction)."""
        alive = alive_particles(self.particles)
        if not alive:
            raise RangeDegenerateError("No alive particles to sample")
        w = [p.weight for p in alive]
        tot = sum(w)
        r = rng.random() * tot
        cum = 0.0
        for p, wi in zip(alive, w):
            cum += wi
            if r <= cum:
                return p.combo
        return alive[-1].combo

    def sample_many_weighted_villain_combos(
        self,
        n: int,
        rng: random.Random,
    ) -> List[HoleCards]:
        return [self.sample_weighted_villain_combo(rng) for _ in range(n)]

    def top_particles(self, k: int = 20) -> List[Particle]:
        alive = alive_particles(self.particles)
        return sorted(alive, key=lambda p: p.weight, reverse=True)[:k]

    def bucket_summary(self) -> dict[str, float]:
        d: dict[str, float] = {}
        for p in alive_particles(self.particles):
            key = p.current_bucket.value
            d[key] = d.get(key, 0.0) + p.weight
        return d

    def summarize(self) -> dict[str, Any]:
        alive = alive_particles(self.particles)
        return {
            "particle_count": len(alive),
            "effective_sample_size": round(effective_sample_size(self.particles), 4),
            "total_weight": round(total_weight(self.particles), 6),
            "street": self.street.value,
            "board_cards": len(self.board_snapshot),
            "last_update_type": self.last_update_type,
            "prior_summary": self.prior_summary,
        }
