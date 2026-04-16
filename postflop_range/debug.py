"""Debug summaries and manual inspection helpers."""

from __future__ import annotations

from typing import Any, List, Optional

from poker_core.models import Action, ActionType, HandState, Player
from poker_core.reconstruction import reconstruct_hand_state

from .particles import Particle
from .range_tracker import VillainParticleTracker


def build_debug_dict(tracker: VillainParticleTracker) -> dict[str, Any]:
    s = tracker.summarize()
    top = tracker.top_particles(15)
    return {
        "particle_count": s["particle_count"],
        "effective_sample_size": s["effective_sample_size"],
        "total_weight": s["total_weight"],
        "top_particles": [
            {
                "combo": p.combo_str,
                "weight": round(p.weight, 5),
                "bucket": p.current_bucket.value,
                "preflop_label": p.preflop_bucket_label,
            }
            for p in top
        ],
        "bucket_weight_summary": tracker.bucket_summary(),
        "last_update_type": tracker.last_update_type,
        "last_update_notes": tracker.last_update_notes,
        "street": s["street"],
        "board_cards": s["board_cards"],
    }


def format_tracker_summary(tracker: VillainParticleTracker) -> str:
    d = build_debug_dict(tracker)
    lines = [
        f"particles={d['particle_count']} ESS={d['effective_sample_size']} "
        f"wtot={d['total_weight']}",
        f"last={d['last_update_type']}: {d['last_update_notes']}",
        f"buckets={d['bucket_weight_summary']}",
        "top:",
    ]
    for row in d["top_particles"][:10]:
        lines.append(f"  {row['combo']} w={row['weight']} {row['bucket']} ({row['preflop_label']})")
    return "\n".join(lines)


def print_top_particles(tracker: VillainParticleTracker, n: int = 20) -> None:
    for p in tracker.top_particles(n):
        print(f"{p.combo_str}  w={p.weight:.5f}  {p.current_bucket.value}  ({p.preflop_bucket_label})")


def print_bucket_summary(tracker: VillainParticleTracker) -> None:
    for k, v in sorted(tracker.bucket_summary().items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:.4f}")


def apply_board_and_print(tracker: VillainParticleTracker, state: HandState) -> None:
    tracker.apply_board_from_state(state)
    print(format_tracker_summary(tracker))


def apply_action_and_print(
    tracker: VillainParticleTracker,
    state_after: HandState,
    action: Action,
) -> None:
    tracker.apply_villain_action(state_after, action=action)
    print(format_tracker_summary(tracker))


def build_tracker_from_state(
    state: HandState,
    *,
    n_particles: int = 300,
    seed: int = 42,
) -> VillainParticleTracker:
    """Initial tracker at flop (3-card board) only."""
    if len(state.board_cards) != 3:
        raise ValueError("build_tracker_from_state expects flop with 3 board cards")
    return VillainParticleTracker.build_initial_from_state(
        state, n_particles=n_particles, seed=seed,
    )


def replay_through_state(
    state: HandState,
    *,
    n_particles: int = 300,
    seed: int = 42,
) -> VillainParticleTracker:
    """Replay postflop history: board deals + villain betting reweights."""
    cfg = state.config
    hist = state.action_history
    flop_idx = None
    for i, a in enumerate(hist):
        if a.action_type == ActionType.DEAL_FLOP:
            flop_idx = i
            break
    if flop_idx is None:
        raise ValueError("No DEAL_FLOP in history")
    st0 = reconstruct_hand_state(cfg, hist[: flop_idx + 1])
    tr = VillainParticleTracker.build_initial_from_state(
        st0, n_particles=n_particles, seed=seed,
    )
    for j in range(flop_idx + 1, len(hist)):
        prefix = hist[: j + 1]
        st = reconstruct_hand_state(cfg, prefix)
        act = prefix[-1]
        if act.action_type in (
            ActionType.DEAL_TURN,
            ActionType.DEAL_RIVER,
        ):
            tr.apply_board_from_state(st)
        elif act.player == Player.VILLAIN and act.action_type in (
            ActionType.CHECK,
            ActionType.CALL,
            ActionType.BET,
            ActionType.RAISE,
        ):
            tr.apply_villain_action(st, action=act)
    return tr
