"""Street-specific action-context labels for postflop EV / debug (v1)."""

from __future__ import annotations

from poker_core.models import HandState, Street

from flop_spot.context import derive_flop_context


def postflop_action_context_label(state: HandState) -> str:
    """Human-readable spot label for turn/river (distinct from :class:`FlopContext`).

    Uses the same IP/OOP and PFR/PFC flags as flop context, but prefixes the
    current street so later-street policy stays explicit in debug output.
    """
    st = state.current_street
    if st not in (Street.TURN, Street.RIVER):
        raise ValueError("postflop_action_context_label is for TURN/RIVER only")

    ctx = derive_flop_context(state)
    pfx = "TURN_" if st == Street.TURN else "RIVER_"
    facing = state.current_bet_to_call_bb > 1e-9

    if facing:
        if state.number_of_raises_this_street >= 2:
            return pfx + "FACING_RAISE"
        return pfx + "FACING_BET"

    if ctx.hero_is_pfr:
        return pfx + ("PFR_IP_CHECKED_TO" if ctx.hero_is_ip else "PFR_OOP_FIRST_TO_ACT")
    return pfx + ("PFC_IP_CHECKED_TO" if ctx.hero_is_ip else "PFC_OOP_FIRST_TO_ACT")
