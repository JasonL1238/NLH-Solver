"""Central state reconstruction from action history.

``reconstruct_hand_state`` is the single canonical derivation path.  Every
mutable field on ``HandState`` is computed here by replaying the action
history against the hand configuration.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .models import (
    Action,
    ActionType,
    BETTING_ACTIONS,
    BETTING_STREETS,
    Card,
    DEAL_ACTIONS,
    HandConfig,
    HandState,
    Player,
    Position,
    Street,
)


class ReconstructionError(Exception):
    """Raised when the action history is structurally invalid."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NEXT_STREET_AFTER_DEAL = {
    ActionType.DEAL_FLOP: Street.FLOP,
    ActionType.DEAL_TURN: Street.TURN,
    ActionType.DEAL_RIVER: Street.RIVER,
}

_DEAL_CARD_COUNTS = {
    ActionType.DEAL_FLOP: 3,
    ActionType.DEAL_TURN: 1,
    ActionType.DEAL_RIVER: 1,
}

_DEAL_REQUIRED_PREV_STREET = {
    ActionType.DEAL_FLOP: Street.PREFLOP,
    ActionType.DEAL_TURN: Street.FLOP,
    ActionType.DEAL_RIVER: Street.TURN,
}


def _opponent(player: Player) -> Player:
    return Player.VILLAIN if player == Player.HERO else Player.HERO


def _first_actor_for_street(street: Street, config: HandConfig) -> Player:
    """Return the player who acts first on the given betting street."""
    if street == Street.PREFLOP:
        return config.btn_player  # BTN/SB acts first preflop
    # Postflop: BB acts first
    return config.bb_player


# ---------------------------------------------------------------------------
# Main reconstruction
# ---------------------------------------------------------------------------

def reconstruct_hand_state(
    config: HandConfig,
    action_history: List[Action],
    *,
    board_cards_mirror: Optional[List[Card]] = None,
) -> HandState:
    """Derive the full canonical ``HandState`` by replaying *action_history*.

    Parameters
    ----------
    config:
        Immutable hand setup (positions, stacks, blinds, hole cards).
    action_history:
        Ordered list of ``Action`` objects from blind posting through the
        current point in the hand.
    board_cards_mirror:
        Optional convenience mirror of board cards.  If provided it is
        validated against the board derived from DEAL_* actions in the
        history.  It is **never** used as a source of truth.

    Returns
    -------
    HandState with every derived field populated.

    Raises
    ------
    ReconstructionError on structurally invalid histories.
    """

    # -- cumulative totals across all streets --
    total_contrib: Dict[Player, float] = {Player.HERO: 0.0, Player.VILLAIN: 0.0}

    # -- per-street tracking (reset on each new street) --
    street = Street.PRE_HAND
    street_contrib: Dict[Player, float] = {Player.HERO: 0.0, Player.VILLAIN: 0.0}
    street_bet_level: float = 0.0
    last_full_raise_size: float = 0.0
    num_raises_this_street: int = 0
    last_aggressor: Optional[Player] = None

    # Who has *voluntarily* acted this street (excludes POST_BLIND)
    street_acted: Dict[Player, bool] = {Player.HERO: False, Player.VILLAIN: False}

    board: List[Card] = []
    hand_over = False
    fold_winner: Optional[Player] = None
    hero_all_in = False
    villain_all_in = False
    current_actor: Optional[Player] = None
    betting_closed = False
    blinds_posted = 0

    def _stack_cap(player: Player) -> float:
        return config.stack_cap_bb(player)

    for idx, action in enumerate(action_history):
        at = action.action_type

        # ------- DEAL actions -------
        if at in DEAL_ACTIONS:
            if hand_over:
                raise ReconstructionError(
                    f"DEAL action at index {idx} after hand is over")
            if not betting_closed:
                raise ReconstructionError(
                    f"DEAL action at index {idx} before betting round closed")

            expected_prev = _DEAL_REQUIRED_PREV_STREET[at]
            if street != expected_prev:
                raise ReconstructionError(
                    f"{at.value} at index {idx} requires street "
                    f"{expected_prev.value}, current is {street.value}")

            cards = action.cards or ()
            expected_count = _DEAL_CARD_COUNTS[at]
            if len(cards) != expected_count:
                raise ReconstructionError(
                    f"{at.value} at index {idx} requires {expected_count} "
                    f"card(s), got {len(cards)}")

            board.extend(cards)
            street = _NEXT_STREET_AFTER_DEAL[at]

            # Reset per-street state
            street_contrib = {Player.HERO: 0.0, Player.VILLAIN: 0.0}
            street_bet_level = 0.0
            last_full_raise_size = 0.0
            num_raises_this_street = 0
            last_aggressor = None
            street_acted = {Player.HERO: False, Player.VILLAIN: False}

            # Determine actor & closure for new street
            if hero_all_in and villain_all_in:
                betting_closed = True
                current_actor = None
            elif hero_all_in or villain_all_in:
                betting_closed = True
                current_actor = None
            else:
                betting_closed = False
                current_actor = _first_actor_for_street(street, config)

            continue

        # ------- POST_BLIND -------
        if at == ActionType.POST_BLIND:
            if blinds_posted >= 2:
                raise ReconstructionError(
                    f"Extra POST_BLIND at index {idx}")
            if action.player is None:
                raise ReconstructionError(
                    f"POST_BLIND at index {idx} missing player")
            if action.amount_to_bb is None:
                raise ReconstructionError(
                    f"POST_BLIND at index {idx} missing amount")

            player = action.player
            amt = action.amount_to_bb

            if blinds_posted == 0:
                if player != config.btn_player:
                    raise ReconstructionError(
                        f"First blind must be posted by BTN/SB "
                        f"({config.btn_player.value}), got {player.value}")
                if abs(amt - config.small_blind_bb) > 1e-9:
                    raise ReconstructionError(
                        f"SB post amount {amt} != small_blind_bb "
                        f"{config.small_blind_bb}")
            else:
                if player != config.bb_player:
                    raise ReconstructionError(
                        f"Second blind must be posted by BB "
                        f"({config.bb_player.value}), got {player.value}")
                if abs(amt - config.big_blind_bb) > 1e-9:
                    raise ReconstructionError(
                        f"BB post amount {amt} != big_blind_bb "
                        f"{config.big_blind_bb}")

            total_contrib[player] += amt
            street_contrib[player] += amt
            street_bet_level = max(street_bet_level, street_contrib[player])
            last_full_raise_size = max(last_full_raise_size, config.big_blind_bb)
            blinds_posted += 1

            if blinds_posted == 2:
                street = Street.PREFLOP
                current_actor = _first_actor_for_street(Street.PREFLOP, config)
                betting_closed = False

            if abs(total_contrib[player] - _stack_cap(player)) < 1e-9:
                if player == Player.HERO:
                    hero_all_in = True
                else:
                    villain_all_in = True

            continue

        # ------- Betting actions -------
        if at not in BETTING_ACTIONS:
            raise ReconstructionError(
                f"Unknown action type {at} at index {idx}")

        if action.player is None:
            raise ReconstructionError(
                f"Betting action at index {idx} missing player")

        player = action.player

        if hand_over:
            raise ReconstructionError(
                f"Action at index {idx} after hand is over")
        if betting_closed:
            raise ReconstructionError(
                f"Action at index {idx} after betting round is closed")
        if street not in BETTING_STREETS:
            raise ReconstructionError(
                f"Betting action at index {idx} on street {street.value}")
        if player != current_actor:
            raise ReconstructionError(
                f"Wrong player at index {idx}: expected "
                f"{current_actor.value if current_actor else 'None'}, "
                f"got {player.value}")

        opp = _opponent(player)
        to_call = max(0.0, street_bet_level - street_contrib[player])
        player_remaining = _stack_cap(player) - total_contrib[player]

        if at == ActionType.FOLD:
            if to_call < 1e-9:
                raise ReconstructionError(
                    f"Cannot fold when there is nothing to call (index {idx})")
            hand_over = True
            fold_winner = opp
            betting_closed = True
            current_actor = None

        elif at == ActionType.CHECK:
            if to_call > 1e-9:
                raise ReconstructionError(
                    f"Cannot check when facing a bet of {to_call} "
                    f"(index {idx})")
            street_acted[player] = True
            # Determine next actor / closure
            current_actor, betting_closed = _next_after_check(
                player, opp, street, config, street_acted,
                hero_all_in, villain_all_in,
            )

        elif at == ActionType.CALL:
            if to_call < 1e-9:
                raise ReconstructionError(
                    f"Cannot call when there is nothing to call (index {idx})")

            actual_call = min(to_call, player_remaining)
            expected_call_to = street_contrib[player] + actual_call

            if action.amount_to_bb is not None:
                if abs(action.amount_to_bb - expected_call_to) > 1e-9:
                    raise ReconstructionError(
                        f"CALL amount_to_bb {action.amount_to_bb} does not "
                        f"match expected {expected_call_to} (index {idx})")

            total_contrib[player] += actual_call
            street_contrib[player] += actual_call

            if abs(total_contrib[player] - _stack_cap(player)) < 1e-9:
                if player == Player.HERO:
                    hero_all_in = True
                else:
                    villain_all_in = True

            street_acted[player] = True
            current_actor, betting_closed = _next_after_call(
                player, opp, street, config, street_acted,
                hero_all_in, villain_all_in,
            )

        elif at == ActionType.BET:
            if to_call > 1e-9:
                raise ReconstructionError(
                    f"Cannot BET when facing a bet (use RAISE); index {idx}")
            if num_raises_this_street > 0:
                raise ReconstructionError(
                    f"Cannot BET after a bet/raise already made this street "
                    f"(use RAISE); index {idx}")
            if action.amount_to_bb is None:
                raise ReconstructionError(
                    f"BET at index {idx} missing amount_to_bb")

            bet_to = action.amount_to_bb
            added = bet_to - street_contrib[player]

            if added > player_remaining + 1e-9:
                raise ReconstructionError(
                    f"BET to {bet_to} exceeds remaining stack "
                    f"{player_remaining + street_contrib[player]} (index {idx})")

            if bet_to < config.big_blind_bb - 1e-9 and added < player_remaining - 1e-9:
                raise ReconstructionError(
                    f"BET to {bet_to} is below minimum bet of "
                    f"{config.big_blind_bb} (index {idx})")

            total_contrib[player] += added
            street_contrib[player] = bet_to
            street_bet_level = bet_to
            last_full_raise_size = bet_to
            num_raises_this_street += 1
            last_aggressor = player
            street_acted[player] = True

            if abs(total_contrib[player] - _stack_cap(player)) < 1e-9:
                if player == Player.HERO:
                    hero_all_in = True
                else:
                    villain_all_in = True

            if opp == Player.HERO and hero_all_in:
                current_actor = None
                betting_closed = True
            elif opp == Player.VILLAIN and villain_all_in:
                current_actor = None
                betting_closed = True
            else:
                current_actor = opp
                betting_closed = False

        elif at == ActionType.RAISE:
            if to_call < 1e-9 and num_raises_this_street == 0:
                raise ReconstructionError(
                    f"Cannot RAISE when there is nothing to call and no "
                    f"prior bet (use BET); index {idx}")
            if action.amount_to_bb is None:
                raise ReconstructionError(
                    f"RAISE at index {idx} missing amount_to_bb")

            raise_to = action.amount_to_bb
            added = raise_to - street_contrib[player]

            if added > player_remaining + 1e-9:
                raise ReconstructionError(
                    f"RAISE to {raise_to} exceeds remaining stack "
                    f"(index {idx})")

            min_legal_raise_to = street_bet_level + last_full_raise_size
            is_all_in_raise = abs(total_contrib[player] + added - _stack_cap(player)) < 1e-9

            if raise_to < street_bet_level + 1e-9 and not is_all_in_raise:
                raise ReconstructionError(
                    f"RAISE to {raise_to} does not exceed current bet "
                    f"{street_bet_level} (index {idx})")

            if raise_to < min_legal_raise_to - 1e-9 and not is_all_in_raise:
                raise ReconstructionError(
                    f"RAISE to {raise_to} is below min raise "
                    f"{min_legal_raise_to} (index {idx})")

            raise_increment = raise_to - street_bet_level
            is_full_raise = raise_increment >= last_full_raise_size - 1e-9

            total_contrib[player] += added
            street_contrib[player] = raise_to
            street_bet_level = raise_to
            if is_full_raise:
                last_full_raise_size = raise_increment
            num_raises_this_street += 1
            last_aggressor = player
            street_acted[player] = True

            if abs(total_contrib[player] - _stack_cap(player)) < 1e-9:
                if player == Player.HERO:
                    hero_all_in = True
                else:
                    villain_all_in = True

            if opp == Player.HERO and hero_all_in:
                current_actor = None
                betting_closed = True
            elif opp == Player.VILLAIN and villain_all_in:
                current_actor = None
                betting_closed = True
            else:
                current_actor = opp
                betting_closed = False

    # --- Post-replay terminal / closure logic ---
    showdown_ready = False
    awaiting_runout = False

    if hand_over:
        pass  # fold already set everything
    elif betting_closed and not hand_over:
        if hero_all_in or villain_all_in:
            if street == Street.RIVER:
                showdown_ready = True
                hand_over = True
                street = Street.SHOWDOWN
            else:
                awaiting_runout = True
        elif street == Street.RIVER:
            showdown_ready = True
            hand_over = True
            street = Street.SHOWDOWN

    # Compute current_bet_to_call_bb for the current actor
    if current_actor is not None:
        cbtc = max(0.0, street_bet_level - street_contrib[current_actor])
    else:
        cbtc = 0.0

    pot = total_contrib[Player.HERO] + total_contrib[Player.VILLAIN]

    # Validate board mirror if provided
    if board_cards_mirror is not None:
        if len(board_cards_mirror) != len(board):
            raise ReconstructionError(
                f"board_cards_mirror length {len(board_cards_mirror)} does "
                f"not match derived board length {len(board)}")
        for i, (m, d) in enumerate(zip(board_cards_mirror, board)):
            if m != d:
                raise ReconstructionError(
                    f"board_cards_mirror[{i}] = {m} != derived {d}")

    return HandState(
        config=config,
        action_history=list(action_history),
        current_street=street,
        board_cards=list(board),
        pot_size_bb=pot,
        hero_contribution_bb=total_contrib[Player.HERO],
        villain_contribution_bb=total_contrib[Player.VILLAIN],
        current_bet_to_call_bb=cbtc,
        current_actor=current_actor,
        last_aggressor=last_aggressor,
        number_of_raises_this_street=num_raises_this_street,
        last_full_raise_size=last_full_raise_size,
        street_contrib_hero=street_contrib[Player.HERO],
        street_contrib_villain=street_contrib[Player.VILLAIN],
        street_bet_level=street_bet_level,
        betting_round_closed=betting_closed,
        hand_over=hand_over,
        showdown_ready=showdown_ready,
        fold_winner=fold_winner,
        awaiting_runout=awaiting_runout,
        hero_all_in=hero_all_in,
        villain_all_in=villain_all_in,
    )


# ---------------------------------------------------------------------------
# Next-actor helpers
# ---------------------------------------------------------------------------

def _next_after_check(
    player: Player,
    opp: Player,
    street: Street,
    config: HandConfig,
    street_acted: Dict[Player, bool],
    hero_ai: bool,
    villain_ai: bool,
) -> tuple:
    """Return ``(next_actor | None, betting_closed)`` after a CHECK."""
    if street_acted[opp]:
        return None, True

    # Opponent hasn't acted yet – they get to act
    opp_is_all_in = (hero_ai if opp == Player.HERO else villain_ai)
    if opp_is_all_in:
        return None, True
    return opp, False


def _next_after_call(
    player: Player,
    opp: Player,
    street: Street,
    config: HandConfig,
    street_acted: Dict[Player, bool],
    hero_ai: bool,
    villain_ai: bool,
) -> tuple:
    """Return ``(next_actor | None, betting_closed)`` after a CALL."""
    # Special case: HU preflop limp.  BTN/SB calls (completes to 1bb) but
    # BB has not yet voluntarily acted, so BB gets an option.
    if not street_acted[opp]:
        opp_ai = hero_ai if opp == Player.HERO else villain_ai
        if opp_ai:
            return None, True
        return opp, False

    # Both players have acted – call closes the round.
    return None, True
