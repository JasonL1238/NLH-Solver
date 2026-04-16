"""Preflop state derivation from action history and strict validation."""

from __future__ import annotations

from typing import List, Optional

from .models import (
    ActionRecord, ActionType, DerivedState, Player, PokerState,
    Position, StackDepthBucket, Street,
)


# ---------------------------------------------------------------------------
# Stack-depth bucketing
# ---------------------------------------------------------------------------

def stack_depth_bucket(effective_stack_bb: float) -> StackDepthBucket:
    if effective_stack_bb <= 5:
        return StackDepthBucket.ULTRA_SHORT
    if effective_stack_bb <= 10:
        return StackDepthBucket.VERY_SHORT
    if effective_stack_bb <= 20:
        return StackDepthBucket.SHORT
    if effective_stack_bb <= 40:
        return StackDepthBucket.MEDIUM
    if effective_stack_bb <= 90:
        return StackDepthBucket.DEEP
    return StackDepthBucket.VERY_DEEP


# ---------------------------------------------------------------------------
# Position helpers
# ---------------------------------------------------------------------------

def _player_for_position(hero_pos: Position) -> dict:
    """Return mapping {Position -> Player}."""
    if hero_pos == Position.BTN_SB:
        return {Position.BTN_SB: Player.HERO, Position.BB: Player.VILLAIN}
    return {Position.BTN_SB: Player.VILLAIN, Position.BB: Player.HERO}


def _position_for_player(hero_pos: Position) -> dict:
    """Return mapping {Player -> Position}."""
    if hero_pos == Position.BTN_SB:
        return {Player.HERO: Position.BTN_SB, Player.VILLAIN: Position.BB}
    return {Player.HERO: Position.BB, Player.VILLAIN: Position.BTN_SB}


# ---------------------------------------------------------------------------
# Derive state fields from action history
# ---------------------------------------------------------------------------

def derive_preflop_state(
    action_history: List[ActionRecord],
    hero_position: Position,
    effective_stack_bb: float,
    small_blind_bb: float = 0.5,
    big_blind_bb: float = 1.0,
) -> dict:
    """Replay the preflop action history and derive all state fields.

    Returns a dict of direct + derived field values that can populate a
    PokerState and DerivedState.
    """
    pos_map = _player_for_position(hero_position)
    btn_player = pos_map[Position.BTN_SB]
    bb_player = pos_map[Position.BB]

    contributions = {Player.HERO: 0.0, Player.VILLAIN: 0.0}
    current_bet_level = 0.0
    last_aggressor: Optional[Player] = None
    num_raises = 0
    last_raise_increment = 0.0
    hand_over = False
    betting_closed = False

    hero_acted = False
    villain_acted = False

    # Track the current bet level to determine raise increments
    previous_bet_level = 0.0

    # Voluntary actions only (excludes POST_BLIND)
    voluntary_actions: List[ActionRecord] = []

    for rec in action_history:
        if rec.street != Street.PREFLOP:
            raise ValidationError("Only PREFLOP actions supported in this module")

        if rec.action_type == ActionType.POST_BLIND:
            contributions[rec.player] += rec.amount_added_bb
            current_bet_level = max(current_bet_level, contributions[rec.player])
            continue

        voluntary_actions.append(rec)

        if rec.player == Player.HERO:
            hero_acted = True
        else:
            villain_acted = True

        if rec.action_type == ActionType.FOLD:
            hand_over = True
            break

        if rec.action_type == ActionType.CHECK:
            pass  # no money change

        elif rec.action_type == ActionType.CALL:
            call_amount = current_bet_level - contributions[rec.player]
            contributions[rec.player] += call_amount

        elif rec.action_type in (ActionType.RAISE, ActionType.BET):
            new_total = contributions[rec.player] + rec.amount_added_bb
            raise_increment = new_total - current_bet_level
            last_raise_increment = raise_increment
            previous_bet_level = current_bet_level
            current_bet_level = new_total
            contributions[rec.player] = new_total
            last_aggressor = rec.player
            num_raises += 1

    pot = sum(contributions.values())

    # Determine current actor and closure
    current_actor: Optional[Player] = None
    if hand_over:
        betting_closed = True
        current_actor = None
    else:
        current_actor, betting_closed = _determine_next_actor(
            voluntary_actions, btn_player, bb_player,
            contributions, current_bet_level, effective_stack_bb,
        )

    # Amount hero needs to call
    hero_to_call = max(0.0, current_bet_level - contributions[Player.HERO])

    # Derived flags
    unopened = (num_raises == 0 and not any(
        a.action_type == ActionType.CALL for a in voluntary_actions
    ))
    facing_limp = False
    facing_open_raise = False
    facing_3bet = False
    facing_4bet = False
    facing_all_in = False

    if not hand_over and current_actor is not None:
        actor_contribution = contributions[current_actor]
        opponent = Player.VILLAIN if current_actor == Player.HERO else Player.HERO
        opponent_contrib = contributions[opponent]

        # Check if opponent is all-in
        if abs(opponent_contrib - effective_stack_bb) < 1e-9:
            facing_all_in = True

        if num_raises == 0:
            # No raise yet -- check for limp (BTN completed to 1bb without raising)
            btn_called = any(
                a.player == btn_player and a.action_type == ActionType.CALL
                for a in voluntary_actions
            )
            if btn_called:
                facing_limp = True
            else:
                unopened = True
        elif num_raises == 1:
            facing_open_raise = True
        elif num_raises == 2:
            facing_3bet = True
        elif num_raises >= 3:
            facing_4bet = True

    # Raise size metrics
    raise_size_in_bb: Optional[float] = None
    raise_size_multiple_bb: Optional[float] = None
    raise_size_multiple_prev: Optional[float] = None

    if num_raises > 0 and last_aggressor is not None:
        raise_size_in_bb = current_bet_level
        raise_size_multiple_bb = current_bet_level / big_blind_bb if big_blind_bb else None
        if previous_bet_level > 0:
            raise_size_multiple_prev = current_bet_level / previous_bet_level
        else:
            raise_size_multiple_prev = None

    # Stack ratios
    stack_to_open: Optional[float] = None
    stack_to_3bet: Optional[float] = None
    if num_raises >= 1:
        first_raise_level = _first_raise_level(action_history, contributions)
        if first_raise_level and first_raise_level > 0:
            stack_to_open = effective_stack_bb / first_raise_level
    if num_raises >= 2:
        if current_bet_level > 0:
            stack_to_3bet = effective_stack_bb / current_bet_level

    bucket = stack_depth_bucket(effective_stack_bb)

    hero_is_btn = (hero_position == Position.BTN_SB)

    derived = DerivedState(
        hero_is_first_to_act_preflop=hero_is_btn,
        hero_is_in_position_postflop_future_flag=hero_is_btn,
        unopened_pot=unopened,
        facing_limp=facing_limp,
        facing_open_raise=facing_open_raise,
        facing_3bet=facing_3bet,
        facing_4bet=facing_4bet,
        facing_all_in=facing_all_in,
        hero_already_acted_this_round=hero_acted,
        villain_already_acted_this_round=villain_acted,
        raise_size_in_bb=raise_size_in_bb,
        raise_size_as_multiple_of_bb=raise_size_multiple_bb,
        raise_size_as_multiple_of_previous_bet=raise_size_multiple_prev,
        stack_to_open_ratio=stack_to_open,
        stack_to_3bet_ratio=stack_to_3bet,
        stack_depth_bucket=bucket,
    )

    return {
        "pot_size_bb": pot,
        "current_bet_to_call_bb": hero_to_call,
        "hero_contribution_bb": contributions[Player.HERO],
        "villain_contribution_bb": contributions[Player.VILLAIN],
        "current_actor": current_actor,
        "betting_round_closed": betting_closed,
        "last_aggressor": last_aggressor,
        "number_of_raises_this_street": num_raises,
        "hand_over": hand_over,
        "derived": derived,
    }


def _first_raise_level(
    history: List[ActionRecord],
    contributions_unused: dict,
) -> Optional[float]:
    """Return the bet level after the first raise/bet in the history."""
    total = {}
    for rec in history:
        if rec.action_type == ActionType.POST_BLIND:
            total[rec.player] = total.get(rec.player, 0.0) + rec.amount_added_bb
        elif rec.action_type in (ActionType.RAISE, ActionType.BET):
            total[rec.player] = total.get(rec.player, 0.0) + rec.amount_added_bb
            return total[rec.player]
    return None


def _determine_next_actor(
    voluntary_actions: List[ActionRecord],
    btn_player: Player,
    bb_player: Player,
    contributions: dict,
    current_bet_level: float,
    effective_stack_bb: float,
) -> tuple:
    """Return (next_actor | None, betting_closed)."""
    if not voluntary_actions:
        # Only blinds posted -- BTN acts first preflop
        return btn_player, False

    last = voluntary_actions[-1]

    # After a call, check if the round closes
    if last.action_type == ActionType.CALL:
        # BB calling after BTN open/raise closes action
        # BTN calling a BB raise also closes
        # Generally: a call closes if both players have acted voluntarily
        # and contributions are equal
        hero_vol = any(a.player == Player.HERO for a in voluntary_actions)
        villain_vol = any(a.player == Player.VILLAIN for a in voluntary_actions)
        if hero_vol and villain_vol:
            return None, True
        # BTN limped (called), BB hasn't acted yet
        if last.player == btn_player:
            return bb_player, False

    if last.action_type == ActionType.CHECK:
        # If BB checks after BTN limp, round closes
        if last.player == bb_player:
            btn_limped = any(
                a.player == btn_player and a.action_type == ActionType.CALL
                for a in voluntary_actions
            )
            if btn_limped:
                return None, True
        # BTN can't check preflop first to act (would need to fold/call/raise)
        # If somehow both checked... shouldn't happen preflop normally
        return None, True

    if last.action_type in (ActionType.RAISE, ActionType.BET):
        opponent = bb_player if last.player == btn_player else btn_player
        # Check if raiser is all-in (opponent can still act)
        if abs(contributions[last.player] - effective_stack_bb) < 1e-9:
            # Raiser is all-in, opponent still gets to respond
            if contributions[opponent] < current_bet_level:
                return opponent, False
            return None, True
        # Check if opponent is all-in
        if abs(contributions[opponent] - effective_stack_bb) < 1e-9:
            return None, True
        return opponent, False

    if last.action_type == ActionType.FOLD:
        return None, True

    return None, True


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ValidationError(Exception):
    pass


def validate_preflop_state(state: PokerState) -> None:
    """Run all preflop validation checks. Raises ValidationError on failure."""
    if state.current_street != Street.PREFLOP:
        raise ValidationError("Only PREFLOP street is supported")

    if state.effective_stack_bb <= 0:
        raise ValidationError("Effective stack must be positive")

    if state.small_blind_bb <= 0 or state.big_blind_bb <= 0:
        raise ValidationError("Blinds must be positive")

    if state.small_blind_bb > state.big_blind_bb:
        raise ValidationError("Small blind must not exceed big blind")

    # Validate positions are complementary
    positions = {state.hero_position, state.villain_position}
    if positions != {Position.BTN_SB, Position.BB}:
        raise ValidationError("Positions must be one BTN_SB and one BB")

    # Validate action history
    _validate_action_sequence(state)

    # Validate contributions are non-negative and within stack
    if state.hero_contribution_bb < 0 or state.villain_contribution_bb < 0:
        raise ValidationError("Contributions cannot be negative")
    if state.hero_contribution_bb > state.effective_stack_bb + 1e-9:
        raise ValidationError("Hero contribution exceeds effective stack")
    if state.villain_contribution_bb > state.effective_stack_bb + 1e-9:
        raise ValidationError("Villain contribution exceeds effective stack")

    # Validate pot consistency
    expected_pot = state.hero_contribution_bb + state.villain_contribution_bb
    if abs(state.pot_size_bb - expected_pot) > 1e-9:
        raise ValidationError(
            f"Pot size {state.pot_size_bb} != hero({state.hero_contribution_bb}) "
            f"+ villain({state.villain_contribution_bb})"
        )


def _validate_action_sequence(state: PokerState) -> None:
    """Validate the preflop action sequence is legal."""
    history = state.action_history
    if not history:
        raise ValidationError("Action history must not be empty (blinds required)")

    pos_map = _position_for_player(state.hero_position)

    # First two actions must be POST_BLIND
    if len(history) < 2:
        raise ValidationError("Need at least 2 actions (blind posts)")

    sb_rec = history[0]
    bb_rec = history[1]

    if sb_rec.action_type != ActionType.POST_BLIND:
        raise ValidationError("First action must be POST_BLIND (SB)")
    if bb_rec.action_type != ActionType.POST_BLIND:
        raise ValidationError("Second action must be POST_BLIND (BB)")

    # SB must be BTN_SB player
    sb_expected = (
        Player.HERO if state.hero_position == Position.BTN_SB else Player.VILLAIN
    )
    bb_expected = (
        Player.HERO if state.hero_position == Position.BB else Player.VILLAIN
    )

    if sb_rec.player != sb_expected:
        raise ValidationError("SB blind must be posted by BTN_SB player")
    if bb_rec.player != bb_expected:
        raise ValidationError("BB blind must be posted by BB player")

    # Validate blind amounts
    if abs(sb_rec.amount_added_bb - state.small_blind_bb) > 1e-9:
        raise ValidationError("SB post amount doesn't match small_blind_bb")
    if abs(bb_rec.amount_added_bb - state.big_blind_bb) > 1e-9:
        raise ValidationError("BB post amount doesn't match big_blind_bb")

    # Validate voluntary actions alternate correctly and are legal
    bet_level = state.big_blind_bb
    contribs = {sb_expected: state.small_blind_bb, bb_expected: state.big_blind_bb}
    expected_actor = sb_expected  # BTN acts first preflop
    hand_done = False

    for i, rec in enumerate(history[2:], start=2):
        if hand_done:
            raise ValidationError(f"Action at index {i} after hand is over")

        if rec.street != Street.PREFLOP:
            raise ValidationError(f"Non-preflop action at index {i}")

        if rec.player != expected_actor:
            raise ValidationError(
                f"Wrong player at index {i}: expected {expected_actor.value}, "
                f"got {rec.player.value}"
            )

        other = bb_expected if rec.player == sb_expected else sb_expected

        if rec.action_type == ActionType.FOLD:
            hand_done = True

        elif rec.action_type == ActionType.CHECK:
            if contribs[rec.player] < bet_level - 1e-9:
                raise ValidationError(
                    f"Cannot check when facing a bet (index {i})"
                )

        elif rec.action_type == ActionType.CALL:
            needed = bet_level - contribs[rec.player]
            if needed < 1e-9:
                raise ValidationError(
                    f"Cannot call when there is nothing to call (index {i})"
                )
            contribs[rec.player] = bet_level

        elif rec.action_type in (ActionType.RAISE, ActionType.BET):
            new_total = contribs[rec.player] + rec.amount_added_bb
            if new_total < bet_level + 1e-9 and new_total < state.effective_stack_bb - 1e-9:
                raise ValidationError(
                    f"Raise to {new_total} is below current bet {bet_level} (index {i})"
                )
            if new_total > state.effective_stack_bb + 1e-9:
                raise ValidationError(
                    f"Raise to {new_total} exceeds effective stack {state.effective_stack_bb} (index {i})"
                )
            bet_level = new_total
            contribs[rec.player] = new_total

        else:
            raise ValidationError(f"Unexpected action type at index {i}: {rec.action_type}")

        if not hand_done:
            expected_actor = other

    # Validate sequence indices are sequential
    for i, rec in enumerate(history):
        if rec.sequence_index != i:
            raise ValidationError(
                f"Sequence index mismatch: expected {i}, got {rec.sequence_index}"
            )
