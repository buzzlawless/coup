"""The rules engine.

The whole game is ``apply(state, decision, rng)``.  ``legal_decisions(state)``
enumerates what the player returned by ``to_act(state)`` may do, and every
decision it returns is accepted by ``apply``.

Randomness enters in exactly three places -- the deal, the replacement card
after a survived challenge, and the two cards drawn for an Exchange -- and all
three go through ``_draw``.  If you later want explicit chance nodes for exact
tree traversal, that is the one function to lift out; the deck is an unordered
bag, so the chance distribution at any point is just its multiset.
"""

from __future__ import annotations

import itertools
import random

from .actions import ACTIONS, EXCHANGE_DRAW, STEAL_AMOUNT, ActionKind
from .cards import FULL_DECK, Card
from .decisions import (
    Block,
    Challenge,
    ChooseAction,
    Decision,
    Discard,
    ExchangeReturn,
    Pass,
)
from .state import (
    GameState,
    LossCause,
    PendingAction,
    PendingBlock,
    Phase,
    PlayerState,
    Resume,
    RuleConfig,
)

MIN_PLAYERS = 2
MAX_PLAYERS = 6
STARTING_COINS = 2
STARTING_INFLUENCE = 2


class IllegalDecision(Exception):
    """Raised when a decision is not available to the player to act."""


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def new_game(
    num_players: int,
    rng: random.Random | None = None,
    config: RuleConfig | None = None,
) -> GameState:
    if not MIN_PLAYERS <= num_players <= MAX_PLAYERS:
        raise ValueError(f"Coup is for {MIN_PLAYERS}-{MAX_PLAYERS} players")
    rng = rng or random.Random()
    config = config or RuleConfig()

    state = GameState(
        config=config,
        players=[PlayerState(seat, coins=STARTING_COINS) for seat in range(num_players)],
        deck=list(FULL_DECK),
    )
    for player in state.players:
        for _ in range(STARTING_INFLUENCE):
            player.influence.append(_draw(state, rng))
    if num_players == MIN_PLAYERS and config.two_player_start_handicap:
        state.players[0].coins = 1

    state.phase = Phase.ACTION
    state.turn = 0
    return state


# ---------------------------------------------------------------------------
# Querying
# ---------------------------------------------------------------------------


def to_act(state: GameState) -> int | None:
    """The player the game is currently waiting on, or None if it is over."""
    if state.phase is Phase.ACTION:
        return state.turn
    if state.phase in (Phase.ACTION_CHALLENGE, Phase.BLOCK, Phase.BLOCK_CHALLENGE):
        return state.responders[0]
    if state.phase is Phase.LOSE_INFLUENCE:
        return state.losses[0][0]
    if state.phase is Phase.EXCHANGE_RETURN:
        assert state.pending_action is not None
        return state.pending_action.actor
    return None


def legal_decisions(state: GameState) -> list[Decision]:
    phase = state.phase

    if phase is Phase.ACTION:
        return _legal_actions(state)

    if phase in (Phase.ACTION_CHALLENGE, Phase.BLOCK_CHALLENGE):
        return [Pass(), Challenge()]

    if phase is Phase.BLOCK:
        assert state.pending_action is not None
        spec = ACTIONS[state.pending_action.kind]
        return [Pass()] + [Block(c) for c in spec.blockers]

    if phase is Phase.LOSE_INFLUENCE:
        seat = state.losses[0][0]
        return [Discard(c) for c in sorted(set(state.players[seat].influence))]

    if phase is Phase.EXCHANGE_RETURN:
        assert state.pending_action is not None
        player = state.players[state.pending_action.actor]
        pool = sorted(player.influence + state.exchange_drawn)
        keep_count = len(player.influence)
        combos = {tuple(c) for c in itertools.combinations(pool, keep_count)}
        return [ExchangeReturn(keep) for keep in sorted(combos)]

    return []


def _legal_actions(state: GameState) -> list[Decision]:
    actor = state.players[state.turn]
    targets = [s for s in state.live_players() if s != actor.seat]

    if actor.coins >= state.config.mandatory_coup_threshold:
        return [ChooseAction(ActionKind.COUP, t) for t in targets]

    moves: list[Decision] = [
        ChooseAction(ActionKind.INCOME),
        ChooseAction(ActionKind.FOREIGN_AID),
        ChooseAction(ActionKind.TAX),
        ChooseAction(ActionKind.EXCHANGE),
    ]
    if actor.coins >= ACTIONS[ActionKind.COUP].cost:
        moves += [ChooseAction(ActionKind.COUP, t) for t in targets]
    if actor.coins >= ACTIONS[ActionKind.ASSASSINATE].cost:
        moves += [ChooseAction(ActionKind.ASSASSINATE, t) for t in targets]
    moves += [
        ChooseAction(ActionKind.STEAL, t)
        for t in targets
        if state.config.allow_stealing_from_zero or state.players[t].coins > 0
    ]
    return moves


# ---------------------------------------------------------------------------
# Stepping
# ---------------------------------------------------------------------------


def apply(
    state: GameState,
    decision: Decision,
    rng: random.Random | None = None,
    validate: bool = True,
) -> GameState:
    """Advance the game by one decision.

    Mutates ``state`` and returns it; clone first if you need the predecessor.

    Validation rebuilds the legal-move list, which roughly doubles the cost of a
    step.  A simulation loop that picks its move *from* ``legal_decisions`` has
    already paid for that, and can pass ``validate=False``; anything taking
    moves from outside should not.
    """
    rng = rng or random.Random()
    if state.game_over:
        raise IllegalDecision("the game is over")
    if validate and decision not in legal_decisions(state):
        raise IllegalDecision(f"{decision!r} is not legal in phase {state.phase.name}")

    phase = state.phase
    if phase is Phase.ACTION:
        _declare_action(state, decision, rng)
    elif phase is Phase.ACTION_CHALLENGE:
        _respond_to_action(state, decision, rng)
    elif phase is Phase.BLOCK:
        _respond_with_block(state, decision, rng)
    elif phase is Phase.BLOCK_CHALLENGE:
        _respond_to_block(state, decision, rng)
    elif phase is Phase.LOSE_INFLUENCE:
        _apply_discard(state, decision, rng)
    elif phase is Phase.EXCHANGE_RETURN:
        _apply_exchange_return(state, decision, rng)
    return state


def _declare_action(state: GameState, decision: ChooseAction, rng: random.Random) -> None:
    spec = ACTIONS[decision.kind]
    actor = state.players[state.turn]

    actor.coins -= spec.cost
    state.pending_action = PendingAction(actor.seat, decision.kind, decision.target, spec.cost)
    state.pending_block = None
    state.history.append(("action", actor.seat, int(decision.kind), decision.target))

    state.resume = Resume.BLOCK_WINDOW
    if spec.challengeable:
        state.responders = _poll_order(state, exclude=actor.seat)
        state.phase = Phase.ACTION_CHALLENGE
        return
    _continue(state, rng)


def _respond_to_action(state: GameState, decision: Decision, rng: random.Random) -> None:
    assert state.pending_action is not None
    challenger = state.responders.pop(0)

    if isinstance(decision, Pass):
        if state.responders:
            return  # keep polling; still in ACTION_CHALLENGE
        _continue(state, rng)  # nobody challenged
        return

    action = state.pending_action
    claim = ACTIONS[action.kind].claim
    assert claim is not None
    honest = _resolve_challenge(state, challenger, action.actor, claim, rng)

    if honest:
        # The claim held: the challenger pays, and the action carries on.
        state.resume = Resume.BLOCK_WINDOW
    else:
        # The actor was bluffing: the action never happened.
        if state.config.refund_cost_on_caught_bluff:
            state.players[action.actor].coins += action.paid
            action.paid = 0
        state.resume = Resume.END_TURN
    _continue(state, rng)


def _respond_with_block(state: GameState, decision: Decision, rng: random.Random) -> None:
    blocker = state.responders.pop(0)

    if isinstance(decision, Pass):
        if state.responders:
            return  # keep polling; still in BLOCK
        _continue(state, rng)  # nobody blocked; resume is RESOLVE_ACTION
        return

    assert isinstance(decision, Block)
    state.pending_block = PendingBlock(blocker, decision.character)
    state.history.append(("block", blocker, int(decision.character)))
    state.responders = _poll_order(state, exclude=blocker)
    state.phase = Phase.BLOCK_CHALLENGE


def _respond_to_block(state: GameState, decision: Decision, rng: random.Random) -> None:
    assert state.pending_block is not None
    block = state.pending_block
    challenger = state.responders.pop(0)

    if isinstance(decision, Pass):
        if state.responders:
            return  # keep polling; still in BLOCK_CHALLENGE
        state.resume = Resume.END_TURN  # the block stands
        _continue(state, rng)
        return

    honest = _resolve_challenge(state, challenger, block.blocker, block.character, rng)
    if honest:
        state.resume = Resume.END_TURN  # block holds; challenger already charged
    else:
        state.resume = Resume.RESOLVE_ACTION  # block was a bluff; the action lands
    _continue(state, rng)


def _apply_discard(state: GameState, decision: Discard, rng: random.Random) -> None:
    seat, _cause = state.losses.pop(0)
    player = state.players[seat]
    player.influence.remove(decision.card)
    player.revealed.append(decision.card)
    state.history.append(("reveal", seat, int(decision.card)))
    if not player.influence:
        _eliminate(state, seat)
    _continue(state, rng)


def _apply_exchange_return(
    state: GameState, decision: ExchangeReturn, rng: random.Random
) -> None:
    assert state.pending_action is not None
    player = state.players[state.pending_action.actor]
    pool = player.influence + state.exchange_drawn
    for card in decision.keep:
        pool.remove(card)
    state.deck.extend(pool)  # the rest go back to the Court deck
    player.influence = list(decision.keep)
    state.exchange_drawn = []
    _continue(state, rng)


# ---------------------------------------------------------------------------
# Flow control
# ---------------------------------------------------------------------------


def _continue(state: GameState, rng: random.Random) -> None:
    """Drive the turn forward until it needs a decision again.

    Pending influence losses always come first: they can eliminate a player and
    so change who is eligible for everything that follows.
    """
    while True:
        if _check_game_over(state):
            return
        if state.losses:
            state.phase = Phase.LOSE_INFLUENCE
            return

        if state.resume is Resume.BLOCK_WINDOW:
            state.resume = Resume.RESOLVE_ACTION
            if _open_block_window(state):
                state.phase = Phase.BLOCK
                return
            continue  # nobody is eligible to block

        if state.resume is Resume.RESOLVE_ACTION:
            state.resume = Resume.END_TURN
            _resolve_action(state, rng)
            if state.phase is Phase.EXCHANGE_RETURN:
                return
            continue

        _begin_turn(state)
        return


def _open_block_window(state: GameState) -> bool:
    """Set up the block poll.  Returns False if no one may block."""
    assert state.pending_action is not None
    action = state.pending_action
    spec = ACTIONS[action.kind]
    if not spec.blockable:
        return False

    if spec.needs_target:
        # Only the victim may block a Steal or an Assassination.
        assert action.target is not None
        eligible = [action.target] if state.players[action.target].alive else []
    else:
        # Anyone may block Foreign Aid with a Duke.
        eligible = _poll_order(state, exclude=action.actor)

    if not eligible:
        return False
    state.responders = eligible
    return True


def _resolve_action(state: GameState, rng: random.Random) -> None:
    assert state.pending_action is not None
    action = state.pending_action
    actor = state.players[action.actor]
    kind = action.kind

    if kind is ActionKind.INCOME:
        actor.coins += 1
    elif kind is ActionKind.FOREIGN_AID:
        actor.coins += 2
    elif kind is ActionKind.TAX:
        actor.coins += 3
    elif kind is ActionKind.STEAL:
        assert action.target is not None
        target = state.players[action.target]
        if target.alive:
            amount = min(STEAL_AMOUNT, target.coins)
            target.coins -= amount
            actor.coins += amount
    elif kind in (ActionKind.COUP, ActionKind.ASSASSINATE):
        assert action.target is not None
        cause = LossCause.COUP if kind is ActionKind.COUP else LossCause.ASSASSINATION
        _queue_loss(state, action.target, cause)
    elif kind is ActionKind.EXCHANGE:
        state.exchange_drawn = [_draw(state, rng) for _ in range(EXCHANGE_DRAW)]
        state.phase = Phase.EXCHANGE_RETURN

    state.history.append(("resolved", action.actor, int(kind)))


def _begin_turn(state: GameState) -> None:
    state.pending_action = None
    state.pending_block = None
    state.responders = []
    state.exchange_drawn = []
    state.resume = Resume.END_TURN

    seat = state.turn
    for _ in range(len(state.players)):
        seat = (seat + 1) % len(state.players)
        if state.players[seat].alive:
            break
    state.turn = seat
    state.phase = Phase.ACTION


def _check_game_over(state: GameState) -> bool:
    live = state.live_players()
    if len(live) > 1:
        return False
    state.winner = live[0] if live else None
    state.phase = Phase.GAME_OVER
    state.responders = []
    state.losses = []
    return True


# ---------------------------------------------------------------------------
# Challenges
# ---------------------------------------------------------------------------


def _resolve_challenge(
    state: GameState,
    challenger: int,
    claimant: int,
    claim: Card,
    rng: random.Random,
) -> bool:
    """Settle one challenge, charging whichever side was wrong.

    Returns whether the claim was honest.

    A claimant who really holds the card reveals it, shuffles it back and draws
    a replacement -- so surviving a challenge leaks the card's identity but does
    not cost it.
    """
    honest = claim in state.players[claimant].influence
    state.history.append(("challenge", challenger, claimant, int(claim), honest))

    if honest:
        player = state.players[claimant]
        player.influence.remove(claim)
        state.deck.append(claim)
        player.influence.append(_draw(state, rng))
        state.history.append(("replace", claimant))
        _queue_loss(state, challenger, LossCause.FAILED_CHALLENGE)
    else:
        _queue_loss(state, claimant, LossCause.CAUGHT_BLUFFING)

    state.responders = []  # one challenge closes the window
    return honest


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def _draw(state: GameState, rng: random.Random) -> Card:
    """Take one card from the Court deck.

    The deck is a bag, not a stack: drawing pops a uniformly random index, so a
    returned card never needs an explicit reshuffle.
    """
    return state.deck.pop(rng.randrange(len(state.deck)))


def _queue_loss(state: GameState, seat: int, cause: LossCause) -> None:
    if state.players[seat].alive:
        state.losses.append((seat, cause))


def _eliminate(state: GameState, seat: int) -> None:
    state.players[seat].coins = 0  # coins go back to the treasury
    state.responders = [s for s in state.responders if s != seat]
    state.losses = [entry for entry in state.losses if entry[0] != seat]
    state.history.append(("eliminated", seat))


def _poll_order(state: GameState, exclude: int) -> list[int]:
    """Live players other than ``exclude``, clockwise starting to their left."""
    n = len(state.players)
    return [
        seat
        for seat in ((exclude + offset) % n for offset in range(1, n))
        if state.players[seat].alive and seat != exclude
    ]
