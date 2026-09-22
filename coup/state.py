"""Game state.

The state is mutable and cloned explicitly (``GameState.clone``) rather than
being rebuilt functionally, because simulation loops care about allocation.
``clone`` is a deep copy of everything a step can touch; nothing is shared with
the original except immutable values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from .actions import ActionKind
from .cards import Card


class Phase(IntEnum):
    """What the game is waiting for."""

    ACTION = 0
    #: Anyone may dispute the character the actor claimed.
    ACTION_CHALLENGE = 1
    #: An eligible player may block the action.
    BLOCK = 2
    #: Anyone may dispute the character the blocker claimed.
    BLOCK_CHALLENGE = 3
    #: A specific player must reveal one of their cards.
    LOSE_INFLUENCE = 4
    #: The exchanging player must choose which cards to keep.
    EXCHANGE_RETURN = 5
    GAME_OVER = 6


class Resume(IntEnum):
    """Where to pick up once the pending influence losses have been settled.

    Losing influence interrupts the turn at several points (a lost challenge, a
    successful Coup), and a single turn can interrupt twice.  Rather than
    recursing, the engine parks a continuation here.
    """

    END_TURN = 0
    BLOCK_WINDOW = 1
    RESOLVE_ACTION = 2


class LossCause(IntEnum):
    """Why a player is losing influence.  Public information."""

    FAILED_CHALLENGE = 0
    CAUGHT_BLUFFING = 1
    COUP = 2
    ASSASSINATION = 3


@dataclass
class RuleConfig:
    """Points where published rulings and house rules disagree.

    These are exposed rather than baked in because two of them change the
    payoff of a bluff, and so change the equilibrium -- they are worth being
    able to flip and re-solve.
    """

    #: A player starting their turn with this many coins must Coup.
    mandatory_coup_threshold: int = 10
    #: In a two-player game the starting player begins with 1 coin, not 2.
    two_player_start_handicap: bool = True
    #: Assassinate costs 3 coins on declaration.  The fee is paid to *attempt*
    #: the hit, so an assassin caught bluffing does not get it back -- a caught
    #: bluff costs an influence and the whole 3 coins.  Set True for the reading
    #: that a cancelled action should never have been charged for.  (A fee lost
    #: to a *Contessa block* is never refunded either way.)
    refund_cost_on_caught_bluff: bool = False
    #: Whether Steal may target a player holding no coins.  It would gain
    #: nothing, so the move is not offered.
    allow_stealing_from_zero: bool = False


@dataclass
class PendingAction:
    actor: int
    kind: ActionKind
    target: int | None
    #: Coins already paid, held so a refund can undo exactly what was charged.
    paid: int = 0


@dataclass
class PendingBlock:
    blocker: int
    character: Card


@dataclass
class PlayerState:
    seat: int
    #: Face-down cards.  Hidden from every other player.
    influence: list[Card] = field(default_factory=list)
    #: Face-up cards, lost for good.  Public.
    revealed: list[Card] = field(default_factory=list)
    coins: int = 0

    @property
    def alive(self) -> bool:
        return bool(self.influence)

    def clone(self) -> PlayerState:
        return PlayerState(self.seat, list(self.influence), list(self.revealed), self.coins)


@dataclass
class GameState:
    config: RuleConfig
    players: list[PlayerState]
    #: The Court deck, held as an unordered bag; draws pop a uniformly random
    #: index, so returning a card never needs a reshuffle.
    deck: list[Card]
    turn: int = 0
    phase: Phase = Phase.ACTION
    pending_action: PendingAction | None = None
    pending_block: PendingBlock | None = None
    #: Players still to be polled in the current challenge or block window, in
    #: clockwise order.  The head is the player to act.
    responders: list[int] = field(default_factory=list)
    #: Queue of (player, cause) influence losses still to be chosen.
    losses: list[tuple[int, LossCause]] = field(default_factory=list)
    resume: Resume = Resume.END_TURN
    #: Cards drawn for an Exchange, not yet merged into a hand.
    exchange_drawn: list[Card] = field(default_factory=list)
    #: Public event log, sufficient to reconstruct everything an observer saw.
    history: list[tuple] = field(default_factory=list)
    winner: int | None = None

    @property
    def game_over(self) -> bool:
        return self.phase is Phase.GAME_OVER

    def live_players(self) -> list[int]:
        return [p.seat for p in self.players if p.alive]

    def clone(self) -> GameState:
        return GameState(
            config=self.config,
            players=[p.clone() for p in self.players],
            deck=list(self.deck),
            turn=self.turn,
            phase=self.phase,
            pending_action=(
                None
                if self.pending_action is None
                else PendingAction(
                    self.pending_action.actor,
                    self.pending_action.kind,
                    self.pending_action.target,
                    self.pending_action.paid,
                )
            ),
            pending_block=(
                None
                if self.pending_block is None
                else PendingBlock(self.pending_block.blocker, self.pending_block.character)
            ),
            responders=list(self.responders),
            losses=list(self.losses),
            resume=self.resume,
            exchange_drawn=list(self.exchange_drawn),
            history=list(self.history),
            winner=self.winner,
        )
