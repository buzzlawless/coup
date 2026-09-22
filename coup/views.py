"""Hidden information.

``GameState`` is the referee's view and knows every card.  A player only ever
sees the projection built here.  Keeping the two apart from the start is what
makes the state usable as an information set later -- retrofitting it is much
harder than paying for it now.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cards import Card
from .state import GameState, Phase


@dataclass(frozen=True)
class PlayerView:
    """What everyone knows about one player."""

    seat: int
    coins: int
    influence_count: int
    revealed: tuple[Card, ...]

    @property
    def alive(self) -> bool:
        return self.influence_count > 0


@dataclass(frozen=True)
class Observation:
    """Everything a single player is entitled to know."""

    seat: int
    #: Only this player's own face-down cards.
    hand: tuple[Card, ...]
    players: tuple[PlayerView, ...]
    deck_size: int
    turn: int
    phase: Phase
    to_act: int | None
    #: Every public event so far, in order.  Two histories that agree here are
    #: indistinguishable to this player, which is exactly the equivalence a
    #: solver needs.
    history: tuple[tuple, ...]

    def key(self) -> tuple:
        """A hashable information-set key with perfect recall."""
        return (
            self.seat,
            self.hand,
            tuple((p.coins, p.influence_count, p.revealed) for p in self.players),
            self.turn,
            int(self.phase),
            self.to_act,
            self.history,
        )


def observe(state: GameState, seat: int) -> Observation:
    from .engine import to_act

    return Observation(
        seat=seat,
        hand=tuple(sorted(state.players[seat].influence)),
        players=tuple(
            PlayerView(
                seat=p.seat,
                coins=p.coins,
                influence_count=len(p.influence),
                revealed=tuple(p.revealed),
            )
            for p in state.players
        ),
        deck_size=len(state.deck),
        turn=state.turn,
        phase=state.phase,
        to_act=to_act(state),
        history=tuple(state.history),
    )
