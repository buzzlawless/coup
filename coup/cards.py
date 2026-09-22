"""The five characters and the composition of the Court deck."""

from __future__ import annotations

from enum import IntEnum


class Card(IntEnum):
    """A character card.

    Values are small ints so that states are cheap to hash and compare, which
    matters once you start enumerating information sets.
    """

    DUKE = 0
    ASSASSIN = 1
    CAPTAIN = 2
    AMBASSADOR = 3
    CONTESSA = 4

    def __str__(self) -> str:
        return self.name.title()


COPIES_PER_CARD = 3

#: Every card in the game, 15 in total. The Court deck starts as this multiset.
FULL_DECK: tuple[Card, ...] = tuple(
    card for card in Card for _ in range(COPIES_PER_CARD)
)
