"""Decisions: the moves a player can be asked to make.

Every node of the game tree asks exactly one player for exactly one of these.
They are frozen and hashable so they can be used as dictionary keys by a
solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from .actions import ActionKind
from .cards import Card


@dataclass(frozen=True)
class ChooseAction:
    """Taken on your own turn."""

    kind: ActionKind
    target: int | None = None


@dataclass(frozen=True)
class Challenge:
    """Dispute the claim just made by an action or a block."""


@dataclass(frozen=True)
class Pass:
    """Decline to challenge, or decline to block."""


@dataclass(frozen=True)
class Block:
    """Claim ``character`` in order to stop the action."""

    character: Card


@dataclass(frozen=True)
class Discard:
    """Choose which influence to lose; the card is revealed and out of play."""

    card: Card


@dataclass(frozen=True)
class ExchangeReturn:
    """Choose which cards to keep after an Exchange; the rest go back."""

    keep: tuple[Card, ...]


Decision = Union[ChooseAction, Challenge, Pass, Block, Discard, ExchangeReturn]

CHALLENGE = Challenge()
PASS = Pass()
