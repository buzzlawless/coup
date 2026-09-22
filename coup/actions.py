"""The action table.

Every action in Coup is a row of the same shape: a cost, an optional character
claim, whether it needs a target, and which characters may block it.  Keeping
this as data rather than code means the challenge and block machinery in
``engine`` is written exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from .cards import Card


class ActionKind(IntEnum):
    INCOME = 0
    FOREIGN_AID = 1
    COUP = 2
    TAX = 3
    ASSASSINATE = 4
    STEAL = 5
    EXCHANGE = 6

    def __str__(self) -> str:
        return self.name.title().replace("_", " ")


@dataclass(frozen=True)
class ActionSpec:
    kind: ActionKind
    cost: int
    #: Character the actor claims to hold.  ``None`` means no claim is made,
    #: and an action with no claim can never be challenged.
    claim: Card | None
    needs_target: bool
    #: Characters that may be claimed in order to block this action.
    blockers: tuple[Card, ...]

    @property
    def challengeable(self) -> bool:
        return self.claim is not None

    @property
    def blockable(self) -> bool:
        return bool(self.blockers)


ACTIONS: dict[ActionKind, ActionSpec] = {
    spec.kind: spec
    for spec in (
        ActionSpec(ActionKind.INCOME, 0, None, False, ()),
        ActionSpec(ActionKind.FOREIGN_AID, 0, None, False, (Card.DUKE,)),
        ActionSpec(ActionKind.COUP, 7, None, True, ()),
        ActionSpec(ActionKind.TAX, 0, Card.DUKE, False, ()),
        ActionSpec(ActionKind.ASSASSINATE, 3, Card.ASSASSIN, True, (Card.CONTESSA,)),
        ActionSpec(ActionKind.STEAL, 0, Card.CAPTAIN, True, (Card.CAPTAIN, Card.AMBASSADOR)),
        ActionSpec(ActionKind.EXCHANGE, 0, Card.AMBASSADOR, False, ()),
    )
}

#: Coins taken from the target by a successful Steal (capped at their balance).
STEAL_AMOUNT = 2

#: Cards drawn from the Court deck by a successful Exchange.
EXCHANGE_DRAW = 2
