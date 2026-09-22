"""Reading the solved-position tables in ``tablebase/``.

A tablebase answers any position, not just an opening: given a position it
returns who wins, in how many plies, and every move that achieves it.

Positions are keyed from the point of view of **the player to move**, so a Duke
on 3 coins facing a Captain on 5 is a single entry however the seats happen to
be numbered.  ``position_key`` is the single definition of that key -- the
builder writes rows with it and ``probe`` looks them up with it, so the two
cannot drift apart.
"""

from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from pathlib import Path

from .decisions import Block, ChooseAction, Discard, ExchangeReturn, Pass
from .engine import to_act
from .state import GameState

_TABLES = Path(__file__).resolve().parent.parent / "tablebase"
DEFAULT_PATH = _TABLES / "heads_up_one_card.csv"
#: The five-card table, where the Exchange makes values probabilities.
AMBASSADOR_PATH = _TABLES / "heads_up_one_card_with_ambassador.csv.gz"

POSITION_FIELDS = [
    "to_act_card",
    "opponent_card",
    "to_act_coins",
    "opponent_coins",
    "phase",
    "pending_action",
    "pending_action_by",
    "pending_block",
    "pending_block_by",
]
RESULT_FIELDS = ["result", "dtm", "best_moves"]
FIELDS = POSITION_FIELDS + RESULT_FIELDS


def describe(decision) -> str:
    """The move as it is written in the table."""
    if isinstance(decision, ChooseAction):
        return str(decision.kind)
    if isinstance(decision, Block):
        return f"block {decision.character}"
    if isinstance(decision, Pass):
        return "pass"
    if isinstance(decision, Discard):
        return f"reveal {decision.card}"
    if isinstance(decision, ExchangeReturn):
        return "exchange " + "+".join(str(c) for c in decision.keep)
    return repr(decision)


def position_key(state: GameState) -> tuple:
    """The lookup key for a live position, relative to the player to move."""
    mover = to_act(state)
    if mover is None:
        raise ValueError("the game is over; there is nothing to look up")
    if len(state.players) != 2:
        raise ValueError("the table covers heads-up positions only")
    opponent = 1 - mover
    for seat in (mover, opponent):
        if len(state.players[seat].influence) != 1:
            raise ValueError("the table covers one-influence positions only")

    action = state.pending_action
    block = state.pending_block
    return (
        str(state.players[mover].influence[0]),
        str(state.players[opponent].influence[0]),
        state.players[mover].coins,
        state.players[opponent].coins,
        state.phase.name,
        "" if action is None else str(action.kind),
        "" if action is None else ("self" if action.actor == mover else "opponent"),
        "" if block is None else str(block.character),
        "" if block is None else ("self" if block.blocker == mover else "opponent"),
    )


def dead_cards(state: GameState) -> str:
    """The two revealed cards, which fix the deck and so every draw probability.

    A player down to one influence has revealed the other, face up and out of
    the deck.  In the Ambassador-free game nothing reads the deck, so this
    cannot affect play and ``position_key`` rightly leaves it out; once an
    Exchange is possible it decides the odds and has to be part of the key.
    """
    return "+".join(sorted(str(c) for p in state.players for c in p.revealed))


def lookup_key(state: GameState) -> tuple:
    """``position_key`` plus the dead cards."""
    return position_key(state) + (dead_cards(state),)


@dataclass(frozen=True)
class Entry:
    #: "win" or "loss", for the player to move.
    result: str
    #: Plies to the end, counting every decision including response windows.
    dtm: int
    #: Every move achieving the result -- the quickest win, or the slowest loss.
    best_moves: tuple[str, ...]

    @property
    def winning(self) -> bool:
        return self.result == "win"


def load(path: Path | str = DEFAULT_PATH) -> dict[tuple, Entry]:
    table: dict[tuple, Entry] = {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["to_act_card"],
                row["opponent_card"],
                int(row["to_act_coins"]),
                int(row["opponent_coins"]),
                row["phase"],
                row["pending_action"],
                row["pending_action_by"],
                row["pending_block"],
                row["pending_block_by"],
            )
            table[key] = Entry(
                result=row["result"],
                dtm=int(row["dtm"]),
                best_moves=tuple(row["best_moves"].split("|")),
            )
    return table


def probe(state: GameState, table: dict[tuple, Entry] | None = None) -> Entry:
    """Look up a live position.  Raises ``KeyError`` if it is out of scope."""
    if table is None:
        table = load()
    key = position_key(state)
    if key not in table:
        raise KeyError(f"position not in the table: {key}")
    return table[key]


# --- the five-card table -------------------------------------------------

AMBASSADOR_POSITION_FIELDS = POSITION_FIELDS + ["dead_cards"]
AMBASSADOR_FIELDS = AMBASSADOR_POSITION_FIELDS + ["win_probability", "best_moves"]


@dataclass(frozen=True)
class Equity:
    """A solved position in the game with the Ambassador in it."""

    #: Probability the player to move wins, under perfect play by both.
    win_probability: float
    #: Every move sharing that value; where there are several they are
    #: genuinely interchangeable.
    best_moves: tuple[str, ...]

    @property
    def certain(self) -> bool:
        """True where the Exchange cannot change the outcome either way."""
        return self.win_probability in (0.0, 1.0)


def load_ambassador(path: Path | str = AMBASSADOR_PATH) -> dict[tuple, Equity]:
    table: dict[tuple, Equity] = {}
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["to_act_card"],
                row["opponent_card"],
                int(row["to_act_coins"]),
                int(row["opponent_coins"]),
                row["phase"],
                row["pending_action"],
                row["pending_action_by"],
                row["pending_block"],
                row["pending_block_by"],
                row["dead_cards"],
            )
            table[key] = Equity(
                win_probability=float(row["win_probability"]),
                best_moves=tuple(row["best_moves"].split("|")),
            )
    return table


def probe_ambassador(
    state: GameState, table: dict[tuple, Equity] | None = None
) -> Equity:
    """Look up a live position in the five-card table."""
    if table is None:
        table = load_ambassador()
    key = lookup_key(state)
    if key not in table:
        raise KeyError(f"position not in the table: {key}")
    return table[key]
