"""Build the tablebase for heads-up Coup with one influence each.

Scope: two players, one card each, both cards public, nobody bluffs (so nobody
challenges), no Ambassador.  Every *legal* position is solved, not merely the
ones reachable from a 0-coin start, the way a chess table covers positions no
sensible game would produce.

The coin range is bounded by the forced-Coup rule.  A turn that begins on 10 or
more coins may only Coup, so any turn that can add coins begins on at most 9,
and the biggest single-turn gain is Tax at +3.  Nobody can therefore ever hold
more than 12, and 0..12 for each player is the whole space.


Rows are written from the point of view of **the player to move**, which is
what collapses the seat symmetry: a Duke on 3 coins facing a Captain on 5 is
one position, not two.  The builder asserts that this projection is injective
-- if two genuinely different positions ever collapsed onto one row, the table
would be silently wrong, so it raises instead.

    python -m analysis.build_tablebase
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from coup import Card, RuleConfig, new_game
from coup.solve import Value, solve_many
from coup.tablebase import FIELDS, POSITION_FIELDS, describe, position_key

CARDS = [Card.DUKE, Card.ASSASSIN, Card.CAPTAIN, Card.CONTESSA]

CONFIG = RuleConfig(
    starting_influence=1,
    starting_coins=0,
    two_player_start_handicap=False,
)

#: A turn beginning on 10+ coins may only Coup, so a turn that adds coins
#: begins on at most 9, and Tax (+3) is the largest single-turn gain.
MAX_COINS = 12

OUT_DIR = Path(__file__).resolve().parent.parent / "tablebase"
CSV_PATH = OUT_DIR / "heads_up_one_card.csv"
META_PATH = OUT_DIR / "heads_up_one_card.meta.json"

def position(first: Card, second: Card, first_coins: int = 0, second_coins: int = 0):
    state = new_game(2, config=CONFIG, hands=[[first], [second]])
    state.players[0].coins = first_coins
    state.players[1].coins = second_coins
    return state


def row_for(solution, key) -> dict:
    node = solution.nodes[key]
    best = solution.best_moves(key)
    top = solution.nodes[best[0][1]].depth
    tied = [d for d, k in best if solution.nodes[k].depth == top]

    row = dict(zip(POSITION_FIELDS, position_key(node.state)))
    row["result"] = "win" if node.value == Value(node.mover) else "loss"
    row["dtm"] = node.depth
    row["best_moves"] = "|".join(describe(d) for d in tied)
    return row


def identity(row: dict) -> tuple:
    """The fields that name a position, as opposed to its solution."""
    return tuple(row[f] for f in POSITION_FIELDS)


def every_legal_position() -> list:
    return [
        position(a, b, ca, cb)
        for a in CARDS
        for b in CARDS
        for ca in range(MAX_COINS + 1)
        for cb in range(MAX_COINS + 1)
    ]


def build() -> list[dict]:
    solution = solve_many(every_legal_position())

    table: dict[tuple, dict] = {}
    for key, node in solution.nodes.items():
        if node.state.game_over:
            continue  # nothing to look up: the game is already decided
        row = row_for(solution, key)
        ident = identity(row)
        seen = table.get(ident)
        if seen is None:
            table[ident] = row
        elif seen != row:
            raise AssertionError(
                "two positions collapsed onto one row with different "
                f"answers, so the schema is missing a field:\n  {seen}\n  {row}"
            )
    return [table[k] for k in sorted(table)]


def main() -> None:
    rows = build()
    OUT_DIR.mkdir(exist_ok=True)

    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    digest = hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()
    META_PATH.write_text(
        json.dumps(
            {
                "game": "Coup, heads-up, one influence each, 0 starting coins",
                "assumptions": [
                    "both cards are public",
                    "no player ever claims a character it does not hold",
                    "no player ever challenges (under honest play a challenge only loses)",
                    "the Ambassador is excluded",
                ],
                "cards": [str(c) for c in CARDS],
                "rules": asdict(CONFIG),
                "rows": len(rows),
                "max_coins": MAX_COINS,
                "coverage": "every legal position",
                "perspective": "every row is written from the point of view of the player to move",
                "dtm": "plies to the end of the game, counting every decision including response windows",
                "sha256": digest,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"{len(rows)} positions -> {CSV_PATH}")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
