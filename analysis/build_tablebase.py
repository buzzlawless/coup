"""Build the tablebase for heads-up Coup with one influence each.

Scope: two players, one card each, both cards public, nobody bluffs (so nobody
challenges), no Ambassador, 0 coins at the start.  Every position reachable
from one of the sixteen openings is solved exactly and written out.

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
from coup.solve import Value, solve
from coup.tablebase import FIELDS, POSITION_FIELDS, describe, position_key

CARDS = [Card.DUKE, Card.ASSASSIN, Card.CAPTAIN, Card.CONTESSA]

CONFIG = RuleConfig(
    starting_influence=1,
    starting_coins=0,
    two_player_start_handicap=False,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "tablebase"
CSV_PATH = OUT_DIR / "heads_up_one_card.csv"
META_PATH = OUT_DIR / "heads_up_one_card.meta.json"

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


def build() -> list[dict]:
    table: dict[tuple, dict] = {}
    for first in CARDS:
        for second in CARDS:
            solution = solve(new_game(2, config=CONFIG, hands=[[first], [second]]))
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
