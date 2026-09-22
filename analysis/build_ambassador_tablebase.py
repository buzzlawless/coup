"""Build the tablebase for the five-card heads-up game, Exchange included.

Scope: two players, one influence each and one card already revealed, both
hands public, nobody bluffs (so nobody challenges).  Every legal position is
solved, not merely the reachable ones.

The Exchange draws from the deck, so positions have win *probabilities* rather
than winners, and the deck is fixed by the four cards out of play -- the two
held and the two revealed.  Both revealed cards are therefore part of the key.

    python -m analysis.build_ambassador_tablebase
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import itertools
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from coup import Card, RuleConfig, new_game
from coup.stochastic import TOLERANCE, best_moves, build, evaluate

CARDS = list(Card)

CONFIG = RuleConfig(
    starting_influence=1,
    starting_coins=0,
    two_player_start_handicap=False,
)

#: A turn beginning on 10+ coins may only Coup, so a turn that adds coins
#: begins on at most 9, and Tax (+3) is the largest single-turn gain.
MAX_COINS = 12

OUT_DIR = Path(__file__).resolve().parent.parent / "tablebase"
CSV_PATH = OUT_DIR / "heads_up_one_card_with_ambassador.csv.gz"
META_PATH = OUT_DIR / "heads_up_one_card_with_ambassador.meta.json"

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
    "dead_cards",
]
FIELDS = POSITION_FIELDS + ["win_probability", "best_moves"]

#: Values are rounded here for output and for the collision check.  Value
#: iteration settles far tighter than this, so the digits printed are real.
DIGITS = 9


def deals():
    """Every legal (hand, hand, revealed, revealed), respecting three copies."""
    for held in itertools.product(CARDS, repeat=2):
        for dead in itertools.combinations_with_replacement(CARDS, 2):
            if all(n <= 3 for n in Counter(held + dead).values()):
                yield held, dead


def every_legal_position():
    for (mine, theirs), (dead_mine, dead_theirs) in deals():
        for my_coins in range(MAX_COINS + 1):
            for their_coins in range(MAX_COINS + 1):
                state = new_game(
                    2,
                    config=CONFIG,
                    hands=[[mine], [theirs]],
                    revealed=[[dead_mine], [dead_theirs]],
                )
                state.players[0].coins = my_coins
                state.players[1].coins = their_coins
                yield state


def build_rows(nodes) -> list[dict]:
    table: dict[tuple, dict] = {}
    for key, node in nodes.items():
        if node.terminal:
            continue  # nothing to look up once the game is decided
        mover_value = node.value if node.mover == 0 else 1.0 - node.value
        row = dict(zip(POSITION_FIELDS, node.key))
        row["win_probability"] = f"{round(mover_value, DIGITS):.{DIGITS}f}"
        row["best_moves"] = "|".join(best_moves(nodes, key))

        seen = table.get(node.key)
        if seen is None:
            table[node.key] = row
        elif seen != row:
            raise AssertionError(
                "two positions collapsed onto one row with different answers, "
                f"so the schema is missing a field:\n  {seen}\n  {row}"
            )
    return [table[k] for k in sorted(table)]


def main() -> None:
    started = time.perf_counter()
    roots = list(every_legal_position())
    print(f"{len(roots):,} legal positions to seed from", flush=True)

    nodes = build(roots)
    print(f"built {len(nodes):,} states in {time.perf_counter()-started:.0f}s", flush=True)

    mark = time.perf_counter()
    sweeps = evaluate(nodes, optimistic=False)
    pessimistic = {key: node.value for key, node in nodes.items()}
    print(f"pessimistic seed: {sweeps} sweeps, {time.perf_counter()-mark:.0f}s", flush=True)

    mark = time.perf_counter()
    sweeps = evaluate(nodes, optimistic=True)
    gap = max(abs(pessimistic[k] - n.value) for k, n in nodes.items())
    print(f"optimistic seed:  {sweeps} sweeps, {time.perf_counter()-mark:.0f}s", flush=True)
    print(f"seed agreement: {gap:.2e} -> the fixpoint is unique", flush=True)
    if gap > 1e-6:
        raise AssertionError(f"seeds disagree by {gap}; some line may run forever")

    rows = build_rows(nodes)
    OUT_DIR.mkdir(exist_ok=True)
    with gzip.open(CSV_PATH, "wt", newline="", compresslevel=9) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    digest = hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()
    META_PATH.write_text(
        json.dumps(
            {
                "game": "Coup, heads-up, one influence each, the Ambassador included",
                "assumptions": [
                    "both held cards are public",
                    "no player ever claims a character it does not hold",
                    "no player ever challenges (under honest play a challenge only loses)",
                    "an Exchange is resolved in the open, so the cards stay public",
                    "each player has one influence left and one card already revealed",
                ],
                "cards": [str(c) for c in CARDS],
                "rules": asdict(CONFIG),
                "rows": len(rows),
                "states_solved": len(nodes),
                "max_coins": MAX_COINS,
                "coverage": "every legal position",
                "value": "win_probability is for the player to move, under perfect play",
                "method": "value iteration; both seeds agree, so no line runs forever",
                "seed_agreement": gap,
                "tolerance": TOLERANCE,
                "sha256": digest,
            },
            indent=2,
        )
        + "\n"
    )
    size = CSV_PATH.stat().st_size / 1e6
    print(f"\n{len(rows):,} rows -> {CSV_PATH.name} ({size:.1f} MB gzipped)")
    print(f"sha256 {digest}")
    print(f"total {time.perf_counter()-started:.0f}s")


if __name__ == "__main__":
    main()
