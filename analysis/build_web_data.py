"""Generate the data the browser UI reads, one file per pair of upcards.

The upcards cannot change inside a game -- with a single influence, revealing a
card ends it -- so every position in a game shares them.  That makes each pair
of upcards a closed subgame which can be solved on its own and shipped as its
own file, so the page downloads a fifteenth of the table rather than all of it.

Each file is self-contained navigation: every position carries its moves, their
EVs and the index of where each one leads.  The page therefore needs no rules
of its own, and cannot disagree with the solver about what is legal.

    python -m analysis.build_web_data
"""

from __future__ import annotations

import gzip
import itertools
import json
from collections import Counter
from pathlib import Path

from coup import Card, RuleConfig, new_game
from coup.solve import state_key, truthful_decisions
from coup.stochastic import build, evaluate
from coup.chance import outcomes
from coup.engine import to_act
from coup.tablebase import describe

OUT = Path(__file__).resolve().parent.parent / "docs" / "data"
MAX_COINS = 12
CONFIG = RuleConfig(starting_influence=1, starting_coins=0, two_player_start_handicap=False)

SHORT = {"Duke": "D", "Assassin": "S", "Captain": "C", "Ambassador": "A", "Contessa": "T"}
PHASE = {
    "ACTION": "a", "ACTION_CHALLENGE": "c", "BLOCK": "b",
    "BLOCK_CHALLENGE": "k", "EXCHANGE_RETURN": "x", "LOSE_INFLUENCE": "l",
}


def slug(dead) -> str:
    return "-".join(SHORT[str(c)] for c in sorted(dead))


def roots_for(dead):
    for held in itertools.product(list(Card), repeat=2):
        if any(n > 3 for n in Counter(held + tuple(dead)).values()):
            continue
        for a in range(MAX_COINS + 1):
            for b in range(MAX_COINS + 1):
                state = new_game(2, config=CONFIG, hands=[[held[0]], [held[1]]],
                                 revealed=[[dead[0]], [dead[1]]])
                state.players[0].coins, state.players[1].coins = a, b
                yield state


def shard(dead) -> dict:
    roots = list(roots_for(dead))
    nodes = build(roots)
    evaluate(nodes)

    order = [k for k in nodes]
    index = {k: i for i, k in enumerate(order)}
    vocab: dict[str, int] = {}

    def word(name: str) -> int:
        return vocab.setdefault(name, len(vocab))

    info, prob, moves = [], [], []
    for key in order:
        node = nodes[key]
        state = None
        if node.terminal:
            info.append(None)
            prob.append(None)
            moves.append(None)
            continue
        k = node.key
        mover = node.mover
        info.append([
            SHORT[k[0]], SHORT[k[1]], k[2], k[3], PHASE[k[4]],
            k[5], k[6], SHORT.get(k[7], ""), k[8],
            "".join(SHORT[c] for c in sorted(k[10].split("+"))) if k[10] else "",
            mover,
        ])
        prob.append(round(node.value if mover == 0 else 1.0 - node.value, 6))
        entry = []
        for name, succ in node.moves:
            ev = sum(p * nodes[c].value for p, c in succ)
            if mover == 1:
                ev = 1.0 - ev
            if len(succ) == 1:
                entry.append([word(name), round(ev, 6), index[succ[0][1]]])
            else:
                entry.append([word(name), round(ev, 6), -1,
                              [[round(p, 6), index[c]] for p, c in succ]])
        moves.append(entry)

    # terminals: who won, relative to nothing -- seat is enough for display
    winners = {index[k]: nodes[k].winner for k in order if nodes[k].terminal}

    starts = {}
    for state in roots:
        k = state_key(state)
        p = nodes[k]
        starts[f"{SHORT[p.key[0]]}{SHORT[p.key[1]]}{p.key[2]}.{p.key[3]}.{p.mover}"] = index[k]

    return {
        "dead": [SHORT[str(c)] for c in sorted(dead)],
        "vocab": [n for n, _ in sorted(vocab.items(), key=lambda kv: kv[1])],
        "info": info,
        "p": prob,
        "moves": moves,
        "winners": winners,
        "starts": starts,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for dead in itertools.combinations_with_replacement(list(Card), 2):
        data = shard(dead)
        path = OUT / f"{slug(dead)}.json.gz"
        raw = json.dumps(data, separators=(",", ":")).encode()
        path.write_bytes(gzip.compress(raw, 9))
        manifest.append({"dead": data["dead"], "file": path.name,
                         "positions": len(data["info"])})
        print(f"  {slug(dead):6} {len(data['info']):>7,} positions  "
              f"{len(raw)/1e6:5.1f} MB raw  {path.stat().st_size/1e6:4.2f} MB gz", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"\n{len(manifest)} shards -> {OUT}")


if __name__ == "__main__":
    main()
