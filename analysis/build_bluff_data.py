"""Generate the data for the bluffing explorer, one file per card P2 holds.

The guarantee sets depend only on P2's card -- the upcards only shape P2's
*belief* about P1's card, which the page applies itself -- so four files cover
every setup.  Each carries the public game graph and the set at every node;
the page solves the small per-node programs for the strategies on the fly.

    python -m analysis.build_bluff_data
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from coup.bluff import TYPES, build, solve

OUT = Path(__file__).resolve().parent.parent / "docs" / "data" / "bluff"
SHORT = {"Duke": "D", "Assassin": "S", "Captain": "C", "Contessa": "T"}
KIND = {"term": "t", "forced": "f", "p1": "1", "p2": "2", "call": "c"}


def export(card) -> dict:
    nodes, starts = build(card)
    sets, sweeps = solve(nodes, 1.0)
    check, _ = solve(nodes, 0.0)
    probes = np.random.default_rng(5).dirichlet(np.ones(4), 64)
    gap = max(float(np.max(np.abs(np.min(a @ probes.T, 0) - np.min(b @ probes.T, 0))))
              for a, b in zip(sets, check))
    if gap > 1e-7:
        raise AssertionError(f"seeds disagree by {gap}: some line may run forever")

    vocab: dict[str, int] = {}
    out_nodes = []
    for n in nodes:
        moves = [[vocab.setdefault(label, len(vocab)), child] for label, child in n.moves]
        out_nodes.append([KIND[n.kind], -1 if n.mover is None else n.mover,
                          list(n.info), moves, -1 if n.claim is None else n.claim])
    return {
        "p2": SHORT[str(card)],
        "types": [SHORT[str(t)] for t in TYPES],
        "vocab": [k for k, _ in sorted(vocab.items(), key=lambda kv: kv[1])],
        "nodes": out_nodes,
        "sets": [np.round(s, 10).tolist() for s in sets],
        "starts": {f"{a}.{b}.{f}": i for (a, b, f), i in starts.items()},
        "sweeps": sweeps,
        "seed_gap": gap,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for card in TYPES:
        data = export(card)
        raw = json.dumps(data, separators=(",", ":")).encode()
        path = OUT / f"{data['p2']}.json.gz"
        path.write_bytes(gzip.compress(raw, 9))
        print(f"  P2 {str(card):9} {len(data['nodes']):5} nodes  {data['sweeps']:2} sweeps  "
              f"seed gap {data['seed_gap']:.0e}  {path.stat().st_size/1e3:5.0f} KB", flush=True)


if __name__ == "__main__":
    main()
