"""Tests for the one-sided bluffing game: P1 may lie, P2 is honest and blind."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from coup import new_game
from coup.bluff import CONFIG, TYPES, build, p1_step, p2_step, solve, value
from coup.engine import to_act
from coup.polytope import intersect, support, union
from coup.tablebase import load, probe

DATA = Path(__file__).resolve().parent.parent / "docs" / "data" / "bluff"
SHORT = {"Duke": "D", "Assassin": "S", "Captain": "C", "Contessa": "T"}


@pytest.fixture(scope="module")
def solved():
    """card -> (nodes, starts, sets), solved from the pessimistic seed."""
    out = {}
    for card in TYPES:
        nodes, starts = build(card)
        sets, _ = solve(nodes, 1.0)
        out[card] = (nodes, starts, sets)
    return out


# --- the polytope helpers ---------------------------------------------------

def test_union_keeps_only_the_undominated_points():
    a = np.array([[1.0, 0, 0, 0]])
    b = np.array([[0.0, 1, 0, 0]])
    worse = np.array([[1.0, 1, 0, 0]])
    u = union([a, b, worse])
    for belief in np.random.default_rng(0).dirichlet(np.ones(4), 50):
        assert support(u, belief) == pytest.approx(min(belief[0], belief[1]))


def test_intersection_of_up_closed_sets():
    """P1 picks the option, so P2 must meet every option's guarantee at once."""
    a = np.array([[1.0, 0, 0, 0], [0.0, 1, 0, 0]])
    b = np.array([[0.0, 0, 1, 0]])
    x = intersect([a, b])
    for belief in np.random.default_rng(1).dirichlet(np.ones(4), 50):
        best = min(belief[0], belief[1]) + belief[2]
        assert support(x, belief) == pytest.approx(best, abs=1e-9)


# --- the solved game --------------------------------------------------------

def test_both_seeds_reach_the_same_sets():
    """Solving down from 1 and up from 0 must agree, or some line runs forever."""
    probes = np.random.default_rng(2).dirichlet(np.ones(4), 64)
    for card in TYPES:
        nodes, _ = build(card)
        hi, _ = solve(nodes, 1.0)
        lo, _ = solve(nodes, 0.0)
        gap = max(float(np.max(np.abs(np.min(a @ probes.T, 0) - np.min(b @ probes.T, 0))))
                  for a, b in zip(hi, lo))
        assert gap < 1e-7, (card, gap)


def test_certainty_reproduces_the_honest_tablebase(solved):
    """If P2 knew P1's card every bluff would be called, so the game is the
    honest one and must match the tablebase at every start."""
    table = load()
    for card, (_nodes, starts, sets) in solved.items():
        for (a, b, first), i in starts.items():
            for t, mine in enumerate(TYPES):
                s = new_game(2, config=CONFIG, hands=[[mine], [card]])
                s.players[0].coins, s.players[1].coins, s.turn = a, b, first
                wins = (probe(s, table).result == "win") == (to_act(s) == 0)
                assert value(sets, i, np.eye(4)[t]) == pytest.approx(float(wins), abs=1e-9)


def test_bluffing_never_hurts_p1(solved):
    """P1 can always play honestly, so under any belief its value is at least
    the belief-weighted honest value."""
    rng = np.random.default_rng(3)
    for _card, (_nodes, starts, sets) in solved.items():
        for i in list(starts.values())[::7]:
            honest = np.array([value(sets, i, np.eye(4)[t]) for t in range(4)])
            for p in rng.dirichlet(np.ones(4), 4):
                assert value(sets, i, p) >= p @ honest - 1e-9


def test_bluffing_pays_against_a_captain(solved):
    """The headline figure: P2 Captain, both upcards Contessa, P1 to move, no
    coins.  Honest play gives P1 2/9; the right bluffs give it 5/9."""
    nodes, starts, sets = solved[TYPES[2]]
    p = np.array([3, 3, 2, 1], float) / 9          # unseen: Duke, Assassin, Captain, Contessa
    i = starts[(0, 0, 0)]
    honest = np.array([value(sets, i, np.eye(4)[t]) for t in range(4)])
    assert p @ honest == pytest.approx(2 / 9)
    assert value(sets, i, p) == pytest.approx(5 / 9, abs=1e-9)


def test_strategies_are_an_equilibrium(solved):
    """Walk lines along the computed strategies: every option a card plays is
    worth exactly its value, and P2's mixes stay within its guarantee."""
    rng = np.random.default_rng(4)
    for card, (nodes, starts, sets) in solved.items():
        for i0 in list(starts.values())[::23]:
            p = rng.dirichlet(np.ones(4))
            y = sets[i0][np.argmin(sets[i0] @ p)]
            i = i0
            for _ in range(40):
                n = nodes[i]
                if n.kind == "term":
                    break
                if n.kind == "forced":
                    i = n.moves[0][1]
                    continue
                assert p @ y == pytest.approx(value(sets, i, p), abs=1e-7)
                if n.kind == "p1":
                    steps = p1_step(n, sets, p, y)
                    for s in steps:
                        assert np.all(s["target"] <= y + 1e-7)
                        for t in range(4):
                            if p[t] > 1e-9 and s["sigma"][t] > 1e-9:
                                assert s["target"][t] == pytest.approx(y[t], abs=1e-7)
                    on = [s for s in steps if (s["sigma"] * p).sum() > 1e-9]
                    w = np.array([(s["sigma"] * p).sum() for s in on])
                    s = on[rng.choice(len(on), p=w / w.sum())]
                    p, y = s["belief"], s["target"]
                    i = dict(n.moves)[s["label"]]
                else:
                    steps = p2_step(n, sets, p, y)
                    mix = sum(s["mu"] * s["target"] for s in steps)
                    assert np.all(mix <= y + 1e-7)
                    assert p @ mix == pytest.approx(p @ y, abs=1e-7)
                    moves = [s for s in steps if s["label"] != "call" and s["mu"] > 1e-9]
                    if not moves:
                        break
                    s = moves[rng.integers(len(moves))]
                    y = s["target"]
                    i = dict(n.moves)[s["label"]]


# --- the page's data --------------------------------------------------------

needs_data = pytest.mark.skipif(
    not (DATA / "D.json.gz").exists(), reason="run `python -m analysis.build_bluff_data`")


@needs_data
def test_every_p2_card_has_a_data_file_covering_every_start():
    for card in TYPES:
        data = json.loads(gzip.decompress((DATA / f"{SHORT[str(card)]}.json.gz").read_bytes()))
        assert data["types"] == ["D", "S", "C", "T"]
        assert len(data["starts"]) == 13 * 13 * 2
        assert data["seed_gap"] < 1e-7
        assert len(data["sets"]) == len(data["nodes"])


@needs_data
def test_data_files_match_a_fresh_solve(solved):
    rng = np.random.default_rng(5)
    for card, (_nodes, starts, sets) in solved.items():
        data = json.loads(gzip.decompress((DATA / f"{SHORT[str(card)]}.json.gz").read_bytes()))
        for (a, b, f), i in starts.items():
            assert data["starts"][f"{a}.{b}.{f}"] == i
        for i in range(0, len(sets), 11):
            shipped = np.array(data["sets"][i])
            for p in rng.dirichlet(np.ones(4), 3):
                assert support(shipped, p) == pytest.approx(value(sets, i, p), abs=1e-8)
