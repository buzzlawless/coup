"""Tests for the heads-up one-card tablebase."""

from __future__ import annotations

import csv
import hashlib
import json

import pytest

from analysis.build_tablebase import (
    CARDS,
    CONFIG,
    CSV_PATH,
    MAX_COINS,
    META_PATH,
    build,
    position,
)
from coup import Card, new_game
from coup.engine import apply
from coup.solve import Value, solve, truthful_decisions
from coup.tablebase import FIELDS, Entry, describe, load, position_key, probe

TABLE = load()


def opening(first: Card, second: Card):
    return new_game(2, config=CONFIG, hands=[[first], [second]])


# --- the artifact on disk -------------------------------------------------


def test_committed_table_matches_a_fresh_build():
    """The file is a build product; it must not drift from the solver."""
    with CSV_PATH.open(newline="") as handle:
        on_disk = list(csv.DictReader(handle))
    rebuilt = [{k: str(v) for k, v in row.items()} for row in build()]
    assert on_disk == rebuilt


def test_metadata_records_the_rules_and_a_matching_digest():
    meta = json.loads(META_PATH.read_text())
    assert meta["sha256"] == hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()
    assert meta["rows"] == len(TABLE)
    assert meta["rules"]["starting_influence"] == 1
    assert meta["rules"]["starting_coins"] == 0


def test_header_is_the_declared_schema():
    with CSV_PATH.open(newline="") as handle:
        assert next(csv.reader(handle)) == FIELDS


def test_every_legal_action_position_is_covered():
    """All 16 matchups x every legal coin pair, not just what a game reaches."""
    import itertools

    stored = {
        key[:4] for key in TABLE if key[4] == "ACTION"
    }  # (mover card, opponent card, mover coins, opponent coins)
    expected = {
        (str(a), str(b), ca, cb)
        for a, b in itertools.product(CARDS, repeat=2)
        for ca in range(MAX_COINS + 1)
        for cb in range(MAX_COINS + 1)
    }
    assert stored == expected
    assert len(expected) == 16 * (MAX_COINS + 1) ** 2 == 2704


def test_twelve_coins_is_the_ceiling():
    """A turn opening on 10+ may only Coup, so it opens on <=9 to earn; Tax is +3."""
    assert MAX_COINS == CONFIG.mandatory_coup_threshold - 1 + 3
    assert max(key[2] for key in TABLE) == MAX_COINS
    assert max(key[3] for key in TABLE) == MAX_COINS


def test_a_forced_coup_wins_on_the_spot():
    """With one influence, anyone who must Coup thereby wins."""
    forced = [
        (key, entry)
        for key, entry in TABLE.items()
        if key[4] == "ACTION" and key[2] >= CONFIG.mandatory_coup_threshold
    ]
    assert len(forced) == 624
    assert all(entry.result == "win" for _key, entry in forced)
    assert all(entry.best_moves == ("Coup",) for _key, entry in forced)


def test_positions_beyond_the_ceiling_are_absent():
    state = position(Card.DUKE, Card.CAPTAIN, MAX_COINS + 1, 0)
    with pytest.raises(KeyError):
        probe(state, TABLE)


def test_no_terminal_positions_are_stored():
    assert all(entry.result in ("win", "loss") for entry in TABLE.values())


# --- lookups --------------------------------------------------------------


def test_every_opening_agrees_with_the_solver():
    for first in CARDS:
        for second in CARDS:
            state = opening(first, second)
            entry = probe(state, TABLE)
            solution = solve(state)
            assert entry.winning is (solution.value is Value.P0_WINS)
            assert entry.dtm == solution.depth


def test_probe_rejects_positions_outside_the_table():
    from coup import RuleConfig

    two_card = new_game(
        2,
        config=RuleConfig(starting_influence=2, starting_coins=0),
        hands=[[Card.DUKE, Card.DUKE], [Card.CAPTAIN, Card.CAPTAIN]],
    )
    with pytest.raises(ValueError):
        probe(two_card, TABLE)


def test_probe_rejects_a_finished_game():
    state = opening(Card.DUKE, Card.CAPTAIN)
    state.players[1].influence.clear()
    state.players[1].revealed.append(Card.CAPTAIN)
    from coup.state import Phase

    state.phase = Phase.GAME_OVER
    with pytest.raises(ValueError):
        probe(state, TABLE)


# --- the tablebase actually plays the game -------------------------------


def decision_named(state, name):
    for decision in truthful_decisions(state):
        if describe(decision) == name:
            return decision
    raise AssertionError(f"{name!r} is not a legal move here")


@pytest.mark.parametrize("first", CARDS)
@pytest.mark.parametrize("second", CARDS)
def test_following_the_table_wins_and_counts_down_exactly(first, second):
    """End to end: every stored move is legal, and dtm ticks down by one a ply."""
    state = opening(first, second)
    predicted = 0 if probe(state, TABLE).winning else 1

    previous = None
    plies = 0
    while not state.game_over:
        entry = probe(state, TABLE)
        if previous is not None:
            assert entry.dtm == previous - 1, "dtm must fall by exactly one per ply"
        previous = entry.dtm
        apply(state, decision_named(state, entry.best_moves[0]))
        plies += 1
        assert plies < 200

    assert state.winner == predicted
    assert previous == 1  # the last stored position was one ply from the end


@pytest.mark.parametrize("first", CARDS)
@pytest.mark.parametrize("second", CARDS)
def test_every_stored_alternative_is_legal_and_equally_good(first, second):
    """Where the table lists several moves, they must all be real and tie."""
    state = opening(first, second)
    entry = probe(state, TABLE)
    for name in entry.best_moves:
        branch = state.clone()
        apply(branch, decision_named(branch, name))
        assert probe(branch, TABLE).dtm == entry.dtm - 1
