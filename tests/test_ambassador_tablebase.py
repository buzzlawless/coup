"""Tests for the five-card tablebase, where the Exchange makes values odds."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import itertools
from collections import Counter
from math import comb

import pytest

from analysis.build_ambassador_tablebase import (
    CONFIG,
    CSV_PATH,
    MAX_COINS,
    META_PATH,
    deals,
)
from coup import Card, new_game
from coup.tablebase import (
    AMBASSADOR_FIELDS,
    load,
    load_ambassador,
    probe,
    probe_ambassador,
)

pytestmark = pytest.mark.skipif(
    not CSV_PATH.exists(), reason="run `python -m analysis.build_ambassador_tablebase`"
)

TABLE = load_ambassador() if CSV_PATH.exists() else {}


def duel(mine, theirs, dead, my_coins=0, their_coins=0):
    state = new_game(
        2,
        config=CONFIG,
        hands=[[mine], [theirs]],
        revealed=[[dead[0]], [dead[1]]],
    )
    state.players[0].coins = my_coins
    state.players[1].coins = their_coins
    return state


# --- the artifact ---------------------------------------------------------


def test_metadata_matches_the_file():
    meta = json.loads(META_PATH.read_text())
    assert meta["sha256"] == hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()
    assert meta["rows"] == len(TABLE)
    assert meta["seed_agreement"] < 1e-6  # no line runs forever


def test_header_is_the_declared_schema():
    with gzip.open(CSV_PATH, "rt", newline="") as handle:
        assert next(csv.reader(handle)) == AMBASSADOR_FIELDS


def test_every_probability_is_a_probability():
    """Regression: float drift once produced a negative zero."""
    assert all(0.0 <= e.win_probability <= 1.0 for e in TABLE.values())


def test_every_legal_action_position_is_covered():
    stored = {k[:4] + (k[9],) for k in TABLE if k[4] == "ACTION"}
    expected = {
        (str(mine), str(theirs), mc, tc, "+".join(sorted(str(c) for c in dead)))
        for (mine, theirs), dead in deals()
        for mc in range(MAX_COINS + 1)
        for tc in range(MAX_COINS + 1)
    }
    assert stored == expected


# --- cross-checks against the independently built four-card table ---------


def test_it_agrees_with_the_four_card_table_where_no_ambassador_is_in_play():
    """Two tables, two algorithms -- retrograde and value iteration -- must
    give the same answer wherever the Exchange cannot happen."""
    plain = load()
    four = [Card.DUKE, Card.ASSASSIN, Card.CAPTAIN, Card.CONTESSA]
    dead = (Card.CONTESSA, Card.CONTESSA)
    checked = 0
    for mine, theirs in itertools.product(four, repeat=2):
        if any(n > 3 for n in Counter([mine, theirs, *dead]).values()):
            continue
        for coins in (0, 3, 7, 11):
            state = duel(mine, theirs, dead, coins, coins)
            exact = probe(state, plain)
            odds = probe_ambassador(state, TABLE)
            assert odds.certain
            assert (odds.win_probability == 1.0) is (exact.result == "win")
            checked += 1
    assert checked > 40


# --- the results this table was built to hold ----------------------------


@pytest.mark.parametrize("captains_dead,dead", [
    (0, (Card.CONTESSA, Card.CONTESSA)),
    (1, (Card.CAPTAIN, Card.CONTESSA)),
    (2, (Card.CAPTAIN, Card.CAPTAIN)),
])
def test_ambassador_against_duke_is_the_odds_of_finding_a_captain(captains_dead, dead):
    state = duel(Card.AMBASSADOR, Card.DUKE, dead)
    left = 3 - captains_dead
    assert probe_ambassador(state, TABLE).win_probability == pytest.approx(
        1 - comb(11 - left, 2) / comb(11, 2), abs=1e-9
    )


def test_the_ambassador_beats_the_captain_moving_first():
    state = duel(Card.AMBASSADOR, Card.CAPTAIN, (Card.DUKE, Card.DUKE))
    assert probe_ambassador(state, TABLE).win_probability == 1.0


def test_the_ambassador_mirror_races_rather_than_rerolling():
    """The one family where holding beats exchanging below certainty: it would
    trade its own re-roll away while the opponent kept theirs."""
    state = duel(Card.AMBASSADOR, Card.AMBASSADOR, (Card.CONTESSA, Card.CONTESSA))
    entry = probe_ambassador(state, TABLE)
    assert "Exchange" not in entry.best_moves
    assert "Foreign Aid" in entry.best_moves
    assert 0.0 < entry.win_probability < 1.0
