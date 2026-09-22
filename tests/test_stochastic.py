"""Tests for the stochastic solver, where the Exchange makes values probabilities."""

from __future__ import annotations

import itertools
from math import comb

import pytest

from coup import Card, RuleConfig, new_game
from coup.solve import Value, solve
from coup.stochastic import build, dead_cards, evaluate, lookup_key, win_probability

CONFIG = RuleConfig(starting_influence=1, starting_coins=0, two_player_start_handicap=False)


def duel(mine, theirs, dead=(Card.CONTESSA, Card.CONTESSA)):
    return new_game(
        2, config=CONFIG, hands=[[mine], [theirs]], revealed=[[dead[0]], [dead[1]]]
    )


def solved(root):
    nodes = build(root)
    evaluate(nodes)
    return nodes


def test_the_ambassador_beats_the_captain_moving_first():
    """It blocks the Steal, so it wins the income race -- no exchange needed."""
    root = duel(Card.AMBASSADOR, Card.CAPTAIN, dead=(Card.DUKE, Card.DUKE))
    assert win_probability(solved(root), root) == pytest.approx(1.0)


def test_the_captain_beats_the_ambassador_moving_first():
    """So it is a pure first-mover matchup, not a dominance."""
    root = duel(Card.CAPTAIN, Card.AMBASSADOR, dead=(Card.DUKE, Card.DUKE))
    assert win_probability(solved(root), root) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "mine,theirs",
    list(itertools.product([Card.DUKE, Card.CAPTAIN], [Card.ASSASSIN, Card.CONTESSA])),
)
def test_it_reproduces_the_exact_solver_where_no_ambassador_is_in_play(mine, theirs):
    """Without an Ambassador nothing reads the deck, so values must be 0 or 1
    and must agree with the retrograde solve exactly."""
    root = duel(mine, theirs)
    probability = win_probability(solved(root), root)
    assert probability in (0.0, 1.0)
    assert (probability == 1.0) is (solve(root).value is Value.P0_WINS)


def test_both_seeds_reach_the_same_fixpoint():
    """Agreement proves no line can be dragged out forever; it is not assumed."""
    root = duel(Card.AMBASSADOR, Card.DUKE, dead=(Card.DUKE, Card.CONTESSA))
    nodes = build(root)
    evaluate(nodes, optimistic=False)
    pessimistic = {k: n.value for k, n in nodes.items()}
    evaluate(nodes, optimistic=True)
    assert max(abs(pessimistic[k] - n.value) for k, n in nodes.items()) < 1e-9


def test_which_cards_are_dead_changes_the_value():
    """The reason the dead cards belong in the lookup key."""
    a = duel(Card.AMBASSADOR, Card.DUKE, dead=(Card.CAPTAIN, Card.CAPTAIN))
    b = duel(Card.AMBASSADOR, Card.DUKE, dead=(Card.CONTESSA, Card.CONTESSA))
    assert win_probability(solved(a), a) != win_probability(solved(b), b)


@pytest.mark.parametrize(
    "captains_dead,dead",
    [
        (0, (Card.CONTESSA, Card.CONTESSA)),
        (1, (Card.CAPTAIN, Card.CONTESSA)),
        (2, (Card.CAPTAIN, Card.CAPTAIN)),
    ],
)
def test_ambassador_against_duke_is_exactly_the_odds_of_finding_a_captain(
    captains_dead, dead
):
    """An analytic check on the whole machine.

    The Ambassador loses this race outright and only a Captain rescues it, so
    its equity is precisely the chance that two cards off an 11-card deck
    contain one -- and that falls as Captains die.
    """
    root = duel(Card.AMBASSADOR, Card.DUKE, dead=dead)
    captains_left = 3 - captains_dead
    expected = 1 - comb(11 - captains_left, 2) / comb(11, 2)
    assert win_probability(solved(root), root) == pytest.approx(expected)


def test_the_lookup_key_records_the_dead_cards():
    state = duel(Card.AMBASSADOR, Card.DUKE, dead=(Card.DUKE, Card.CONTESSA))
    assert dead_cards(state) == "Contessa+Duke"
    assert lookup_key(state)[-1] == "Contessa+Duke"
    assert lookup_key(state)[:4] == ("Ambassador", "Duke", 0, 0)
