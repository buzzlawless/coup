"""Tests for enumerating deck draws."""

from __future__ import annotations

from collections import Counter
from math import comb

import pytest

from coup import ActionKind, Card, ChooseAction, Pass, RuleConfig, new_game
from coup.chance import draw_count, outcomes
from coup.engine import apply

CONFIG = RuleConfig(starting_influence=1, starting_coins=0, two_player_start_handicap=False)


def duel(mine, theirs, dead_mine, dead_theirs):
    return new_game(
        2, config=CONFIG, hands=[[mine], [theirs]], revealed=[[dead_mine], [dead_theirs]]
    )


def test_a_revealed_card_leaves_the_deck():
    """One influence means one card *lost*, and a lost card is out of play."""
    state = duel(Card.AMBASSADOR, Card.CAPTAIN, Card.DUKE, Card.DUKE)
    assert len(state.deck) == 11
    assert Counter(state.deck)[Card.DUKE] == 1  # three, minus the two face up
    assert [p.revealed for p in state.players] == [[Card.DUKE], [Card.DUKE]]


def test_most_decisions_draw_nothing():
    state = duel(Card.DUKE, Card.CAPTAIN, Card.CONTESSA, Card.CONTESSA)
    assert draw_count(state, ChooseAction(ActionKind.INCOME)) == 0
    certain = outcomes(state, ChooseAction(ActionKind.INCOME))
    assert len(certain) == 1 and certain[0][0] == 1.0


def test_an_exchange_draws_two_when_it_resolves():
    state = duel(Card.AMBASSADOR, Card.CAPTAIN, Card.DUKE, Card.CONTESSA)
    assert draw_count(state, ChooseAction(ActionKind.EXCHANGE)) == 0  # only declared
    apply(state, ChooseAction(ActionKind.EXCHANGE))
    assert draw_count(state, Pass()) == 2  # the pass resolves it, and that draws


def test_exchange_outcomes_are_a_probability_distribution():
    state = duel(Card.AMBASSADOR, Card.CAPTAIN, Card.DUKE, Card.CONTESSA)
    apply(state, ChooseAction(ActionKind.EXCHANGE))
    results = outcomes(state, Pass())
    assert sum(p for p, _ in results) == pytest.approx(1.0, abs=1e-12)
    assert all(p > 0 for p, _ in results)
    assert all(len(child.exchange_drawn) == 2 for _p, child in results)
    # distinct multisets of 2 from the 11-card deck
    assert len({tuple(sorted(c.exchange_drawn)) for _p, c in results}) == len(results)


def test_probabilities_are_hypergeometric_in_the_remaining_deck():
    state = duel(Card.AMBASSADOR, Card.CAPTAIN, Card.DUKE, Card.CONTESSA)
    held = Counter(state.deck)
    apply(state, ChooseAction(ActionKind.EXCHANGE))
    results = {tuple(sorted(c.exchange_drawn)): p for p, c in outcomes(state, Pass())}

    total = comb(11, 2)
    pair = tuple(sorted((Card.ASSASSIN, Card.DUKE)))
    assert results[pair] == pytest.approx(held[Card.ASSASSIN] * held[Card.DUKE] / total)
    same = (Card.ASSASSIN, Card.ASSASSIN)
    assert results[same] == pytest.approx(comb(held[Card.ASSASSIN], 2) / total)


def test_which_cards_are_dead_changes_the_odds():
    """The whole reason the revealed cards have to be part of the state."""

    def odds(dead_mine, dead_theirs):
        state = duel(Card.AMBASSADOR, Card.CAPTAIN, dead_mine, dead_theirs)
        apply(state, ChooseAction(ActionKind.EXCHANGE))
        return {
            tuple(sorted(c.exchange_drawn)): p for p, c in outcomes(state, Pass())
        }

    two_dukes_dead = odds(Card.DUKE, Card.DUKE)
    no_dukes_dead = odds(Card.CONTESSA, Card.CONTESSA)
    pair = (Card.DUKE, Card.DUKE)
    assert pair not in two_dukes_dead  # only one Duke left; cannot draw two
    assert no_dukes_dead[pair] > 0
