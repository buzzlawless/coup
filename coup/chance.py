"""Enumerating the deck draws, so the Exchange can be solved rather than sampled.

The engine resolves a draw inline from an injected source of randomness.  For
analysis we need the opposite: every outcome with its probability.  Both
helpers here work by handing ``apply`` a stand-in for that source -- one that
counts the draws a decision makes, and one that dictates them -- so the engine
itself needs no notion of a chance node.

The deck is an unordered bag, so the distribution over a draw of *k* cards is
hypergeometric in the bag's contents, and the bag's contents are fixed by the
cards held and revealed.
"""

from __future__ import annotations

import itertools
import random
from collections import Counter
from math import comb, prod

from .engine import apply
from .state import GameState


class _CountingDraws:
    """A random source that records how many cards a decision draws."""

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng
        self.count = 0

    def randrange(self, n: int) -> int:
        self.count += 1
        return self._rng.randrange(n)


class _ScriptedDraws:
    """A random source that deals chosen cards, in order."""

    def __init__(self, state: GameState, cards) -> None:
        self._state = state
        self._cards = list(cards)
        self._next = 0

    def randrange(self, _n: int) -> int:
        card = self._cards[self._next]
        self._next += 1
        return self._state.deck.index(card)


def draw_count(state: GameState, decision) -> int:
    """How many cards applying ``decision`` takes from the deck."""
    counter = _CountingDraws(random.Random(0))
    apply(state.clone(), decision, counter, validate=False)
    return counter.count


def outcomes(state: GameState, decision) -> list[tuple[float, GameState]]:
    """Every result of applying ``decision``, as (probability, state).

    A decision that touches no cards returns a single certain outcome, so this
    is safe to use for every move rather than only the drawing ones.  That is
    also the overwhelmingly common case, so it is taken on the first attempt:
    the decision is applied with a counting source and the result kept if it
    turned out to draw nothing, rather than probing first and throwing the work
    away.
    """
    child = state.clone()
    counter = _CountingDraws(random.Random(0))
    apply(child, decision, counter, validate=False)
    drawn = counter.count
    if drawn == 0:
        return [(1.0, child)]

    held = Counter(state.deck)
    total = comb(len(state.deck), drawn)
    results = []
    for combination in sorted(
        {tuple(sorted(c)) for c in itertools.combinations(state.deck, drawn)}
    ):
        ways = prod(comb(held[card], n) for card, n in Counter(combination).items())
        child = state.clone()
        apply(child, decision, _ScriptedDraws(child, combination), validate=False)
        results.append((ways / total, child))
    return results
