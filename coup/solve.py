"""Exact solution of small Coup positions by retrograde analysis.

Coup with two players, one influence each and no bluffing is a finite
zero-sum game of perfect information, so it has a definite value: a win for
one side, or a draw.

Two things stop plain minimax from working:

* **The graph has cycles.**  A blocked action changes nothing -- Foreign Aid
  into a Duke, a Steal into a Captain -- so a player can pass in all but name,
  and a losing player may prefer to do so forever.  Backward induction from the
  terminals handles this; a state that never gets resolved is a draw by
  infinite play.
* **There is no depth bound**, so the search has to run over the state *graph*
  rather than the game tree.  ``state_key`` collapses transpositions.

``solve`` therefore runs the standard breadth-first retrograde sweep, which
also yields distance-to-win: a node is won as soon as one successor is won for
the mover, and lost only once every successor has been shown lost.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from enum import IntEnum

from .actions import ACTIONS
from .decisions import Block, Challenge, ChooseAction, Decision
from .engine import apply, legal_decisions, to_act
from .state import GameState


class Value(IntEnum):
    """Who wins the position under perfect play."""

    P0_WINS = 0
    P1_WINS = 1
    DRAW = 2


def state_key(state: GameState) -> tuple:
    """A canonical key: everything that can affect play, and nothing else.

    The deck and the history are excluded.  The history cannot affect future
    play, and the deck is only touched by a draw -- so this is a sound
    collapse exactly as long as no draw can occur, which ``solve`` asserts.
    """
    return (
        tuple(
            (tuple(sorted(p.influence)), tuple(sorted(p.revealed)), p.coins)
            for p in state.players
        ),
        state.turn,
        int(state.phase),
        None
        if state.pending_action is None
        else (
            state.pending_action.actor,
            int(state.pending_action.kind),
            state.pending_action.target,
            state.pending_action.paid,
        ),
        None
        if state.pending_block is None
        else (state.pending_block.blocker, int(state.pending_block.character)),
        tuple(state.responders),
        tuple(state.losses),
        int(state.resume),
    )


def truthful_decisions(state: GameState) -> list[Decision]:
    """Legal moves, minus every bluff and every challenge.

    Dropping bluffs is the premise.  Dropping challenges follows from it: if it
    is common knowledge that nobody bluffs then every claim is true, so a
    challenge is a losing move and no solution would ever choose one.  Removing
    them keeps the deck untouched, which is what lets the analysis treat each
    player's card as fixed.
    """
    seat = to_act(state)
    if seat is None:
        return []
    hand = state.players[seat].influence

    out: list[Decision] = []
    for decision in legal_decisions(state):
        if isinstance(decision, Challenge):
            continue
        if isinstance(decision, ChooseAction):
            claim = ACTIONS[decision.kind].claim
            if claim is not None and claim not in hand:
                continue
        if isinstance(decision, Block) and decision.character not in hand:
            continue
        out.append(decision)
    return out


@dataclass
class Node:
    state: GameState
    mover: int | None
    moves: list[tuple[Decision, tuple]]  # (decision, successor key)
    value: Value | None = None
    #: Plies to the end under perfect play, once resolved.
    depth: int = 0


@dataclass
class Solution:
    root: tuple
    nodes: dict[tuple, Node]

    @property
    def value(self) -> Value:
        # Not `or`: Value.P0_WINS is 0 and would be swallowed as falsy.
        value = self.nodes[self.root].value
        return Value.DRAW if value is None else value

    @property
    def depth(self) -> int:
        return self.nodes[self.root].depth

    def best_moves(self, key: tuple) -> list[tuple[Decision, tuple]]:
        """Moves preserving the node's value, shortest win / longest loss first."""
        node = self.nodes[key]
        if node.mover is None:
            return []
        wins = Value(node.mover)
        keep = [(d, k) for d, k in node.moves if self.nodes[k].value is node.value]
        reverse = node.value is not wins
        return sorted(keep, key=lambda dk: self.nodes[dk[1]].depth, reverse=reverse)

    def turns(self, seat: int) -> int:
        """How many of ``seat``'s own turns the principal variation takes.

        ``depth`` counts plies, and a single turn spans several -- declaring,
        each response window, and any reveal -- so it is not a turn count.
        """
        return sum(
            1
            for mover, decision in self.principal_variation()
            if mover == seat and isinstance(decision, ChooseAction)
        )

    def principal_variation(self) -> list[tuple[int, Decision]]:
        """One optimal line from the root, as (mover, decision) pairs."""
        line: list[tuple[int, Decision]] = []
        key, seen = self.root, set()
        while key not in seen:
            seen.add(key)
            node = self.nodes[key]
            best = self.best_moves(key)
            if not best:
                break
            decision, key = best[0]
            assert node.mover is not None
            line.append((node.mover, decision))
        return line


def build_graph(root: GameState, decisions=truthful_decisions) -> dict[tuple, Node]:
    """Expand every position reachable from ``root``."""
    rng = random.Random(0)  # unused: the reduced game makes no draws
    deck_size = len(root.deck)
    nodes: dict[tuple, Node] = {}
    queue = deque([root])
    nodes[state_key(root)] = Node(root, to_act(root), [])

    while queue:
        state = queue.popleft()
        node = nodes[state_key(state)]
        if state.game_over:
            continue
        for decision in decisions(state):
            child = apply(state.clone(), decision, rng, validate=False)
            if len(child.deck) != deck_size:
                raise AssertionError(
                    f"{decision!r} touched the deck; state_key would be unsound"
                )
            child_key = state_key(child)
            if child_key not in nodes:
                nodes[child_key] = Node(child, to_act(child), [])
                queue.append(child)
            node.moves.append((decision, child_key))
    return nodes


def solve(root: GameState, decisions=truthful_decisions) -> Solution:
    nodes = build_graph(root, decisions)

    predecessors: dict[tuple, list[tuple]] = {k: [] for k in nodes}
    unresolved: dict[tuple, int] = {}
    frontier: deque[tuple] = deque()

    for key, node in nodes.items():
        if node.state.game_over:
            node.value = Value(node.state.winner) if node.state.winner is not None else Value.DRAW
            node.depth = 0
            frontier.append(key)
            continue
        unresolved[key] = len(node.moves)
        for _decision, child in node.moves:
            predecessors[child].append(key)
        if not node.moves:  # no legal move: cannot arise in Coup, but be safe
            node.value = Value.DRAW
            frontier.append(key)

    # Breadth-first retrograde sweep.  A node is won the moment one successor is
    # known won for its mover; it is lost only once every successor is spent.
    while frontier:
        key = frontier.popleft()
        resolved = nodes[key]
        for parent_key in predecessors[key]:
            parent = nodes[parent_key]
            if parent.value is not None:
                continue
            if resolved.value == Value(parent.mover):
                parent.value = resolved.value
                parent.depth = resolved.depth + 1
                frontier.append(parent_key)
            else:
                unresolved[parent_key] -= 1
                if unresolved[parent_key] == 0:
                    parent.value = resolved.value
                    parent.depth = resolved.depth + 1
                    frontier.append(parent_key)

    for node in nodes.values():
        if node.value is None:
            node.value = Value.DRAW  # never forced either way: infinite play

    return Solution(state_key(root), nodes)
