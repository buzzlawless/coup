"""Exact solution of small Coup positions by retrograde analysis.

Coup with two players, one influence each and no bluffing is a finite zero-sum
game of perfect information, so every position is a win for one side.  There is
no draw: the game has no stalemate rule, no turn limit and no agreement.

Two things still stop plain minimax from working:

* **The graph has cycles.**  A blocked action changes nothing -- Foreign Aid
  into a Duke, a Steal into a Captain -- so a player can pass in all but name
  and positions repeat.  Minimax would recurse forever.
* **There is no depth bound**, so the search has to run over the state *graph*
  rather than the game tree.  ``state_key`` collapses transpositions.

``solve`` therefore runs the standard breadth-first retrograde sweep back from
the terminals, which also yields distance-to-win: a node is won as soon as one
successor is won for the mover, and lost only once every successor has been
shown lost.

Cycles being real does not mean a cycle can ever be *chosen*.  A position the
sweep never resolves would be one neither player can force to an end, and since
Coup cannot end in a draw that is a finding or a bug, not a result -- so it
raises ``NonTerminating`` rather than quietly reporting one.
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
    """Who wins the position under perfect play.

    There is no third value.  Coup has no draw mechanism -- no stalemate, no
    turn limit, no agreement -- so the only alternative to a win for one side
    is a game that never ends, and that is an exception rather than a result.
    """

    P0_WINS = 0
    P1_WINS = 1


class NonTerminating(Exception):
    """Neither player could be shown to force the game to an end.

    This should be impossible: Income is always legal, cannot be blocked and
    makes no claim, so every player can add a coin every turn no matter what
    the opponent does, and the only way to take coins away -- Steal -- moves
    them rather than destroying them.  Somebody's bank therefore grows without
    bound, reaches Coup range, and ends the game.

    So this firing means either a real non-terminating line, which is worth
    knowing about, or a bug in the graph.  Neither should be quietly reported
    as a draw.
    """


def state_key(state: GameState) -> tuple:
    """A canonical key: everything that can affect play, and nothing else.

    The deck and the history are excluded.  The history cannot affect future
    play, and the deck's contents are fixed by the cards held and revealed,
    which are included -- so two positions with the same key face the same
    draw distribution.
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
        tuple(sorted(state.exchange_drawn)),
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
    nodes: dict[tuple, Node]
    #: The positions the solve started from.
    roots: tuple = ()

    #: Positions left unresolved, non-empty only when ``solve`` was told to
    #: tolerate them.
    nonterminating: frozenset = frozenset()

    @property
    def root(self) -> tuple:
        if len(self.roots) != 1:
            raise ValueError(f"this solution has {len(self.roots)} roots, not one")
        return self.roots[0]

    @property
    def value(self) -> Value | None:
        # Not `or`: Value.P0_WINS is 0 and would be swallowed as falsy.
        return self.nodes[self.root].value

    @property
    def depth(self) -> int:
        return self.nodes[self.root].depth

    def best_moves(self, key: tuple) -> list[tuple[Decision, tuple]]:
        """Moves preserving the node's value, shortest win / longest loss first."""
        node = self.nodes[key]
        if node.mover is None:
            return []
        wins = Value(node.mover)
        keep = [
            (d, k)
            for d, k in node.moves
            if self.nodes[k].value is node.value and node.value is not None
        ]
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


def build_graph(roots, decisions=truthful_decisions) -> dict[tuple, Node]:
    """Expand every position reachable from ``roots``.

    ``roots`` is one ``GameState`` or an iterable of them.  Passing many at once
    shares the transposition table between them, which is how a whole tablebase
    gets solved in a single sweep instead of once per starting position.
    """
    if isinstance(roots, GameState):
        roots = [roots]
    roots = list(roots)
    if not roots:
        raise ValueError("need at least one root position")

    rng = random.Random(0)  # unused: the reduced game makes no draws
    deck_size = len(roots[0].deck)
    nodes: dict[tuple, Node] = {}
    queue: deque[GameState] = deque()
    for root in roots:
        if len(root.deck) != deck_size:
            raise ValueError("roots must all have the same deck size")
        key = state_key(root)
        if key not in nodes:
            nodes[key] = Node(root, to_act(root), [])
            queue.append(root)

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


def solve(
    root: GameState,
    decisions=truthful_decisions,
    allow_nonterminating: bool = False,
) -> Solution:
    """Solve a single two-player position exactly.

    Raises ``NonTerminating`` if any reachable position cannot be forced to an
    end by either side.  Pass ``allow_nonterminating=True`` to collect those
    positions on the solution instead of raising -- useful when exploring a
    variant that really can stall.
    """
    return solve_many([root], decisions, allow_nonterminating)


def solve_many(
    roots,
    decisions=truthful_decisions,
    allow_nonterminating: bool = False,
) -> Solution:
    """Solve many positions in one sweep, sharing the transposition table."""
    roots = list(roots)
    for root in roots:
        if len(root.players) != 2:
            raise ValueError("the solver handles heads-up positions only")
    nodes = build_graph(roots, decisions)

    predecessors: dict[tuple, list[tuple]] = {k: [] for k in nodes}
    unresolved: dict[tuple, int] = {}
    frontier: deque[tuple] = deque()

    for key, node in nodes.items():
        if node.state.game_over:
            if node.state.winner is None:
                raise AssertionError("a heads-up game ended with nobody alive")
            node.value = Value(node.state.winner)
            node.depth = 0
            frontier.append(key)
            continue
        if not node.moves:
            raise AssertionError("a live position with no legal move")
        unresolved[key] = len(node.moves)
        for _decision, child in node.moves:
            predecessors[child].append(key)

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

    stuck = frozenset(key for key, node in nodes.items() if node.value is None)
    if stuck and not allow_nonterminating:
        example = nodes[next(iter(stuck))].state
        coins = tuple(p.coins for p in example.players)
        raise NonTerminating(
            f"{len(stuck)} of {len(nodes)} positions resolve to neither win; "
            f"e.g. coins={coins} phase={example.phase.name}. "
            "Coup has no draw mechanism, so this is a finding or a bug."
        )

    return Solution(nodes, tuple(state_key(r) for r in roots), nonterminating=stuck)
