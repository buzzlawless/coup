"""Solving positions where the Exchange makes the game stochastic.

Once an Ambassador is in play a position no longer has a win/loss value but a
win *probability*, because the exchange draws from the deck.  That rules out
the retrograde sweep in ``solve``: with chance nodes there is no "won as soon
as one successor is won".  What replaces it is value iteration.

The structure that keeps this cheap is a stratification.  A position with no
Ambassador in either hand cannot reach one -- no other action touches the deck
-- so those positions form a closed, purely deterministic subgame whose values
are exactly 0 or 1.  They are the boundary condition for everything above them,
and iteration only has to run over positions where an Ambassador is actually
held.

Nodes do not retain their ``GameState``.  At a few hundred thousand positions
that would cost hundreds of megabytes for nothing: everything later needed --
who moves, the lookup key, the moves and where they lead -- is extracted while
the state is in hand and the state is then dropped.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .cards import Card
from .chance import outcomes
from .engine import to_act
from .solve import state_key, truthful_decisions
from .state import GameState
from .tablebase import describe, position_key

def dead_cards(state: GameState) -> str:
    """The two revealed cards, which fix the deck and so every draw probability.

    A player down to one influence has revealed the other, face up and out of
    the deck.  In the Ambassador-free game nothing reads the deck, so this
    cannot affect play and ``position_key`` rightly leaves it out; once an
    Exchange is possible it decides the odds and has to be part of the key.
    """
    return "+".join(sorted(str(c) for p in state.players for c in p.revealed))


def lookup_key(state: GameState) -> tuple:
    """``position_key`` plus the dead cards."""
    return position_key(state) + (dead_cards(state),)


#: Absolute convergence bound on the value of every position.
TOLERANCE = 1e-12
MAX_SWEEPS = 100_000


@dataclass(slots=True)
class Node:
    #: Seat to move, or None at a terminal position.
    mover: int | None
    #: The lookup key, relative to the mover and including the dead cards;
    #: None at a terminal position.
    key: tuple | None
    #: ((move name, ((probability, successor key), ...)), ...)
    moves: tuple = ()
    #: Set only at a terminal position.
    winner: int | None = None
    #: Probability that seat 0 wins, under perfect play by both.
    value: float = 0.0

    @property
    def terminal(self) -> bool:
        return self.mover is None

    #: True once the position can never reach an Exchange, so its value is exact.
    deterministic: bool = field(default=False)


def _holds_ambassador(state: GameState) -> bool:
    return any(Card.AMBASSADOR in p.influence for p in state.players)


def build(roots, decisions=truthful_decisions) -> dict[tuple, Node]:
    """Expand every position reachable from ``roots``, chance included."""
    if isinstance(roots, GameState):
        roots = [roots]

    nodes: dict[tuple, Node] = {}
    queue: deque[GameState] = deque()
    for root in roots:
        key = state_key(root)
        if key not in nodes:
            nodes[key] = Node(mover=None, key=None)  # filled in below
            queue.append(root)

    while queue:
        state = queue.popleft()
        key = state_key(state)
        node = nodes[key]

        if state.game_over:
            node.mover, node.key, node.winner = None, None, state.winner
            node.value = 1.0 if state.winner == 0 else 0.0
            node.deterministic = True
            continue

        node.mover = to_act(state)
        node.key = lookup_key(state)
        node.deterministic = not _holds_ambassador(state)

        moves = []
        for decision in decisions(state):
            successors = []
            for probability, child in outcomes(state, decision):
                child_key = state_key(child)
                if child_key not in nodes:
                    nodes[child_key] = Node(mover=None, key=None)
                    queue.append(child)
                successors.append((probability, child_key))
            moves.append((describe(decision), tuple(successors)))
        node.moves = tuple(moves)

    return nodes


def _sweep(nodes: dict[tuple, Node], order: list[tuple]) -> float:
    """One Gauss-Seidel pass.  Returns the largest change made."""
    worst = 0.0
    for key in order:
        node = nodes[key]
        best = None
        for _name, successors in node.moves:
            expected = 0.0
            for probability, child in successors:
                expected += probability * nodes[child].value
            if best is None:
                best = expected
            elif node.mover == 0:
                best = max(best, expected)
            else:
                best = min(best, expected)
        change = abs(best - node.value)
        if change > worst:
            worst = change
        node.value = best
    return worst


def evaluate(nodes: dict[tuple, Node], optimistic: bool = False) -> int:
    """Value-iterate to a fixpoint.  Returns the number of sweeps taken.

    ``optimistic`` seeds undecided positions at 1 instead of 0.  Both seeds
    reach the same fixpoint exactly when no line can go on forever; running
    both is therefore a proof of termination rather than an assumption of it.
    """
    order = [key for key, node in nodes.items() if not node.terminal]
    for key in order:
        nodes[key].value = 1.0 if optimistic else 0.0

    for sweep in range(1, MAX_SWEEPS + 1):
        if _sweep(nodes, order) <= TOLERANCE:
            return sweep
    raise RuntimeError(f"value iteration did not settle in {MAX_SWEEPS} sweeps")


def win_probability(nodes: dict[tuple, Node], state: GameState) -> float:
    """The probability that the player to move wins, under perfect play."""
    node = nodes[state_key(state)]
    return node.value if node.mover == 0 else 1.0 - node.value


def best_moves(nodes: dict[tuple, Node], key: tuple) -> tuple[str, ...]:
    node = nodes[key]
    scored = []
    for name, successors in node.moves:
        expected = sum(p * nodes[c].value for p, c in successors)
        scored.append((expected, name))
    target = max(s for s, _ in scored) if node.mover == 0 else min(s for s, _ in scored)
    return tuple(name for score, name in scored if abs(score - target) <= TOLERANCE)
