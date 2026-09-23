"""The one-sided bluffing game, solved exactly.

Rules, on top of the heads-up one-influence game without the Ambassador:

* P1 (seat 0) may claim any character, for actions and for blocks.
* P2 (seat 1) is honest, and cannot see P1's card.  P1 sees everything.
* P2 may challenge any claim P1 makes.  With one influence a challenge ends
  the game: P1 wins it if the claim was true, loses it if not.  P1 never
  challenges, because P2 never lies.

**Why this is tractable.**  P2 remembers every claim, so its decisions depend
on the whole history -- and the tree of histories is far too large to walk.
But P1's hidden card matters at exactly one moment: when a challenge is
resolved.  Everything else is public.  So for each *public* position we can
compute the set of outcomes P2 can guarantee, written as a vector with one
entry per card P1 might hold, and that set does not depend on how the
position was reached.  History only enters through P2's belief about P1's
card, and a belief turns a set into a number by a weighted minimum.

The sets are built backwards with three rules (see ``coup.polytope``):

* P1 chooses, and P2 sees what -- *intersection* of the children's sets,
  because P2 must have an answer ready for every choice;
* P2 chooses, and may randomise -- *convex hull of the union*;
* P2 challenges a claim of card c -- the call outcome is the vector with a 1 at
  c and 0 elsewhere, since P1 survives exactly when it holds c.

That is enough to capture mixed bluffing exactly.  The value of the game for a
prior ``p`` is ``min over z in U(start) of p . z``.

Cycles in the public graph (a blocked action changes nothing) are handled as
before: iterate from a pessimistic and an optimistic seed and require the two
to meet, which proves no line can be dragged out forever.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .actions import ACTIONS, ActionKind
from .cards import Card
from .decisions import Block, Challenge, ChooseAction, Pass
from .engine import apply, legal_decisions, new_game, to_act
from .polytope import intersect, prune, support, union
from .solve import state_key, truthful_decisions
from .state import Phase, RuleConfig
from .tablebase import describe

#: The cards P1 might hold, in the order used by every vector.
TYPES = [Card.DUKE, Card.ASSASSIN, Card.CAPTAIN, Card.CONTESSA]
INDEX = {c: i for i, c in enumerate(TYPES)}
MAX_COINS = 12

CONFIG = RuleConfig(starting_influence=1, starting_coins=0, two_player_start_handicap=False)

#: P1's card never matters except when a challenge is resolved, which the
#: solver handles itself; the engine still needs *some* card in P1's hand.
PLACEHOLDER = Card.DUKE


@dataclass
class Node:
    kind: str                      # term | forced | p1 | p2 | call
    mover: int | None
    info: tuple                    # public description, for display
    moves: list = field(default_factory=list)   # [(label, child index)]
    claim: int | None = None       # call nodes: the card P1 claimed
    vec: np.ndarray | None = None  # terminals: P1's result, the same for every card


def p1_claim(state):
    """The card P1 is claiming in the current challenge window, if any."""
    if state.phase is Phase.ACTION_CHALLENGE and state.pending_action.actor == 0:
        return ACTIONS[state.pending_action.kind].claim
    if state.phase is Phase.BLOCK_CHALLENGE and state.pending_block.blocker == 0:
        return state.pending_block.character
    return None


def p1_moves(state):
    """Anything P1 may say.  No Ambassador exists here, so claiming one would
    be a certain lie; and P1 never challenges an honest P2."""
    out = []
    for d in legal_decisions(state):
        if isinstance(d, Challenge):
            continue
        if isinstance(d, ChooseAction) and d.kind is ActionKind.EXCHANGE:
            continue
        if isinstance(d, Block) and d.character is Card.AMBASSADOR:
            continue
        out.append(d)
    return out


def public_info(state) -> tuple:
    a, b = state.players
    act, blk = state.pending_action, state.pending_block
    return (
        a.coins, b.coins, state.turn, to_act(state), state.phase.name,
        "" if act is None else str(act.kind), -1 if act is None else act.actor,
        "" if blk is None else str(blk.character), -1 if blk is None else blk.blocker,
    )


def start(p2_card: Card, p1_coins: int = 0, p2_coins: int = 0, first: int = 0):
    state = new_game(2, config=CONFIG, hands=[[PLACEHOLDER], [p2_card]])
    state.players[0].coins, state.players[1].coins, state.turn = p1_coins, p2_coins, first
    return state


def build(p2_card: Card):
    """Every public position reachable from any legal start, for one P2 card."""
    roots = [start(p2_card, a, b, f)
             for a in range(MAX_COINS + 1) for b in range(MAX_COINS + 1) for f in (0, 1)]
    index: dict[tuple, int] = {}
    nodes: list[Node] = []
    queue: deque = deque()

    def intern(state) -> int:
        key = state_key(state)
        if key not in index:
            index[key] = len(nodes)
            nodes.append(None)
            queue.append((state, index[key]))
        return index[key]

    starts = {}
    for r in roots:
        starts[(r.players[0].coins, r.players[1].coins, r.turn)] = intern(r)

    while queue:
        state, i = queue.popleft()
        if state.game_over:
            vec = np.ones(4) if state.winner == 0 else np.zeros(4)
            nodes[i] = Node("term", None, public_info(state), vec=vec)
            continue
        mover = to_act(state)
        claim = p1_claim(state)
        if mover == 1 and claim is not None:
            passed = apply(state.clone(), Pass(), validate=False)
            nodes[i] = Node("call", 1, public_info(state),
                            moves=[("pass", intern(passed))], claim=INDEX[claim])
            continue
        if mover == 1:
            decisions = truthful_decisions(state)
        elif state.phase in (Phase.ACTION_CHALLENGE, Phase.BLOCK_CHALLENGE):
            decisions = [Pass()]          # P2's claim is true; calling it only loses
        else:
            decisions = p1_moves(state)
        moves = [(describe(d), intern(apply(state.clone(), d, validate=False)))
                 for d in decisions]
        kind = "forced" if len(moves) == 1 else ("p1" if mover == 0 else "p2")
        nodes[i] = Node(kind, mover, public_info(state), moves=moves)
    return nodes, starts


def _probes(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fixed = [np.eye(4)[i] for i in range(4)] + [np.full(4, 0.25)]
    return np.vstack(fixed + [rng.dirichlet(np.ones(4)) for _ in range(24)])


def solve(nodes, seed_value: float, tolerance: float = 1e-9, max_sweeps: int = 400):
    """Iterate the guarantee sets to a fixpoint.  Returns (sets, sweeps)."""
    probes = _probes()
    sets = [n.vec[None, :] if n.kind == "term" else np.full((1, 4), seed_value) for n in nodes]
    order = [i for i in reversed(range(len(nodes))) if nodes[i].kind != "term"]
    for sweep in range(1, max_sweeps + 1):
        worst = 0.0
        for i in order:
            n = nodes[i]
            kids = [sets[c] for _, c in n.moves]
            if n.kind == "forced":
                new = kids[0]
            elif n.kind == "p1":
                new = intersect(kids)
            elif n.kind == "p2":
                new = union(kids)
            else:  # call
                call = np.zeros((1, 4)); call[0, n.claim] = 1.0
                new = union([call, kids[0]])
            before = np.min(sets[i] @ probes.T, axis=0)
            after = np.min(new @ probes.T, axis=0)
            worst = max(worst, float(np.max(np.abs(after - before))))
            sets[i] = new
        if worst <= tolerance:
            return sets, sweep
    raise RuntimeError(f"guarantee sets did not settle in {max_sweeps} sweeps (last change {worst})")


def value(sets, i, belief) -> float:
    return support(sets[i], belief)


# --- playing the equilibrium ---------------------------------------------
#
# The guarantee sets give values.  To *play*, each node needs a strategy, and
# every one of them is a small linear program.  A line is followed with two
# pieces of state: P2's belief ``p`` about P1's card, and the guarantee ``y``
# P2 is committed to -- a point of the current set with p . y equal to the
# value.  Carrying ``y`` rather than re-deriving it at every node is what keeps
# the two sides' mixes consistent where P2 is indifferent, which in a bluffing
# game is exactly where the interesting play is.

from scipy.optimize import linprog  # noqa: E402

TOL = 1e-9


def options(node, sets):
    """(label, vertex set) for every choice at a node, calls included."""
    out = [(label, sets[child]) for label, child in node.moves]
    if node.kind == "call":
        call = np.zeros((1, 4))
        call[0, node.claim] = 1.0
        out.insert(0, ("call", call))
    return out


def p1_split(node, sets, p):
    """P1's equilibrium mix at a node where it chooses: q[a, t] is the
    probability that P1 holds card t *and* plays option a.

    This is the dual of the value LP: P1 splits P2's belief across its options
    so that, with P2 answering each one as well as it can, the total is as
    large as possible.
    """
    opts = options(node, sets)
    A = len(opts)
    nq = 4 * A
    c = np.concatenate([np.zeros(nq), -np.ones(A)])          # maximise sum of w_a
    rows, rhs = [], []
    for a, (_label, verts) in enumerate(opts):
        for v in verts:                                       # w_a <= q_a . v
            r = np.zeros(nq + A)
            r[4 * a:4 * a + 4] = -v
            r[nq + a] = 1.0
            rows.append(r); rhs.append(0.0)
    eq = np.zeros((4, nq + A))
    for t in range(4):
        eq[t, [4 * a + t for a in range(A)]] = 1.0             # sum over a of q[a, t] = p_t
    res = linprog(c, A_ub=np.array(rows), b_ub=np.array(rhs), A_eq=eq, b_eq=np.asarray(p, float),
                  bounds=[(0, None)] * nq + [(0, 1)] * A, method="highs")
    if not res.success:
        raise RuntimeError(f"P1 split failed: {res.message}")
    return res.x[:nq].reshape(A, 4), -res.fun


def target(verts, belief, bound):
    """The point of conv(verts) + R^4_+ that is at most ``bound`` and best for
    P2 under ``belief``: how P2 honours a commitment after seeing a move."""
    k = len(verts)
    res = linprog(verts @ np.asarray(belief, float),
                  A_ub=verts.T, b_ub=np.asarray(bound, float) + TOL,
                  A_eq=np.ones((1, k)), b_eq=[1.0], bounds=[(0, None)] * k, method="highs")
    if not res.success:
        return None
    return res.x @ verts


def supporting_belief(verts, point):
    """A belief under which ``point`` is P2's best reply -- for moves the
    equilibrium never makes, where Bayes' rule has nothing to say."""
    # maximise the worst margin  p . (v - point)  over the vertices
    k = len(verts)
    c = np.concatenate([np.zeros(4), [-1.0]])
    A_ub = np.hstack([-(verts - point), np.ones((k, 1))])
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(k), A_eq=[[1, 1, 1, 1, 0]], b_eq=[1.0],
                  bounds=[(0, None)] * 4 + [(None, None)], method="highs")
    return res.x[:4] if res.success else np.full(4, 0.25)


def p1_step(node, sets, p, y):
    """Everything needed to show and follow a P1 decision.

    Returns, per option: the probability each card plays it, P2's belief after
    seeing it, and the guarantee P2 then commits to -- whose entry for a card
    is that card's EV for the option.
    """
    q, _ = p1_split(node, sets, p)
    p = np.asarray(p, float)
    steps = []
    for a, (label, verts) in enumerate(options(node, sets)):
        mass = q[a].sum()
        sigma = np.divide(q[a], p, out=np.zeros(4), where=p > TOL)
        if mass > TOL:
            post = q[a] / mass
            ya = target(verts, post, y)
        else:
            post = None
            ya = target(verts, np.ones(4), y)                  # lowest point within the bound
        if ya is None:                                         # numerical slack: fall back
            post_ = post if post is not None else p
            ya = verts[np.argmin(verts @ post_)]
        if post is None:
            post = supporting_belief(verts, ya)
        steps.append({"label": label, "sigma": sigma, "belief": post, "target": ya})
    return steps


def p2_step(node, sets, p, y):
    """P2's equilibrium mix at a node where it chooses, and the guarantee it
    carries into each option."""
    opts = options(node, sets)
    stacked = np.vstack([v for _l, v in opts])
    owner = np.concatenate([[b] * len(v) for b, (_l, v) in enumerate(opts)])
    lam = None
    k = len(stacked)
    res = linprog(stacked @ np.asarray(p, float), A_ub=stacked.T,
                  b_ub=np.asarray(y, float) + TOL, A_eq=np.ones((1, k)), b_eq=[1.0],
                  bounds=[(0, None)] * k, method="highs")
    if res.success:
        lam = res.x
    steps = []
    for b, (label, verts) in enumerate(opts):
        if lam is not None and lam[owner == b].sum() > TOL:
            mu = lam[owner == b].sum()
            yb = lam[owner == b] @ stacked[owner == b] / mu
        else:
            mu = 0.0
            yb = verts[np.argmin(verts @ np.asarray(p, float))]
        steps.append({"label": label, "mu": float(mu), "target": yb})
    return steps
