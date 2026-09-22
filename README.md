# coup

A rules engine for [Coup](https://en.wikipedia.org/wiki/Coup_(card_game)) — base
game, 2–6 players. No agents, no search, no UI: just a state machine you can
drive.

```python
import random
from coup import new_game, legal_decisions, apply, to_act, observe

rng = random.Random(0)
state = new_game(4, rng)
while not state.game_over:
    state = apply(state, rng.choice(legal_decisions(state)), rng)
print(state.winner)
```

## The model

**Actions are data, not code.** `coup/actions.py` holds one row per action —
cost, character claimed, whether it needs a target, which characters block it.
Challenges and blocks are written once against that table.

| Action | Cost | Claims | Target | Blocked by | Challengeable |
|---|---|---|---|---|---|
| Income | 0 | — | — | — | no |
| Foreign Aid | 0 | — | — | Duke | no |
| Coup | 7 | — | yes | — | no |
| Tax | 0 | Duke | — | — | yes |
| Assassinate | 3 | Assassin | yes | Contessa | yes |
| Steal | 0 | Captain | yes | Captain, Ambassador | yes |
| Exchange | 0 | Ambassador | — | — | yes |

A block is the same shape one level down: it claims a character, costs nothing,
and is itself challengeable but never blockable.

**A turn is a nested negotiation, not an atomic action.** `Phase` makes the
pending state explicit:

```
ACTION → ACTION_CHALLENGE → BLOCK → BLOCK_CHALLENGE → (resolve)
                 ↘ LOSE_INFLUENCE ↙        ↘ EXCHANGE_RETURN
```

Losing influence interrupts the turn and a single turn can interrupt twice — a
challenger who calls a real Assassin pays once for the challenge and again for
the assassination. So influence loss is a *queue*, and where to pick up
afterwards is parked in `Resume` rather than being held on the call stack.

**Hidden information is separated from the start.** `GameState` is the
referee's view. `observe(state, seat)` is what one player is entitled to know,
and `Observation.key()` is a hashable perfect-recall information-set key.

## API

| | |
|---|---|
| `new_game(n, rng, config)` | deal and start |
| `to_act(state)` | which seat the game is waiting on |
| `legal_decisions(state)` | every move available to that seat |
| `apply(state, decision, rng, validate=True)` | advance one decision; **mutates** `state` |
| `state.clone()` | independent deep copy |
| `observe(state, seat)` | that seat's information set |

`apply` accepts exactly what `legal_decisions` returns and raises
`IllegalDecision` otherwise. `state.history` is the public event log — enough to
reconstruct everything an observer saw.

Validation rebuilds the legal-move list, which roughly doubles the cost of a
step. A loop that already picks its move *from* `legal_decisions` can pass
`validate=False`: about 3.3k → 4.7k random 4-player games/sec single-threaded.

## Rulings that vary between tables

Published rulings genuinely disagree on these. They are `RuleConfig` flags
rather than baked in because two of them change the payoff of a bluff, and so
change the equilibrium — they are worth being able to flip and re-solve:

| Flag | Default | |
|---|---|---|
| `refund_cost_on_caught_bluff` | `False` | The 3 coins buy the *attempt*, so an assassin caught bluffing loses an influence **and** the whole fee. Set `True` for the reading that a cancelled action should never have been charged for. A fee lost to a *Contessa block* is never refunded either way. |
| `allow_stealing_from_zero` | `False` | Steal is not offered against a player with no coins, since it would gain nothing. |
| `two_player_start_handicap` | `True` | In a 2-player game the starting player begins with 1 coin. |
| `mandatory_coup_threshold` | `10` | Coins at which Coup becomes the only legal action. |

## Modeling choices worth knowing

These are decisions the physical game leaves to the table, resolved here so the
tree is well-defined:

- **Response order is sequential, not simultaneous.** Challenge and block
  windows poll live players clockwise from the claimant; the first to act ends
  the window. At a real table this is whoever speaks first.
- **One challenge per claim.** Once a claim has been challenged and settled,
  that window closes — nobody else may challenge it.
- **One block per action.** If a block is defeated as a bluff, the action
  resolves; a second player does not get to block in its place. This only ever
  comes up for Foreign Aid, where more than one player is eligible.
- **A target who loses a challenge may still block**, if they are still alive.
  They lost a card, not their right to block.

## Randomness

It enters in exactly three places — the deal, the replacement card after a
survived challenge, and the two cards drawn for an Exchange — and all three go
through `engine._draw`. The deck is an unordered *bag*, not a stack: drawing
pops a uniformly random index, so a returned card never needs a reshuffle, and
the chance distribution at any point is just the deck's multiset.

That is the lift point for solver work. `_draw` currently resolves inline from
the injected `rng`, which suits sampling methods directly; turning those three
sites into explicit chance nodes for exact traversal is a change localized to
that one function.

## Analysis

`coup/solve.py` solves small positions exactly by retrograde analysis. The
state graph has **cycles** — a blocked action changes nothing, so a player can
pass in all but name — which is why it is a breadth-first sweep back from the
terminals rather than minimax, which would recurse forever.

There is no draw value. Coup has no draw mechanism, and none is reachable
anyway: Income cannot be blocked and makes no claim, so a player can add a coin
every turn whatever the opponent does, and Steal moves coins rather than
destroying them — so somebody always reaches Coup range. A position the sweep
cannot resolve is therefore a finding or a bug, and `solve` raises
`NonTerminating` rather than quietly calling it a draw.

`analysis/heads_up_one_card.py` uses it on the reduced game: two players, one
influence each, 0 coins, both cards public, nobody bluffs (so nobody
challenges), no Ambassador. Run it with `python -m analysis.heads_up_one_card`.

## Tests

```
python -m pytest tests -q
```

Weighted towards the cases naive implementations get wrong: double influence
loss from a failed challenge against an assassin, bluffed blocks, the
assassination fee, exchange on one influence, and card conservation across
random play.
