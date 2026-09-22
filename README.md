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

## The Ambassador, and chance

An Ambassador in play makes the game stochastic: the Exchange draws from the
deck, so a position has a win *probability* rather than a winner.

Two things follow. First, `coup/chance.py` enumerates a draw's outcomes with
their probabilities instead of sampling one, by handing `apply` a stand-in for
its source of randomness — so the engine needs no notion of a chance node.
Second, `coup/stochastic.py` replaces the retrograde sweep with value
iteration, since with chance there is no "won as soon as one successor is won".
Seeding the iteration pessimistically and optimistically and getting the same
fixpoint is a *proof* that no line runs forever, rather than an assumption.

**One influence means one card lost.** A player down to a single influence has
revealed the other, face up and out of the deck — so the deck holds 11 cards,
not 13, and *which* two are dead changes every draw. In the Ambassador-free
game nothing reads the deck, so this provably cannot matter and `position_key`
leaves it out; once an Exchange is possible it decides the odds, and
`stochastic.lookup_key` adds it.

How much it matters: an Ambassador facing a Duke from an even start cannot win
while holding its own card at all — every bit of its equity is the exchange.
Exchanging costs the turn, so the Duke moves next, and from that tempo only a
Captain still wins the race. Its value is therefore exactly the chance of
finding a Captain in two cards off an 11-card deck — 27/55, 19/55 or 10/55 as
zero, one or two Captains are already dead. Give it four coins in hand and it
beats the Duke outright without exchanging, so this is a fact about the even
start rather than about the card.

## Tablebase

`tablebase/heads_up_one_card.csv` is the solved table for the reduced game:
**every legal position**, 9,335 of them, with who wins, in how many plies, and
every move that achieves it. Not merely the ones a game from the 0-coin start
reaches — a chess table covers positions no sensible game would produce, and so
does this one.

The coin range is bounded by the forced-Coup rule, not by an arbitrary cap. A
turn beginning on 10 or more coins may only Coup, so any turn that can *add*
coins begins on at most 9, and the largest single-turn gain is Tax at +3.
Nobody can ever hold more than **12**, which makes the space 16 matchups × 13 ×
13 coin pairs = 2,704 action positions, plus the mid-turn positions that follow
from them.

```python
from coup import Card, new_game
from coup.tablebase import load, probe
from analysis.build_tablebase import CONFIG

table = load()
probe(new_game(2, config=CONFIG, hands=[[Card.ASSASSIN], [Card.CAPTAIN]]), table)
# Entry(result='loss', dtm=22, best_moves=('Foreign Aid',))
```

Rows are written from the point of view of **the player to move**, which
collapses the seat symmetry: a Duke on 3 coins facing a Captain on 5 is one
position, not two. The builder asserts that projection is injective, so a
schema that lost a distinguishing field would fail the build rather than emit a
table that is quietly wrong.

| Column | |
|---|---|
| `to_act_card`, `opponent_card` | the two cards, mover first |
| `to_act_coins`, `opponent_coins` | banks, mover first |
| `phase` | `ACTION`, `BLOCK`, `ACTION_CHALLENGE`, `BLOCK_CHALLENGE`, `LOSE_INFLUENCE` |
| `pending_action`, `pending_action_by` | the action on the table, and whether it is the mover's (`self`/`opponent`) |
| `pending_block`, `pending_block_by` | likewise for a declared block |
| `result` | `win` or `loss`, **for the player to move** |
| `dtm` | plies to the end; a ply is one decision, so a turn spans several |
| `best_moves` | `\|`-separated, all tying for best — quickest win, or slowest loss |

Terminal positions are not stored: there is nothing to look up once the game is
decided. All 624 positions where the mover holds 10+ coins are wins playing
Coup — with one influence, being forced to Coup is being handed the game.

**Why CSV.** Real chess tablebases are binary because they are terabytes and
have to be mmapped and compressed; at 430 KB none of that applies. What does
apply is that this is a build product that has to stay trustworthy: CSV diffs
line by line in git, so flipping a `RuleConfig` flag shows exactly which
positions changed value instead of producing an opaque new blob. It also loads
with no dependency at all — `csv`, pandas, SQLite's `.import`, or a
spreadsheet. Rows are sorted deterministically so the diff is meaningful.

`tablebase/heads_up_one_card.meta.json` records the rules, the assumptions, the
row count and a SHA-256 of the CSV, because the table is only valid for the
`RuleConfig` it was solved under.

Regenerate with `python -m analysis.build_tablebase`. A test rebuilds it and
compares against the committed file, so it cannot drift from the solver.

## Tests

```
python -m pytest tests -q
```

Weighted towards the cases naive implementations get wrong: double influence
loss from a failed challenge against an assassin, bluffed blocks, the
assassination fee, exchange on one influence, and card conservation across
random play.
