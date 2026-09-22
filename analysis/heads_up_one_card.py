"""Heads-up Coup with one influence each, no bluffing: solved exactly.

The reduced game: two players, one card each, both cards public, 0 coins, and
nobody ever claims a character they do not hold (so nobody ever challenges).
The Ambassador is left out.  What remains is a finite perfect-information race
-- either reach 7 coins and Coup, or reach 3 and Assassinate someone who cannot
answer with a Contessa -- and it has an exact value.

    python -m analysis.heads_up_one_card
"""

from __future__ import annotations

from coup import Card, GameState, RuleConfig, new_game
from coup.actions import ACTIONS
from coup.decisions import Block, ChooseAction, Discard, ExchangeReturn, Pass
from coup.solve import Value, solve

CARDS = [Card.DUKE, Card.ASSASSIN, Card.CAPTAIN, Card.CONTESSA]

CONFIG = RuleConfig(
    starting_influence=1,
    starting_coins=0,
    two_player_start_handicap=False,
)


def position(first: Card, second: Card) -> GameState:
    return new_game(2, config=CONFIG, hands=[[first], [second]])


def describe(decision) -> str:
    if isinstance(decision, ChooseAction):
        name = str(decision.kind)
        return f"{name} vs P{decision.target}" if decision.target is not None else name
    if isinstance(decision, Block):
        return f"block ({decision.character})"
    if isinstance(decision, Pass):
        return "pass"
    if isinstance(decision, Discard):
        return f"reveal {decision.card}"
    if isinstance(decision, ExchangeReturn):
        return "exchange"
    return repr(decision)


def line(solution) -> str:
    parts = []
    for mover, decision in solution.principal_variation():
        text = describe(decision)
        if text != "pass":  # passing on a block window is noise in the line
            parts.append(f"P{mover}: {text}")
    return "  ".join(parts)


def coins_at_start(state) -> tuple[int, int]:
    return tuple(p.coins for p in state.players)


def main() -> None:
    results = {}
    for first in CARDS:
        for second in CARDS:
            results[first, second] = solve(position(first, second))

    width = max(len(str(c)) for c in CARDS) + 2
    print("Winner, with the first player's card down the side.")
    print("'1st' = the player to move first wins; number is how many of their turns it takes.\n")
    header = " " * (width + 2) + "".join(str(c).ljust(width + 4) for c in CARDS)
    print(header)
    for first in CARDS:
        row = [str(first).ljust(width + 2)]
        for second in CARDS:
            solution = results[first, second]
            if solution.value is Value.DRAW:
                cell = "draw"
            else:
                seat = 0 if solution.value is Value.P0_WINS else 1
                cell = f"{'1st' if seat == 0 else '2nd'} ({solution.turns(seat)})"
            row.append(cell.ljust(width + 4))
        print("".join(row))

    print("\n\nOptimal lines\n" + "-" * 13)
    for first in CARDS:
        for second in CARDS:
            solution = results[first, second]
            verdict = (
                "draw"
                if solution.value is Value.DRAW
                else f"P0 ({first}) wins"
                if solution.value is Value.P0_WINS
                else f"P1 ({second}) wins"
            )
            nodes = len(solution.nodes)
            print(f"\n{first} (first) vs {second}: {verdict}, {solution.depth} plies, {nodes} states")
            print(f"  {line(solution)}")


if __name__ == "__main__":
    main()
