"""Tests for the exact solver on the reduced heads-up game."""

from __future__ import annotations

import pytest

from coup import ActionKind, Card, ChooseAction, RuleConfig, new_game
from coup.decisions import Block, Challenge
from coup.solve import NonTerminating, Value, solve, truthful_decisions

CARDS = [Card.DUKE, Card.ASSASSIN, Card.CAPTAIN, Card.CONTESSA]

CONFIG = RuleConfig(
    starting_influence=1,
    starting_coins=0,
    two_player_start_handicap=False,
)


def duel(first: Card, second: Card):
    return solve(new_game(2, config=CONFIG, hands=[[first], [second]]))


def winner(first: Card, second: Card) -> int:
    return 0 if duel(first, second).value is Value.P0_WINS else 1


# --- the reduced game itself ---------------------------------------------


def test_one_card_zero_coin_deal():
    state = new_game(2, config=CONFIG, hands=[[Card.DUKE], [Card.CAPTAIN]])
    assert [p.influence for p in state.players] == [[Card.DUKE], [Card.CAPTAIN]]
    assert [p.coins for p in state.players] == [0, 0]
    assert len(state.deck) == 13


def test_hands_must_match_starting_influence():
    with pytest.raises(ValueError):
        new_game(2, config=CONFIG, hands=[[Card.DUKE, Card.DUKE], [Card.CAPTAIN]])


def test_cannot_deal_a_card_the_deck_lacks():
    config = RuleConfig(starting_influence=4, starting_coins=0)
    with pytest.raises(ValueError):
        new_game(2, config=config, hands=[[Card.DUKE] * 4, [Card.CAPTAIN] * 4])


# --- the honest-play restriction ------------------------------------------


def test_truthful_play_drops_bluffs_and_challenges():
    state = new_game(2, config=CONFIG, hands=[[Card.CONTESSA], [Card.DUKE]])
    moves = truthful_decisions(state)
    assert ChooseAction(ActionKind.TAX) not in moves       # no Duke to back it
    assert ChooseAction(ActionKind.INCOME) in moves
    assert ChooseAction(ActionKind.FOREIGN_AID) in moves
    assert not any(isinstance(m, Challenge) for m in moves)


def test_only_a_real_holder_may_block():
    state = new_game(2, config=CONFIG, hands=[[Card.CAPTAIN], [Card.CONTESSA]])
    from coup.engine import apply

    apply(state, ChooseAction(ActionKind.FOREIGN_AID))
    assert not any(isinstance(m, Block) for m in truthful_decisions(state))


# --- solved values --------------------------------------------------------


def test_every_position_resolves_to_a_win_for_someone():
    """Coup has no draw, so solve() raises rather than inventing one.

    Income cannot be blocked and makes no claim, so a player can always add a
    coin whatever the opponent does, and Steal moves coins rather than
    destroying them.  Somebody always reaches Coup range.
    """
    for first in CARDS:
        for second in CARDS:
            solution = duel(first, second)  # raises NonTerminating if it stalls
            assert solution.value in (Value.P0_WINS, Value.P1_WINS)
            assert not solution.nonterminating


def test_the_state_graph_really_is_cyclic():
    """Justifies the retrograde sweep: minimax would recurse forever here."""
    from coup.solve import build_graph

    nodes = build_graph(new_game(2, config=CONFIG, hands=[[Card.DUKE], [Card.DUKE]]))
    colour = dict.fromkeys(nodes, 0)
    back_edges = 0
    for start in nodes:
        if colour[start]:
            continue
        colour[start] = 1
        stack = [(start, iter(nodes[start].moves))]
        while stack:
            key, moves = stack[-1]
            nxt = next(moves, None)
            if nxt is None:
                colour[key] = 2
                stack.pop()
                continue
            child = nxt[1]
            if colour[child] == 1:
                back_edges += 1
            elif colour[child] == 0:
                colour[child] = 1
                stack.append((child, iter(nodes[child].moves)))
    assert back_edges > 0


def test_the_solver_is_heads_up_only():
    config = RuleConfig(starting_influence=1, starting_coins=0)
    state = new_game(3, config=config, hands=[[Card.DUKE], [Card.CAPTAIN], [Card.CONTESSA]])
    with pytest.raises(ValueError):
        solve(state)


def test_a_first_player_win_is_reported_as_such():
    """Value.P0_WINS is 0; a falsy-zero slip would report it as a draw."""
    solution = duel(Card.DUKE, Card.DUKE)
    assert solution.value is Value.P0_WINS
    assert solution.turns(0) == 4


def test_captain_wins_every_matchup_from_either_seat():
    for other in CARDS:
        if other is not Card.CAPTAIN:
            assert winner(Card.CAPTAIN, other) == 0
            assert winner(other, Card.CAPTAIN) == 1


def test_duke_beats_contessa_even_moving_second():
    assert winner(Card.DUKE, Card.CONTESSA) == 0
    assert winner(Card.CONTESSA, Card.DUKE) == 1


@pytest.mark.parametrize(
    "first,second",
    [
        (Card.DUKE, Card.ASSASSIN),
        (Card.ASSASSIN, Card.DUKE),
        (Card.ASSASSIN, Card.CONTESSA),
        (Card.CONTESSA, Card.ASSASSIN),
        (Card.DUKE, Card.DUKE),
        (Card.ASSASSIN, Card.ASSASSIN),
        (Card.CONTESSA, Card.CONTESSA),
    ],
)
def test_the_remaining_matchups_go_to_whoever_moves_first(first, second):
    assert winner(first, second) == 0


def test_the_assassin_is_fastest_against_another_assassin():
    assert duel(Card.ASSASSIN, Card.ASSASSIN).turns(0) == 3


def test_a_contessa_blocks_the_knife_and_wins_the_coup_race():
    solution = duel(Card.CONTESSA, Card.ASSASSIN)
    assert solution.value is Value.P0_WINS
    assert any(
        isinstance(d, Block) and d.character is Card.CONTESSA
        for _mover, d in solution.principal_variation()
    )


def test_neither_disputed_ruling_changes_the_reduced_game():
    """No bluffing means no challenges, so the refund never fires; and a steal
    from a penniless player is a wasted turn even where it is legal."""
    from dataclasses import replace

    for flag in ("allow_stealing_from_zero", "refund_cost_on_caught_bluff"):
        config = replace(CONFIG, **{flag: True})
        for first in CARDS:
            for second in CARDS:
                alt = solve(new_game(2, config=config, hands=[[first], [second]]))
                assert alt.value is duel(first, second).value
