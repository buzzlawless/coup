"""Rules tests, weighted towards the cases naive implementations get wrong."""

from __future__ import annotations

import random

import pytest

from coup import (
    ACTIONS,
    FULL_DECK,
    ActionKind,
    Block,
    Card,
    Challenge,
    ChooseAction,
    Discard,
    ExchangeReturn,
    GameState,
    IllegalDecision,
    Pass,
    Phase,
    PlayerState,
    RuleConfig,
    apply,
    legal_decisions,
    new_game,
    observe,
    to_act,
)


def game(hands, coins=None, deck=None, config=None, turn=0) -> GameState:
    """Build a state with known hands, so scenarios are reproducible."""
    remaining = list(FULL_DECK)
    for hand in hands:
        for card in hand:
            remaining.remove(card)
    state = GameState(
        config=config or RuleConfig(),
        players=[
            PlayerState(seat, list(hand), [], 2 if coins is None else coins[seat])
            for seat, hand in enumerate(hands)
        ],
        deck=remaining if deck is None else list(deck),
    )
    state.turn = turn
    state.phase = Phase.ACTION
    return state


def run(state, *decisions, rng=None):
    rng = rng or random.Random(0)
    for decision in decisions:
        apply(state, decision, rng)
    return state


def card_total(state) -> int:
    return (
        len(state.deck)
        + len(state.exchange_drawn)
        + sum(len(p.influence) + len(p.revealed) for p in state.players)
    )


# --- setup ----------------------------------------------------------------


def test_deck_composition():
    assert len(FULL_DECK) == 15
    assert all(FULL_DECK.count(c) == 3 for c in Card)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_deal(n):
    state = new_game(n, random.Random(n))
    assert all(len(p.influence) == 2 for p in state.players)
    assert len(state.deck) == 15 - 2 * n
    assert to_act(state) == 0


def test_two_player_handicap():
    assert new_game(2, random.Random(1)).players[0].coins == 1
    config = RuleConfig(two_player_start_handicap=False)
    assert new_game(2, random.Random(1), config).players[0].coins == 2


@pytest.mark.parametrize("n", [1, 7])
def test_player_count_bounds(n):
    with pytest.raises(ValueError):
        new_game(n)


# --- plain actions --------------------------------------------------------


def test_income_and_foreign_aid_and_tax():
    state = game([[Card.DUKE, Card.DUKE], [Card.CAPTAIN, Card.CAPTAIN]])
    run(state, ChooseAction(ActionKind.INCOME))
    assert state.players[0].coins == 3
    assert state.turn == 1

    state = game([[Card.DUKE, Card.DUKE], [Card.CAPTAIN, Card.CAPTAIN]])
    run(state, ChooseAction(ActionKind.FOREIGN_AID), Pass())
    assert state.players[0].coins == 4

    state = game([[Card.DUKE, Card.DUKE], [Card.CAPTAIN, Card.CAPTAIN]])
    run(state, ChooseAction(ActionKind.TAX), Pass())
    assert state.players[0].coins == 5


def test_coup_costs_seven_and_removes_an_influence():
    state = game(
        [[Card.DUKE, Card.DUKE], [Card.CAPTAIN, Card.AMBASSADOR]], coins=[7, 2]
    )
    run(state, ChooseAction(ActionKind.COUP, 1), Discard(Card.CAPTAIN))
    assert state.players[0].coins == 0
    assert state.players[1].influence == [Card.AMBASSADOR]
    assert state.players[1].revealed == [Card.CAPTAIN]


def test_coup_is_mandatory_at_ten_coins():
    state = game([[Card.DUKE, Card.DUKE], [Card.CAPTAIN, Card.CAPTAIN]], coins=[10, 2])
    assert legal_decisions(state) == [ChooseAction(ActionKind.COUP, 1)]
    with pytest.raises(IllegalDecision):
        apply(state, ChooseAction(ActionKind.INCOME))


def test_coup_is_not_offered_below_seven_coins():
    state = game([[Card.DUKE, Card.DUKE], [Card.CAPTAIN, Card.CAPTAIN]], coins=[6, 2])
    assert ChooseAction(ActionKind.COUP, 1) not in legal_decisions(state)


def test_steal_is_capped_at_the_targets_balance():
    state = game([[Card.CAPTAIN, Card.DUKE], [Card.DUKE, Card.DUKE]], coins=[2, 1])
    run(state, ChooseAction(ActionKind.STEAL, 1), Pass(), Pass())
    assert (state.players[0].coins, state.players[1].coins) == (3, 0)


def test_stealing_from_zero_can_be_forbidden():
    config = RuleConfig(allow_stealing_from_zero=False)
    state = game(
        [[Card.CAPTAIN, Card.DUKE], [Card.DUKE, Card.DUKE]], coins=[2, 0], config=config
    )
    assert ChooseAction(ActionKind.STEAL, 1) not in legal_decisions(state)


# --- challenges -----------------------------------------------------------


def test_surviving_a_challenge_replaces_the_revealed_card():
    state = game([[Card.DUKE, Card.ASSASSIN], [Card.CAPTAIN, Card.CAPTAIN]])
    deck_before = len(state.deck)
    run(state, ChooseAction(ActionKind.TAX), Challenge(), Discard(Card.CAPTAIN))
    assert state.players[0].coins == 5  # tax still went through
    assert len(state.players[0].influence) == 2  # card returned and redrawn
    assert len(state.deck) == deck_before  # net zero: one in, one out
    assert state.players[1].revealed == [Card.CAPTAIN]  # challenger paid


def test_caught_bluffing_cancels_the_action():
    state = game([[Card.CAPTAIN, Card.CAPTAIN], [Card.DUKE, Card.DUKE]])
    run(state, ChooseAction(ActionKind.TAX), Challenge(), Discard(Card.CAPTAIN))
    assert state.players[0].coins == 2  # no tax
    assert state.players[0].revealed == [Card.CAPTAIN]
    assert state.turn == 1


def test_a_failed_challenge_against_an_assassin_costs_two_cards():
    """The challenger pays for the bad challenge, then the knife still lands."""
    state = game(
        [[Card.ASSASSIN, Card.DUKE], [Card.CAPTAIN, Card.AMBASSADOR]], coins=[3, 2]
    )
    run(
        state,
        ChooseAction(ActionKind.ASSASSINATE, 1),
        Challenge(),                  # target challenges and is wrong
        Discard(Card.CAPTAIN),        # pays for the challenge
        Pass(),                       # may still block, but declines
        Discard(Card.AMBASSADOR),     # pays for the assassination
    )
    assert not state.players[1].alive
    assert state.game_over and state.winner == 0


def test_a_target_who_loses_a_challenge_may_still_block():
    state = game(
        [[Card.ASSASSIN, Card.DUKE], [Card.CONTESSA, Card.AMBASSADOR]], coins=[3, 2]
    )
    run(state, ChooseAction(ActionKind.ASSASSINATE, 1), Challenge(), Discard(Card.AMBASSADOR))
    assert state.phase is Phase.BLOCK
    assert Block(Card.CONTESSA) in legal_decisions(state)
    run(state, Block(Card.CONTESSA), Pass())
    assert state.players[1].influence == [Card.CONTESSA]  # survived on one card


def test_assassination_fee_is_refunded_when_the_assassin_is_caught():
    state = game([[Card.DUKE, Card.DUKE], [Card.CAPTAIN, Card.CAPTAIN]], coins=[3, 2])
    run(state, ChooseAction(ActionKind.ASSASSINATE, 1), Challenge(), Discard(Card.DUKE))
    assert state.players[0].coins == 3
    assert len(state.players[1].influence) == 2  # assassination never happened


def test_assassination_fee_refund_can_be_switched_off():
    config = RuleConfig(refund_cost_on_caught_bluff=False)
    state = game(
        [[Card.DUKE, Card.DUKE], [Card.CAPTAIN, Card.CAPTAIN]], coins=[3, 2], config=config
    )
    run(state, ChooseAction(ActionKind.ASSASSINATE, 1), Challenge(), Discard(Card.DUKE))
    assert state.players[0].coins == 0


# --- blocks ---------------------------------------------------------------


def test_foreign_aid_can_be_blocked_by_any_player():
    state = game(
        [
            [Card.CAPTAIN, Card.CAPTAIN],
            [Card.AMBASSADOR, Card.AMBASSADOR],
            [Card.DUKE, Card.ASSASSIN],
        ]
    )
    run(state, ChooseAction(ActionKind.FOREIGN_AID), Pass())
    assert to_act(state) == 2  # the poll reached the third player
    run(state, Block(Card.DUKE), Pass(), Pass())
    assert state.players[0].coins == 2  # blocked


def test_only_the_target_may_block_a_steal():
    state = game(
        [
            [Card.CAPTAIN, Card.DUKE],
            [Card.AMBASSADOR, Card.AMBASSADOR],
            [Card.CAPTAIN, Card.CAPTAIN],
        ]
    )
    run(state, ChooseAction(ActionKind.STEAL, 1), Pass(), Pass())
    assert state.phase is Phase.BLOCK
    assert to_act(state) == 1
    run(state, Pass())
    assert state.players[0].coins == 4


def test_a_bluffed_block_costs_the_blocker_and_lets_the_action_land():
    state = game(
        [[Card.ASSASSIN, Card.DUKE], [Card.CAPTAIN, Card.AMBASSADOR]], coins=[3, 2]
    )
    run(
        state,
        ChooseAction(ActionKind.ASSASSINATE, 1),
        Pass(),                      # nobody challenges the assassin
        Block(Card.CONTESSA),        # a bluff
        Challenge(),                 # and it is called
        Discard(Card.CAPTAIN),       # pays for the bluff
        Discard(Card.AMBASSADOR),    # the assassination proceeds
    )
    assert not state.players[1].alive


def test_a_true_block_survives_its_challenge_and_stops_the_action():
    state = game(
        [[Card.ASSASSIN, Card.DUKE], [Card.CONTESSA, Card.AMBASSADOR]], coins=[3, 2]
    )
    run(
        state,
        ChooseAction(ActionKind.ASSASSINATE, 1),
        Pass(),
        Block(Card.CONTESSA),
        Challenge(),
        Discard(Card.DUKE),          # the challenger pays
    )
    assert len(state.players[1].influence) == 2  # unharmed
    assert state.players[0].coins == 0           # the fee is still gone
    assert state.turn == 1


def test_a_block_that_nobody_challenges_stands():
    state = game([[Card.CAPTAIN, Card.DUKE], [Card.DUKE, Card.DUKE]], coins=[2, 5])
    run(state, ChooseAction(ActionKind.STEAL, 1), Pass(), Block(Card.CAPTAIN), Pass())
    assert (state.players[0].coins, state.players[1].coins) == (2, 5)


# --- exchange -------------------------------------------------------------


def test_exchange_keeps_hand_size_and_conserves_cards():
    state = game([[Card.AMBASSADOR, Card.DUKE], [Card.CAPTAIN, Card.CAPTAIN]])
    run(state, ChooseAction(ActionKind.EXCHANGE), Pass())
    assert state.phase is Phase.EXCHANGE_RETURN
    assert len(state.exchange_drawn) == 2
    options = legal_decisions(state)
    assert all(len(o.keep) == 2 for o in options)
    run(state, options[0])
    assert len(state.players[0].influence) == 2
    assert card_total(state) == 15


def test_exchange_on_one_influence_keeps_one_of_three():
    state = game([[Card.AMBASSADOR], [Card.CAPTAIN, Card.CAPTAIN]])
    state.players[0].revealed = [Card.DUKE]
    state.deck.remove(Card.DUKE)
    run(state, ChooseAction(ActionKind.EXCHANGE), Pass())
    assert all(len(o.keep) == 1 for o in legal_decisions(state))


def test_exchange_return_must_come_from_the_pool():
    state = game([[Card.AMBASSADOR, Card.DUKE], [Card.CAPTAIN, Card.CAPTAIN]])
    run(state, ChooseAction(ActionKind.EXCHANGE), Pass())
    with pytest.raises(IllegalDecision):
        apply(state, ExchangeReturn((Card.CONTESSA, Card.CONTESSA)))


# --- elimination ----------------------------------------------------------


def test_elimination_returns_coins_to_the_treasury():
    state = game([[Card.DUKE, Card.DUKE], [Card.CAPTAIN]], coins=[7, 6])
    state.players[1].revealed = [Card.CAPTAIN]
    state.deck.remove(Card.CAPTAIN)
    run(state, ChooseAction(ActionKind.COUP, 1), Discard(Card.CAPTAIN))
    assert state.players[1].coins == 0
    assert state.game_over and state.winner == 0


def test_turn_order_skips_eliminated_players():
    state = game(
        [[Card.DUKE, Card.DUKE], [Card.CAPTAIN], [Card.AMBASSADOR, Card.AMBASSADOR]],
        coins=[7, 2, 2],
    )
    state.players[1].revealed = [Card.CAPTAIN]
    state.deck.remove(Card.CAPTAIN)
    run(state, ChooseAction(ActionKind.COUP, 1), Discard(Card.CAPTAIN))
    assert state.turn == 2


# --- hidden information ---------------------------------------------------


def test_observation_hides_other_hands():
    state = game([[Card.DUKE, Card.ASSASSIN], [Card.CAPTAIN, Card.CONTESSA]])
    view = observe(state, 0)
    assert view.hand == (Card.DUKE, Card.ASSASSIN)
    assert view.players[1].influence_count == 2
    assert view.players[1].revealed == ()
    assert Card.CONTESSA not in view.hand
    assert isinstance(hash(view.key()), int)


def test_observations_differ_by_seat_but_agree_on_public_state():
    state = game([[Card.DUKE, Card.ASSASSIN], [Card.CAPTAIN, Card.CONTESSA]])
    a, b = observe(state, 0), observe(state, 1)
    assert a.hand != b.hand
    assert a.players == b.players
    assert a.history == b.history


def test_clone_is_independent():
    state = game([[Card.DUKE, Card.DUKE], [Card.CAPTAIN, Card.CAPTAIN]])
    snapshot = state.clone()
    run(state, ChooseAction(ActionKind.INCOME))
    assert snapshot.players[0].coins == 2
    assert snapshot.turn == 0
    assert snapshot.deck is not state.deck


# --- invariants over random play -----------------------------------------


@pytest.mark.parametrize("seed", range(40))
def test_random_games_terminate_and_conserve_cards(seed):
    rng = random.Random(seed)
    state = new_game(rng.choice([2, 3, 4, 5, 6]), rng)
    for _ in range(10_000):
        if state.game_over:
            break
        assert card_total(state) == 15
        assert all(p.coins >= 0 for p in state.players)
        options = legal_decisions(state)
        assert options
        apply(state, rng.choice(options), rng)
    assert state.game_over
    assert state.winner is not None
    assert card_total(state) == 15


def test_every_action_is_in_the_table():
    assert set(ACTIONS) == set(ActionKind)
