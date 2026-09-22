"""A rules engine for Coup (base game, 2-6 players).

The engine is a pure state machine: ``new_game`` builds a state,
``legal_decisions`` enumerates the moves available to ``to_act``, and ``apply``
advances the game by one of them.  There is no I/O, no agent and no search --
those go on top.

    >>> import random
    >>> from coup import new_game, legal_decisions, apply, to_act
    >>> rng = random.Random(0)
    >>> state = new_game(4, rng)
    >>> while not state.game_over:
    ...     state = apply(state, rng.choice(legal_decisions(state)), rng)
    >>> state.winner is not None
    True
"""

from .actions import ACTIONS, ActionKind, ActionSpec
from .cards import COPIES_PER_CARD, FULL_DECK, Card
from .decisions import (
    Block,
    Challenge,
    ChooseAction,
    Decision,
    Discard,
    ExchangeReturn,
    Pass,
)
from .engine import IllegalDecision, apply, legal_decisions, new_game, to_act
from .state import (
    GameState,
    LossCause,
    Phase,
    PlayerState,
    Resume,
    RuleConfig,
)
from .views import Observation, PlayerView, observe

__all__ = [
    "ACTIONS",
    "COPIES_PER_CARD",
    "FULL_DECK",
    "ActionKind",
    "ActionSpec",
    "Block",
    "Card",
    "Challenge",
    "ChooseAction",
    "Decision",
    "Discard",
    "ExchangeReturn",
    "GameState",
    "IllegalDecision",
    "LossCause",
    "Observation",
    "Pass",
    "Phase",
    "PlayerState",
    "PlayerView",
    "Resume",
    "RuleConfig",
    "apply",
    "legal_decisions",
    "new_game",
    "observe",
    "to_act",
]
