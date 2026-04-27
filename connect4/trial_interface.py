from __future__ import annotations

import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

from connect4.grading_state import GradingConnectState, legal_actions
from connect4.seeded_random_player import SeededRandomPlayer

if TYPE_CHECKING:
    from connect4.policy import Policy


class Connect4TrialInterface:
    def __init__(
        self,
        seed: int | None = None,
        opponent_seed: int | None = None,
        opponent_policy: "Policy | None" = None,
    ):
        self.rs = np.random.RandomState(seed)
        self.opponent_seed = opponent_seed
        if opponent_policy is not None:
            self.opponent_policy = opponent_policy
        else:
            if opponent_seed is None:
                raise ValueError(
                    "Connect4TrialInterface requires either opponent_seed or opponent_policy."
                )
            self.opponent_policy = SeededRandomPlayer(opponent_seed)
        self.opponent_name = getattr(
            self.opponent_policy,
            "variant_name",
            type(self.opponent_policy).__name__,
        )

    @staticmethod
    def _as_board(state: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=int).copy()

    def get_player_in_state(self, state: np.ndarray) -> int:
        board = self._as_board(state)
        red_count = int(np.count_nonzero(board == -1))
        yellow_count = int(np.count_nonzero(board == 1))
        return -1 if red_count <= yellow_count else 1

    def get_reward(self, state: np.ndarray) -> float:
        winner = self.get_winner(state)
        if winner == -1:
            return 1.0
        if winner == 1:
            return -1.0
        return 0.0

    def draw_init_state(self) -> tuple[np.ndarray, float]:
        state = np.zeros((GradingConnectState.ROWS, GradingConnectState.COLS), dtype=int)
        return state, self.get_reward(state)

    def get_random_state(self) -> tuple[np.ndarray, float]:
        state, reward = self.draw_init_state()
        rollout_length = int(self.rs.randint(0, 8))
        for _ in range(rollout_length):
            if self.is_terminal_state(state):
                break
            actions = self.get_actions_in_state(state)
            action = actions[self.rs.choice(range(len(actions)))]
            state, reward = self.exec_action(state, action)
        return state, reward

    def get_actions_in_state(self, state: np.ndarray) -> list[int]:
        board = self._as_board(state)
        return legal_actions(board)

    def exec_action(self, state: np.ndarray, action: int) -> tuple[np.ndarray, float]: 
        board = self._as_board(state)
        if self.is_terminal_state(board):
            raise ValueError("Cannot execute an action in a terminal state.")

        player = self.get_player_in_state(board)
        next_state = GradingConnectState(board, player).transition(action)
        if next_state.is_final():
            return next_state.board.copy(), self.get_reward(next_state.board)

        opponent_action = self.opponent_policy.act(next_state.board.copy())
        after_opponent = GradingConnectState(next_state.board, next_state.player).transition(
            opponent_action
        )
        return after_opponent.board.copy(), self.get_reward(after_opponent.board)

    def exec_policy(self, pi, s: np.ndarray | None = None) -> pd.DataFrame:
        if s is None:
            state, reward = self.draw_init_state()
        else:
            state = self._as_board(s)
            reward = self.get_reward(state)

        rows = []
        while not self.is_terminal_state(state):
            action = pi(state.copy())
            rows.append([state.copy(), action, reward])
            state, reward = self.exec_action(state, action)
        rows.append([state.copy(), None, reward])
        return pd.DataFrame(rows, columns=["state", "action", "reward"])

    def is_terminal_state(self, state: np.ndarray) -> bool:
        board = self._as_board(state)
        return GradingConnectState(board, self.get_player_in_state(board)).is_final()

    def get_winner(self, state: np.ndarray) -> int:
        board = self._as_board(state)
        return int(GradingConnectState(board, self.get_player_in_state(board)).get_winner())
