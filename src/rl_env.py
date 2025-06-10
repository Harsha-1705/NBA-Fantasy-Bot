# src/rl_env.py
from __future__ import annotations

import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd


class FantasyEnv(gym.Env):
    """
    A **tiny** draft RL environment; only enough to compile & step.
    Replace dummy logic with real calculations later.
    """

    metadata = {"render.modes": ["ansi"]}

    def __init__(self, weekly_df: pd.DataFrame, bank_start: float = 50_000.0):
        super().__init__()
        self.weekly = weekly_df.copy()
        self.bank_start = bank_start

        # Roster space
        self.player_ids = sorted(self.weekly["PLAYER_ID"].unique().tolist())
        self.n_players = len(self.player_ids)

        # --- Gym spaces ---
        self.action_space = gym.spaces.Discrete(1 + 2 * self.n_players)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        # --- internal ---
        self.week = 0
        self.bank = bank_start
        self.roster: set[int] = set()

    # --------------------------------------------------------------------- #
    # Core API                                                               #
    # --------------------------------------------------------------------- #
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.week = 0
        self.bank = self.bank_start
        self.roster.clear()
        obs = self._get_state()
        info = {}
        return obs, info

    def step(self, action: int):
        """
        Dummy logic:
        * BUY → add pid to roster
        * SELL → remove pid if in roster
        * Reward = random small noise (placeholder)
        """
        if action != 0:
            idx = (action - 1) % self.n_players
            pid = self.player_ids[idx]
            if action <= self.n_players:
                self.roster.add(pid)
            else:
                self.roster.discard(pid)

        reward = random.uniform(-1, 1)   # placeholder
        self.week += 1
        term = self.week >= 10           # stop after 10 ticks for now
        obs = self._get_state()
        info = {}
        return obs, reward, term, False, info

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    def _get_state(self) -> np.ndarray:
        """Return a 6-element float32 vector (placeholder values)."""
        return np.array(
            [
                30.0,              # rolling FP
                32.0,              # rolling MIN
                2.0,               # days rest
                6_000.0,           # salary
                self.bank,         # bank
                self.week,         # week_no
            ],
            dtype=np.float32,
        )

    def render(self):
        print(f"Week {self.week} | Bank: {self.bank:.0f} | Roster size: {len(self.roster)}")
