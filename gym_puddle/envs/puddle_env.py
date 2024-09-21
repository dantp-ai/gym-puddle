from typing import Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class PuddleEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, start: np.ndarray | None = None, goal: tuple[float, float] | None = None, goal_threshold: float = 0.1, noise: float = 0.025, thrust: float = 0.05, puddle_center: list[np.ndarray] | None = None, puddle_width: list[np.ndarray] | None = None, render_mode: str | None = None):
        self.start = start if start is not None else np.array([0.2, 0.4])
        self.goal = goal if goal is not None else np.array([1.0, 1.0])
        self.goal_threshold = goal_threshold
        self.noise = noise
        self.thrust = thrust
        self.puddle_center = puddle_center if puddle_center is not None else [np.array(center) for center in [[.3, .6], [.4, .5], [.8, .9]]]
        self.puddle_width = puddle_width if puddle_width is not None else [np.array(width) for width in [[.1, .03], [.03, .1], [.03, .1]]]

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(2,))

        self.actions = [np.zeros(2) for _ in range(5)]
        # [(-0.05, 0), (0.05, 0), (0, -0.05), (0, 0.05), (0, 0)]
        for i in range(4):
            self.actions[i][i//2] = thrust * (i%2 * 2 - 1)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.pos = self._get_initial_obs()

    def step(self, action: np.int64) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action), f"{action}, {type(action)} invalid"

        self.pos += self.actions[action] + self.np_random.uniform(low=-self.noise, high=self.noise, size=(2,))
        self.pos = np.clip(self.pos, 0.0, 1.0)

        reward = self._get_reward(self.pos)

        terminated = np.linalg.norm((self.pos - self.goal), ord=1) < self.goal_threshold

        return self.pos, reward, terminated, False, {}

    def _get_reward(self, pos):
        reward = -1.
        for cen, wid in zip(self.puddle_center, self.puddle_width):
            reward -= 2. * self._gaussian1d(pos[0], cen[0], wid[0]) * \
                self._gaussian1d(pos[1], cen[1], wid[1])

        return reward

    def _get_initial_obs(self) -> np.ndarray:
        if self.start is None:
            pos = self.observation_space.sample()
        else:
            pos = self.start.copy()

        return pos

    def _gaussian1d(self, p, mu, sig):
        return np.exp(-((p - mu)**2)/(2.*sig**2)) / (sig*np.sqrt(2.*np.pi))

    def reset(self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None):
        super().reset(seed=seed)

        self.pos = self._get_initial_obs()

        return self.pos, {}
