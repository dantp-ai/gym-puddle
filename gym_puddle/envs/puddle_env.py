import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any


class PuddleEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, start=[0.2, 0.4], goal=[1.0, 1.0], goal_threshold=0.1,
            noise=0.025, thrust=0.05, puddle_center=[[.3, .6], [.4, .5], [.8, .9]],
            puddle_width=[[.1, .03], [.03, .1], [.03, .1]], render_mode = None):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.goal_threshold = goal_threshold
        self.noise = noise
        self.thrust = thrust
        self.puddle_center = [np.array(center) for center in puddle_center]
        self.puddle_width = [np.array(width) for width in puddle_width]

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(2,))

        self.actions = [np.zeros(2) for i in range(5)]
        for i in range(4):
            self.actions[i][i//2] = thrust * (i%2 * 2 - 1)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.pos = None

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        self.pos += self.actions[action] + self.np_random.uniform(low=-self.noise, high=self.noise, size=(2,))
        self.pos = np.clip(self.pos, 0.0, 1.0)

        reward = self._get_reward(self.pos)

        done = np.linalg.norm((self.pos - self.goal), ord=1) < self.goal_threshold

        return self.pos, reward, done, {}

    def _get_reward(self, pos):
        reward = -1.
        for cen, wid in zip(self.puddle_center, self.puddle_width):
            reward -= 2. * self._gaussian1d(pos[0], cen[0], wid[0]) * \
                self._gaussian1d(pos[1], cen[1], wid[1])

        return reward

    def _gaussian1d(self, p, mu, sig):
        return np.exp(-((p - mu)**2)/(2.*sig**2)) / (sig*np.sqrt(2.*np.pi))

    def reset(self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None):
        super().reset(seed=seed)

        if self.start is None:
            self.pos = self.observation_space.sample()
        else:
            self.pos = np.copy(self.start)

        return self.pos, {}
