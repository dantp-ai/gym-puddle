from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pygame import Surface as pygSurface
from pygame.time import Clock as pygClock


@dataclass
class Puddle:
    center: np.ndarray
    width: np.ndarray


class PuddleEnv(gym.Env):
    """
    ## Description

    The `PuddleEnv` environment corresponds to the continuous grid-world environment described by Degris Thomas, Martha White, and Richard S. Sutton
    in ["Off-policy actor-critic" arXiv preprint arXiv:1205.4839 (2012).](https://arxiv.org/abs/1205.4839).

    The agent is placed in a two-dimensional continuous grid-world in a start position,
    and has to reliably reach the goal position by maximizing the discounted cumulative sum of rewards (while avoiding negatively rewarding puddles).


    ## Observation Space

    The observation is a `ndarray` with shape `(2,)` with the floating-point values corresponding to coordinates in the two-dimensional grid in the range `[0, 1]`.

    **Note**: Actions (described below in the section "Actions Space") can lead to next observations outside of the grid, in which case the coordinates are clipped to the range `[0, 1]`.


    ## Action Space

    The action is a `ndarray` with shape `(5, )` which can take integer values in `{0, 4}`.

    The first four integers correspond to moving in the four cardinal directions (west, east, south, north).
    The last integer value corresponds to no move (standing still).
    The amount of moving in one of the four directions is given by the configuration parameter `thrust`
    (which is one of the input arguments to the environment, described below in the section "Arguments").
    Hence, to summarize (using thrust = 0.05):

    - 0: (-0.05, 0), move west
    - 1: (0.05, 0),  move east
    - 2: (0, -0.05), move north
    - 3: (0, 0.05),  move south
    - 4: (0, 0),     no move

    **Note**: The next observation is determined by the current agent position plus the direction from one of the five actions **plus**
    a noise drawn from an uniform distribution.
    - The lower and upper bound of the distribution is configurable using the input argument `noise` (see section "Arguments").


    ## Rewards

    At every step where the agent's position (p_x, p_y) is not within the `goal_threshold` (see section "Episode End" below), it receives:

    * -1.0 (if no puddles)
    * -1 + (-2) * (N(p_x, .3, .1) * N(p_y, .6, .03) + N(p_x, .4, .03) * N(p_y, .5, .1) + N(p_x, .8, .03) * N(p_y, .9, .1)) (if puddles exist)

    **Note**:

    - `N(., mu, rho)` = is a one-dimensional normal distribution with mean `mu` and variance `rho`.
    - By default there are three puddles. Each puddle defines a 1-D normal distribution with mean = puddle-center and variance = puddle-width:
        - puddle 1: center=[0.3, 0.6], width=[0.1, 0.03]
        - puddle 2: center=[0.4, 0.5], width=[0.03, 0.1]
        - puddle 3: center=[0.8, 0.9], width=[0.03, 0.1]
    - The number of puddles is also configurable (see input argument `puddles` in the section "Arguments").


    ## Starting State

    The starting state is [0.2, 0.4] (it is configurable by the input argument `start`).


    ## Episode End

    The episode ends with:

    - Termination: The L1-norm of the difference between the agent's position and the goal position is less than 0.1 (the value is configurable through the input argument `threshold`).

    - Truncation: Episode length is greater than e.g., 5000 time-steps. This can be achieved by passing an integer value to the `max_episode_steps=5000` input argument during env init:

    ```python
    import gymnasium as gym
    import gym_puddle

    >>> env = gym.make("PuddleWorld-v0", max_episode_steps=5000)
    >>> env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<PuddleEnv<PuddleWorld-v0>>>>>
    ```

    Note that Gymnasium environments are by default wrapped with `TimeLimit` (https://gymnasium.farama.org/main/api/wrappers/misc_wrappers/#gymnasium.wrappers.TimeLimit),
    so it suffices to provide a non-None value to `max_episode_steps`. No additional `TimeLimit` wrapping is necessary.


    ## Arguments

    - `start`: (list) List with 2-D start position of agent (default: [0.2, 0.4]).
    - `goal`: (list) List with 2-D goal position (default: [1., 1.]).
    - `goal_threshold`: (float) Threshold for the L1-norm of the difference between agent position and `goal` (default: 0.1).
    - `noise`: (float) Defines the lower and higher bound of the uniform distribution that adds noise to the next observation (default: 0.025).
    - `thrust`: (float) Amout of movement in each of the four directions (default: 0.05).
    - `puddles`: (list[Puddle]) List of puddles. When None, the three default puddles are initialized (see section "Reward" above).
    - `render_mode`: (str) Render modes supported are "human" and "rgb_array".
    - `max_episode_steps`: (int) Number of timesteps after which to truncate the current episode (caller should reset the environment immediately afterwards).

    """

    metadata: dict[str, Any] = {
        "render_modes": [
            "human",
            "rgb_array",
        ],
        "render_fps": 50.0,
    }

    def __init__(
        self,
        start: list[float] | None = [0.2, 0.4],  # noqa: B006
        goal: list[float] = [1.0, 1.0],  # noqa: B006
        goal_threshold: float = 0.1,
        noise: float = 0.025,
        thrust: float = 0.05,
        puddles: list[Puddle] | None = None,
        render_mode: str | None = None,
    ):
        self.goal_threshold = goal_threshold
        self.noise = noise
        self.thrust = thrust
        default_puddles = [
            Puddle(center=np.array([0.3, 0.6]), width=np.array([0.1, 0.03])),
            Puddle(center=np.array([0.4, 0.5]), width=np.array([0.03, 0.1])),
            Puddle(center=np.array([0.8, 0.9]), width=np.array([0.03, 0.1])),
        ]
        self.puddles = puddles if puddles is not None else default_puddles

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(2,))

        self.start = np.array(start, dtype=np.float32) if start is not None else None
        self.goal = np.array(goal, dtype=np.float32)

        if not self.observation_space.contains(
            self.start
        ) or not self.observation_space.contains(self.goal):
            raise ValueError(
                "The `start` or `goal` is invalid. Only values in [0., 1.] are allowed."
            )

        self.actions = [np.zeros(2, dtype=np.float32) for _ in range(5)]
        for i in range(4):
            self.actions[i][i // 2] = thrust * (i % 2 * 2 - 1)

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                "`render_mode` is invalid. "
                "Check `metadata['render_modes'] for valid values."
            )

        self.render_mode = render_mode
        self.viewer: pygSurface | None = None
        self.screen_width = 600
        self.screen_height = 400
        self.env_img_pixels = self._draw_image()
        self.clock: pygClock | None = None
        self.pos = self._get_initial_obs()

    def step(
        self,
        action: np.int64,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"{action}, {type(action)} invalid")

        self.pos += self.actions[action] + self.np_random.uniform(
            low=-self.noise,
            high=self.noise,
            size=(2,),
        )
        self.pos = np.clip(self.pos, 0.0, 1.0)

        reward = self._get_reward(self.pos)

        terminated = bool(
            np.linalg.norm((self.pos - self.goal), ord=1) < self.goal_threshold,
        )

        if self.render_mode == "human":
            self.render()

        return self.pos, reward, terminated, False, {}

    def _get_reward(self, pos: np.ndarray) -> float:
        reward = -1.0
        for puddle in self.puddles:
            reward -= (
                2.0
                * self._gaussian1d(pos[0], puddle.center[0], puddle.width[0])
                * self._gaussian1d(pos[1], puddle.center[1], puddle.width[1])
            )

        return reward

    def _get_initial_obs(self) -> np.ndarray:
        if self.start is None:
            pos = self.observation_space.sample()
        else:
            pos = self.start.copy()

        return pos

    def _gaussian1d(self, p: float, mu: float, sig: float) -> float:
        return np.exp(-((p - mu) ** 2) / (2.0 * sig**2)) / (sig * np.sqrt(2.0 * np.pi))

    def _draw_image(
        self,
        img_width: int = 100,
        img_height: int = 100,
        n_channels: int = 3,
    ) -> np.ndarray:
        x = np.linspace(0.0, 1.0, img_width, endpoint=False)
        y = np.linspace(0.0, 1.0, img_height, endpoint=False)
        xx, yy = np.meshgrid(x, y)
        positions = np.stack([xx, yy], axis=2)
        get_reward_vec = np.vectorize(self._get_reward, signature="(n)->()")
        rewards = get_reward_vec(positions)
        pixels = np.repeat(rewards[:, :, np.newaxis], n_channels, axis=2)

        pixels -= pixels.min()

        if pixels.max() == 0:  # should occur only when no puddles
            return np.ones_like(pixels) * 255.0

        pixels *= 255.0 / pixels.max()

        return np.floor(pixels)

    def render(self):  # type: ignore
        if self.render_mode is None:
            gym.logger.warn(
                "You are rendering without specifying render mode."
                "Specify `render_mode` at initialization. "
                "Check suitable values in `env.metadata['render_modes']`.",
            )
            return None

        import pygame

        if self.viewer is None:
            pygame.init()  # pylint: disable=E1101
            if self.render_mode == "human":
                pygame.display.init()
                self.viewer = pygame.display.set_mode(
                    (self.screen_width, self.screen_height),
                )
            else:
                self.viewer = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.screen_width, self.screen_height))
        canvas.fill((255, 255, 255))

        puddle_surface = pygame.surfarray.make_surface(self.env_img_pixels)
        puddle_surface = pygame.transform.scale(
            puddle_surface,
            (self.screen_width, self.screen_height),
        )
        canvas.blit(puddle_surface, (0, 0))

        # Render agent
        agent_size = 10
        agent_pos = (
            int(self.pos[0] * self.screen_width),
            int(self.pos[1] * self.screen_height),
        )
        pygame.draw.rect(canvas, (0, 255, 0), (*agent_pos, agent_size, agent_size))

        # Render goal
        goal_size = 10
        goal_pos = (
            int(self.goal[0] * self.screen_width),
            int(self.goal[1] * self.screen_height),
        )
        pygame.draw.rect(canvas, (255, 0, 0), (*goal_pos, goal_size, goal_size))

        canvas = pygame.transform.flip(canvas, False, True)
        self.viewer.blit(canvas, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            framerate = float(self.metadata["render_fps"])  # type: ignore
            self.clock.tick(framerate)
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.viewer)),
                axes=(1, 0, 2),
            )
        return None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        pos = self._get_initial_obs()

        if self.render_mode == "human":
            self.render()

        return pos, {}

    def close(self) -> None:
        if self.viewer is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()  # pylint: disable=E1101
            self.viewer = None
            self.clock = None
