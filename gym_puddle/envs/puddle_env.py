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
    metadata: dict[str, Any] = {
        "render_modes": [
            "human",
            "rgb_array",
        ],
        "render_fps": 50.0,
    }

    def __init__(
        self,
        start: np.ndarray | None = None,
        goal: np.ndarray | None = None,
        goal_threshold: float = 0.1,
        noise: float = 0.025,
        thrust: float = 0.05,
        puddles: list[Puddle] | None = None,
        render_mode: str | None = None,
    ):
        self.start = (
            start if start is not None else np.array([0.2, 0.4], dtype=np.float32)
        )
        self.goal = goal if goal is not None else np.array([1.0, 1.0], dtype=np.float32)
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

        self.actions = [np.zeros(2, dtype=np.float32) for _ in range(5)]
        # [(-0.05, 0), (0.05, 0), (0, -0.05), (0, 0.05), (0, 0)]
        for i in range(4):
            self.actions[i][i // 2] = thrust * (i % 2 * 2 - 1)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
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

        self.pos = self._get_initial_obs()

        if self.render_mode == "human":
            self.render()

        return self.pos, {}

    def close(self) -> None:
        if self.viewer is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()  # pylint: disable=E1101
            self.viewer = None
            self.clock = None
