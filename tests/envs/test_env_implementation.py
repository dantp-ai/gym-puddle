import gymnasium as gym
import numpy as np
import pytest
from gymnasium.wrappers.time_limit import TimeLimit

import gym_puddle  # noqa: F401
from gym_puddle.envs import PuddleEnv


def test_constant_minus_one_reward_no_puddles() -> None:
    puddle_world: gym.Env = PuddleEnv(puddles=[])
    puddle_world = TimeLimit(puddle_world, max_episode_steps=5000)

    puddle_world.reset()
    while True:
        action = puddle_world.action_space.sample()
        observation, reward, terminated, truncated, _ = puddle_world.step(action)

        assert reward == -1.0

        if terminated or truncated:
            puddle_world.reset()
            break

    puddle_world.close()


def test_invalid_action() -> None:
    env = gym.make("PuddleWorld-v0")
    env.reset()

    action = np.array(5, dtype=np.int64)

    with pytest.raises(ValueError):
        env.step(action)
        env.close()


def test_initial_position_after_reset() -> None:
    start = np.array([0.4, 0.5], dtype=np.float32)
    env = gym.make("PuddleWorld-v0", start=start)
    initial_obs, _ = env.reset()

    assert np.array_equal(initial_obs, start)

    env.close()
