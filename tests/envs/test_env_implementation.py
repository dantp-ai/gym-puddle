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
    start = [0.4, 0.5]
    start_arr = np.array(start, dtype=np.float32)
    env = gym.make("PuddleWorld-v0", start=start)
    initial_obs, _ = env.reset()

    assert np.array_equal(initial_obs, start_arr)

    env.close()


def test_random_start_position_after_two_consecutive_episodes() -> None:
    env = gym.make("PuddleWorld-v0")
    o1, _ = env.reset(seed=32)
    o12, _ = env.reset()

    env = gym.make("PuddleWorld-v0")
    o2, _ = env.reset(seed=32)
    o22, _ = env.reset()

    env.close()

    assert np.array_equal(o1, o2)
    assert np.array_equal(o12, o22)
