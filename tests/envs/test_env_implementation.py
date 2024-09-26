import gymnasium
from gymnasium.wrappers.time_limit import TimeLimit

from gym_puddle.envs import PuddleEnv


def test_constant_minus_one_reward_no_puddles() -> None:
    puddle_world: gymnasium.Env = PuddleEnv(puddles=[])
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
