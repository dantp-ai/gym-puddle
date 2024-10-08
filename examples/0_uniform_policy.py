import gymnasium as gym

import gym_puddle  # noqa: F401


def main() -> None:
    seed = 43
    env = gym.make("PuddleWorld-v0", render_mode="human", goal=[0.96, 0.96])
    observation, _ = env.reset(seed=seed)

    env.action_space.seed(seed=seed)
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, _ = env.step(action)
        env.render()

        if terminated or truncated:
            env.reset()
            break

    env.close()


if __name__ == "__main__":
    main()
