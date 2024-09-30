# gym-puddle

The grid-world environment with continuous state space and discrete action space described by Degris Thomas, Martha White, and Richard S. Sutton in ["Off-policy actor-critic" arXiv preprint arXiv:1205.4839 (2012)](https://arxiv.org/abs/1205.4839) for Gymnasium.

<kbd>
  <img src='screenshot.png'/>
</kbd>

## Setup

The `gym-puddle` package is managed by [uv](https://docs.astral.sh/uv/getting-started/installation/). To install the package (in edit mode by default) and all its extra dependencies, do:

```shell
uv sync --all-extras
```

This will install the project and its dependencies in a virtual environment under `./.venv`.

### Running the tests

To run the pytest tests, simply do:

```shell
pytest tests/
```

## Usage

Below is a simple example of using a random policy for a maximum of 1000 time-steps.

```python

import gymnasium as gym
import gym_puddle

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
```

**Notes**:

- In the above example, the agent-environment interaction is rendered visually on a canvas (since we've set the `render_mode=human`). To disable it, remove the input argument.
- To truncate the episode beyond a number of time steps (even before having reached the goal state), pass `max_episode_steps` to the input arguments of `make()`. Note that the caller needs to reset the environment immediately after truncation or termination (see example above).
- Rendering is fast, but disabling it will make the code even faster and is highly recommended to do for training agents.

## References
- https://github.com/EhsanEI/gym-puddle

- Off-Policy Actor-Critic. Thomas Degris, Martha White, Richard S. Sutton. In *Proceedings of the Twenty-Ninth International Conference on Machine Learning (ICML)*, 2012.


## Acknowledgments

- The code is based on and forked from [EhsanEI](https://github.com/EhsanEI/gym-puddle)'s implementation.
