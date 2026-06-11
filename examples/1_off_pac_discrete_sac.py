"""Train and evaluate a Discrete-SAC off-policy actor-critic on PuddleWorld.

Context:
    The PuddleWorld environment in this repo is the continuous gridworld used
    by Degris, White, and Sutton in "Off-policy actor-critic" (2012,
    arXiv:1205.4839) as one of three benchmarks. The original Off-PAC
    algorithm uses linear features (tile coding) + GTD(lambda) critic and is
    a fundamentally different algorithm from modern deep off-policy AC. This
    example does *not* reproduce the 2012 paper; instead it trains the
    canonical modern off-policy actor-critic for discrete action spaces,
    Discrete SAC, on the same environment. See issue #35 for the rationale
    and the out-of-scope items that may follow in later PRs.

What the script does:
    1. Build PuddleWorld via torchRL's GymEnv wrapper.
    2. Build a small MLP actor (categorical logits) and Q-value network.
    3. Wire torchRL's DiscreteSACLoss + SoftUpdate.
    4. Collect frames into a replay buffer with torchRL's Collector, train
       for a fixed budget of environment frames.
    5. Evaluate the trained deterministic (argmax) policy against a uniform
       random baseline and print a comparison table.
    6. Assert a performance bar (Discrete SAC must clearly beat random) so
       regressions fail loudly.

Run:
    source .venv/bin/activate
    uv sync --extra rl-examples
    python examples/1_off_pac_discrete_sac.py
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from tensordict.nn import TensorDictModule
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.envs import EnvBase, TransformedEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import RewardScaling
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP, OneHotCategorical, ProbabilisticActor
from torchrl.objectives import DiscreteSACLoss, SoftUpdate

import gym_puddle  # noqa: F401  # registers PuddleWorld-v0

# --- Environment ----------------------------------------------------------
ENV_ID = "PuddleWorld-v0"
OBS_DIM = 2
NUM_ACTIONS = 5
MAX_EPISODE_STEPS = 500

# --- Network architecture -------------------------------------------------
HIDDEN_DIM = 64

# --- Training budget ------------------------------------------------------
TOTAL_FRAMES = 100_000
FRAMES_PER_BATCH = 200
INIT_RANDOM_FRAMES = 1_000
BATCH_SIZE = 256
BUFFER_SIZE = 100_000
UPDATES_PER_BATCH = 100

# --- Discrete SAC ---------------------------------------------------------
LR = 1e-4
GAMMA = 0.99
TAU = 0.005
# PuddleWorld's per-step rewards live in [~-10, -1]. Scaling rewards down on
# the *training* env keeps Q-values in a numerically sane range (Q-loss
# otherwise grows quadratically with episode return magnitude and destabilizes
# learning). Eval is run on the unscaled env so reported metrics are on the
# environment's native reward scale.
REWARD_SCALE = 0.1

# --- Evaluation -----------------------------------------------------------
EVAL_EPISODES = 50

# --- Reproducibility ------------------------------------------------------
SEED = 43

# Progress is printed every PRINT_EVERY_BATCHES collector iterations.
PRINT_EVERY_BATCHES = 10


@dataclass
class EvalMetrics:
    policy_name: str
    mean_return: float
    mean_length: float
    success_rate: float


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_env(seed: int) -> EnvBase:
    env = GymEnv(ENV_ID, max_episode_steps=MAX_EPISODE_STEPS)
    env.set_seed(seed)
    return env


def build_train_env(seed: int) -> EnvBase:
    """Training env: same as eval but with reward scaled by REWARD_SCALE."""
    return TransformedEnv(
        build_env(seed),
        RewardScaling(loc=0.0, scale=REWARD_SCALE),
    )


def build_actor(env: EnvBase) -> ProbabilisticActor:
    net = MLP(
        in_features=OBS_DIM,
        out_features=NUM_ACTIONS,
        num_cells=[HIDDEN_DIM, HIDDEN_DIM],
    )
    module = TensorDictModule(net, in_keys=["observation"], out_keys=["logits"])
    return ProbabilisticActor(
        module=module,
        in_keys=["logits"],
        out_keys=["action"],
        spec=env.action_spec,
        distribution_class=OneHotCategorical,
        return_log_prob=True,
    )


def build_qvalue(env: EnvBase) -> TensorDictModule:
    del env  # kept in the signature for symmetry with build_actor / build_loss
    net = MLP(
        in_features=OBS_DIM,
        out_features=NUM_ACTIONS,
        num_cells=[HIDDEN_DIM, HIDDEN_DIM],
    )
    return TensorDictModule(net, in_keys=["observation"], out_keys=["action_value"])


def build_loss(
    env: EnvBase,
    actor: ProbabilisticActor,
    qvalue: TensorDictModule,
) -> tuple[DiscreteSACLoss, SoftUpdate]:
    loss_module = DiscreteSACLoss(
        actor_network=actor,
        qvalue_network=qvalue,
        action_space=env.action_spec,
        num_actions=NUM_ACTIONS,
        num_qvalue_nets=2,
        loss_function="l2",
        delay_qvalue=True,
    )
    loss_module.make_value_estimator(gamma=GAMMA)
    target_updater = SoftUpdate(loss_module, tau=TAU)
    return loss_module, target_updater


def _rollout_metrics(
    env: EnvBase, policy: ProbabilisticActor | None
) -> tuple[float, int, int]:
    """Run one episode, return (return, length, terminated_flag)."""
    td = env.rollout(MAX_EPISODE_STEPS, policy=policy)
    rewards = float(td["next", "reward"].sum().item())
    length = int(td.shape[0])
    terminated = int(td["next", "terminated"][-1].item())
    return rewards, length, terminated


def evaluate(
    env: EnvBase,
    actor: ProbabilisticActor,
    n_episodes: int,
    policy_name: str,
) -> EvalMetrics:
    returns: list[float] = []
    lengths: list[int] = []
    successes: list[int] = []
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        for _ in range(n_episodes):
            r, length, term = _rollout_metrics(env, actor)
            returns.append(r)
            lengths.append(length)
            successes.append(term)
    return EvalMetrics(
        policy_name=policy_name,
        mean_return=float(np.mean(returns)),
        mean_length=float(np.mean(lengths)),
        success_rate=float(np.mean(successes)),
    )


def random_baseline(env: EnvBase, n_episodes: int) -> EvalMetrics:
    returns: list[float] = []
    lengths: list[int] = []
    successes: list[int] = []
    for _ in range(n_episodes):
        # `env.rollout` with policy=None samples actions uniformly from the action spec.
        r, length, term = _rollout_metrics(env, None)
        returns.append(r)
        lengths.append(length)
        successes.append(term)
    return EvalMetrics(
        policy_name="random",
        mean_return=float(np.mean(returns)),
        mean_length=float(np.mean(lengths)),
        success_rate=float(np.mean(successes)),
    )


def train(
    actor: ProbabilisticActor,
    loss_module: DiscreteSACLoss,
    target_updater: SoftUpdate,
) -> ProbabilisticActor:
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=LR)

    collector = Collector(
        create_env_fn=lambda: build_train_env(SEED),
        policy=actor,
        frames_per_batch=FRAMES_PER_BATCH,
        total_frames=TOTAL_FRAMES,
        init_random_frames=INIT_RANDOM_FRAMES,
        auto_register_policy_transforms=True,
    )
    buffer = TensorDictReplayBuffer(storage=LazyTensorStorage(BUFFER_SIZE))

    frames_collected = 0
    batch_index = 0
    last_losses: dict[str, float] = {}
    start = time.monotonic()
    for data in collector:
        batch_index += 1
        buffer.extend(data.reshape(-1))
        frames_collected += data.numel()

        if len(buffer) < BATCH_SIZE:
            continue

        for _ in range(UPDATES_PER_BATCH):
            sample = buffer.sample(BATCH_SIZE)
            loss_td = loss_module(sample)
            total_loss = (
                loss_td["loss_actor"] + loss_td["loss_qvalue"] + loss_td["loss_alpha"]
            )
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            target_updater.step()

        last_losses = {
            "actor": float(loss_td["loss_actor"].detach()),
            "qvalue": float(loss_td["loss_qvalue"].detach()),
            "alpha": float(loss_td["alpha"].detach()),
            "entropy": float(loss_td["entropy"].detach()),
        }

        if batch_index % PRINT_EVERY_BATCHES == 0:
            elapsed = time.monotonic() - start
            print(
                f"  frames={frames_collected:>6}/{TOTAL_FRAMES}"
                f"  loss_actor={last_losses['actor']:>+7.3f}"
                f"  loss_q={last_losses['qvalue']:>7.3f}"
                f"  alpha={last_losses['alpha']:>5.3f}"
                f"  H={last_losses['entropy']:>5.3f}"
                f"  elapsed={elapsed:>5.1f}s",
            )

    collector.shutdown()
    return actor


def _print_metrics_table(metrics: list[EvalMetrics]) -> None:
    print()
    print(
        "| policy         | mean_return | mean_length | success_rate |",
    )
    print(
        "| -------------- | ----------- | ----------- | ------------ |",
    )
    for m in metrics:
        print(
            f"| {m.policy_name:<14} |"
            f" {m.mean_return:>+11.2f} |"
            f" {m.mean_length:>11.2f} |"
            f" {m.success_rate:>11.2%} |",
        )
    print()


def main() -> None:
    _set_global_seed(SEED)

    train_env = build_env(SEED)
    eval_env = build_env(SEED + 1)

    actor = build_actor(train_env)
    qvalue = build_qvalue(train_env)
    loss_module, target_updater = build_loss(train_env, actor, qvalue)

    print(f"[1/3] Evaluating random baseline ({EVAL_EPISODES} episodes)...")
    random_metrics = random_baseline(eval_env, EVAL_EPISODES)

    print(f"[2/3] Training Discrete SAC for {TOTAL_FRAMES} frames...")
    actor = train(actor, loss_module, target_updater)

    print(f"[3/3] Evaluating trained Discrete SAC ({EVAL_EPISODES} episodes)...")
    sac_metrics = evaluate(eval_env, actor, EVAL_EPISODES, "discrete-sac")

    train_env.close()
    eval_env.close()

    _print_metrics_table([random_metrics, sac_metrics])

    bar = random_metrics.mean_return + 0.5 * abs(random_metrics.mean_return)
    if sac_metrics.mean_return <= bar:
        raise AssertionError(
            f"Discrete SAC mean_return ({sac_metrics.mean_return:.2f}) failed "
            f"to beat the random baseline bar ({bar:.2f}, i.e. 50% improvement "
            f"over |random|={abs(random_metrics.mean_return):.2f}).",
        )
    print("Performance bar PASSED:")
    print(
        f"  discrete-sac mean_return = {sac_metrics.mean_return:.2f}"
        f" > {bar:.2f} = random + 0.5 * |random|",
    )


if __name__ == "__main__":
    main()
