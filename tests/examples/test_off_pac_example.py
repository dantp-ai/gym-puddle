"""Smoke tests for the Discrete-SAC Off-PAC example.

These verify that the example's component builders wire together correctly
without running the full training loop. The training loop itself is too slow
and too seed-sensitive to test in CI; we only assert structural correctness
here.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

# Skip the entire module when the rl-examples extra is not installed.
torchrl = pytest.importorskip("torchrl")
tensordict = pytest.importorskip("tensordict")
torch = pytest.importorskip("torch")

_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[2] / "examples" / "1_off_pac_discrete_sac.py"
)
_EXAMPLE_MODULE_NAME = "off_pac_discrete_sac_example"


def _load_example() -> ModuleType:
    """Load the numerically-prefixed example file as a module by path.

    Registers in sys.modules so @dataclass decorators inside the example
    can look up their own module during class creation.
    """
    spec = importlib.util.spec_from_file_location(_EXAMPLE_MODULE_NAME, _EXAMPLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[_EXAMPLE_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


example = _load_example()


def test_module_imports_without_running_main() -> None:
    """Importing the example module must not trigger training."""
    assert hasattr(example, "main")
    assert callable(example.main)


def test_build_env_returns_torchrl_env_with_expected_specs() -> None:
    env = example.build_env(seed=example.SEED)
    try:
        assert env.observation_spec["observation"].shape[-1] == example.OBS_DIM
        assert env.action_spec.space.n == example.NUM_ACTIONS
        td = env.reset()
        assert "observation" in td
        assert td["observation"].shape == (example.OBS_DIM,)
    finally:
        env.close()


def test_build_actor_emits_action_with_correct_spec() -> None:
    env = example.build_env(seed=example.SEED)
    try:
        actor = example.build_actor(env)
        td = env.reset()
        td = actor(td)
        assert "logits" in td
        assert td["logits"].shape == (example.NUM_ACTIONS,)
        assert "action" in td
        # OneHot action: shape [NUM_ACTIONS], exactly one element equal to 1.
        action = td["action"]
        assert action.shape == (example.NUM_ACTIONS,)
        assert int(action.sum().item()) == 1
    finally:
        env.close()


def test_build_qvalue_emits_action_value_vector() -> None:
    env = example.build_env(seed=example.SEED)
    try:
        qvalue = example.build_qvalue(env)
        td = env.reset()
        td = qvalue(td)
        assert "action_value" in td
        assert td["action_value"].shape == (example.NUM_ACTIONS,)
    finally:
        env.close()


def test_build_loss_accepts_a_real_rollout_and_returns_scalar_losses() -> None:
    env = example.build_env(seed=example.SEED)
    try:
        actor = example.build_actor(env)
        qvalue = example.build_qvalue(env)
        loss_module, target_updater = example.build_loss(env, actor, qvalue)

        td = env.rollout(8, policy=actor)
        out = loss_module(td)

        for key in ("loss_actor", "loss_qvalue", "loss_alpha", "alpha", "entropy"):
            assert key in out, f"missing {key} in loss output"
            assert out[key].shape == torch.Size(
                []
            ), f"expected scalar for {key}, got {out[key].shape}"

        # SoftUpdate.step() must not raise on a fresh loss module.
        target_updater.step()
    finally:
        env.close()
