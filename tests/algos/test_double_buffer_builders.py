"""Forwarding tests for the double-buffer runner builder helpers.

The builders are the only public construction path for
``DoubleBufferOffPolicyRunner`` from Hydra owner configs; every runner kwarg
the builders accept must reach the runner unchanged (issue #1481 added
``backend_device_binder`` after UniLab had to set it post-construction).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from omegaconf import OmegaConf


class _FakeEnv:
    obs_groups_spec = {"obs": 4, "critic": 6}
    action_space = SimpleNamespace(shape=(2,))

    def close(self):
        return None


def _fake_env_factory(num_envs, env_cfg_override):
    del num_envs, env_cfg_override
    return _FakeEnv()


class _FakeLearner:
    last_kwargs: dict[str, Any] = {}

    def __init__(self, *args, **kwargs):
        del args
        type(self).last_kwargs = kwargs


class _FakeRunner:
    def __init__(self, *args, **kwargs):
        del args
        self.kwargs = kwargs


def _binder(backend: str) -> str | None:
    del backend
    return None


def _learner_name(algo: str) -> str:
    return "FastSACLearner" if algo == "sac" else "FlashSACLearner"


def _training_cfg() -> dict[str, Any]:
    return {
        "task_name": "FakeTask",
        "sim_backend": "fake",
        "env_steps_per_sync": 1,
        "use_amp": False,
        "trace_enabled": False,
        "trace_output_dir": "logs",
        "trace_thread_time": False,
        "trace_cuda_events": False,
        "log_interval": 3,
    }


def _algo_cfg(extra: dict[str, Any]) -> dict[str, Any]:
    base = {
        "num_envs": 4,
        "replay_buffer_n": 8,
        "batch_size": 4,
        "learning_starts": 4,
        "updates_per_step": 1,
        "policy_frequency": 1,
        "seed": 1,
        "gamma": 0.99,
        "tau": 0.005,
        "actor_lr": 1e-3,
        "critic_lr": 1e-3,
        "actor_hidden_dim": 8,
        "critic_hidden_dim": 8,
        "num_atoms": 1,
        "obs_normalization": False,
    }
    base.update(extra)
    return base


def _sac_cfg() -> Any:
    return OmegaConf.create(
        {
            "training": _training_cfg(),
            "algo": _algo_cfg(
                {
                    "use_layer_norm": False,
                    "algo_params": {
                        "alpha_lr": 1e-3,
                        "alpha_init": 1.0,
                        "target_entropy_ratio": 1.0,
                        "max_grad_norm": 1.0,
                        "amp_dtype": "bf16",
                        "use_compile": False,
                    },
                }
            ),
        }
    )


def _flashsac_cfg() -> Any:
    return OmegaConf.create(
        {
            "training": _training_cfg(),
            "algo": _algo_cfg(
                {
                    "algo_params": {
                        "actor_num_blocks": 1,
                        "critic_num_blocks": 1,
                        "critic_min_v": -10.0,
                        "critic_max_v": 10.0,
                        "temp_initial_value": 1.0,
                        "temp_target_sigma": 1.0,
                        "temp_target_entropy": 1.0,
                        "actor_bc_alpha": 1.0,
                        "actor_noise_zeta_mu": 0.0,
                        "actor_noise_zeta_max": 0.0,
                        "learning_rate_init": 1e-3,
                        "learning_rate_peak": 1e-3,
                        "learning_rate_end": 1e-3,
                        "learning_rate_warmup_steps": 1,
                        "learning_rate_decay_steps": 1,
                        "normalize_reward": False,
                        "normalized_g_max": 1.0,
                        "n_step": 1,
                        "amp_dtype": "bf16",
                        "use_compile": False,
                        "compile_full_objectives": True,
                    },
                }
            ),
        }
    )


@pytest.mark.parametrize("with_binder", [False, True])
def test_sac_builder_forwards_backend_device_binder(
    monkeypatch: pytest.MonkeyPatch, with_binder: bool
) -> None:
    import uni_rl.algos.fast_sac.double_buffer as module

    monkeypatch.setattr(module, "FastSACLearner", _FakeLearner)
    monkeypatch.setattr(module, "DoubleBufferOffPolicyRunner", _FakeRunner)

    kwargs: dict[str, Any] = {}
    if with_binder:
        kwargs["backend_device_binder"] = _binder
    runner = module.build_sac_double_buffer_runner(
        _sac_cfg(),
        env_factory=_fake_env_factory,
        env_cfg_override=None,
        replay_prefetch_mode="one_tick",
        device="cpu",
        **kwargs,
    )

    assert runner.kwargs["backend_device_binder"] is (_binder if with_binder else None)
    assert runner.kwargs["log_interval"] == 3
    assert runner.kwargs["inference_placement"].collector_tensor_native is False
    assert runner.kwargs["inference_placement"].mode.value == "cpu"
    settings = runner.kwargs["tensor_runtime_settings"]
    assert settings.inference_slot_capacity == 1
    assert settings.collector_metrics_interval == 1
    assert settings.replay_ingress_depth == 2
    assert settings.replay_ingress_slot_rows == 4


@pytest.mark.parametrize("with_binder", [False, True])
def test_flashsac_builder_forwards_backend_device_binder(
    monkeypatch: pytest.MonkeyPatch, with_binder: bool
) -> None:
    import uni_rl.algos.flash_sac.double_buffer as module

    monkeypatch.setattr(module, "FlashSACLearner", _FakeLearner)
    monkeypatch.setattr(module, "DoubleBufferOffPolicyRunner", _FakeRunner)
    # CPU-only test host: bypass the CUDA/MPS replay-device gate and seeding.
    monkeypatch.setattr(
        module,
        "require_offpolicy_replay_device",
        lambda device: device,
        raising=False,
    )
    monkeypatch.setattr(module, "apply_training_seed", lambda *args, **kwargs: None)

    kwargs: dict[str, Any] = {}
    if with_binder:
        kwargs["backend_device_binder"] = _binder
    runner = module.build_flashsac_double_buffer_runner(
        _flashsac_cfg(),
        env_factory=_fake_env_factory,
        env_cfg_override=None,
        replay_prefetch_mode="one_tick",
        device="cpu",
        **kwargs,
    )

    assert runner.kwargs["backend_device_binder"] is (_binder if with_binder else None)
    assert runner.kwargs["log_interval"] == 3
    assert runner.kwargs["inference_placement"].collector_tensor_native is False
    assert runner.kwargs["inference_placement"].mode.value == "cpu"
    settings = runner.kwargs["tensor_runtime_settings"]
    assert settings.inference_slot_capacity == 1
    assert settings.collector_metrics_interval == 1
    assert settings.replay_ingress_depth == 2
    assert settings.replay_ingress_slot_rows == 4
    assert _FakeLearner.last_kwargs["compile_full_objectives"] is True


@pytest.mark.parametrize("algo", ["sac", "flashsac"])
def test_double_buffer_builders_forward_tensor_runtime_settings(
    monkeypatch: pytest.MonkeyPatch, algo: str
) -> None:
    if algo == "sac":
        import uni_rl.algos.fast_sac.double_buffer as module

        cfg = _sac_cfg()
        learner_name = "FastSACLearner"
    else:
        import uni_rl.algos.flash_sac.double_buffer as module

        cfg = _flashsac_cfg()
        learner_name = "FlashSACLearner"

    monkeypatch.setattr(module, learner_name, _FakeLearner)
    monkeypatch.setattr(module, "DoubleBufferOffPolicyRunner", _FakeRunner)
    if algo == "flashsac":
        monkeypatch.setattr(module, "require_offpolicy_replay_device", lambda device: device)
    if algo == "flashsac":
        monkeypatch.setattr(module, "apply_training_seed", lambda *args, **kwargs: None)
    cfg.env = {"tensor_runtime": False}
    cfg.training.inference_slot_capacity = 3
    cfg.training.collector_metrics_interval = 7
    cfg.training.replay_ingress_depth = 4
    cfg.training.replay_ingress_slot_rows = 2

    if algo == "sac":
        runner = module.build_sac_double_buffer_runner(
            cfg,
            env_factory=_fake_env_factory,
            env_cfg_override=None,
            replay_prefetch_mode="one_tick",
            device="cpu",
        )
    else:
        runner = module.build_flashsac_double_buffer_runner(
            cfg,
            env_factory=_fake_env_factory,
            env_cfg_override=None,
            replay_prefetch_mode="one_tick",
            device="cpu",
        )

    settings = runner.kwargs["tensor_runtime_settings"]
    assert settings.inference_slot_capacity == 3
    assert settings.collector_metrics_interval == 7
    assert settings.replay_ingress_depth == 4
    assert settings.replay_ingress_slot_rows == 2
    assert settings.batch_size == 4
    assert settings.updates_per_step == 1
    assert settings.learner_sample_count == 4


@pytest.mark.parametrize("algo", ["sac", "flashsac"])
def test_double_buffer_builders_reject_runtime_bounds_before_env_probe(
    monkeypatch: pytest.MonkeyPatch, algo: str
) -> None:
    if algo == "sac":
        import uni_rl.algos.fast_sac.double_buffer as module

        cfg = _sac_cfg()
        build = module.build_sac_double_buffer_runner
    else:
        import uni_rl.algos.flash_sac.double_buffer as module

        cfg = _flashsac_cfg()
        build = module.build_flashsac_double_buffer_runner
        monkeypatch.setattr(module, "require_offpolicy_replay_device", lambda device: device)
        monkeypatch.setattr(module, "apply_training_seed", lambda *args, **kwargs: None)
    cfg.env = {"tensor_runtime": False}
    cfg.training.replay_ingress_depth = 17

    def _fail_factory(*args, **kwargs):
        raise AssertionError("invalid bounds must fail before env probing")

    monkeypatch.setattr(module, _learner_name(algo), _FakeLearner)
    with pytest.raises(ValueError, match="replay_ingress_depth.*17"):
        build(
            cfg,
            env_factory=_fail_factory,
            env_cfg_override=None,
            replay_prefetch_mode="one_tick",
            device="cpu",
        )


def test_flashsac_builder_resolves_tensor_runtime_before_collector_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uni_rl.algos.flash_sac.double_buffer as module

    monkeypatch.setattr(module, "FlashSACLearner", _FakeLearner)
    monkeypatch.setattr(module, "DoubleBufferOffPolicyRunner", _FakeRunner)
    monkeypatch.setattr(module, "require_offpolicy_replay_device", lambda device: device)
    monkeypatch.setattr(module, "apply_training_seed", lambda *args, **kwargs: None)

    cfg = _flashsac_cfg()
    cfg.env = {"tensor_runtime": False}
    runner = module.build_flashsac_double_buffer_runner(
        cfg,
        env_factory=_fake_env_factory,
        env_cfg_override=None,
        replay_prefetch_mode="one_tick",
        device="cpu",
    )
    assert runner.kwargs["inference_placement"].collector_tensor_native is False
    assert runner.kwargs["inference_placement"].mode.value == "cpu"


def test_sac_builder_resolves_tensor_runtime_before_collector_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uni_rl.algos.fast_sac.double_buffer as module

    monkeypatch.setattr(module, "FastSACLearner", _FakeLearner)
    monkeypatch.setattr(module, "DoubleBufferOffPolicyRunner", _FakeRunner)
    cfg = _sac_cfg()
    cfg.env = {"tensor_runtime": False}

    runner = module.build_sac_double_buffer_runner(
        cfg,
        env_factory=_fake_env_factory,
        env_cfg_override=None,
        replay_prefetch_mode="one_tick",
        device="cpu",
    )

    assert runner.kwargs["inference_placement"].collector_tensor_native is False
    assert runner.kwargs["inference_placement"].mode.value == "cpu"


def test_sac_builder_rejects_tensor_runtime_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uni_rl.algos.fast_sac.double_buffer as module

    monkeypatch.setattr(module, "FastSACLearner", _FakeLearner)
    monkeypatch.setattr(module, "DoubleBufferOffPolicyRunner", _FakeRunner)
    cfg = _sac_cfg()
    cfg.env = {"tensor_runtime": True}

    with pytest.raises(
        ValueError, match="FastSAC CUDA inference transport requires CUDA env and learner devices"
    ):
        module.build_sac_double_buffer_runner(
            cfg,
            env_factory=_fake_env_factory,
            env_cfg_override=None,
            replay_prefetch_mode="one_tick",
            device="cpu",
        )


def test_flashsac_builder_rejects_tensor_runtime_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uni_rl.algos.flash_sac.double_buffer as module

    monkeypatch.setattr(module, "require_offpolicy_replay_device", lambda device: device)
    cfg = _flashsac_cfg()
    cfg.env = {"tensor_runtime": True}

    with pytest.raises(
        ValueError, match="CUDA inference transport requires CUDA env and learner devices"
    ):
        module.build_flashsac_double_buffer_runner(
            cfg,
            env_factory=_fake_env_factory,
            env_cfg_override=None,
            replay_prefetch_mode="one_tick",
            device="cpu",
        )


def test_sac_builder_forwards_custom_runtime_preparation_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uni_rl.algos.fast_sac.double_buffer as module
    from uni_rl.offpolicy.runtime import OffPolicyRuntime

    def prepare_hook(learner, context):
        del learner, context

    runtime = OffPolicyRuntime(learner_cls=_FakeLearner, learner_prepare_hook=prepare_hook)
    monkeypatch.setattr(module, "FastSACLearner", _FakeLearner)
    monkeypatch.setattr(module, "DoubleBufferOffPolicyRunner", _FakeRunner)
    monkeypatch.setattr(module, "resolve_custom_offpolicy_runtime", lambda rl_cfg: runtime)

    with pytest.warns(DeprecationWarning, match="inference_request_timeout_sec is deprecated"):
        cfg = _sac_cfg()
        cfg.training.inference_request_timeout_sec = 17.0
        runner = module.build_sac_double_buffer_runner(
            cfg,
            env_factory=_fake_env_factory,
            env_cfg_override=None,
            replay_prefetch_mode="one_tick",
            device="cpu",
        )

    assert runner.kwargs["learner_prepare_hook"] is prepare_hook
    assert "inference_request_timeout_sec" not in runner.kwargs
