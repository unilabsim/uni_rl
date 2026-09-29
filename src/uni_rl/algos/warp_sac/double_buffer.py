"""WarpSAC builder for the device-authoritative off-policy runtime."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig

from uni_rl.algos.warp_sac.learner import WarpSACLearner
from uni_rl.algos.warp_sac.replay import WarpSACReplayPipeline
from uni_rl.env_contract import EnvFactory
from uni_rl.ipc.replay_pipelines.gpu_resident import require_offpolicy_replay_device
from uni_rl.offpolicy.double_buffer_runner import DoubleBufferOffPolicyRunner
from uni_rl.utils.device import get_default_device
from uni_rl.utils.nan_guard import NanGuardCfg
from uni_rl.utils.observations import get_obs_dims
from uni_rl.utils.seed import apply_training_seed
from uni_rl.utils.tensor_runtime import (
    InferencePlacement,
    resolve_inference_transport,
    resolve_tensor_runtime_settings,
)

if TYPE_CHECKING:
    from uni_rl.ipc.dp_sync import DpParameterSync


def _param(algo_cfg: DictConfig, name: str, default: Any) -> Any:
    params = algo_cfg.get("algo_params", {})
    if name in params:
        return params[name]
    return algo_cfg.get(name, default)


def _validate_warpsac_runtime(cfg: DictConfig) -> None:
    if int(_param(cfg.algo, "n_step", 1)) != 1:
        raise ValueError("WarpSAC device replay requires n_step=1")


def build_warpsac_double_buffer_runner(
    cfg: DictConfig,
    *,
    env_factory: EnvFactory,
    env_cfg_override: dict[str, Any] | None,
    replay_prefetch_mode: str,
    device: str | None = None,
    nan_guard_cfg: NanGuardCfg | None = None,
    torch_thread_runtime: dict[str, Any] | None = None,
    collector_cpu_ids: list[int] | None = None,
    dp_sync: DpParameterSync | None = None,
    backend_device_binder: Callable[[str], str | None] | None = None,
) -> Any:
    """Build WarpSAC with bounded ingress and regime-aware device replay."""
    device = require_offpolicy_replay_device(device or get_default_device())
    apply_training_seed(cfg.algo.seed, torch_runtime=True, cuda=True)
    if replay_prefetch_mode != "one_tick":
        raise ValueError("WarpSAC device replay requires replay_prefetch_mode='one_tick'")
    _validate_warpsac_runtime(cfg)
    inference_placement = resolve_inference_transport(cfg, device=device, algo_name="WarpSAC")
    tensor_runtime_settings = resolve_tensor_runtime_settings(
        cfg,
        algo_name="WarpSAC",
        num_envs=cfg.algo.num_envs,
    )

    if "inference_request_timeout_sec" in cfg.training:
        warnings.warn(
            "training.inference_request_timeout_sec is deprecated and ignored; "
            "remove it from owner YAML.",
            DeprecationWarning,
            stacklevel=2,
        )
    env = env_factory(1, env_cfg_override)
    try:
        obs_dim, critic_obs_dim = get_obs_dims(dict(env.obs_groups_spec))
        action_shape = env.action_space.shape
        assert action_shape is not None
        action_dim = int(action_shape[0])
    finally:
        env.close()

    learner = WarpSACLearner(
        device=device,
        obs_dim=obs_dim,
        action_dim=action_dim,
        critic_obs_dim=critic_obs_dim,
        gamma=cfg.algo.gamma,
        tau=cfg.algo.tau,
        actor_lr=cfg.algo.actor_lr,
        critic_lr=cfg.algo.critic_lr,
        actor_hidden_dim=cfg.algo.actor_hidden_dim,
        critic_hidden_dim=cfg.algo.critic_hidden_dim,
        actor_num_blocks=cfg.algo.algo_params.actor_num_blocks,
        critic_num_blocks=cfg.algo.algo_params.critic_num_blocks,
        num_atoms=cfg.algo.num_atoms,
        critic_min_v=cfg.algo.algo_params.critic_min_v,
        critic_max_v=cfg.algo.algo_params.critic_max_v,
        temp_initial_value=cfg.algo.algo_params.temp_initial_value,
        temp_target_sigma=cfg.algo.algo_params.temp_target_sigma,
        temp_target_entropy=cfg.algo.algo_params.temp_target_entropy,
        actor_bc_alpha=cfg.algo.algo_params.actor_bc_alpha,
        actor_noise_zeta_mu=cfg.algo.algo_params.actor_noise_zeta_mu,
        actor_noise_zeta_max=cfg.algo.algo_params.actor_noise_zeta_max,
        actor_normalize_parameters=bool(_param(cfg.algo, "actor_normalize_parameters", True)),
        critic_normalize_parameters=bool(_param(cfg.algo, "critic_normalize_parameters", True)),
        learning_rate_init=cfg.algo.algo_params.learning_rate_init,
        learning_rate_peak=cfg.algo.algo_params.learning_rate_peak,
        learning_rate_end=cfg.algo.algo_params.learning_rate_end,
        learning_rate_warmup_steps=cfg.algo.algo_params.learning_rate_warmup_steps,
        learning_rate_decay_steps=cfg.algo.algo_params.learning_rate_decay_steps,
        normalize_reward=cfg.algo.algo_params.normalize_reward,
        normalized_g_max=cfg.algo.algo_params.normalized_g_max,
        n_step=cfg.algo.algo_params.n_step,
        obs_normalization=cfg.algo.obs_normalization,
        use_amp=cfg.training.use_amp,
        amp_dtype=cfg.algo.algo_params.amp_dtype,
        use_compile=cfg.algo.algo_params.use_compile,
        compile_full_objectives=bool(_param(cfg.algo, "compile_full_objectives", False)),
    )

    return DoubleBufferOffPolicyRunner(
        learner=learner,
        env_name=cfg.training.task_name,
        algo_type="warpsac",
        env_factory=env_factory,
        num_envs=cfg.algo.num_envs,
        replay_buffer_n=cfg.algo.replay_buffer_n,
        batch_size=tensor_runtime_settings.batch_size,
        learning_starts=cfg.algo.learning_starts,
        updates_per_step=tensor_runtime_settings.updates_per_step,
        policy_frequency=cfg.algo.policy_frequency,
        target_frequency=int(_param(cfg.algo, "target_frequency", 1)),
        policy_before_critic=True,
        env_steps_per_sync=cfg.training.env_steps_per_sync,
        device=device,
        obs_normalization=cfg.algo.obs_normalization,
        sim_backend=cfg.training.sim_backend,
        env_cfg_override=env_cfg_override,
        seed=cfg.algo.seed,
        trace_enabled=cfg.training.trace_enabled,
        trace_output_dir=cfg.training.trace_output_dir,
        trace_thread_time=cfg.training.trace_thread_time,
        trace_cuda_events=cfg.training.trace_cuda_events,
        replay_prefetch_mode=replay_prefetch_mode,
        nan_guard_cfg=nan_guard_cfg,
        torch_thread_runtime=torch_thread_runtime,
        collector_cpu_ids=collector_cpu_ids,
        dp_sync=dp_sync,
        backend_device_binder=backend_device_binder,
        inference_placement=inference_placement,
        tensor_runtime_settings=tensor_runtime_settings,
        log_interval=int(cfg.training.log_interval),
        replay_pipeline_factory=partial(
            WarpSACReplayPipeline,
            decay_step=int(_param(cfg.algo, "decay_step", 0)),
            min_weight=float(
                _param(cfg.algo, "replay_min_weight", _param(cfg.algo, "min_weight", 0.1))
            ),
            num_buckets=int(_param(cfg.algo, "replay_num_buckets", 2000)),
        ),
    )


__all__ = ["build_warpsac_double_buffer_runner"]
