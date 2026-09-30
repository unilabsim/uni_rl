"""Linux-friendly scheduling policy for off-policy runtime roles.

The knobs are intentionally advisory: process/container runtimes and platform
limits remain authoritative.  A policy may fail, but it must never change the
RL algorithm or fail an otherwise healthy training run.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

try:
    from _thread import _setthreadlocalicon as _set_thread_cpu_ids  # type: ignore[attr-defined]
except ImportError:  # Python < 3.14 has no thread-local CPU affinity.
    _set_thread_cpu_ids = None


@dataclass(frozen=True)
class RoleSchedulingPolicy:
    """Validated CPU scheduling request for one runtime role."""

    nice: int | None = None
    cpu_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.nice is not None and (type(self.nice) is not int or self.nice < 0):
            raise ValueError(
                f"Role scheduling nice must be a non-negative integer or null, got {self.nice!r}"
            )
        if self.cpu_ids is not None:
            if not self.cpu_ids:
                raise ValueError("Role scheduling cpu_ids must be a non-empty list or null")
            invalid = next(
                (cpu_id for cpu_id in self.cpu_ids if type(cpu_id) is not int or cpu_id < 0),
                None,
            )
            if invalid is not None:
                raise ValueError(
                    "Role scheduling cpu_ids entries must be non-negative integers, "
                    f"got {invalid!r}"
                )
            if len(dict.fromkeys(self.cpu_ids)) != len(self.cpu_ids):
                raise ValueError("Role scheduling cpu_ids entries must be unique")

    def manifest(self) -> dict[str, object]:
        return {
            "nice": self.nice,
            "cpu_ids": list(self.cpu_ids) if self.cpu_ids is not None else None,
        }


@dataclass(frozen=True)
class RoleSchedulingSettings:
    """Effective scheduling requests resolved before collector spawn."""

    learner: RoleSchedulingPolicy = RoleSchedulingPolicy()
    buffer: RoleSchedulingPolicy = RoleSchedulingPolicy()
    collector: RoleSchedulingPolicy = RoleSchedulingPolicy()
    configured_learner_nice: int | None = None
    configured_buffer_nice: int | None = None
    configured_collector_nice: int | None = None

    def __post_init__(self) -> None:
        configured_values = (
            ("learner", self.configured_learner_nice, self.learner.nice),
            ("buffer", self.configured_buffer_nice, self.buffer.nice),
            ("collector", self.configured_collector_nice, self.collector.nice),
        )
        for role, configured, effective in configured_values:
            if configured is None:
                continue
            if configured != effective:
                raise ValueError(
                    f"configured {role} scheduling nice must match its effective value "
                    f"({configured!r} != {effective!r})"
                )

    def manifest(self) -> dict[str, dict[str, object]]:
        return {
            "learner": self.learner.manifest(),
            "buffer": self.buffer.manifest(),
            "collector": self.collector.manifest(),
        }


def _resolve_optional_nice(value: object, *, key: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError(f"training.{key} must be a non-negative integer or null, got {value!r}")
    return value


def _resolve_policy(
    value: object,
    *,
    key: str,
    default_cpu_ids: tuple[int, ...] | None = None,
) -> RoleSchedulingPolicy:
    if value is None:
        return RoleSchedulingPolicy(cpu_ids=default_cpu_ids)
    if not isinstance(value, Mapping):
        raise TypeError(f"training.{key} must be a mapping or null, got {value!r}")
    nice = _resolve_optional_nice(value.get("nice"), key=f"{key}.nice")
    cpu_ids = value.get("cpu_ids", default_cpu_ids)
    if cpu_ids is None:
        return RoleSchedulingPolicy(nice=nice)
    if isinstance(cpu_ids, (str, bytes)) or not isinstance(cpu_ids, (list, tuple, set)):
        raise TypeError(
            f"training.{key}.cpu_ids must be a list of integers or null, got {cpu_ids!r}"
        )
    return RoleSchedulingPolicy(nice=nice, cpu_ids=tuple(cpu_ids))


def _apply_cpu_ids(cpu_ids: tuple[int, ...], *, role: str) -> bool:
    if _set_thread_cpu_ids is not None:
        _set_thread_cpu_ids(cpu_ids)
        return True
    available = os.sched_getaffinity(0)
    unknown = set(cpu_ids) - available
    if unknown:
        raise OSError(
            f"requested unavailable CPUs {sorted(unknown)}; process affinity is {sorted(available)}"
        )
    os.sched_setaffinity(0, cpu_ids)
    return True


def resolve_role_scheduling_settings(cfg: Any) -> RoleSchedulingSettings:
    """Resolve scheduling requests from the single owner config tree."""
    from omegaconf import OmegaConf

    def select(key: str) -> Any:
        value = OmegaConf.select(cfg, f"training.{key}", default=None)
        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
        return value

    learner = _resolve_policy(
        select("learner_scheduling"),
        key="learner_scheduling",
    )
    buffer = _resolve_policy(
        select("buffer_scheduling"),
        key="buffer_scheduling",
    )
    collector = _resolve_policy(
        select("collector_scheduling"),
        key="collector_scheduling",
    )
    return RoleSchedulingSettings(
        learner=learner,
        buffer=buffer,
        collector=collector,
        configured_learner_nice=learner.nice,
        configured_buffer_nice=buffer.nice,
        configured_collector_nice=collector.nice,
    )


def apply_role_scheduling_policy(
    policy: RoleSchedulingPolicy,
    *,
    role: str,
) -> dict[str, object]:
    """Apply one policy to the current process or thread and report evidence."""

    applied: dict[str, Any] = {
        "role": role,
        "nice": policy.nice,
        "cpu_ids": list(policy.cpu_ids) if policy.cpu_ids is not None else None,
        "cpu_affinity_scope": "thread" if _set_thread_cpu_ids is not None else "process",
        "applied": {},
    }
    if policy.nice is not None:
        try:
            os.nice(policy.nice)
            applied["applied"]["nice"] = True
        except (AttributeError, OSError, PermissionError) as exc:
            print(
                f"[offpolicy.scheduling] unable to apply {role} nice={policy.nice}: {exc}",
                file=sys.stderr,
            )
            applied["applied"]["nice"] = False
    if policy.cpu_ids:
        try:
            applied["applied"]["cpu_ids"] = _apply_cpu_ids(policy.cpu_ids, role=role)
        except (AttributeError, OSError, PermissionError) as exc:
            print(
                f"[offpolicy.scheduling] unable to apply {role} cpu_ids="
                f"{list(policy.cpu_ids)}: {exc}",
                file=sys.stderr,
            )
            applied["applied"]["cpu_ids"] = False
    return applied
