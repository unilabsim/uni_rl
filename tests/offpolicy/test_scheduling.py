"""Contracts for off-policy role scheduling resolution and application."""

from __future__ import annotations

import os
from typing import Any

import pytest
from omegaconf import OmegaConf

from uni_rl.offpolicy.scheduling import (
    RoleSchedulingPolicy,
    RoleSchedulingSettings,
    apply_role_scheduling_policy,
    resolve_role_scheduling_settings,
)


def _cfg(training: dict[str, Any] | None = None) -> Any:
    return OmegaConf.create({"training": training or {}})


def test_role_scheduling_defaults_to_no_policy() -> None:
    settings = resolve_role_scheduling_settings(_cfg())

    assert settings.manifest() == {
        "learner": {"nice": None, "cpu_ids": None},
        "buffer": {"nice": None, "cpu_ids": None},
        "collector": {"nice": None, "cpu_ids": None},
    }


def test_role_scheduling_resolves_owner_requests() -> None:
    settings = resolve_role_scheduling_settings(
        _cfg(
            {
                "learner_scheduling": {"nice": 0},
                "buffer_scheduling": {"nice": 0, "cpu_ids": [0, 1]},
                "collector_scheduling": {"nice": 5},
            }
        )
    )

    assert settings.learner == RoleSchedulingPolicy(nice=0)
    assert settings.buffer == RoleSchedulingPolicy(nice=0, cpu_ids=(0, 1))
    assert settings.collector == RoleSchedulingPolicy(nice=5)
    assert settings.manifest()["buffer"] == {"nice": 0, "cpu_ids": [0, 1]}


@pytest.mark.parametrize("role", ["learner", "buffer", "collector"])
def test_role_scheduling_rejects_negative_nice(role: str) -> None:
    with pytest.raises(ValueError, match=f"training\\.{role}_scheduling\\.nice"):
        resolve_role_scheduling_settings(_cfg({f"{role}_scheduling": {"nice": -1}}))


@pytest.mark.parametrize("value", [True, 1.0, "0"])
def test_role_scheduling_rejects_non_integer_nice(value: Any) -> None:
    with pytest.raises(ValueError, match="training\\.learner_scheduling\\.nice"):
        resolve_role_scheduling_settings(_cfg({"learner_scheduling": {"nice": value}}))


@pytest.mark.parametrize("value", [[], [-1], [0, 0], "0", 2])
def test_role_scheduling_rejects_invalid_cpu_ids(value: Any) -> None:
    error = TypeError if isinstance(value, (str, int)) else ValueError
    with pytest.raises(error, match="cpu_ids"):
        resolve_role_scheduling_settings(_cfg({"buffer_scheduling": {"cpu_ids": value}}))


def test_role_scheduling_rejects_non_mapping_policy() -> None:
    with pytest.raises(TypeError, match="training\\.collector_scheduling"):
        resolve_role_scheduling_settings(_cfg({"collector_scheduling": 5}))


def test_direct_settings_reject_inconsistent_configured_evidence() -> None:
    with pytest.raises(ValueError, match="configured collector scheduling nice"):
        RoleSchedulingSettings(
            collector=RoleSchedulingPolicy(),
            configured_collector_nice=3,
        )


def test_apply_policy_reports_unavailable_cpu_ids_without_failing(capsys) -> None:
    policy = RoleSchedulingPolicy(cpu_ids=(2**31 - 1,))

    evidence = apply_role_scheduling_policy(policy, role="buffer")

    assert evidence["cpu_ids"] == [2**31 - 1]
    assert evidence["applied"] == {"cpu_ids": False}
    assert "unable to apply buffer cpu_ids" in capsys.readouterr().err


def test_apply_collector_nice_only_reports_manifested_request() -> None:
    policy = RoleSchedulingPolicy(nice=0)

    evidence = apply_role_scheduling_policy(policy, role="collector")

    assert evidence == {
        "role": "collector",
        "nice": 0,
        "cpu_ids": None,
        "cpu_affinity_scope": "process",
        "applied": {"nice": True},
    }
    assert os.nice(0) >= 0
