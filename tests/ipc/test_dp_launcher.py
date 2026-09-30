from __future__ import annotations

from typing import Any

import pytest

from uni_rl.ipc import dp_launcher
from uni_rl.ipc.dp_launcher import (
    rank_local_cuda_device,
    resolve_dp_rank_device,
    visible_cuda_entries,
)


class _FakePopen:
    calls: list["_FakePopen"] = []

    def __init__(self, command: list[str], *, env: dict[str, str], start_new_session: bool) -> None:
        self.command = command
        self.env = env
        self.start_new_session = start_new_session
        self.pid = 123456
        self.returncode = 0
        type(self).calls.append(self)

    def poll(self) -> int:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return self.returncode


def test_dp_rank_supervisor_reuses_downstream_entry_script(monkeypatch: Any) -> None:
    """Spawned ranks must re-run the owner application's script.

    ``uni_rl`` deliberately does not ship a ``scripts/`` package; the entry
    point is supplied by the downstream consumer such as UniLab.
    """
    _FakePopen.calls.clear()
    monkeypatch.setattr(dp_launcher.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(dp_launcher.DpRankSupervisor, "_install_signal_handlers", lambda self: None)
    monkeypatch.setattr(dp_launcher.DpRankSupervisor, "_restore_signal_handlers", lambda self: None)
    monkeypatch.setattr(dp_launcher, "_process_group_exists", lambda child: False)
    monkeypatch.setattr(
        dp_launcher.sys,
        "argv",
        ["/workspace/UniLab/src/unilab/scripts/train_sac.py", "task=g1_walk_flat", "--debug"],
    )

    with dp_launcher.DpRankSupervisor((0, 1), "/tmp/run"):
        pass

    assert len(_FakePopen.calls) == 1
    assert _FakePopen.calls[0].command == [
        dp_launcher.sys.executable,
        "/workspace/UniLab/src/unilab/scripts/train_sac.py",
        "task=g1_walk_flat",
        "--debug",
    ]


def test_visible_cuda_entries_accepts_opaque_rank_local_tokens() -> None:
    assert visible_cuda_entries(None) == ()
    assert visible_cuda_entries("") == ()
    assert visible_cuda_entries("-1") == ()
    assert visible_cuda_entries("0") == ("0",)
    assert visible_cuda_entries("GPU-AAAA, MIG-BBBB") == ("GPU-AAAA", "MIG-BBBB")


def test_single_visible_gpu_is_rank_local_cuda_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-uuid-7")

    assert rank_local_cuda_device() == "cuda:0"


def test_rank_device_uses_visibility_without_training_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.delenv(dp_launcher.UNILAB_DP_WORLD_SIZE, raising=False)

    assert resolve_dp_rank_device(None, 0) == "cuda:0"


def test_single_rank_visibility_conflicting_with_training_devices_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.delenv(dp_launcher.UNILAB_DP_WORLD_SIZE, raising=False)

    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES is authoritative"):
        resolve_dp_rank_device((3,), 0)


def test_spawned_rank_uses_local_zero_even_with_parent_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    monkeypatch.setenv(dp_launcher.UNILAB_DP_WORLD_SIZE, "2")

    assert resolve_dp_rank_device((4, 5), 1) == "cuda:0"


def test_supervisor_children_each_receive_one_visible_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakePopen.calls.clear()
    monkeypatch.setattr(dp_launcher.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(dp_launcher.DpRankSupervisor, "_install_signal_handlers", lambda self: None)
    monkeypatch.setattr(dp_launcher.DpRankSupervisor, "_restore_signal_handlers", lambda self: None)
    monkeypatch.setattr(dp_launcher, "_process_group_exists", lambda child: False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b,GPU-c")

    with dp_launcher.DpRankSupervisor((1, 2), "/tmp/run"):
        pass

    assert len(_FakePopen.calls) == 1
    assert _FakePopen.calls[0].env["CUDA_VISIBLE_DEVICES"] == "GPU-b"
    assert _FakePopen.calls[0].env[dp_launcher.UNILAB_DP_DEVICES] == "GPU-b,GPU-c"
