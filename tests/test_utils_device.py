"""The §5 environment probe: a CPU fallback on GPU-intended work must STOP.

These run on any box, with or without CUDA — `probe_compute_device` is fed a
`DeviceSpec` through a patched `resolve_device`, so the test asserts the *policy*
and never needs a GPU. That is the whole point: the check that catches a missing
GPU cannot itself require one.
"""

from __future__ import annotations

import pytest

from mcreid.utils import device as device_module
from mcreid.utils.device import DeviceSpec, probe_compute_device

CPU = DeviceSpec(kind="cpu", index=None, name="cpu", total_memory_mb=None, use_half=False)
CUDA = DeviceSpec(kind="cuda", index=0, name="RTX 4060", total_memory_mb=8188, use_half=True)


@pytest.fixture
def resolves_to(monkeypatch):
    """Pin what `resolve_device` returns, and record how it was called."""

    def _install(spec: DeviceSpec) -> list[tuple[str, bool]]:
        calls: list[tuple[str, bool]] = []

        def fake(requested: str = "auto", allow_half: bool = True) -> DeviceSpec:
            calls.append((requested, allow_half))
            return spec

        monkeypatch.setattr(device_module, "resolve_device", fake)
        return calls

    return _install


def test_cpu_is_a_stop_not_a_slow_run(resolves_to):
    resolves_to(CPU)
    with pytest.raises(RuntimeError) as excinfo:
        probe_compute_device("auto", "mcreid-wildtrack footpoint")
    message = str(excinfo.value)
    # The task must be named: the error is meant to be paste-able into a
    # blockers.md row without going back to find out what was running.
    assert "mcreid-wildtrack footpoint" in message
    assert "cpu" in message
    assert "--allow-cpu" in message


def test_allow_cpu_is_the_escape_hatch_and_returns_the_spec(resolves_to):
    resolves_to(CPU)
    assert probe_compute_device("auto", "a slow thing on purpose", allow_cpu=True) is CPU


def test_cuda_passes_and_returns_the_resolved_spec(resolves_to):
    resolves_to(CUDA)
    assert probe_compute_device("auto", "mcreid-wildtrack run") is CUDA


def test_the_request_is_forwarded_verbatim(resolves_to):
    calls = resolves_to(CUDA)
    probe_compute_device("cuda:1", "task", allow_half=False)
    assert calls == [("cuda:1", False)]


def test_a_cpu_request_still_stops_unless_allow_cpu_says_otherwise(resolves_to):
    """Asking for `--device cpu` is not the same as accepting the consequence.

    `resolve_device("cpu")` is a legitimate request and returns without error;
    the probe still refuses, because on a long GPU job the flag is far more often
    a leftover than an intention. `--allow-cpu` is where the intention lives.
    """
    resolves_to(CPU)
    with pytest.raises(RuntimeError):
        probe_compute_device("cpu", "mcreid-wildtrack run")
