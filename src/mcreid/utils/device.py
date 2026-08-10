"""Explicit device selection with a hard CPU fallback.

torch is an optional dependency (perception extra); everything here degrades to
``cpu`` when torch is absent so the core pipeline stays importable.
"""

from __future__ import annotations

from dataclasses import dataclass

from mcreid.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class DeviceSpec:
    """Resolved compute device."""

    kind: str  # "cuda" | "cpu"
    index: int | None
    name: str
    total_memory_mb: int | None
    use_half: bool

    @property
    def torch_device(self) -> str:
        return self.kind if self.index is None else f"{self.kind}:{self.index}"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        mem = f", {self.total_memory_mb} MiB" if self.total_memory_mb else ""
        return f"{self.torch_device} ({self.name}{mem}, half={self.use_half})"


def resolve_device(requested: str = "auto", allow_half: bool = True) -> DeviceSpec:
    """Resolve ``requested`` ("auto" | "cpu" | "cuda" | "cuda:N") to a DeviceSpec.

    Raises:
        RuntimeError: if CUDA was explicitly requested but is unavailable (fail fast —
            silently degrading to CPU would hide a 20x slowdown behind a green demo).
    """
    requested = requested.strip().lower()
    if requested not in {"auto", "cpu"} and not requested.startswith("cuda"):
        raise ValueError(f"unsupported device string: {requested!r}")

    if requested == "cpu":
        return DeviceSpec(kind="cpu", index=None, name="cpu", total_memory_mb=None, use_half=False)

    try:
        import torch
    except ImportError:
        if requested.startswith("cuda"):
            raise RuntimeError(
                "device='cuda' requested but torch is not installed. "
                "Install the perception extra: uv pip install -e '.[perception]'"
            ) from None
        logger.info("torch unavailable — falling back to cpu")
        return DeviceSpec(kind="cpu", index=None, name="cpu", total_memory_mb=None, use_half=False)

    if not torch.cuda.is_available():
        if requested.startswith("cuda"):
            raise RuntimeError("device='cuda' requested but torch.cuda.is_available() is False")
        logger.info("CUDA unavailable — falling back to cpu")
        return DeviceSpec(kind="cpu", index=None, name="cpu", total_memory_mb=None, use_half=False)

    index = 0
    if requested.startswith("cuda:"):
        index = int(requested.split(":", 1)[1])
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"cuda:{index} requested but only {torch.cuda.device_count()} found")

    props = torch.cuda.get_device_properties(index)
    spec = DeviceSpec(
        kind="cuda",
        index=index,
        name=props.name,
        total_memory_mb=int(props.total_memory // (1024 * 1024)),
        use_half=allow_half,
    )
    logger.info("resolved device: %s", spec)
    return spec


def probe_compute_device(
    requested: str, task: str, allow_cpu: bool = False, allow_half: bool = True
) -> DeviceSpec:
    """Resolve, log, and — unless ``allow_cpu`` — refuse to start on CPU.

    `refactored_method.md` §5, environment probe before compute: before any run
    estimated at more than ~5 minutes, assert CUDA is actually there and log the
    device. **A CPU fallback on GPU-intended work is a STOP, not a slow run.**
    Grinding hours on CPU for a job the GPU does in minutes is the canonical
    violation, and it is invisible without this check because the run looks
    healthy — it is just 20x slower and nothing says so.

    ``resolve_device("auto")`` deliberately degrades to CPU so the torch-free
    core stays importable. That is right for a library and wrong for a long
    command, which is why the refusal lives here and not there.

    Args:
        requested: device string, as `resolve_device`.
        task: what is about to run, named in the error so the blockers row
            writes itself.
        allow_cpu: escape hatch for deliberately running the slow path.

    Raises:
        RuntimeError: CPU resolved and ``allow_cpu`` is False.
    """
    spec = resolve_device(requested, allow_half=allow_half)
    if spec.kind != "cuda" and not allow_cpu:
        # ASCII on purpose. This string is read on a Windows console that mangles
        # the em-dashes and section signs used everywhere else in this repo, and a
        # STOP message full of replacement characters reads like a second bug.
        raise RuntimeError(
            f"{task}: environment probe FAILED - resolved to {spec.torch_device}, not cuda.\n"
            "This is a STOP (refactored_method.md section 5), not a slow run: on CPU this job "
            "is ~20x longer and every number it produces arrives too late to be worth having.\n"
            "Diagnose first - driver, the perception extra, CUDA_VISIBLE_DEVICES - then either "
            "fix it or file a reports/blockers.md row. Pass --allow-cpu only if the slow path "
            "is genuinely what you want."
        )
    logger.info("env probe OK — %s runs on %s", task, spec)
    return spec
