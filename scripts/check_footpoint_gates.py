"""The single arbiter for `plan-footpoint.md`'s gates.

Every gate is a command that exits 0 or 1, reading committed JSON. Nothing here
computes a metric — it only compares numbers that a measurement already wrote
down, so a gate can never be accidentally satisfied by the same code that
produced the number.

    uv run python scripts/check_footpoint_gates.py            # all gates
    uv run python scripts/check_footpoint_gates.py --gate g_fp2

Gates, and the plan section that defines each:

    G_FP0a  instrument proof   the refactor did not move the measurement
    G_FP0b  design ceiling     what each arm scores on GT boxes; reported, not thresholded
    G_FP1   strictly better    every arm vs the box-bottom baseline, both p50 and p90
    G_FP2   the surviving band mean <= 0.48 m, derived from D-015 read against D-014
    G_FP3   the pipeline runs  demo end to end, FPS within 20 % of baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
ARMS_JSON = REPO / "docs" / "artifacts" / "footpoint_estimators.json"
RUNTIME_JSON = REPO / "docs" / "artifacts" / "footpoint_runtime.json"
LEGACY = {
    "0.5": REPO / "docs" / "artifacts" / "footpoint_iou0.5.json",
    "0.3": REPO / "docs" / "artifacts" / "footpoint_iou0.3.json",
    "0.1": REPO / "docs" / "artifacts" / "footpoint_iou0.1.json",
}

# plan-footpoint.md §3 G_FP2. NOT "the merge radius", NOT "the clustering
# radius" — this repo has three different 0.35 m values and none of them is this.
# 0.48 m is the largest gap on D-015's RECORDED curve at which D-014's approved
# >= 90 % one-person-cross-view hold still holds (0.48 -> 98 %, next sampled
# point 1.02 -> 33 %). The curve is unsampled in between, so the gate takes the
# last measured passing point rather than interpolating a threshold nobody
# measured. Changing it is a dated reports/deviation-log.md row with an
# authority named — and if the change would make a failing gate pass, it is a
# blockers.md proposal instead.
BAND_M = 0.48
D015_CURVE = {0.08: 99, 0.48: 98, 1.02: 33, 1.48: 13}  # gap m -> 1-holds-1 %
D014_REQUIRED_HOLD_PCT = 90

# G_FP0a tolerance. The bbox arm re-derives the legacy numbers through a new code
# path on the same frames and detections, so agreement should be exact; 1e-9
# allows for nothing more than float reassociation.
EXACT = 1e-9
RUNTIME_BUDGET = 0.8  # fps_pose >= 0.8 * fps_bbox


class GateResult:
    def __init__(self, name: str) -> None:
        self.name = name
        self.lines: list[str] = []
        self.ok = True
        self.skipped = False

    def check(self, condition: bool, message: str) -> None:
        self.lines.append(f"    {'PASS' if condition else 'FAIL'}  {message}")
        self.ok = self.ok and condition

    def note(self, message: str) -> None:
        self.lines.append(f"    ....  {message}")

    def skip(self, message: str) -> None:
        self.skipped = True
        self.lines.append(f"    SKIP  {message}")

    def report(self) -> None:
        status = "SKIP" if self.skipped else ("PASS" if self.ok else "FAIL")
        print(f"[{status}] {self.name}")
        for line in self.lines:
            print(line)


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def gate_g0a() -> GateResult:
    """The refactor did not move the measurement.

    The `bbox` arm goes through the new pluggable estimator; the legacy artifacts
    went through the hard-coded rule. Same frames, same detections, same
    homographies — so any difference at all is the refactor, and a foot-point
    comparison whose baseline drifted is not a comparison.
    """
    r = GateResult("G_FP0a  instrument proof: bbox arm reproduces the committed artifacts")
    arms = _load(ARMS_JSON)
    if arms is None:
        r.skip(f"{ARMS_JSON.relative_to(REPO)} not written yet")
        return r

    for iou, legacy_path in LEGACY.items():
        legacy = _load(legacy_path)
        if legacy is None:
            r.skip(f"{legacy_path.name} missing")
            continue
        new = arms["arms"]["detector"].get(iou, {}).get("bbox")
        if new is None:
            r.check(False, f"IoU {iou}: no bbox arm in the new artifact")
            continue
        for key in ("mean_m", "p50_m", "p90_m", "n_pairs"):
            delta = abs(float(new[key]) - float(legacy["detector_boxes"][key]))
            r.check(delta <= EXACT, f"IoU {iou} detector {key}: delta {delta:.3e}")

    legacy_gt = _load(LEGACY["0.5"])
    new_gt = arms["arms"]["gt"]["gt"].get("bbox")
    if legacy_gt is not None and new_gt is not None:
        for key in ("mean_m", "p50_m", "p90_m", "n_pairs"):
            delta = abs(float(new_gt[key]) - float(legacy_gt["gt_boxes"][key]))
            r.check(delta <= EXACT, f"GT boxes {key}: delta {delta:.3e}")
    return r


def gate_g0b() -> GateResult:
    """Each arm's ceiling, on GT boxes. Reported, never thresholded.

    This is the check that decides whether a failure is a design REJECTION or a
    VOID one. An arm that cannot reach the band even with perfect boxes was
    structurally incapable, and charging the kill counter for it is the error the
    operator ruled on in 103 on 2026-08-08c.
    """
    r = GateResult(
        "G_FP0b  design ceiling on GT boxes (informational - never increments the kill counter)"
    )
    arms = _load(ARMS_JSON)
    if arms is None:
        r.skip(f"{ARMS_JSON.relative_to(REPO)} not written yet")
        return r

    for arm, stats in sorted(arms["arms"]["gt"]["gt"].items()):
        verdict = "could pass" if float(stats["mean_m"]) <= BAND_M else "STRUCTURALLY INCAPABLE"
        r.note(
            f"{arm:8s} on GT boxes: mean {stats['mean_m']:.3f} m, "
            f"p50 {stats['p50_m']:.3f}, p90 {stats['p90_m']:.3f}  -> {verdict}"
        )
    fallback = arms.get("pose_fallback_fraction", {})
    for source, value in fallback.items():
        if value is not None:
            r.note(f"pose fell back to the box bottom on {value * 100:.1f} % of {source} boxes")
    return r


def gate_g1() -> GateResult:
    """Strictly better than box-bottom, in p50 AND p90, at every IoU column.

    "At every column" is not extra strictness — it is D-004. The statistic IS the
    sweep, so an arm that improves the lenient column while the strict one
    worsens has moved the reporting gate, not the foot point.
    """
    r = GateResult("G_FP1   strictly better than box-bottom (p50 and p90, every IoU column)")
    arms = _load(ARMS_JSON)
    if arms is None:
        r.skip(f"{ARMS_JSON.relative_to(REPO)} not written yet")
        return r

    detector = arms["arms"]["detector"]
    any_arm_passes = False
    for arm in ("pose", "stature"):
        arm_ok = True
        for iou in sorted(detector, key=float, reverse=True):
            base = detector[iou]["bbox"]
            got = detector[iou].get(arm)
            if got is None:
                arm_ok = False
                continue
            for key in ("p50_m", "p90_m"):
                better = float(got[key]) < float(base[key])
                arm_ok = arm_ok and better
                r.lines.append(
                    f"    {'PASS' if better else 'FAIL'}  {arm:8s} IoU {iou} {key}: "
                    f"{got[key]:.3f} vs baseline {base[key]:.3f}"
                )
        any_arm_passes = any_arm_passes or arm_ok
    r.ok = any_arm_passes
    r.note("gate needs AT LEAST ONE arm strictly better on all 6 comparisons")
    return r


def gate_g2() -> GateResult:
    """Lands in the band where the calibration gate survives."""
    r = GateResult(f"G_FP2   mean <= {BAND_M} m at every IoU column")
    arms = _load(ARMS_JSON)
    if arms is None:
        r.skip(f"{ARMS_JSON.relative_to(REPO)} not written yet")
        return r

    curve = ", ".join(f"{g:.2f} m -> {h} %" for g, h in sorted(D015_CURVE.items()))
    r.note(f"D-015 curve: {curve}")
    r.note(
        f"D-014 approved criterion: 1-holds-1 >= {D014_REQUIRED_HOLD_PCT} %. "
        f"Largest measured gap still meeting it = {BAND_M} m. Unsampled above it, so no "
        "interpolation."
    )

    detector = arms["arms"]["detector"]
    any_arm_passes = False
    for arm in ("pose", "stature"):
        worst = -1.0
        for iou in sorted(detector, key=float, reverse=True):
            got = detector[iou].get(arm)
            if got is None:
                worst = float("inf")
                continue
            worst = max(worst, float(got["mean_m"]))
            r.lines.append(
                f"    {'PASS' if float(got['mean_m']) <= BAND_M else 'FAIL'}  "
                f"{arm:8s} IoU {iou} mean {got['mean_m']:.3f} m"
            )
        any_arm_passes = any_arm_passes or (0.0 <= worst <= BAND_M)
    r.ok = any_arm_passes
    return r


def gate_g3() -> GateResult:
    """The pipeline still runs, and is not appreciably slower."""
    r = GateResult(
        f"G_FP3   demo runs with --footpoint pose, FPS >= {RUNTIME_BUDGET:.0%} of baseline"
    )
    runtime = _load(RUNTIME_JSON)
    if runtime is None:
        r.skip(f"{RUNTIME_JSON.relative_to(REPO)} not written yet")
        return r

    for arm in ("bbox", "pose"):
        r.check(bool(runtime[arm]["exit_ok"]), f"{arm} arm ran end to end without crashing")
    ratio = float(runtime["pose"]["fps"]) / float(runtime["bbox"]["fps"])
    r.check(
        ratio >= RUNTIME_BUDGET,
        f"pose {runtime['pose']['fps']:.2f} FPS vs bbox {runtime['bbox']['fps']:.2f} "
        f"= {ratio:.0%} of baseline",
    )
    return r


GATES = {
    "g0a": gate_g0a,
    "g0b": gate_g0b,
    "g_fp1": gate_g1,
    "g_fp2": gate_g2,
    "g_fp3": gate_g3,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", choices=sorted(GATES), help="Run one gate. Default: all.")
    args = parser.parse_args()

    selected = [args.gate] if args.gate else list(GATES)
    results = [GATES[name]() for name in selected]
    for result in results:
        result.report()

    # A skipped gate is not a passing gate. Exiting 0 on "the measurement has not
    # been run yet" is how a plan reports green on nothing at all.
    failed = [r.name for r in results if not r.ok and not r.skipped]
    skipped = [r.name for r in results if r.skipped]
    print()
    if failed:
        print(f"FAILED: {len(failed)} gate(s)")
    if skipped:
        print(f"SKIPPED (no evidence on disk): {len(skipped)} gate(s)")
    if not failed and not skipped:
        print("all gates PASS")
    return 1 if (failed or skipped) else 0


if __name__ == "__main__":
    sys.exit(main())
