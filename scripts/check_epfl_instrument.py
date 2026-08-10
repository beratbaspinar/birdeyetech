"""G0e — the instrument proof for the EPFL transform chain.

`refactored_method.md` §3: **no stage gate is trusted before the eval pipeline is
proved self-consistent.** This arm learned that the hard way. The metric-scale
derivation had a pre-registered check and stopped correctly the moment it failed;
the **rig construction had none**, so a calibration parser that slid one camera's
ground matrix into another's head slot, and a homography applied in the wrong
direction, produced a plausible-looking rig and a demo of an empty floor instead
of an error. An instrument proof is owed by every transform in the chain, not
only by the one that looks like a measurement.

Two checks, both machine-checkable, both failing loudly:

  A. **Coverage.** Every annotated ground-truth position must project INSIDE the
     360x288 image for at least two cameras. Four overlapping cameras watch one
     room; a position no two of them can see is not something this rig can fuse,
     and a transform that sends everything off-frame fails here immediately.

  B. **Agreement.** Detected foot points pushed through the chain must land near
     an annotated person. The bound is **3.0 cells — the birth-cluster radius the
     fusion stage actually uses** (deviation-log row 3), not a number read off the
     result: a projection error larger than the radius that decides "same person"
     makes every downstream identity claim arbitrary.

Check B is the one with teeth, and it is the one that would have caught the
direction defect on the first run. Inverting the homography puts detected feet
88+ pixels from anybody at every scale swept; the correct direction puts them
2.05 cells from the nearest annotated person, where distinct people sit ~10.6
cells apart.

    uv run python scripts/check_epfl_instrument.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from mcreid.calib.epfl import DEFAULT_GRID, build_rig, parse_calibration, parse_ground_truth
from mcreid.calib.geometry import ground_to_image, image_to_ground
from mcreid.cli.public_demo import EPFL_BIRTH_CLUSTER_CELLS, EPFL_CELL_UNIT, EPFL_IMAGE_SIZE
from mcreid.track.gpu_view import GpuPerViewBackend, GpuViewConfig
from mcreid.utils.device import probe_compute_device
from mcreid.utils.logging import setup_logging

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "artifacts" / "epfl_instrument.json"

MIN_CAMERAS_SEEING = 2
MIN_COVERAGE_FRACTION = 0.80
# The bound IS the birth-cluster radius. A projection error above the radius that
# decides "same person" makes every identity claim downstream arbitrary.
MAX_MEDIAN_CELLS = EPFL_BIRTH_CLUSTER_CELLS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / "data" / "epfl_lab")
    parser.add_argument("--sequence", default="6p")
    parser.add_argument("--frames", type=int, default=20, help="Annotated frames for check B.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    setup_logging("INFO")
    probe_compute_device(args.device, "check_epfl_instrument", allow_cpu=args.allow_cpu)

    calibs = parse_calibration(args.root / f"calibration-{args.sequence}.txt")
    positions, header = parse_ground_truth(args.root / f"gt_lab_{args.sequence}.txt")
    grid = (header.get("grid_w", DEFAULT_GRID[0]), header.get("grid_h", DEFAULT_GRID[1]))
    rig = build_rig(calibs, EPFL_CELL_UNIT, EPFL_IMAGE_SIZE, grid)
    width, height = EPFL_IMAGE_SIZE
    rows = sorted(positions)

    # --- A. coverage, over EVERY annotated frame -----------------------------
    seen_counts = []
    for row in rows:
        for position_id in positions[row].values():
            world = np.array(
                [
                    [
                        (position_id % grid[0]) * EPFL_CELL_UNIT,
                        (position_id // grid[0]) * EPFL_CELL_UNIT,
                    ]
                ]
            )
            n_seeing = 0
            for cam in rig.cameras:
                pixels, valid = ground_to_image(cam, world)
                if not bool(valid[0]) or not np.all(np.isfinite(pixels[0])):
                    continue
                u, v = pixels[0]
                if 0 <= u < width and 0 <= v < height:
                    n_seeing += 1
            seen_counts.append(n_seeing)
    seen = np.asarray(seen_counts)
    coverage = float(np.mean(seen >= MIN_CAMERAS_SEEING))
    a_ok = coverage >= MIN_COVERAGE_FRACTION

    # --- B. agreement, against the detector ----------------------------------
    backends = {
        cam.camera_id: GpuPerViewBackend(
            cam.camera_id,
            GpuViewConfig(weights=Path("weights/yolo11x.pt"), imgsz=640, device=args.device),
        )
        for cam in rig.cameras
    }
    distances: list[float] = []
    for index, cam in enumerate(rig.cameras):
        capture = cv2.VideoCapture(str(args.root / f"{args.sequence}-c{index}.avi"))
        for row in rows[: args.frames]:
            capture.set(cv2.CAP_PROP_POS_FRAMES, row)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            boxes, _ = backends[cam.camera_id].detect(np.asarray(frame, dtype=np.uint8))
            if len(boxes) == 0:
                continue
            feet = np.array([[(b[0] + b[2]) / 2.0, b[3]] for b in boxes], dtype=np.float64)
            world, valid = image_to_ground(cam, feet)
            truth = np.array(
                [
                    [(p % grid[0]) * EPFL_CELL_UNIT, (p // grid[0]) * EPFL_CELL_UNIT]
                    for p in positions[row].values()
                ]
            )
            for point, good in zip(world, valid, strict=True):
                if not good or not np.all(np.isfinite(point)):
                    continue
                distances.append(float(np.min(np.linalg.norm(truth - point, axis=1))))
        capture.release()

    median_cells = float(np.median(distances)) if distances else float("inf")
    b_ok = bool(distances) and median_cells <= MAX_MEDIAN_CELLS

    result = {
        "what_this_proves": (
            "the EPFL transform chain (calibration parse -> homography direction -> "
            "top-view scaling -> world) is self-consistent with the dataset's own "
            "ground truth, BEFORE any identity number is trusted. G0 pattern, "
            "refactored_method.md section 3."
        ),
        "units": "grid cells",
        "A_coverage": {
            "annotated_positions": int(seen.size),
            "min_cameras_required": MIN_CAMERAS_SEEING,
            "fraction_seen_by_enough_cameras": coverage,
            "required": MIN_COVERAGE_FRACTION,
            "mean_cameras_seeing": float(seen.mean()),
            "pass": bool(a_ok),
        },
        "B_agreement": {
            "n_detections": len(distances),
            "median_cells_to_nearest_gt": median_cells,
            "max_allowed_cells": MAX_MEDIAN_CELLS,
            "why_that_bound": (
                "it is the birth-cluster radius the fusion stage uses; a projection "
                "error above the radius that decides 'same person' makes every "
                "downstream identity claim arbitrary"
            ),
            "reference_distinct_person_separation_cells_p50": 10.6,
            "pass": bool(b_ok),
        },
        "pass": bool(a_ok and b_ok),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"[{'PASS' if a_ok else 'FAIL'}] A coverage: {coverage:.1%} of {seen.size} annotated "
          f"positions seen by >= {MIN_CAMERAS_SEEING} cameras (need {MIN_COVERAGE_FRACTION:.0%}), "
          f"mean {seen.mean():.2f} cameras")
    print(f"[{'PASS' if b_ok else 'FAIL'}] B agreement: median {median_cells:.2f} cells from a "
          f"detected foot to the nearest annotated person (max {MAX_MEDIAN_CELLS}), "
          f"n={len(distances)}")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
