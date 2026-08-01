"""N-camera ground calibration from floor markers seen by every camera.

This is the cheapest calibration that puts several cameras in **one** world
frame: mark a handful of points on the floor, record where each camera sees
each one, and fit a per-camera image->world homography against the *same* list
of world coordinates. Sharing the world list is the whole mechanism — it is what
makes "1.2 m from the door" mean the same thing to every camera, which is what
the fusion stage needs and what an uncalibrated rig cannot provide.

**Why the gate needs more than four markers.** Four correspondences give an
*exact* homography: `cv2.getPerspectiveTransform` reproduces them to ~1e-15, so
a residual computed on the fitting points is identically zero and a gate built
on it can never fail. Worse, the cross-camera agreement on those same points is
also identically zero, because every camera was fitted to map them onto the same
world coordinates. A four-point rig is not un-gateable by accident; it is
un-gateable in principle.

So the gate is **leave-one-out**: with N >= 6 markers, each camera's homography
is re-fitted N times on N-1 of them and scored on the one held out. That
measures what actually matters — how well the calibration generalises to a floor
point it was not fitted to — and the cross-camera spread on held-out points is a
real disagreement rather than an algebraic identity.

The bounds are set from this project's own measurements, not invented: on
WILDTRACK, ground-truth boxes place one person within **0.12 m** from any two
cameras and never beyond the 0.35 m clustering radius, while detector boxes miss
it 58-65 % of the time. A calibration whose own geometry is worse than the
detector noise it has to survive is not worth running, so the default ceiling
sits below the fusion stage's 0.75 m merge radius with room to spare.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from mcreid.calib.geometry import apply_homography
from mcreid.calib.homography import fit_ground_homography, ground_plane_from_correspondences
from mcreid.calib.schema import CameraCalib, Intrinsics, RigCalib
from mcreid.utils.logging import get_logger

logger = get_logger(__name__)

FloatArray = npt.NDArray[np.float64]

MIN_MARKERS = 6
"""Fewer than this and the residual gate is an identity, not a measurement.

Four is the minimum to FIT a homography and the maximum at which the fit is
exact, which is exactly the wrong place to stand: the gate would pass every rig
including a badly mismarked one. Six leaves two degrees of freedom after the
four a homography consumes, so a held-out point can genuinely disagree."""

MAX_HELDOUT_ERROR_M = 0.25
"""Ceiling on a camera's leave-one-out floor error.

Below the 0.35 m birth-clustering radius and well below the 0.75 m merge radius,
so a calibration that passes cannot by itself push two views of one person past
the gates that fuse them. WILDTRACK ground truth achieves 0.12 m across seven
cameras, so this is roughly 2x the achievable figure — loose enough for hand-
marked points on a phone photo, tight enough to be worth passing."""

MAX_CROSS_CAMERA_DISAGREEMENT_M = 0.35
"""Ceiling on how far two cameras may place the same held-out floor point apart.

This is the birth-clustering radius exactly. Two cameras that disagree by more
than the distance at which the fusion stage stops believing they are looking at
one person have not been calibrated into a shared frame in any useful sense."""


@dataclass(frozen=True)
class FloorMarkers:
    """Shared floor points, and where each camera sees them."""

    world_points: FloatArray
    """(N, 2) metres on the floor. The SAME physical markers for every camera."""
    image_points: dict[str, FloatArray]
    """camera_id -> (N, 2) pixels, in the same order as `world_points`."""
    image_sizes: dict[str, tuple[int, int]]
    """camera_id -> (width, height), so a rig cannot be built against a
    resolution the camera never produced."""

    def __post_init__(self) -> None:
        if self.world_points.ndim != 2 or self.world_points.shape[1] != 2:
            raise ValueError(f"world_points must be (N, 2), got {self.world_points.shape}")
        n = self.world_points.shape[0]
        if n < MIN_MARKERS:
            raise ValueError(
                f"need at least {MIN_MARKERS} floor markers, got {n}. Four would fit a "
                "homography exactly, which makes the residual gate an identity that "
                "cannot fail — see mcreid.calib.floor."
            )
        if not self.image_points:
            raise ValueError("no cameras in the marker set")
        for camera_id, pts in self.image_points.items():
            if pts.shape != (n, 2):
                raise ValueError(
                    f"{camera_id}: expected {n} image points to match the world list, "
                    f"got {pts.shape}"
                )
            if camera_id not in self.image_sizes:
                raise ValueError(f"{camera_id}: no image_size declared")

    @property
    def camera_ids(self) -> list[str]:
        return sorted(self.image_points)

    @property
    def n_markers(self) -> int:
        return int(self.world_points.shape[0])


def load_floor_markers(path: Path) -> FloorMarkers:
    """Read a floor-marker YAML.

    Expected shape::

        world_points:            # metres, shared by every camera
          - [0.0, 0.0]
          - [2.0, 0.0]
          ...
        cameras:
          cam0:
            image_size: [1280, 720]
            image_points: [[u, v], ...]   # same order as world_points
          cam1:
            image_size: [640, 480]
            image_points: [[u, v], ...]
    """
    import yaml

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"floor marker file not found: {path}")
    data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("world_points", "cameras"):
        if key not in data:
            raise ValueError(f"{path}: missing required key {key!r}")

    world = np.asarray(data["world_points"], dtype=np.float64)
    image_points: dict[str, FloatArray] = {}
    image_sizes: dict[str, tuple[int, int]] = {}
    for camera_id, entry in data["cameras"].items():
        for key in ("image_size", "image_points"):
            if key not in entry:
                raise ValueError(f"{path}: camera {camera_id} missing {key!r}")
        image_points[camera_id] = np.asarray(entry["image_points"], dtype=np.float64)
        size = entry["image_size"]
        image_sizes[camera_id] = (int(size[0]), int(size[1]))
    return FloorMarkers(
        world_points=world, image_points=image_points, image_sizes=image_sizes
    )


@dataclass(frozen=True)
class CameraAgreement:
    """One camera's leave-one-out generalisation error."""

    camera_id: str
    mean_error_m: float
    max_error_m: float
    per_marker_m: FloatArray
    """(N,) error at each marker when that marker was held out of the fit."""

    @property
    def passes(self) -> bool:
        return self.max_error_m <= MAX_HELDOUT_ERROR_M


@dataclass(frozen=True)
class RigAgreement:
    """Whether the cameras agree about where the floor is."""

    per_camera: list[CameraAgreement]
    max_disagreement_m: float
    """Largest distance between two cameras' estimates of one held-out marker."""
    worst_pair: tuple[str, str] | None
    worst_marker: int | None
    overlap_m2: float
    """Area of floor both cameras' marker hulls cover. Zero means no shared view."""

    @property
    def passes(self) -> bool:
        return (
            all(cam.passes for cam in self.per_camera)
            and self.max_disagreement_m <= MAX_CROSS_CAMERA_DISAGREEMENT_M
        )

    def report(self) -> list[str]:
        lines = [
            f"floor calibration agreement (leave-one-out over "
            f"{len(self.per_camera[0].per_marker_m)} markers):"
        ]
        for cam in self.per_camera:
            verdict = "OK  " if cam.passes else "FAIL"
            lines.append(
                f"  {verdict} {cam.camera_id}: held-out error mean {cam.mean_error_m:.3f} m, "
                f"max {cam.max_error_m:.3f} m (ceiling {MAX_HELDOUT_ERROR_M:.2f} m)"
            )
        verdict = "OK  " if self.max_disagreement_m <= MAX_CROSS_CAMERA_DISAGREEMENT_M else "FAIL"
        where = ""
        if self.worst_pair is not None and self.worst_marker is not None:
            where = f" — {self.worst_pair[0]} vs {self.worst_pair[1]} at marker {self.worst_marker}"
        lines.append(
            f"  {verdict} cross-camera disagreement: max {self.max_disagreement_m:.3f} m "
            f"(ceiling {MAX_CROSS_CAMERA_DISAGREEMENT_M:.2f} m){where}"
        )
        lines.append(f"       shared floor area covered by the markers: {self.overlap_m2:.1f} m^2")
        return lines


def _leave_one_out(image_pts: FloatArray, world_pts: FloatArray) -> FloatArray:
    """(N,) world error at each marker when fitted without it."""
    n = world_pts.shape[0]
    errors = np.zeros(n, dtype=np.float64)
    for held in range(n):
        keep = [i for i in range(n) if i != held]
        H, _, _ = fit_ground_homography(image_pts[keep], world_pts[keep])
        projected, valid = apply_homography(H, image_pts[held : held + 1])
        if not valid.all():
            errors[held] = np.inf
            continue
        errors[held] = float(np.linalg.norm(projected[0] - world_pts[held]))
    return errors


def _heldout_world_estimates(
    image_pts: FloatArray, world_pts: FloatArray
) -> FloatArray:
    """(N, 2) where the camera thinks each marker is, fitted without that marker."""
    n = world_pts.shape[0]
    out = np.full((n, 2), np.nan, dtype=np.float64)
    for held in range(n):
        keep = [i for i in range(n) if i != held]
        H, _, _ = fit_ground_homography(image_pts[keep], world_pts[keep])
        projected, valid = apply_homography(H, image_pts[held : held + 1])
        if valid.all():
            out[held] = projected[0]
    return out


def _hull_area(points: FloatArray) -> float:
    """Area of the convex hull of (N, 2) points — the floor the markers span."""
    if points.shape[0] < 3:
        return 0.0
    import cv2

    hull = cv2.convexHull(points.astype(np.float32))
    return float(cv2.contourArea(hull))


def measure_agreement(markers: FloorMarkers) -> RigAgreement:
    """Leave-one-out error per camera, and the cross-camera spread on held-out points."""
    per_camera = []
    estimates: dict[str, FloatArray] = {}
    for camera_id in markers.camera_ids:
        image_pts = markers.image_points[camera_id]
        errors = _leave_one_out(image_pts, markers.world_points)
        estimates[camera_id] = _heldout_world_estimates(image_pts, markers.world_points)
        per_camera.append(
            CameraAgreement(
                camera_id=camera_id,
                mean_error_m=float(np.mean(errors)),
                max_error_m=float(np.max(errors)),
                per_marker_m=errors,
            )
        )

    worst = 0.0
    worst_pair: tuple[str, str] | None = None
    worst_marker: int | None = None
    for left, right in combinations(markers.camera_ids, 2):
        spread = np.linalg.norm(estimates[left] - estimates[right], axis=1)
        spread = np.where(np.isfinite(spread), spread, np.inf)
        index = int(np.argmax(spread))
        if spread[index] > worst:
            worst = float(spread[index])
            worst_pair = (left, right)
            worst_marker = index

    # Every camera saw every marker by construction, so the shared floor is the
    # hull of the markers themselves. Reported because a rig can pass every
    # residual bound and still have the cameras looking at 2 m^2 of floor, which
    # gives cross-view fusion almost nothing to work with.
    overlap = _hull_area(markers.world_points)
    return RigAgreement(
        per_camera=per_camera,
        max_disagreement_m=worst,
        worst_pair=worst_pair,
        worst_marker=worst_marker,
        overlap_m2=overlap,
    )


def rig_from_floor_markers(markers: FloorMarkers, margin_m: float = 1.0) -> RigCalib:
    """Fit one ground homography per camera against the shared world points."""
    cameras: list[CameraCalib] = []
    for camera_id in markers.camera_ids:
        width, height = markers.image_sizes[camera_id]
        # No lens model is estimated here: a floor-marker calibration has no
        # information about distortion, and inventing coefficients would make
        # `undistort_points` move the very pixels that were measured. A rig that
        # needs distortion correction should come from `mcreid-calibrate rig`.
        intrinsics = Intrinsics(
            fx=float(width),
            fy=float(width),
            cx=width / 2.0,
            cy=height / 2.0,
            dist_coeffs=[0.0] * 5,
            image_width=width,
            image_height=height,
            rms_reproj_px=0.0,
            n_views=0,
        )
        ground = ground_plane_from_correspondences(
            intrinsics,
            markers.image_points[camera_id],
            markers.world_points,
            method="four_point",
            margin_m=margin_m,
        )
        cameras.append(
            CameraCalib(
                camera_id=camera_id,
                intrinsics=intrinsics,
                ground=ground,
                notes=(
                    f"floor-marker calibration, {markers.n_markers} shared points, "
                    "no lens model estimated"
                ),
            )
        )
    return RigCalib(
        cameras=cameras,
        world_notes=(
            "Floor-marker calibration: every camera fitted to ONE shared list of world "
            "points, so positions are comparable between cameras and metric on the floor "
            "plane. Off-plane geometry is not modelled."
        ),
    )


def build_and_gate(markers: FloorMarkers) -> tuple[RigCalib, RigAgreement]:
    """Fit the rig and measure it. Raises if the cameras do not agree.

    The gate refuses rather than warns, on the same reasoning as
    `mcreid-calibrate report`: a rig that silently ships a bad floor produces a
    demo that runs, looks plausible and is wrong, which is more expensive than a
    failure at calibration time.
    """
    agreement = measure_agreement(markers)
    for line in agreement.report():
        logger.info(line)
    if not agreement.passes:
        raise ValueError(
            "floor calibration REFUSED — the cameras do not agree about where the floor "
            "is:\n" + "\n".join(agreement.report())
        )
    return rig_from_floor_markers(markers), agreement
