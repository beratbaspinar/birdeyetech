"""Where a person actually touches the floor.

The shipped rule is the bottom-centre of the detection box, and it is right only
when the feet are visible. In a crowd the box truncates at whoever stands in
front, so the "foot point" lands short and the same person's world position
disagrees between cameras by far more than the calibration error — measured at
0.62 m to 2.17 m across the box-attribution sweep, against 0.123 m from
ground-truth boxes (`docs/artifacts/footpoint_iou*.json`, D-004).

Three estimators, switchable by name, all returning the same thing: a pixel that
the *existing* Z=0 ground homography can be applied to unchanged.

``bbox``
    Bottom-centre. The shipped rule, kept as the default and as the baseline
    every other arm is measured against.
``pose``
    Ankle keypoints, projected from the plane an ankle actually sits on
    (~9 cm above the sole) rather than from the floor.
``stature``
    The box's **top** centre — the head — projected from the plane a head sits
    on, i.e. a person's height. The head is the edge that survives a crowd:
    heads stick out, feet are what gets occluded. That is the exact inverse of
    the failure mode being fixed, which is why this is the fallback and not an
    arbitrary second option.

**Everything here is pure geometry over inputs the caller supplies** — no torch,
no detector, no dataset — so the whole module is unit-testable against the
analytic virtual-camera rig in ``mcreid.sim``. Keypoints come from
``mcreid.track.pose``, which is the only file that needs a GPU.

## The one piece of real maths, and why it is not an approximation

A ground homography maps the image to the plane Z = 0. Both new arms observe a
point that is *not* on that plane — an ankle is above it, a head is far above
it — so projecting either through the floor homography puts the person too far
away. The fix is the homography of the plane the point is actually on.

``H_img2world`` gives ``H_world2img = K [r1 r2 t]`` up to scale. Recover the
scale from ``‖K⁻¹h₁‖``, take ``r3 = r1 × r2``, and the plane ``Z = h`` has its
own exact homography ``K [r1 r2 (r3 h + t)]``. At ``h = 0`` it reduces to the
original, which is asserted in the tests rather than assumed.

This is the classical head-foot homology, derived from the calibration we
already carry instead of estimated from vanishing points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from mcreid.calib.geometry import feet_point, ground_to_image, image_to_ground
from mcreid.calib.schema import CameraCalib
from mcreid.utils.logging import get_logger

logger = get_logger(__name__)

FloatArray = npt.NDArray[np.float64]

# Ankle-joint height above the sole, metres. ANTHROPOMETRIC, not tuned: the
# lateral malleolus sits at roughly 5-6% of stature, i.e. ~9 cm on a 1.70 m
# adult. It is ablated (Z=0 vs Z=0.09) rather than trusted — plan-footpoint.md §7.
ANKLE_HEIGHT_M = 0.09

# Stature for the head-based arm, metres. Fixed for v1 on purpose: a per-track
# height estimate would be fitted from the very boxes this module exists because
# we distrust. YAGNI until the fixed one has been measured (plan §8 decision 4).
DEFAULT_STATURE_M = 1.70

# Keypoint confidence below which an ankle is not evidence. A forced
# low-confidence ankle is worse than an honest fallback — plan §7.
DEFAULT_MIN_KEYPOINT_CONF = 0.5

ESTIMATOR_NAMES = ("bbox", "pose", "stature")


@dataclass(frozen=True)
class FootPoints:
    """Estimated ground-contact pixels, plus how each one was arrived at.

    ``points_px`` is always safe to feed to ``image_to_ground`` with the ordinary
    Z=0 homography: rows that were observed off the floor have already been
    mapped through their own plane and re-projected onto the floor's. That is
    what makes every arm a drop-in replacement rather than a new interface.

    ``source`` is per row and is the honest part — an arm that quietly degraded
    to ``bbox`` on 90 % of its detections has not been measured, it has been
    disguised, so the fallback rate is reported alongside every number.
    """

    points_px: FloatArray
    source: tuple[str, ...]
    confidence: FloatArray

    def __post_init__(self) -> None:
        n = self.points_px.shape[0]
        if self.points_px.ndim != 2 or self.points_px.shape[1] != 2:
            raise ValueError(f"points_px must be (N, 2), got {self.points_px.shape}")
        if len(self.source) != n or self.confidence.shape[0] != n:
            raise ValueError(
                f"ragged FootPoints: {n} points, {len(self.source)} sources, "
                f"{self.confidence.shape[0]} confidences"
            )

    @property
    def fallback_fraction(self) -> float:
        """Fraction of rows that fell back to the box bottom. 0.0 when empty."""
        if not self.source:
            return 0.0
        return sum(s == "bbox" for s in self.source) / len(self.source)


def plane_homography(cam: CameraCalib, height_m: float) -> FloatArray:
    """Homography mapping undistorted pixels to world XY on the plane Z=height_m.

    At ``height_m == 0`` this is ``cam.ground.H`` up to scale, which the tests
    assert rather than assume.

    Raises:
        ValueError: the decomposition is degenerate (singular K or homography).
    """
    K = np.asarray(cam.intrinsics.K, dtype=np.float64)
    H_world2img = np.asarray(cam.ground.H_inv, dtype=np.float64)

    A = np.linalg.inv(K) @ H_world2img
    n1, n2 = float(np.linalg.norm(A[:, 0])), float(np.linalg.norm(A[:, 1]))
    if n1 < 1e-12 or n2 < 1e-12:
        raise ValueError("degenerate ground homography: zero-norm rotation column")
    scale = 0.5 * (n1 + n2)

    r1, r2, t = A[:, 0] / scale, A[:, 1] / scale, A[:, 2] / scale
    # Sign disambiguation. A homography is defined up to scale INCLUDING its
    # sign, and the wrong sign sends a positive height the wrong way — a head
    # would project further from the camera instead of nearer. `context.md`
    # fixes the convention: cameras are above the floor, so the camera centre
    # C = -R^T t must have positive Z.
    r3 = np.cross(r1, r2)
    R = np.stack([r1, r2, r3], axis=1)
    if float((-R.T @ t)[2]) < 0.0:
        r1, r2, t = -r1, -r2, -t
        r3 = np.cross(r1, r2)

    H_world2img_h = K @ np.stack([r1, r2, r3 * float(height_m) + t], axis=1)
    if abs(float(np.linalg.det(H_world2img_h))) < 1e-12:
        raise ValueError(f"plane Z={height_m} projects singularly for {cam.camera_id}")
    return np.asarray(np.linalg.inv(H_world2img_h), dtype=np.float64)


def _project_via_plane(
    cam: CameraCalib, pts_img: FloatArray, height_m: float
) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
    """Map observed pixels through the Z=height plane, then back onto the floor.

    Returns the *floor-equivalent pixel*, not the world point, so that every
    caller of the shipped `image_to_ground` path keeps working untouched. The
    round trip is exact up to floating point; the tests pin it.
    """
    if pts_img.size == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=bool)

    from mcreid.calib.geometry import apply_homography, horizon_sign, undistort_points

    H_h = plane_homography(cam, height_m)
    ideal = undistort_points(cam, pts_img)
    sign = horizon_sign(H_h, cam.intrinsics.image_size)
    world, ok = apply_homography(H_h, ideal, valid_sign=sign)

    floor_px = np.full_like(pts_img, np.nan, dtype=np.float64)
    if np.any(ok):
        back, back_ok = ground_to_image(cam, world[ok])
        idx = np.flatnonzero(ok)
        floor_px[idx] = back
        ok = ok.copy()
        ok[idx] = back_ok & np.all(np.isfinite(back), axis=1)
    return floor_px, ok


class FootPointEstimator(Protocol):
    """Drop-in replacement for the box-bottom rule.

    Every implementation takes the same arguments and returns `FootPoints` whose
    pixels go through the unchanged Z=0 projection path. `ankles` is ignored by
    arms that do not use keypoints, so a caller can pass them unconditionally.
    """

    # Read-only on purpose: the implementations are frozen dataclasses, and a
    # settable `name: str` on the protocol would exclude every one of them.
    @property
    def name(self) -> str: ...

    def __call__(
        self,
        boxes_xyxy: npt.ArrayLike,
        cam: CameraCalib | None = None,
        ankles: npt.ArrayLike | None = None,
    ) -> FootPoints: ...


def _as_boxes(boxes_xyxy: npt.ArrayLike) -> FloatArray:
    boxes = np.asarray(boxes_xyxy, dtype=np.float64)
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"expected (N, 4) xyxy boxes, got shape {boxes.shape}")
    return boxes


@dataclass(frozen=True)
class BboxFootPoint:
    """The shipped rule: bottom-centre of the box. The baseline, not the goal."""

    name: str = "bbox"

    def __call__(
        self,
        boxes_xyxy: npt.ArrayLike,
        cam: CameraCalib | None = None,
        ankles: npt.ArrayLike | None = None,
    ) -> FootPoints:
        boxes = _as_boxes(boxes_xyxy)
        pts = feet_point(boxes)
        return FootPoints(
            points_px=pts,
            source=("bbox",) * pts.shape[0],
            confidence=np.ones(pts.shape[0], dtype=np.float64),
        )


@dataclass(frozen=True)
class PoseFootPoint:
    """Ankle keypoints, projected from the plane an ankle sits on.

    Both ankles confident → their midpoint, which is the point between the feet
    and is what a person's ground position means. One ankle confident → that
    one, accepting the half-stance offset, because one real ankle beats a
    truncated box. Neither → **fall back to the box bottom and say so.**

    Degrading is not failing. The pipeline must never crash because a person
    turned their back on a camera, and a run whose keypoints are all missing has
    to come out exactly equal to the baseline rather than come out wrong.
    """

    min_conf: float = DEFAULT_MIN_KEYPOINT_CONF
    ankle_height_m: float = ANKLE_HEIGHT_M
    name: str = "pose"

    def __call__(
        self,
        boxes_xyxy: npt.ArrayLike,
        cam: CameraCalib | None = None,
        ankles: npt.ArrayLike | None = None,
    ) -> FootPoints:
        boxes = _as_boxes(boxes_xyxy)
        n = boxes.shape[0]
        base = BboxFootPoint()(boxes)
        if ankles is None or cam is None or n == 0:
            return base

        kp = np.asarray(ankles, dtype=np.float64)
        if kp.shape != (n, 2, 3):
            raise ValueError(f"ankles must be (N, 2, 3) as [x, y, conf], got {kp.shape}")

        good = (kp[..., 2] >= self.min_conf) & np.all(np.isfinite(kp[..., :2]), axis=-1)
        n_good = good.sum(axis=1)
        # Mean over the confident ankles only: with one ankle this is that
        # ankle, with two it is the midpoint. A plain mean would drag a
        # single-ankle row toward a NaN or a zero.
        weights = good.astype(np.float64)
        totals = np.where(n_good[:, None] > 0, weights[..., None].sum(axis=1), 1.0)
        observed = np.nansum(np.where(good[..., None], kp[..., :2], 0.0), axis=1) / totals

        usable = n_good > 0
        points = base.points_px.copy()
        source = list(base.source)
        conf = base.confidence.copy()

        if np.any(usable):
            projected, ok = _project_via_plane(cam, observed[usable], self.ankle_height_m)
            idx = np.flatnonzero(usable)
            for slot, row in enumerate(idx):
                if not ok[slot]:
                    # Beyond the horizon on its own plane: keep the box bottom.
                    continue
                points[row] = projected[slot]
                source[row] = "pose"
                conf[row] = float(kp[row, :, 2][good[row]].mean())

        return FootPoints(points_px=points, source=tuple(source), confidence=conf)


@dataclass(frozen=True)
class StatureFootPoint:
    """The head, projected from the plane a head sits on.

    Uses the box **top** centre and a fixed stature. The premise is the measured
    failure mode read backwards: occlusion truncates a box from the BOTTOM,
    because the occluder stands between the camera and the feet. The top edge is
    the one that survives.

    It buys that robustness with an assumption — that everyone is
    `stature_m` tall and standing upright — and that assumption is wrong for
    every individual and roughly right for a population. Which is why this arm
    is measured rather than assumed better, and why it is second.
    """

    stature_m: float = DEFAULT_STATURE_M
    name: str = "stature"

    def __call__(
        self,
        boxes_xyxy: npt.ArrayLike,
        cam: CameraCalib | None = None,
        ankles: npt.ArrayLike | None = None,
    ) -> FootPoints:
        boxes = _as_boxes(boxes_xyxy)
        n = boxes.shape[0]
        base = BboxFootPoint()(boxes)
        if cam is None or n == 0:
            return base

        heads = np.stack([(boxes[:, 0] + boxes[:, 2]) * 0.5, boxes[:, 1]], axis=1)
        projected, ok = _project_via_plane(cam, heads, self.stature_m)

        points = base.points_px.copy()
        source = list(base.source)
        for row in range(n):
            if ok[row]:
                points[row] = projected[row]
                source[row] = "stature"
        return FootPoints(
            points_px=points, source=tuple(source), confidence=base.confidence.copy()
        )


def resolve_estimator(
    name: str,
    *,
    min_conf: float = DEFAULT_MIN_KEYPOINT_CONF,
    ankle_height_m: float = ANKLE_HEIGHT_M,
    stature_m: float = DEFAULT_STATURE_M,
) -> FootPointEstimator:
    """Name -> estimator. The one place a `--footpoint` flag becomes an object.

    Kept a pure function on purpose: `context.md` records a whole live session
    lost to a CLI flag that never reached the config it named, and the rule that
    came out of it is that flag-to-config resolution lives in a pure function
    with a test per value.
    """
    if name == "bbox":
        return BboxFootPoint()
    if name == "pose":
        return PoseFootPoint(min_conf=min_conf, ankle_height_m=ankle_height_m)
    if name == "stature":
        return StatureFootPoint(stature_m=stature_m)
    raise ValueError(f"unknown foot-point estimator {name!r}; available: {list(ESTIMATOR_NAMES)}")


def world_points(
    cam: CameraCalib, estimate: FootPoints
) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
    """Floor positions for an estimate, through the unchanged Z=0 path."""
    return image_to_ground(cam, estimate.points_px)
