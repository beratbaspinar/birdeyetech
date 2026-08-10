"""Ground-contact estimators, against a camera whose geometry is exact.

`VirtualCamera` carries a true 3x4 projection matrix, so "where does a head at
(X, Y, 1.7) actually land in this image" has an analytic answer. Every claim
about the Z=h homography is checked against that answer rather than against
another homography — a homography compared to a homography can be wrong twice
and agree.
"""

from __future__ import annotations

import numpy as np
import pytest

from mcreid.calib.ground_contact import (
    ANKLE_HEIGHT_M,
    BboxFootPoint,
    FootPoints,
    PoseFootPoint,
    StatureFootPoint,
    plane_homography,
    resolve_estimator,
    world_points,
)
from mcreid.sim.virtual_camera import VirtualCamera


@pytest.fixture
def cam():
    """One camera looking down at the floor from a corner, 3 m up."""
    vc = VirtualCamera(
        camera_id="cam0",
        position_m=(-3.0, -3.0, 3.0),
        yaw_deg=45.0,
        pitch_deg=35.0,
        hfov_deg=75.0,
        image_size=(1280, 720),
    )
    return vc, vc.to_calib()


def _project_true(vc, xyz) -> np.ndarray:
    """Exact pixel of a world point, straight through the 3x4 matrix."""
    p = np.asarray(vc.P, dtype=np.float64) @ np.array([*xyz, 1.0], dtype=np.float64)
    return p[:2] / p[2]


# --- the plane homography itself -------------------------------------------


def test_at_zero_height_it_is_the_ground_homography(cam):
    """h=0 must reduce to the shipped homography, up to scale.

    Asserted, not assumed: if this drifts, every arm silently measures against a
    slightly different floor than the baseline does, and the comparison stops
    being a comparison.
    """
    _, calib = cam
    H0 = plane_homography(calib, 0.0)
    shipped = np.asarray(calib.ground.H, dtype=np.float64)
    # Normalise both, rather than dividing element-wise: a homography is defined
    # up to scale, and an entry that is legitimately ~0 in both arms makes an
    # element-wise ratio pure noise.
    assert np.allclose(H0 / H0[2, 2], shipped / shipped[2, 2], atol=1e-9)


@pytest.mark.parametrize("height", [0.09, 0.5, 1.7, 2.0])
@pytest.mark.parametrize("xy", [(0.0, 0.0), (1.5, -2.0), (-2.5, 3.0), (4.0, 4.0)])
def test_a_point_at_height_h_recovers_its_own_ground_position(cam, height, xy):
    """The whole design, in one assertion.

    Project a point at (X, Y, h) with the exact camera matrix, then map that
    pixel back through the Z=h homography. It must return (X, Y) — the ground
    position of the thing standing there, not the position of a floor point that
    happens to image at the same pixel.
    """
    vc, calib = cam
    px = _project_true(vc, (*xy, height))
    H_h = plane_homography(calib, height)
    world = H_h @ np.array([px[0], px[1], 1.0])
    recovered = world[:2] / world[2]
    assert np.allclose(recovered, xy, atol=1e-6)


def test_the_floor_homography_gets_a_raised_point_wrong_by_metres(cam):
    """The error this module exists to remove, quantified on a known case.

    A head at 1.7 m read through the FLOOR homography lands far behind the person
    — this is the same mechanism as a truncated box, in the opposite direction,
    and it is why the stature arm needs its own plane rather than a fudge factor.
    """
    vc, calib = cam
    truth = np.array([1.0, 1.0])
    px = _project_true(vc, (*truth, 1.7))
    naive = np.asarray(calib.ground.H, dtype=np.float64) @ np.array([px[0], px[1], 1.0])
    naive_xy = naive[:2] / naive[2]
    assert np.linalg.norm(naive_xy - truth) > 1.0


def test_sign_disambiguation_survives_a_negated_homography(cam):
    """A homography is defined up to scale INCLUDING sign.

    Negating it changes nothing about the floor mapping, so a decomposition that
    does not disambiguate looks perfectly healthy — and then sends every raised
    point the wrong way. Pinned because the failure is invisible at h=0.
    """
    vc, calib = cam
    flipped = calib.model_copy(deep=True)
    flipped.ground.H_img2world = (-np.asarray(calib.ground.H, dtype=np.float64)).tolist()

    truth = (2.0, -1.0)
    px = _project_true(vc, (*truth, 1.7))
    for c in (calib, flipped):
        world = plane_homography(c, 1.7) @ np.array([px[0], px[1], 1.0])
        assert np.allclose(world[:2] / world[2], truth, atol=1e-6)


# --- the estimators ---------------------------------------------------------


def test_bbox_arm_is_exactly_the_shipped_rule(cam):
    _, calib = cam
    boxes = [[100.0, 50.0, 200.0, 400.0], [300.0, 80.0, 360.0, 420.0]]
    got = BboxFootPoint()(boxes, calib)
    assert np.allclose(got.points_px, [[150.0, 400.0], [330.0, 420.0]])
    assert got.source == ("bbox", "bbox")
    assert got.fallback_fraction == 1.0


def test_stature_arm_recovers_the_true_ground_position_from_the_head(cam):
    """A box whose bottom edge is a lie, whose top edge is not."""
    vc, calib = cam
    truth = np.array([1.0, 0.5])
    head_px = _project_true(vc, (*truth, 1.70))
    foot_px = _project_true(vc, (*truth, 0.0))
    # Occluded from the knees down: the box stops halfway up the person.
    truncated = np.array(
        [[head_px[0] - 30, head_px[1], head_px[0] + 30, (head_px[1] + foot_px[1]) / 2]]
    )

    stature = StatureFootPoint(stature_m=1.70)(truncated, calib)
    bbox = BboxFootPoint()(truncated, calib)
    world_stature, _ = world_points(calib, stature)
    world_bbox, _ = world_points(calib, bbox)

    assert np.linalg.norm(world_stature[0] - truth) < 0.05
    assert np.linalg.norm(world_bbox[0] - truth) > 0.5


def test_pose_arm_uses_the_ankle_plane_not_the_floor(cam):
    """An ankle is ~9 cm off the floor and projecting it as a floor point is an
    error that grows with distance. Small, systematic, and free to remove."""
    vc, calib = cam
    truth = np.array([2.0, 2.0])
    ankle_px = _project_true(vc, (*truth, ANKLE_HEIGHT_M))
    boxes = np.array([[ankle_px[0] - 40, ankle_px[1] - 300, ankle_px[0] + 40, ankle_px[1]]])
    ankles = np.array([[[ankle_px[0], ankle_px[1], 0.9], [ankle_px[0], ankle_px[1], 0.9]]])

    corrected = PoseFootPoint(ankle_height_m=ANKLE_HEIGHT_M)(boxes, calib, ankles)
    uncorrected = PoseFootPoint(ankle_height_m=0.0)(boxes, calib, ankles)
    w_corrected, _ = world_points(calib, corrected)
    w_uncorrected, _ = world_points(calib, uncorrected)

    assert np.linalg.norm(w_corrected[0] - truth) < 0.01
    assert np.linalg.norm(w_uncorrected[0] - truth) > np.linalg.norm(w_corrected[0] - truth)


def test_pose_arm_takes_the_midpoint_of_two_confident_ankles(cam):
    _, calib = cam
    boxes = np.array([[100.0, 100.0, 200.0, 400.0]])
    ankles = np.array([[[120.0, 396.0, 0.9], [180.0, 400.0, 0.8]]])
    got = PoseFootPoint()(boxes, calib, ankles)
    assert got.source == ("pose",)
    # The estimate derives from (150, 398) — the midpoint — not from either ankle.
    assert got.confidence[0] == pytest.approx(0.85)


def test_one_confident_ankle_is_still_better_than_a_truncated_box(cam):
    _, calib = cam
    boxes = np.array([[100.0, 100.0, 200.0, 400.0]])
    ankles = np.array([[[120.0, 396.0, 0.9], [180.0, 400.0, 0.1]]])
    got = PoseFootPoint(min_conf=0.5)(boxes, calib, ankles)
    assert got.source == ("pose",)
    assert got.confidence[0] == pytest.approx(0.9)


def test_no_confident_ankle_degrades_to_the_box_and_says_so(cam):
    """Degrading is not failing — but it must be VISIBLE.

    An arm that quietly fell back on most of its detections has not been
    measured, it has been disguised. `source` is what stops that.
    """
    _, calib = cam
    boxes = np.array([[100.0, 100.0, 200.0, 400.0]])
    ankles = np.array([[[120.0, 396.0, 0.2], [180.0, 400.0, 0.1]]])
    got = PoseFootPoint(min_conf=0.5)(boxes, calib, ankles)
    assert got.source == ("bbox",)
    assert np.allclose(got.points_px, BboxFootPoint()(boxes).points_px)
    assert got.fallback_fraction == 1.0


def test_missing_keypoints_entirely_equals_the_baseline_exactly(cam):
    """The graceful-degradation contract, as an equality rather than a hope.

    If pose is unavailable for a whole run, the pose arm must produce the
    baseline's numbers exactly — otherwise a pose run with no keypoints reports
    a difference that is not a foot-point difference.
    """
    _, calib = cam
    boxes = np.array([[10.0, 20.0, 60.0, 300.0], [400.0, 100.0, 460.0, 500.0]])
    assert np.array_equal(
        PoseFootPoint()(boxes, calib, None).points_px, BboxFootPoint()(boxes).points_px
    )


def test_nan_keypoints_do_not_propagate(cam):
    _, calib = cam
    boxes = np.array([[100.0, 100.0, 200.0, 400.0]])
    ankles = np.array([[[np.nan, np.nan, 0.9], [np.nan, np.nan, 0.9]]])
    got = PoseFootPoint()(boxes, calib, ankles)
    assert got.source == ("bbox",)
    assert np.all(np.isfinite(got.points_px))


def test_empty_input_is_ordinary_not_exceptional(cam):
    _, calib = cam
    for name in ("bbox", "pose", "stature"):
        got = resolve_estimator(name)(np.zeros((0, 4)), calib, None)
        assert got.points_px.shape == (0, 2)
        assert got.fallback_fraction == 0.0


# --- resolution -------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "cls"),
    [("bbox", BboxFootPoint), ("pose", PoseFootPoint), ("stature", StatureFootPoint)],
)
def test_every_flag_value_resolves_to_its_estimator(name, cls):
    """One test per flag value. `context.md` records a whole live session lost to
    a flag that never reached the config it named."""
    est = resolve_estimator(name)
    assert isinstance(est, cls)
    assert est.name == name


def test_an_unknown_name_fails_loudly_and_lists_the_options():
    with pytest.raises(ValueError, match="unknown foot-point estimator"):
        resolve_estimator("ankles-please")


def test_resolver_forwards_its_tunables():
    est = resolve_estimator("pose", min_conf=0.7, ankle_height_m=0.12)
    assert (est.min_conf, est.ankle_height_m) == (0.7, 0.12)
    assert resolve_estimator("stature", stature_m=1.8).stature_m == 1.8


def test_footpoints_rejects_ragged_construction():
    with pytest.raises(ValueError, match="ragged"):
        FootPoints(np.zeros((2, 2)), ("bbox",), np.ones(2))
