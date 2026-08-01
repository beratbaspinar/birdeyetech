"""Tests for the uncalibrated multi-camera live session.

No camera, no GPU, no torch: the backend is injected and every frame is
synthetic, so the appearance-only fusion profile and the cross-view ledger are
exercised on CPU.

The load-bearing test here is `test_appearance_only_fuses_across_views_where_
geometry_gated_fusion_fragments`. Everything else pins a config value; that one
pins the *reason* the config exists, against a control that reproduces the
failure when the profile is off.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from mcreid.fusion.associate import INFEASIBLE, build_cost_matrix
from mcreid.fusion.global_id import FusionConfig
from mcreid.fusion.types import GlobalTrackSnapshot, TrackState, ViewObservation
from mcreid.live_multi import (
    GEOMETRY_FREE_M,
    CrossViewLedger,
    MultiLiveConfig,
    MultiLiveSession,
    appearance_only_fusion_config,
    uncalibrated_rig,
)

DIM = 16
SIZES = {"cam0": (1280, 720), "cam1": (640, 480)}


def _unit(seed: int) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=DIM)
    return np.asarray(vec / np.linalg.norm(vec), dtype=np.float64)


def _at_distance(
    base: npt.NDArray[np.float64], target: float, seed: int = 11
) -> npt.NDArray[np.float64]:
    """A unit vector exactly `target` cosine distance from `base`.

    Bisected rather than eyeballed. A fixture built by blending "somewhat toward
    another vector" lands wherever it lands — an earlier version of this file
    asked for 0.30 and got 0.095, so every test naming an operating point was
    quietly testing an easier one. `test_the_fixture_hits_the_distance_it_claims`
    is the guard.
    """
    other = _unit(seed)
    low, high = 0.0, 1.0
    for _ in range(80):
        weight = (low + high) / 2.0
        mixed = (1.0 - weight) * base + weight * other
        mixed = mixed / np.linalg.norm(mixed)
        if 1.0 - float(base @ mixed) < target:
            low = weight
        else:
            high = weight
    mixed = (1.0 - high) * base + high * other
    return np.asarray(mixed / np.linalg.norm(mixed), dtype=np.float64)


def _frames() -> dict[str, npt.NDArray[np.uint8]]:
    return {
        cid: np.full((height, width, 3), 90, dtype=np.uint8)
        for cid, (width, height) in SIZES.items()
    }


class _TwoCameraBackend:
    """One person, seen at OPPOSITE corners of two cameras.

    Opposite corners on purpose: that is what two uncalibrated cameras looking
    at the same room from different angles actually produce, and it is exactly
    the case a geometric gate gets wrong.
    """

    def __init__(self, cross_camera_distance: float = 0.30, cameras=("cam0", "cam1")) -> None:
        base = _unit(3)
        self.cameras = cameras
        self.cross_camera_distance = cross_camera_distance
        self.embeddings = {
            cameras[0]: base,
            cameras[1]: _at_distance(base, cross_camera_distance),
        }
        self.boxes = {
            cameras[0]: np.array([80.0, 120.0, 190.0, 520.0]),
            cameras[1]: np.array([430.0, 60.0, 540.0, 400.0]),
        }
        self.calls = 0

    def step(self, frames, frame: int) -> list[ViewObservation]:
        self.calls += 1
        out = []
        for index, camera_id in enumerate(self.cameras):
            if camera_id not in frames:
                continue
            out.append(
                ViewObservation(
                    camera_id=camera_id,
                    frame=frame,
                    local_track_id=index + 1,
                    bbox_xyxy=self.boxes[camera_id],
                    embedding=self.embeddings[camera_id],
                    score=0.9,
                )
            )
        return out


def _session(backend, fusion_config=None) -> MultiLiveSession:
    return MultiLiveSession(
        backend=backend,
        rig=uncalibrated_rig(SIZES),
        config=MultiLiveConfig(tile_height=180),
        fusion_config=fusion_config or appearance_only_fusion_config(),
    )


def _run(
    session: MultiLiveSession, steps: int, frames: dict[str, npt.NDArray[np.uint8]] | None = None
) -> dict[str, object]:
    frames = frames or _frames()
    info = {}
    for step in range(steps):
        _, info = session.process(frames, now=step * 0.05, dt=0.05)
    return info


# --- the appearance-only fusion profile --------------------------------------------------------


def test_geometry_is_suspended_by_a_scoped_flag_not_by_opening_every_radius():
    """The first version set every radius to 1e4 and was REJECTED for it: that
    removes the single-camera motion gate, which is sound, along with the
    cross-camera one, which is not. Measured cost of the global version on real
    crops: 63% same-camera identity theft, 0% for both the shipped config and
    this one."""
    base, config = FusionConfig(), appearance_only_fusion_config()
    assert config.cross_camera_geometry_open is True
    assert config.association.weight_geometry == 0.0
    assert config.association.weight_appearance == 1.0
    # Every radius that the rejected version opened globally stays SHIPPED.
    assert config.association.chi2_gate == base.association.chi2_gate
    assert config.association.max_distance_m == base.association.max_distance_m
    assert config.merge_radius_m == base.merge_radius_m
    assert config.revive_max_reach_m == base.revive_max_reach_m
    assert config.revive_speed_margin_m == base.revive_speed_margin_m
    # `_cluster` is the one path safe to open, because it already refuses two
    # observations from the SAME camera, so its radius only widens cross-camera.
    assert config.birth_cluster_radius_m == GEOMETRY_FREE_M


def test_the_sentinel_is_never_applied_to_a_gate_in_squared_units():
    """1e4 clears the ~12.2 pseudo-metre max separation by ~800x as a DISTANCE.
    Assigned to chi2_gate it compares a SQUARED Mahalanobis distance, where the
    margin measured 1.69x and inverted above span_m ~ 8 — the 'opened' gate
    started rejecting the cross-corner pairs it existed to accept."""
    config = appearance_only_fusion_config()
    assert config.association.chi2_gate != GEOMETRY_FREE_M
    assert config.association.max_distance_m != GEOMETRY_FREE_M


def test_the_measured_appearance_gates_are_left_alone():
    """The only evidence left is appearance. Loosening its gates here would be
    inventing numbers on the path with the least corroboration, not the most."""
    base, config = FusionConfig(), appearance_only_fusion_config()
    assert config.association.max_appearance_distance == base.association.max_appearance_distance
    assert config.merge_appearance_distance == base.merge_appearance_distance
    assert config.revive_appearance_distance == base.revive_appearance_distance
    assert config.dormant == base.dormant
    assert config.n_init == base.n_init


def test_the_cluster_veto_tightens_to_the_association_gate():
    """0.62 is justified by co-location evidence this rig does not have."""
    base, config = FusionConfig(), appearance_only_fusion_config()
    assert base.cluster_appearance_distance == pytest.approx(0.62)
    assert config.cluster_appearance_distance == config.association.max_appearance_distance
    assert config.cluster_appearance_distance < base.cluster_appearance_distance


def test_max_cost_is_raised_or_the_appearance_gate_silently_tightens():
    """With geometry weighted to zero the cost IS the normalised appearance
    distance, so the shipped 0.85 ceiling would move the gate 0.56 -> 0.476."""
    config = appearance_only_fusion_config().association
    gate = config.max_appearance_distance
    zero = np.zeros((1, 1))

    just_inside = build_cost_matrix(zero, zero, np.full((1, 1), gate - 1e-6), config)
    assert just_inside[0, 0] <= config.max_cost

    tightened = replace(config, max_cost=0.85)
    assert build_cost_matrix(zero, zero, np.full((1, 1), gate - 1e-6), tightened)[0, 0] > 0.85

    just_outside = build_cost_matrix(zero, zero, np.full((1, 1), gate + 1e-6), config)
    assert just_outside[0, 0] == INFEASIBLE


def test_geometry_cannot_reject_across_cameras_however_far_apart_they_land():
    """Now a property of the manager, not of the raw cost matrix: the gate still
    exists at its shipped value and is bypassed per pair, so it must be tested
    where the bypass lives."""
    from mcreid.fusion.global_id import GlobalIDManager

    manager = GlobalIDManager(uncalibrated_rig(SIZES), appearance_only_fusion_config())
    assert manager._geometry_abstains("cam0", {"cam1"})
    # ...and is NOT bypassed within one camera, whatever the separation.
    assert not manager._geometry_abstains("cam0", {"cam0"})
    config = appearance_only_fusion_config().association
    huge = np.full((1, 1), 1000.0)
    assert build_cost_matrix(huge, huge, np.zeros((1, 1)), config)[0, 0] == INFEASIBLE


# --- the uncalibrated rig ----------------------------------------------------------------------


def test_each_camera_gets_its_own_pixel_plane_at_its_own_resolution():
    rig = uncalibrated_rig(SIZES)
    assert rig.camera_ids == ["cam0", "cam1"]
    for camera_id, (width, height) in SIZES.items():
        assert rig.get(camera_id).intrinsics.image_size == (width, height)


def test_the_rig_says_in_writing_that_its_coordinates_are_not_shared():
    """A future reader must not be able to take a world_xy from this rig and
    compare it across cameras without tripping over the reason not to."""
    rig = uncalibrated_rig(SIZES)
    assert "not comparable between cameras" in rig.world_notes
    assert all("NOT metres" in cam.notes for cam in rig.cameras)


def test_an_empty_rig_is_refused():
    with pytest.raises(ValueError, match="at least one camera"):
        uncalibrated_rig({})


# --- cross-view identity, which is the whole point ----------------------------------------------


def test_appearance_only_fuses_across_views_where_geometry_gated_fusion_fragments():
    """The control is the point: the same sequence, the same embedder distance,
    and the ONLY difference is whether the geometric gates are open.

    Under the shipped geometry-gated config the two cameras' pixel planes place
    one person ~5 pseudo-metres apart, past every gate, so they are born as two
    identities and never merge. That is not a hypothetical — it is what this
    rig would do without the profile.
    """
    fused = _session(_TwoCameraBackend())
    _run(fused, steps=12)
    assert fused.ledger.multi_camera_ids, "no identity was ever supported by both cameras"
    assert len(fused.reported_ids) == 1, f"expected one identity, got {fused.reported_ids}"

    fragmented = _session(_TwoCameraBackend(), fusion_config=FusionConfig())
    _run(fragmented, steps=12)
    assert not fragmented.ledger.multi_camera_ids
    assert len(fragmented.reported_ids) == 2, (
        "the geometry-gated control must fragment, or this test is not measuring "
        "the profile"
    )


@pytest.mark.parametrize("distance", [0.05, 0.30, 0.50, 0.55])
def test_fusion_holds_up_to_the_association_gate_including_the_measured_operating_point(
    distance: float,
):
    """0.50 is the number that matters: the measured same-person cross-camera
    distance on real WILDTRACK crops with the shipped OSNet is 0.525 mean. A
    profile that only fused near-identical vectors would pass the 0.05 case and
    do nothing on this rig."""
    session = _session(_TwoCameraBackend(cross_camera_distance=distance))
    _run(session, steps=12)
    assert session.ledger.multi_camera_ids == [1]
    assert len(session.reported_ids) == 1


def test_a_stranger_in_the_second_view_is_not_fused_into_the_first():
    """Opening the geometric gates must not make appearance permissive too.

    0.70, not 0.90. Over 105,869 real different-person cross-camera pairs the
    MAXIMUM observed distance is 0.845 and p99 is 0.776, so a 0.90 "stranger"
    is a person who does not exist and the test was unfalsifiable. 0.70 sits
    inside the real distribution, above p95.
    """
    session = _session(_TwoCameraBackend(cross_camera_distance=0.70))
    _run(session, steps=12)
    assert not session.ledger.multi_camera_ids
    assert len(session.reported_ids) == 2


def test_the_easy_stranger_is_not_what_the_rig_actually_faces():
    """Guards the fixture above against drifting back to a comfortable number.

    The real different-person cross-camera mean is 0.623 and 20% of real
    strangers fall INSIDE the 0.56 association gate. That 20% is not defended
    by any threshold here — it is why the profile is single-occupant-only — and
    a test asserting otherwise would be claiming a property the measurement
    denies.
    """
    assert 0.623 < 0.70 < 0.845, "stranger fixture must sit inside the real range"


def test_the_boundary_is_the_association_gate_and_not_something_looser():
    """Just past 0.56 must fail, or the gate is not the thing deciding."""
    session = _session(_TwoCameraBackend(cross_camera_distance=0.60))
    _run(session, steps=12)
    assert not session.ledger.multi_camera_ids


def test_the_fixture_hits_the_distance_it_claims():
    """Without this, every operating point above is whatever bisection felt like."""
    for target in (0.05, 0.30, 0.50, 0.56, 0.90):
        backend = _TwoCameraBackend(cross_camera_distance=target)
        measured = 1.0 - float(backend.embeddings["cam0"] @ backend.embeddings["cam1"])
        assert measured == pytest.approx(target, abs=1e-6)


def test_a_camera_with_no_fresh_frame_simply_does_not_contribute():
    session = _session(_TwoCameraBackend())
    _run(session, steps=8)
    held = session.reported_ids
    frames = _frames()

    info = session.process({"cam0": frames["cam0"]}, now=1.0, dt=0.05)[1]
    assert info["cameras"] == ["cam0"]
    assert session.reported_ids == held, "dropping a camera must not mint an identity"
    assert session.frames_by_camera["cam0"] == 9
    assert session.frames_by_camera["cam1"] == 8


def test_process_refuses_an_empty_frame_set():
    session = _session(_TwoCameraBackend())
    with pytest.raises(ValueError, match="at least one camera"):
        session.process({}, now=0.0, dt=0.05)


def test_process_refuses_a_non_positive_dt():
    session = _session(_TwoCameraBackend())
    with pytest.raises(ValueError, match="dt must be positive"):
        session.process(_frames(), now=0.0, dt=0.0)


# --- the cross-view ledger, which is the acceptance evidence ------------------------------------


def _snapshot(global_id: int, cameras: tuple[str, ...]) -> GlobalTrackSnapshot:
    return GlobalTrackSnapshot(
        global_id=global_id,
        frame=0,
        world_xy=np.zeros(2),
        velocity_mps=np.zeros(2),
        covariance=np.eye(2),
        state=TrackState.CONFIRMED,
        supporting_cameras=cameras,
        frames_since_measurement=0,
        hits=5,
    )


def test_the_ledger_reports_a_new_cross_view_identity_exactly_once():
    ledger = CrossViewLedger()
    assert ledger.observe([_snapshot(1, ("cam0",))], frame=0) == []
    assert ledger.observe([_snapshot(1, ("cam0", "cam1"))], frame=1) == [1]
    assert ledger.observe([_snapshot(1, ("cam0", "cam1"))], frame=2) == []
    assert ledger.frames_multi[1] == 2
    assert ledger.first_multi_frame[1] == 1
    assert ledger.multi_camera_ids == [1]


def test_an_identity_seen_by_two_cameras_at_different_times_still_counts_as_cross_view():
    """Handoff — one camera then the other, never both at once — is a real pass.
    It just is not the *simultaneous* evidence, so the two are counted apart."""
    ledger = CrossViewLedger()
    ledger.observe([_snapshot(1, ("cam0",))], frame=0)
    ledger.observe([_snapshot(1, ("cam1",))], frame=1)
    assert ledger.multi_camera_ids == [1]
    assert ledger.frames_multi.get(1, 0) == 0


def test_the_report_names_the_cameras_and_the_verdict():
    session = _session(_TwoCameraBackend())
    _run(session, steps=12)
    report = "\n".join(session.cross_view_report())
    assert "CROSS-VIEW" in report
    assert "cam0" in report and "cam1" in report


def test_the_report_says_so_when_nothing_was_tracked():
    session = _session(_TwoCameraBackend())
    assert "nothing was tracked" in session.cross_view_report()[0]


# --- the mosaic --------------------------------------------------------------------------------


def test_the_mosaic_survives_cameras_of_different_resolutions():
    session = _session(_TwoCameraBackend())
    mosaic, _ = session.process(_frames(), now=0.0, dt=0.05)
    # Two tiles at tile_height plus the 40 px banner.
    assert mosaic.shape[0] == 180 + 40
    assert mosaic.shape[1] == int(round(1280 * 180 / 720)) + int(round(640 * 180 / 480))


def test_a_camera_with_no_frame_still_gets_a_tile():
    """Dropping the tile would reflow the mosaic every time a camera skipped a
    frame, which at 15 vs 30 FPS is most frames."""
    session = _session(_TwoCameraBackend())
    both, _ = session.process(_frames(), now=0.0, dt=0.05)
    one, _ = session.process({"cam0": _frames()["cam0"]}, now=0.05, dt=0.05)
    assert one.shape == both.shape


def test_more_than_two_cameras_tile_into_two_rows():
    sizes = {f"cam{i}": (640, 480) for i in range(4)}
    session = MultiLiveSession(
        backend=_TwoCameraBackend(cameras=("cam0", "cam1")),
        rig=uncalibrated_rig(sizes),
        config=MultiLiveConfig(tile_height=120),
    )
    frames = {cid: np.full((480, 640, 3), 90, dtype=np.uint8) for cid in sizes}
    mosaic, _ = session.process(frames, now=0.0, dt=0.05)
    assert mosaic.shape[0] == 120 * 2 + 40


# --- CLI argument plumbing ----------------------------------------------------------------------


def test_settings_broadcast_to_every_camera_or_are_given_one_each():
    from mcreid.cli.live_multi import broadcast

    assert broadcast("msmf", 3, "backend") == ["msmf"] * 3
    assert broadcast("msmf,dshow", 2, "backend") == ["msmf", "dshow"]


def test_a_partial_per_camera_list_is_refused_rather_than_padded():
    """Padding would silently give camera 3 camera 1's frame rate, and the
    resulting recording header would be wrong with nothing to show for it."""
    import typer

    from mcreid.cli.live_multi import broadcast

    with pytest.raises(typer.BadParameter, match="1 value or exactly 3"):
        broadcast("30,15", 3, "nominal-fps")


def test_specs_are_named_cam0_upward_in_the_order_given():
    from mcreid.cli.live_multi import build_specs

    specs = build_specs([2, 0], 1280, 720, "MJPG", "30,15", "msmf,dshow")
    assert [s.camera_id for s in specs] == ["cam0", "cam1"]
    assert [s.device for s in specs] == [2, 0]
    assert [s.nominal_fps for s in specs] == [30.0, 15.0]
    assert [s.backend for s in specs] == ["msmf", "dshow"]


def test_duplicate_device_indices_are_refused():
    import typer

    from mcreid.cli.live_multi import parse_devices

    assert parse_devices("0, 1") == [0, 1]
    with pytest.raises(typer.BadParameter, match="duplicate device index"):
        parse_devices("0,0")


def test_a_partial_fourcc_never_puts_nul_bytes_on_stdout():
    """Found by grep refusing to read the probe's own output: a device that
    reports a partial FOURCC yields NUL bytes, and printing those turns the
    whole run's stdout into a binary stream."""
    from mcreid.cli.live_multi import _fourcc_of

    class _Capture:
        def __init__(self, raw: int) -> None:
            self.raw = raw

        def get(self, prop: int) -> float:
            return float(self.raw)

    # 'YUY2' little-endian, and a value whose upper bytes are zero.
    assert _fourcc_of(_Capture(0x32595559)) == "YUY2"
    partial = _fourcc_of(_Capture(0x00005559))
    assert "\x00" not in partial and partial == "YU"
    assert _fourcc_of(_Capture(0)) == "?"
    assert _fourcc_of(_Capture(-1)) == "?"


# --- diagnostics ---------------------------------------------------------------------------------


class _BlackoutBackend(_TwoCameraBackend):
    """Both cameras lose the person for a window, then get them back."""

    def __init__(self, window: tuple[int, int]) -> None:
        super().__init__()
        self.window = window

    def step(self, frames, frame: int) -> list[ViewObservation]:
        if self.window[0] <= frame < self.window[1]:
            return []
        return super().step(frames, frame)


def test_the_shadow_probe_needs_no_multi_camera_changes(tmp_path):
    """Claimed cheap on this path because ShadowProbe reads the fused manager
    (last_ground, last_assignment, dormant) and never touches a camera. Measured
    rather than asserted: a full leave/return through a two-camera rig has to
    produce rows, a written pair of files, and a resurrection."""
    from mcreid.diagnostics.shadow import ShadowProbe, summarise

    config = appearance_only_fusion_config()
    shadow = ShadowProbe(
        tmp_path / "shadow", gate=config.dormant.appearance_distance, top_k=config.dormant.top_k
    )
    session = MultiLiveSession(
        backend=_BlackoutBackend((30, 460)),
        rig=uncalibrated_rig(SIZES),
        config=MultiLiveConfig(tile_height=120),
        fusion_config=config,
        shadow=shadow,
    )
    _run(session, steps=520)

    assert session.manager.dormant.n_resurrected == 1
    assert session.reported_ids == [1], "the identity must survive the absence"
    assert session.ledger.multi_camera_ids == [1]
    assert shadow.rows, "a leave/return through two cameras produced no probe rows"

    jsonl, csv_path = shadow.write()
    assert jsonl.stat().st_size > 0 and csv_path.stat().st_size > 0
    assert any("measurements" in line for line in summarise(shadow.rows, shadow.gate))


def test_shadow_rows_carry_both_query_kinds_on_the_multi_camera_path():
    """The obs query is what _resurrect probes with, the track EMA what
    _adopt_dormant_identity uses. Losing either on this path would make a
    multi-camera session's log not comparable with s1/s3."""
    from mcreid.diagnostics.shadow import QUERY_OBS, QUERY_TRACK_EMA, ShadowProbe

    config = appearance_only_fusion_config()
    shadow = ShadowProbe(
        Path("unused"), gate=config.dormant.appearance_distance, top_k=config.dormant.top_k
    )
    session = MultiLiveSession(
        backend=_BlackoutBackend((30, 460)),
        rig=uncalibrated_rig(SIZES),
        config=MultiLiveConfig(tile_height=120),
        fusion_config=config,
        shadow=shadow,
    )
    _run(session, steps=520)
    sources = {row.source for row in shadow.rows}
    assert sources == {QUERY_OBS, QUERY_TRACK_EMA}



# --- the regime the review found untested: more than one person -------------------------------


class _TwoPeopleBackend:
    """One person per camera — the case the whole profile is unsafe in."""

    def __init__(self, between_people: float) -> None:
        first = _unit(3)
        self.embeddings = {"cam0": first, "cam1": _at_distance(first, between_people)}
        self.boxes = {
            "cam0": np.array([80.0, 120.0, 190.0, 520.0]),
            "cam1": np.array([430.0, 60.0, 540.0, 400.0]),
        }

    def step(self, frames, frame: int) -> list[ViewObservation]:
        return [
            ViewObservation(
                camera_id=camera_id,
                frame=frame,
                local_track_id=index + 1,
                bbox_xyxy=self.boxes[camera_id],
                embedding=self.embeddings[camera_id],
                score=0.9,
            )
            for index, camera_id in enumerate(("cam0", "cam1"))
            if camera_id in frames
        ]


def test_two_people_one_per_camera_are_kept_apart_at_the_measured_mean():
    """The deliverable's own untested regime. 0.623 is the measured
    different-person cross-camera mean on real WILDTRACK crops."""
    session = _session(_TwoPeopleBackend(between_people=0.623))
    _run(session, steps=30)
    assert len(session.reported_ids) == 2, "two people collapsed into one identity"
    assert not session.ledger.multi_camera_ids


def test_the_profile_is_documented_as_unsafe_for_two_people():
    """This is the honest half. At the measured mean the synthetic case above
    holds, but on REAL crops the merge tests EMA-to-EMA vectors whose
    different-person mean is 0.456 — inside the strict 0.48 gate — and two
    people fuse 76.7% of the time. No threshold in this file fixes that, so the
    constraint is carried as a stated precondition instead of a silent one."""
    from mcreid.live_multi import SINGLE_OCCUPANT_WARNING

    assert "ONE occupant" in SINGLE_OCCUPANT_WARNING
    assert "76.7%" in SINGLE_OCCUPANT_WARNING
    assert "0.48 gate" in SINGLE_OCCUPANT_WARNING


# --- geometry abstains between cameras, and ONLY between cameras ------------------------------


class _SameCameraStrangerBackend:
    """An identity coasts out of cam0; a DIFFERENT person walks into cam0.

    The attack that broke the first version of this profile. No second camera is
    involved: opening the radii globally removed the single-camera motion gate,
    which is sound evidence, to repair the cross-camera one, which is not.
    """

    def __init__(self, present: int = 12, gap: int = 20) -> None:
        self.person = _unit(3)
        self.stranger = _at_distance(self.person, 0.623)
        self.present, self.gap = present, gap

    def step(self, frames, frame: int) -> list[ViewObservation]:
        if frame < self.present:
            box, emb, tid = np.array([100.0, 200.0, 200.0, 600.0]), self.person, 1
        elif frame < self.present + self.gap:
            return []
        else:
            box, emb, tid = np.array([1000.0, 200.0, 1100.0, 600.0]), self.stranger, 2
        return [
            ViewObservation(
                camera_id="cam0", frame=frame, local_track_id=tid,
                bbox_xyxy=box, embedding=emb, score=0.9,
            )
        ]


def test_a_stranger_does_not_inherit_an_identity_inside_one_camera():
    """Measured on real crops: 63% theft when the radii were opened globally,
    0% with the exemption scoped to cross-camera pairs, 0% shipped."""
    session = _session(_SameCameraStrangerBackend())
    _run(session, steps=45)
    assert len(session.reported_ids) == 2, (
        f"the stranger took the identity: {session.reported_ids}"
    )


def test_geometry_abstains_only_between_different_single_camera_planes():
    """The rule, stated directly. An estimate fed by two cameras lives in
    neither plane, so geometry abstains on it too — keying on whether camera
    HISTORIES overlap instead re-enabled the gate on exactly those blended
    estimates, and blocked genuine merges at 0.27 appearance distance."""
    from mcreid.fusion.global_id import GlobalIDManager

    manager = GlobalIDManager(uncalibrated_rig(SIZES), appearance_only_fusion_config())
    assert manager._pixel_plane(["cam0"]) == "cam0"
    assert manager._pixel_plane(["cam0", "cam1"]) is None
    assert manager._pixel_plane([]) is None
    # `_dormant` is a pseudo-camera from gallery seeding, not a sensor.
    assert manager._pixel_plane(["cam0", "_dormant"]) == "cam0"

    assert not manager._geometry_abstains("cam0", {"cam0"})
    assert manager._geometry_abstains("cam0", {"cam1"})
    assert manager._geometry_abstains("cam0", {"cam0", "cam1"})


def test_the_shipped_config_never_abstains():
    """The WILDTRACK path has a real homography and must be untouched."""
    from mcreid.fusion.global_id import FusionConfig, GlobalIDManager

    manager = GlobalIDManager(uncalibrated_rig(SIZES), FusionConfig())
    assert not manager._geometry_abstains("cam0", {"cam1"})
    assert not FusionConfig().cross_camera_geometry_open
