"""Tests for threaded multi-camera capture and the raw recording.

No camera: `CameraStream` takes an opener, so a fake `VideoCapture` drives the
whole thread, the recorder and the latest-wins handoff.

The recording is the part that must not be allowed to rot quietly. A session
where it silently wrote nothing looks identical, at the console, to one where it
worked — which is why `n_written` and the timestamp CSV are asserted rather than
the absence of an exception.
"""

from __future__ import annotations

import csv
import threading
import time
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

from mcreid.capture import (
    CAPTURE_BACKENDS,
    DEFAULT_BACKEND,
    MP4V_BYTES_PER_PIXEL_FRAME,
    CameraRig,
    CameraSpec,
    CameraStream,
    StreamRecorder,
    build_rig,
    check_disk_space,
    estimated_bytes_per_second,
)

SPEC = CameraSpec(camera_id="cam0", device=0, width=64, height=48, nominal_fps=30.0)


class _FakeCapture:
    """Yields distinct frames on demand, then blocks until released."""

    def __init__(
        self, n_frames: int = 1000, size: tuple[int, int] = (48, 64), fail_every: int = 0
    ) -> None:
        self.n_frames = n_frames
        self.size = size
        self.fail_every = fail_every
        self.served = 0
        self.released = False
        self._gate = threading.Semaphore(0)

    def allow(self, n: int = 1) -> None:
        for _ in range(n):
            self._gate.release()

    def read(self) -> tuple[bool, npt.NDArray[np.uint8] | None]:
        if not self._gate.acquire(timeout=1.0):
            return False, None
        self.served += 1
        if self.fail_every and self.served % self.fail_every == 0:
            return False, None
        # Frame content encodes its index, so a test can tell frames apart.
        frame = np.full((*self.size, 3), self.served % 256, dtype=np.uint8)
        return True, frame

    def release(self) -> None:
        self.released = True


def _stream(capture: _FakeCapture, recorder=None, spec: CameraSpec = SPEC) -> CameraStream:
    return CameraStream(spec, recorder=recorder, opener=lambda _: capture)


def _wait_for(predicate: Callable[[], bool], timeout: float = 3.0) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


# --- CameraSpec validation ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"camera_id": ""}, "camera_id"),
        ({"device": -1}, "device index"),
        ({"width": 0}, "invalid requested size"),
        ({"nominal_fps": 0.0}, "nominal_fps"),
        ({"fourcc": "MJP"}, "fourcc"),
    ],
)
def test_bad_camera_specs_fail_at_construction(kwargs, match):
    base = {"camera_id": "cam0", "device": 0}
    with pytest.raises(ValueError, match=match):
        CameraSpec(**{**base, **kwargs})


# --- latest-wins handoff -----------------------------------------------------------------------


def test_latest_returns_a_frame_once_and_then_reports_nothing_new():
    capture = _FakeCapture()
    stream = _stream(capture)
    stream.start()
    try:
        capture.allow(1)
        assert _wait_for(lambda: stream.stats.frames_read >= 1)
        first = stream.latest()
        assert first is not None
        assert stream.latest() is None, "a frame must not be processed twice"
    finally:
        stream.stop()


def test_the_processing_loop_gets_the_newest_frame_not_the_oldest():
    """A queue here would grow a lag between what the person does and what the
    overlay shows. Latest-wins is the design; this pins it."""
    capture = _FakeCapture()
    stream = _stream(capture)
    stream.start()
    try:
        capture.allow(5)
        assert _wait_for(lambda: stream.stats.frames_read >= 5)
        frame = stream.latest()
        assert frame is not None
        assert frame.seq == 4, f"got seq {frame.seq}, expected the newest"
    finally:
        stream.stop()


def test_capture_keeps_running_at_full_rate_while_nobody_consumes():
    """The recording must not be throttled by the processing loop — that is the
    entire reason capture lives on its own thread."""
    capture = _FakeCapture()
    stream = _stream(capture)
    stream.start()
    try:
        capture.allow(40)
        assert _wait_for(lambda: stream.stats.frames_read >= 40)
        assert stream.stats.frames_read == 40
    finally:
        stream.stop()


def test_a_failed_read_is_counted_and_does_not_stop_the_stream():
    capture = _FakeCapture(fail_every=3)
    stream = _stream(capture)
    stream.start()
    try:
        capture.allow(9)
        assert _wait_for(lambda: stream.stats.frames_read >= 6)
        assert stream.stats.frames_failed == 3
        assert stream.error is None
    finally:
        stream.stop()


def test_a_dead_capture_thread_surfaces_instead_of_looking_like_an_empty_room():
    """The failure this rig cannot afford to misread: no detections because the
    camera died, reported as no detections because nobody walked in."""
    capture = _FakeCapture(fail_every=1)
    stream = _stream(capture)
    stream.start()
    try:
        capture.allow(31)
        assert _wait_for(lambda: stream.error is not None)
        rig = CameraRig([stream])
        with pytest.raises(RuntimeError, match="capture thread for cam0 died"):
            rig.raise_for_errors()
    finally:
        stream.stop()


def test_starting_twice_is_refused():
    capture = _FakeCapture()
    stream = _stream(capture)
    stream.start()
    try:
        with pytest.raises(RuntimeError, match="already started"):
            stream.start()
    finally:
        stream.stop()


def test_stop_releases_the_device():
    capture = _FakeCapture()
    stream = _stream(capture)
    stream.start()
    stream.stop()
    assert capture.released


# --- recording ---------------------------------------------------------------------------------


def test_every_captured_frame_is_recorded_with_a_timestamp_row(tmp_path):
    capture = _FakeCapture()
    recorder = StreamRecorder(
        tmp_path / "cam0.mp4", tmp_path / "cam0_timestamps.csv", fps=30.0
    )
    stream = _stream(capture, recorder=recorder)
    stream.start()
    try:
        capture.allow(25)
        assert _wait_for(lambda: stream.stats.frames_read >= 25)
    finally:
        stream.stop()

    assert recorder.n_written == 25
    assert (tmp_path / "cam0.mp4").stat().st_size > 0

    with (tmp_path / "cam0_timestamps.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 25, "a frame without a timestamp cannot be synced offline"
    assert [int(row["seq"]) for row in rows] == list(range(25))
    assert all(int(row["width"]) == 64 and int(row["height"]) == 48 for row in rows)
    times = [float(row["t_capture_s"]) for row in rows]
    assert times == sorted(times), "capture timestamps must be monotonic"
    assert all(float(row["t_wall_unix"]) > 1.7e9 for row in rows)


def test_the_recorder_reports_the_size_the_camera_actually_delivered(tmp_path):
    """A camera that downgrades 720p to VGA must not produce a file whose header
    claims 720p — the offline pass would letterbox every frame."""
    capture = _FakeCapture(size=(48, 64))
    spec = CameraSpec(camera_id="cam0", device=0, width=1280, height=720)
    recorder = StreamRecorder(tmp_path / "c.mp4", tmp_path / "c.csv", fps=30.0)
    stream = _stream(capture, recorder=recorder, spec=spec)
    stream.start()
    try:
        capture.allow(3)
        assert _wait_for(lambda: recorder.n_written >= 3)
    finally:
        stream.stop()
    with (tmp_path / "c.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert all(int(row["width"]) == 64 for row in rows)


def test_a_recorder_with_no_frames_closes_cleanly(tmp_path):
    recorder = StreamRecorder(tmp_path / "c.mp4", tmp_path / "c.csv", fps=30.0)
    recorder.close()
    assert recorder.n_written == 0


def test_a_zero_fps_recording_is_refused(tmp_path):
    with pytest.raises(ValueError, match="fps must be positive"):
        StreamRecorder(tmp_path / "c.mp4", tmp_path / "c.csv", fps=0.0)


# --- disk pre-flight ---------------------------------------------------------------------------


def test_the_disk_estimate_scales_with_pixels_and_rate():
    small = [CameraSpec(camera_id="a", device=0, width=640, height=480, nominal_fps=15.0)]
    big = [CameraSpec(camera_id="a", device=0, width=1280, height=720, nominal_fps=30.0)]
    assert estimated_bytes_per_second(small) == pytest.approx(
        640 * 480 * 15 * MP4V_BYTES_PER_PIXEL_FRAME
    )
    assert estimated_bytes_per_second(big) > 5 * estimated_bytes_per_second(small)


def test_the_estimate_has_headroom_over_this_projects_own_measured_bitrate():
    """0.0250 and 0.0257 bytes/pixel/frame, measured on reports/live_*.mp4. An
    estimate below the measurement would let a session run out of disk."""
    assert MP4V_BYTES_PER_PIXEL_FRAME > 0.0257


def test_the_disk_check_refuses_a_session_that_would_not_fit(tmp_path):
    absurd = [
        CameraSpec(camera_id=f"c{i}", device=i, width=4096, height=4096, nominal_fps=120.0)
        for i in range(8)
    ]
    needed_mb, free_mb, ok = check_disk_space(tmp_path, absurd, minutes=600.0)
    assert needed_mb > free_mb
    assert not ok


def test_a_modest_session_passes_the_disk_check(tmp_path):
    needed_mb, free_mb, ok = check_disk_space(tmp_path, [SPEC], minutes=1.0)
    assert needed_mb < 10.0
    assert ok is (free_mb - needed_mb > 5000.0)


def test_a_non_positive_duration_is_refused(tmp_path):
    with pytest.raises(ValueError, match="minutes must be positive"):
        check_disk_space(tmp_path, [SPEC], minutes=0.0)


# --- the rig -----------------------------------------------------------------------------------


def test_the_rig_hands_back_only_cameras_with_a_fresh_frame():
    captures = {"cam0": _FakeCapture(), "cam1": _FakeCapture()}
    specs = [
        CameraSpec(camera_id=cid, device=i, width=64, height=48)
        for i, cid in enumerate(captures)
    ]
    streams = [CameraStream(s, opener=lambda spec: captures[spec.camera_id]) for s in specs]
    rig = CameraRig(streams)
    rig.start()
    try:
        captures["cam0"].allow(2)
        assert _wait_for(lambda: streams[0].stats.frames_read >= 2)
        fresh = rig.latest()
        assert set(fresh) == {"cam0"}, "cam1 has produced nothing and must be absent"

        captures["cam1"].allow(1)
        assert _wait_for(lambda: streams[1].stats.frames_read >= 1)
        assert set(rig.latest()) == {"cam1"}, "cam0's frame was already consumed"
    finally:
        rig.stop()


def test_a_camera_that_fails_to_open_releases_the_ones_already_started():
    """USB bandwidth runs out on camera 2, not camera 1. Leaving camera 1 held
    makes the next attempt fail for a different reason than the real one."""
    good = _FakeCapture()

    def opener(spec: CameraSpec) -> _FakeCapture:
        if spec.camera_id == "cam1":
            raise RuntimeError("USB bandwidth exceeded")
        return good

    specs = [CameraSpec(camera_id=f"cam{i}", device=i, width=64, height=48) for i in range(2)]
    rig = CameraRig([CameraStream(s, opener=opener) for s in specs])
    with pytest.raises(RuntimeError, match="USB bandwidth"):
        rig.start()
    assert good.released, "the camera that did open was never released"


def test_duplicate_camera_ids_are_refused():
    specs = [CameraSpec(camera_id="cam0", device=i, width=64, height=48) for i in range(2)]
    with pytest.raises(ValueError, match="unique"):
        CameraRig([CameraStream(s, opener=lambda _: _FakeCapture()) for s in specs])


def test_an_empty_rig_is_refused():
    with pytest.raises(ValueError, match="at least one camera"):
        CameraRig([])


def test_build_rig_names_recordings_per_camera_and_session(tmp_path):
    specs = [CameraSpec(camera_id=f"cam{i}", device=i, width=64, height=48) for i in range(2)]
    rig = build_rig(specs, tmp_path, "20260801_120000", opener=lambda _: _FakeCapture())
    paths = [s.recorder.video_path.name for s in rig.streams]
    assert paths == ["20260801_120000_cam0.mp4", "20260801_120000_cam1.mp4"]
    stamps = [s.recorder.timestamps_path.name for s in rig.streams]
    assert stamps == [
        "20260801_120000_cam0_timestamps.csv",
        "20260801_120000_cam1_timestamps.csv",
    ]


def test_build_rig_without_a_directory_records_nothing(tmp_path):
    specs = [CameraSpec(camera_id="cam0", device=0, width=64, height=48)]
    rig = build_rig(specs, None, "s", opener=lambda _: _FakeCapture())
    assert rig.streams[0].recorder is None


def test_measured_fps_is_computed_from_capture_timestamps():
    capture = _FakeCapture()
    stream = _stream(capture)
    assert stream.stats.measured_fps == 0.0  # nothing captured yet
    stream.start()
    try:
        capture.allow(10)
        assert _wait_for(lambda: stream.stats.frames_read >= 10)
    finally:
        stream.stop()
    assert stream.stats.measured_fps > 0.0


# --- capture backend selection ------------------------------------------------------------------


def test_the_default_backend_is_msmf_on_measured_evidence():
    """Not a preference. On this rig the Microsoft LifeCam VX-2000 negotiates
    5.0 FPS through dshow and 14.9 through msmf, at 640x480, 320x240 AND
    160x120 — so it is which media type each backend picks, not bandwidth."""
    assert DEFAULT_BACKEND == "msmf"
    assert CameraSpec(camera_id="cam0", device=0).backend == "msmf"


def test_every_named_backend_maps_to_a_real_opencv_constant():
    import cv2

    assert CAPTURE_BACKENDS["msmf"] == cv2.CAP_MSMF
    assert CAPTURE_BACKENDS["dshow"] == cv2.CAP_DSHOW
    assert CAPTURE_BACKENDS["any"] == cv2.CAP_ANY


def test_an_unknown_backend_is_refused_at_construction():
    with pytest.raises(ValueError, match="unknown backend"):
        CameraSpec(camera_id="cam0", device=0, backend="v4l2")


def test_the_backend_is_per_camera_not_global():
    """Two devices on one machine do not necessarily have the same best backend."""
    fast = CameraSpec(camera_id="cam0", device=0, backend="msmf")
    legacy = CameraSpec(camera_id="cam1", device=1, backend="dshow")
    assert (fast.backend, legacy.backend) == ("msmf", "dshow")
