"""Threaded multi-camera capture with unconditional raw recording.

Two things this module exists to keep apart:

**Capture rate is not processing rate.** Two USB cameras on one laptop do not
run at the same rate, and neither runs at the rate the detector can consume. A
single-threaded `read()` loop over N cameras runs at the *slowest* camera's
rate — measured on this rig, two cameras that individually deliver 30 and 15 FPS
read at 14.3 combined, because `read()` blocks. One thread per camera decouples
them and each device keeps its own rate.

**The recording is the deliverable, not the live view.** Detection may drop the
session to single-digit FPS; the recording must not care. Each capture thread
writes every frame it reads to its own file, alongside a per-frame timestamp
row, before the processing loop has any say. A session is therefore replayable
offline at full sensor rate no matter what the live loop managed.

Frames are handed to the processing loop *latest-wins*: `latest()` returns the
newest frame once and then reports empty until a new one arrives, so the
processing loop never works on a stale frame and never processes one twice.
"""

from __future__ import annotations

import csv
import shutil
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import numpy.typing as npt

from mcreid.utils.logging import get_logger

logger = get_logger(__name__)

Image = npt.NDArray[np.uint8]

# Measured on this project's own webcam clips: reports/live_*.mp4 are real
# 1280x760 webcam footage at 0.0250 and 0.0257 bytes per pixel per frame under
# the mp4v encoder. 0.030 is that with ~20 % headroom, so the disk estimate
# errs toward refusing a session rather than filling the disk mid-run.
MP4V_BYTES_PER_PIXEL_FRAME = 0.030

# Windows capture backends, by name. MSMF is the default on measured evidence,
# not on preference: on this rig's Microsoft LifeCam VX-2000, MSMF delivers
# 30.4 FPS and DSHOW 14.6 — a ~2x gap at every resolution tried, including
# 160x120, so it is which media type each backend negotiates and not bandwidth.
# The integrated camera is 30 FPS on either, and both cameras hold full rate
# together under MSMF. DSHOW is kept because some devices only enumerate there.
#
# BEWARE THE OTHER VARIABLE. An earlier measurement of this same camera read
# 15.1 (MSMF) and 5.0 (DSHOW) and was written up as a hardware ceiling. It was
# not: the camera was pointed at ceiling lights, saturated (mean luma 248), and
# its auto-exposure had roughly halved BOTH backends. Aimed at the room (luma
# 117) both roughly double. The backend gap is real and reproducible; any
# absolute FPS figure for this camera is only valid with the scene stated.
CAPTURE_BACKENDS: dict[str, int] = {
    "msmf": cv2.CAP_MSMF,
    "dshow": cv2.CAP_DSHOW,
    "any": cv2.CAP_ANY,
}
DEFAULT_BACKEND = "msmf"

# Consecutive failed reads before a stream is declared dead. The previous guard
# required `frames_read == 0` as well, so it could only ever fire BEFORE the
# first successful frame: a camera unplugged mid-session busy-spun forever and
# `raise_for_errors()` never raised. `frames_failed` is cumulative and a camera
# may drop the odd frame legitimately, so the counter that matters is the run of
# failures since the last success.
MAX_CONSECUTIVE_FAILURES = 30


class VideoCaptureLike(Protocol):
    """The slice of `cv2.VideoCapture` this module uses."""

    def read(self) -> tuple[bool, Any]: ...
    def release(self) -> None: ...


@dataclass(frozen=True)
class CameraSpec:
    """One physical camera and what we ask of it."""

    camera_id: str
    device: int
    width: int = 1280
    height: int = 720
    fourcc: str = "MJPG"
    """Requested pixel format. Ignored by cameras that only offer YUY2 — the
    negotiated format is reported at open time, never assumed."""
    nominal_fps: float | None = 30.0
    """Container-header rate, or None to MEASURE it from the first frames.

    None is the better default for an unfamiliar camera and is what the CLI
    passes for `--nominal-fps auto`. The timestamp CSV is authoritative either
    way, but a video that needs a sidecar to play at the right speed is one most
    people will play at the wrong speed."""
    backend: str = DEFAULT_BACKEND
    """Capture backend by name. Per camera, because two devices on one machine
    do not necessarily negotiate their best mode on the same one."""

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ValueError("camera_id must be non-empty")
        if self.device < 0:
            raise ValueError(f"device index must be >= 0, got {self.device}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"invalid requested size {self.width}x{self.height}")
        if self.nominal_fps is not None and self.nominal_fps <= 0:
            raise ValueError(f"nominal_fps must be positive, got {self.nominal_fps}")
        if len(self.fourcc) != 4:
            raise ValueError(f"fourcc must be 4 characters, got {self.fourcc!r}")
        if self.backend not in CAPTURE_BACKENDS:
            raise ValueError(
                f"unknown backend {self.backend!r}; available: {sorted(CAPTURE_BACKENDS)}"
            )


@dataclass(frozen=True)
class CapturedFrame:
    """One frame plus everything needed to place it on a timeline later."""

    camera_id: str
    image: Image
    seq: int
    t_capture: float
    """Monotonic clock (perf_counter), for intervals within the session."""
    t_wall: float
    """Unix time, for lining this session up against anything outside it."""


def estimated_bytes_per_second(specs: Sequence[CameraSpec]) -> float:
    """Disk rate the raw recordings will consume, all cameras together."""
    # An unmeasured camera is budgeted at 30 fps: the estimate exists to refuse
    # a session that would not fit, so guessing high is the safe direction.
    return sum(
        spec.width * spec.height * (spec.nominal_fps or 30.0) * MP4V_BYTES_PER_PIXEL_FRAME
        for spec in specs
    )


def check_disk_space(
    out_dir: Path, specs: Sequence[CameraSpec], minutes: float, reserve_gb: float = 5.0
) -> tuple[float, float, bool]:
    """(estimated MB, free MB, ok) for a `minutes`-long recording.

    `reserve_gb` is headroom left for the rest of the machine. Windows gets
    unpleasant well before a volume is genuinely full, and a capture session is
    not worth finding that boundary.
    """
    if minutes <= 0:
        raise ValueError(f"minutes must be positive, got {minutes}")
    out_dir.mkdir(parents=True, exist_ok=True)
    needed_mb = estimated_bytes_per_second(specs) * minutes * 60.0 / 1e6
    free_mb = shutil.disk_usage(out_dir).free / 1e6
    return needed_mb, free_mb, free_mb - needed_mb > reserve_gb * 1000.0


class StreamRecorder:
    """Writes one camera's raw frames plus a per-frame timestamp CSV.

    Opened lazily on the first frame, because the negotiated frame size is only
    known once a frame has actually arrived — a camera that silently downgrades
    720p to VGA would otherwise produce a video whose header lies about it.
    """

    def __init__(
        self,
        video_path: Path,
        timestamps_path: Path,
        fps: float | None = None,
        calibration_frames: int = 20,
    ) -> None:
        """`fps=None` measures the real rate from the first `calibration_frames`.

        Measuring beats declaring, because a declared rate is wrong exactly when
        it matters. This rig's USB camera was believed to top out at 15 FPS and
        actually delivers 30 under Media Foundation, so a session recorded with
        `--nominal-fps 15` plays back at half speed with nothing in the file
        saying so. The timestamp CSV was always authoritative, but a video that
        needs a sidecar to play correctly is a video most people will play
        wrongly.

        The calibration frames are held in memory and flushed once the rate is
        known, so nothing is dropped: 20 frames of 720p is ~55 MB, transient,
        and one camera at a time.
        """
        if fps is not None and fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        if calibration_frames < 2:
            raise ValueError("need at least 2 frames to measure a rate")
        self.video_path = Path(video_path)
        self.timestamps_path = Path(timestamps_path)
        self.fps = fps
        self.declared_fps = fps
        """What the caller asked for, or None. `fps` becomes the measured rate."""
        self.calibration_frames = calibration_frames
        self.n_written = 0
        self._writer: Any = None
        self._csv: Any = None
        self._csv_writer: Any = None
        self._size: tuple[int, int] | None = None
        self._pending: list[CapturedFrame] = []

    def _measure(self, frames: list[CapturedFrame]) -> float:
        """Rate from the MEDIAN inter-frame interval, not the endpoint span.

        The first frames out of a freshly opened camera arrive faster than the
        steady state — they were already buffered — so a span estimate reads
        high. Measured on this rig: 33.2 fps from the span against a 27.0 fps
        session mean. The median ignores that head, and one stalled frame with
        it, at no extra memory.
        """
        intervals = [
            b.t_capture - a.t_capture
            # strict=False is the point: pairing consecutive frames means the
            # second sequence is deliberately one shorter.
            for a, b in zip(frames, frames[1:], strict=False)
            if b.t_capture > a.t_capture
        ]
        if not intervals:
            return 30.0
        return 1.0 / float(np.median(intervals))

    def _open(self, image: Image) -> None:
        assert self.fps is not None, "_open called before the rate was known"
        height, width = image.shape[:2]
        self._size = (width, height)
        self.video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(self.video_path), cv2.VideoWriter.fourcc(*"mp4v"), self.fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"could not open video writer for {self.video_path}")
        self._writer = writer
        self._csv = self.timestamps_path.open("w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv)
        self._csv_writer.writerow(["seq", "t_capture_s", "t_wall_unix", "width", "height"])
        logger.info("recording %dx%d @ %.1f fps -> %s", width, height, self.fps, self.video_path)

    def write(self, frame: CapturedFrame) -> None:
        if self._writer is None:
            if self.fps is None:
                # Hold frames until the rate is known, then flush. Dropping them
                # instead would silently lose the first second of every session.
                self._pending.append(frame)
                if len(self._pending) < self.calibration_frames:
                    return
                self.fps = self._measure(self._pending)
                logger.info(
                    "measured %.1f fps over %d frames -> %s",
                    self.fps,
                    len(self._pending),
                    self.video_path.name,
                )
                buffered, self._pending = self._pending, []
                self._open(buffered[0].image)
                for held in buffered:
                    self._write_open(held)
                return
            self._open(frame.image)
        self._write_open(frame)

    def _write_open(self, frame: CapturedFrame) -> None:
        assert self._size is not None and self._writer is not None
        assert self._csv is not None and self._csv_writer is not None
        height, width = frame.image.shape[:2]
        if (width, height) != self._size:
            # A mid-session resolution change would silently produce a file the
            # writer drops every frame of. Say so instead.
            raise RuntimeError(
                f"{self.video_path}: frame size changed {self._size} -> {(width, height)}"
            )
        self._writer.write(frame.image)
        self._csv_writer.writerow(
            [frame.seq, f"{frame.t_capture:.6f}", f"{frame.t_wall:.6f}", width, height]
        )
        self.n_written += 1
        if self.n_written % 60 == 0:
            self._csv.flush()

    def close(self) -> None:
        # A session shorter than the calibration window still has to produce a
        # playable file rather than an empty one.
        if self._writer is None and self._pending:
            self.fps = self._measure(self._pending) if len(self._pending) > 1 else 30.0
            buffered, self._pending = self._pending, []
            self._open(buffered[0].image)
            for held in buffered:
                self._write_open(held)
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._csv is not None:
            self._csv.flush()
            self._csv.close()
            self._csv = None


def open_capture(spec: CameraSpec) -> VideoCaptureLike:
    """Open one device on its configured backend and negotiate a mode.

    The backend is not a detail. Measured on this rig, the same USB camera at
    the same resolution reads 5.0 FPS through DSHOW and 14.9 through MSMF, and
    the gap holds at 640x480, 320x240 and 160x120 — so it is which media type
    each backend negotiates, not bandwidth. `mcreid-live-multi probe` measures
    both so the choice is made from numbers rather than from this comment.
    """
    capture = cv2.VideoCapture(spec.device, CAPTURE_BACKENDS[spec.backend])
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"could not open camera device {spec.device} on backend {spec.backend}. "
            "Check the index with `mcreid-live-multi probe`, and that no other app "
            "holds the camera."
        )
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*spec.fourcc))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, spec.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, spec.height)
    ok, frame = capture.read()
    if not ok or frame is None:
        capture.release()
        raise RuntimeError(
            f"camera {spec.device} opened on {spec.backend} but returned no frames"
        )
    return capture


@dataclass
class StreamStats:
    """What a stream actually did, for the end-of-run report."""

    frames_read: int = 0
    frames_failed: int = 0
    first_t: float | None = None
    last_t: float | None = None
    sizes: set[tuple[int, int]] = field(default_factory=set)

    @property
    def measured_fps(self) -> float:
        if self.first_t is None or self.last_t is None or self.frames_read < 2:
            return 0.0
        span = self.last_t - self.first_t
        return (self.frames_read - 1) / span if span > 0 else 0.0


class CameraStream:
    """One camera on its own thread: read -> stamp -> record -> publish."""

    def __init__(
        self,
        spec: CameraSpec,
        recorder: StreamRecorder | None = None,
        opener: Callable[[CameraSpec], VideoCaptureLike] = open_capture,
    ) -> None:
        self.spec = spec
        self.recorder = recorder
        self._opener = opener
        self._capture: VideoCaptureLike | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._pending: CapturedFrame | None = None
        self.stats = StreamStats()
        self.error: BaseException | None = None
        """Set when the capture thread dies. The owner must surface it — a
        thread that quietly exits looks exactly like a camera nobody walked in
        front of, and that is the one failure this rig cannot afford to
        misread."""

    # --- lifecycle --------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError(f"{self.spec.camera_id} already started")
        self._capture = self._opener(self.spec)
        self._thread = threading.Thread(
            target=self._pump, name=f"capture-{self.spec.camera_id}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        if self.recorder is not None:
            self.recorder.close()

    def _pump(self) -> None:
        assert self._capture is not None
        seq = 0
        consecutive = 0
        try:
            while not self._stop.is_set():
                ok, image = self._capture.read()
                if not ok or image is None:
                    self.stats.frames_failed += 1
                    consecutive += 1
                    if consecutive > MAX_CONSECUTIVE_FAILURES:
                        raise RuntimeError(
                            f"{self.spec.camera_id}: {consecutive} consecutive failed "
                            "reads — the camera is gone"
                        )
                    # Without this a failing read() returns instantly and the
                    # loop spins a whole core. Measured on an unplug: 6.5 MILLION
                    # failed reads in one second, on the machine whose deliverable
                    # is >=15 FPS from a GPU pipeline on the same CPU.
                    time.sleep(0.01)
                    continue
                consecutive = 0
                now = time.perf_counter()
                frame = CapturedFrame(
                    camera_id=self.spec.camera_id,
                    image=np.asarray(image, dtype=np.uint8),
                    seq=seq,
                    t_capture=now,
                    t_wall=time.time(),
                )
                seq += 1
                self.stats.frames_read += 1
                self.stats.sizes.add((frame.image.shape[1], frame.image.shape[0]))
                if self.stats.first_t is None:
                    self.stats.first_t = now
                self.stats.last_t = now

                # Record BEFORE publishing. If the processing loop is the thing
                # that falls over, the recording is already complete up to here.
                if self.recorder is not None:
                    self.recorder.write(frame)
                with self._lock:
                    self._pending = frame
        except BaseException as exc:  # noqa: BLE001 - re-raised by the owner
            self.error = exc
            logger.exception("capture thread for %s died", self.spec.camera_id)

    # --- consumption ------------------------------------------------------

    def latest(self) -> CapturedFrame | None:
        """The newest unconsumed frame, or None if nothing new has arrived.

        Latest-wins on purpose: when the detector is slower than the camera the
        right frame to process is the current one, not the oldest queued one. A
        queue here would build a growing lag between what the person is doing
        and what the overlay shows.
        """
        with self._lock:
            frame, self._pending = self._pending, None
        return frame


class CameraRig:
    """Every stream in one object, so the CLI has one thing to open and close."""

    def __init__(self, streams: Sequence[CameraStream]) -> None:
        if not streams:
            raise ValueError("a rig needs at least one camera")
        ids = [s.spec.camera_id for s in streams]
        if len(set(ids)) != len(ids):
            raise ValueError(f"camera_ids must be unique, got {ids}")
        self.streams = list(streams)

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return tuple(s.spec.camera_id for s in self.streams)

    def start(self) -> None:
        started: list[CameraStream] = []
        try:
            for stream in self.streams:
                stream.start()
                started.append(stream)
        except BaseException:
            # Opening camera 2 can fail on USB bandwidth *after* camera 1 is
            # live. Leaving one thread running and one device held would make
            # the next attempt fail for a different reason than the real one.
            for stream in started:
                stream.stop()
            raise

    def stop(self) -> None:
        for stream in self.streams:
            stream.stop()

    def latest(self) -> dict[str, CapturedFrame]:
        """Fresh frames only, keyed by camera_id. Cameras with nothing new are absent."""
        out: dict[str, CapturedFrame] = {}
        for stream in self.streams:
            frame = stream.latest()
            if frame is not None:
                out[stream.spec.camera_id] = frame
        return out

    def raise_for_errors(self) -> None:
        for stream in self.streams:
            if stream.error is not None:
                raise RuntimeError(
                    f"capture thread for {stream.spec.camera_id} died"
                ) from stream.error


def build_rig(
    specs: Sequence[CameraSpec],
    record_dir: Path | None,
    session: str,
    opener: Callable[[CameraSpec], VideoCaptureLike] = open_capture,
) -> CameraRig:
    """Streams for `specs`, each recording into `record_dir` when given."""
    streams = []
    for spec in specs:
        recorder = None
        if record_dir is not None:
            recorder = StreamRecorder(
                video_path=record_dir / f"{session}_{spec.camera_id}.mp4",
                timestamps_path=record_dir / f"{session}_{spec.camera_id}_timestamps.csv",
                fps=spec.nominal_fps,
            )
        streams.append(CameraStream(spec, recorder=recorder, opener=opener))
    return CameraRig(streams)
