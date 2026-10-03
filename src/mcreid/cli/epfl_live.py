"""Interactive EPFL playback: consecutive frames through the existing pipeline.

This is a window, not a second tracker. Each step reads the next frame from all
four files, then runs the same detector, embedder, per-view tracker, and
grid-cell fusion the benchmark uses. Nothing here writes the benchmark mp4,
gif, JSON, or composite, and ground-truth positions are not an input.

Frames are not dropped to keep up with the file's 25 fps. The number on screen
is the measured inference rate. Scrubbing with the arrow keys only moves
through frames this process has already inferred, or infers exactly one new
frame.
"""

from __future__ import annotations

import hashlib
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from mcreid.calib.geometry import feet_point
from mcreid.calib.schema import RigCalib
from mcreid.fusion.global_id import FusionConfig, GlobalIDManager
from mcreid.fusion.types import GlobalTrackSnapshot, ViewObservation
from mcreid.track.gpu_view import GpuPerViewBackend, GpuViewConfig
from mcreid.track.per_view import PerViewConfig
from mcreid.track.reid_models import build_embedder
from mcreid.utils.device import resolve_device
from mcreid.utils.logging import get_logger
from mcreid.viz.bev import BevRenderer
from mcreid.viz.palette import TEXT_COLOR, camera_color, id_color

logger = get_logger(__name__)

Image = npt.NDArray[np.uint8]

# The live window never substitutes ground truth for a track, and it never
# skips a source frame to pretend it is running at the video rate.
GROUND_TRUTH_POSITIONS_USED = False
SKIPS_SOURCE_FRAMES = False

WINDOW_NAME = "mcreid EPFL live"
PREVIEW_W = 1920
PREVIEW_H = 1080
HEADER_H = 86
GAP = 8
TILE_W = 600
TILE_H = 480
BEV_CANVAS = (640, 640)
TRAIL_FRAMES = 25
HISTORY_LIMIT = 48
# Benchmark cap is 4.5 cells. A person who fills the 288px frame has a box on
# the image border; fusion multiplies that foot's variance by 9 and the 4.5 cap
# then discards it, so the map coasts while the cameras still see the person.
# 8.5 keeps that down-weighted foot (about 2.5 cells, times 3) and still drops
# a foot near the horizon, whose sigma is tens of cells. The benchmark command
# does not use this value.
LIVE_POSITION_SIGMA_CELLS = 8.5

_BG: tuple[int, int, int] = (16, 17, 20)
_FG: tuple[int, int, int] = (236, 237, 239)
_MUTED: tuple[int, int, int] = (168, 172, 178)
_WARN: tuple[int, int, int] = (80, 200, 240)
_BAD: tuple[int, int, int] = (60, 60, 220)

# Extended codes only. 81/83 are also 'Q'/'S' from waitKey, so arrows that
# arrive as those bytes are not bound; a/d are the reliable step keys.
_LEFT_KEYS = {63234, 2424832, 65361, 0xFF51}
_RIGHT_KEYS = {63235, 2555904, 65363, 0xFF53}


@dataclass(frozen=True)
class PreviewLayout:
    """Pixel boxes for the 2x2 camera grid and the map column."""

    tiles: tuple[tuple[int, int, int, int], ...]
    bev_origin: tuple[int, int]
    list_origin: tuple[int, int]
    right_x: int
    right_w: int


@dataclass(frozen=True)
class LiveHud:
    """Status drawn on one preview. Display only — not a tracker input."""

    weights_name: str
    device: str
    source_frame: int
    frame_count: int
    video_fps: float
    inference_fps: float | None
    last_ms: float | None
    synced: bool
    paused: bool
    at_end: bool
    benchmark_detector: bool
    viewing_cached: bool
    confirmed_ids: int
    presentation: bool = False
    show_frustums: bool = True


@dataclass(frozen=True)
class LiveStep:
    """One inferred quartet, kept so pause/scrub does not re-run the detector."""

    source_frame: int
    frames: dict[str, Image]
    observations: dict[str, list[ViewObservation]]
    assignment: dict[tuple[str, int], int]
    bev: Image
    snapshots: list[GlobalTrackSnapshot]
    inference_s: float
    synced: bool


def action_for_key(key: int) -> str | None:
    """Map an OpenCV key code to quit, pause, restart, left, or right."""
    if key < 0:
        return None
    if key in _LEFT_KEYS:
        return "left"
    if key in _RIGHT_KEYS:
        return "right"
    low = key & 0xFF
    if low in (ord("q"), ord("Q"), 27):
        return "quit"
    if low == ord(" "):
        return "pause"
    if low in (ord("r"), ord("R")):
        return "restart"
    if low in (ord("a"), ord("A"), ord(",")):
        return "left"
    if low in (ord("d"), ord("D"), ord(".")):
        return "right"
    if low in (ord("m"), ord("M")):
        return "map"
    if low in (ord("f"), ord("F")):
        return "frusta"
    return None


def assert_grid_units(units: str) -> None:
    """EPFL has no recoverable metre scale. Refuse a label that invents one."""
    folded = units.casefold()
    if "metre" in folded or "meter" in folded:
        raise ValueError(
            f"EPFL world unit is grid cells, not {units!r}. "
            "The metric scale is not recoverable from this dataset."
        )


def live_fusion_config(config: FusionConfig) -> FusionConfig:
    """Published cell-unit fusion, with the live-view position cap.

    Returns ``config`` unchanged when its cap is already at least as wide.
    """
    if config.max_position_sigma_m >= LIVE_POSITION_SIGMA_CELLS:
        return config
    return replace(config, max_position_sigma_m=LIVE_POSITION_SIGMA_CELLS)


def is_benchmark_detector(weights: Path) -> bool:
    """The published EPFL numbers were produced with yolo11x, imgsz 640."""
    return Path(weights).name == "yolo11x.pt"


def preview_layout(bev_width: int, bev_height: int) -> PreviewLayout:
    """Where the four tiles and the map sit inside the 1920x1080 window."""
    if bev_width < 1 or bev_height < 1:
        raise ValueError(f"BEV must be non-empty, got {bev_width}x{bev_height}")
    tiles = (
        (GAP, HEADER_H, TILE_W, TILE_H),
        (GAP + TILE_W + GAP, HEADER_H, TILE_W, TILE_H),
        (GAP, HEADER_H + TILE_H + GAP, TILE_W, TILE_H),
        (GAP + TILE_W + GAP, HEADER_H + TILE_H + GAP, TILE_W, TILE_H),
    )
    right_x = GAP + TILE_W + GAP + TILE_W + GAP
    right_w = PREVIEW_W - right_x - GAP
    origin_x = right_x + max((right_w - bev_width) // 2, 0)
    list_y = HEADER_H + bev_height + 12
    return PreviewLayout(
        tiles=tiles,
        bev_origin=(origin_x, HEADER_H),
        list_origin=(right_x, list_y),
        right_x=right_x,
        right_w=right_w,
    )


def _fit_scale(text: str, max_px: int, scale: float, thickness: int) -> float:
    while scale > 0.35:
        (width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        if width <= max_px:
            return scale
        scale -= 0.05
    return scale


def _text(
    canvas: Image,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
    max_px: int | None = None,
) -> None:
    used = scale if max_px is None else _fit_scale(text, max_px, scale, thickness)
    cv2.putText(
        canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, used, color, thickness, cv2.LINE_AA
    )


def _draw_tile(
    frame: Image,
    observations: list[ViewObservation],
    assignment: Mapping[tuple[str, int], int],
    camera_id: str,
    camera_index: int,
    source_frame: int,
) -> Image:
    """Upscale one camera, then draw the box, local id, global id, and score."""
    tile = cv2.resize(frame, (TILE_W, TILE_H), interpolation=cv2.INTER_LINEAR)
    scale_x = TILE_W / float(frame.shape[1])
    scale_y = TILE_H / float(frame.shape[0])
    for obs in observations:
        box = np.asarray(obs.bbox_xyxy, dtype=np.float64)
        p0 = (int(round(box[0] * scale_x)), int(round(box[1] * scale_y)))
        p1 = (int(round(box[2] * scale_x)), int(round(box[3] * scale_y)))
        gid = assignment.get((camera_id, obs.local_track_id))
        colour = id_color(gid) if gid is not None else (110, 110, 110)
        cv2.rectangle(tile, p0, p1, colour, 2, cv2.LINE_AA)
        if gid is None:
            label = f"L{obs.local_track_id}  {obs.score:.2f}"
        else:
            label = f"G{gid}  L{obs.local_track_id}  {obs.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(p0[1] - th - 8, 0)
        cv2.rectangle(tile, (p0[0], top), (p0[0] + tw + 8, top + th + 8), colour, -1)
        cv2.putText(
            tile,
            label,
            (p0[0] + 4, top + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        foot = feet_point(box)[0]
        cv2.drawMarker(
            tile,
            (int(round(foot[0] * scale_x)), int(round(foot[1] * scale_y))),
            colour,
            cv2.MARKER_CROSS,
            14,
            2,
            cv2.LINE_AA,
        )
    border = camera_color(camera_index)
    cv2.rectangle(tile, (1, 1), (TILE_W - 2, TILE_H - 2), border, 2, cv2.LINE_AA)
    banner = f"{camera_id}  f{source_frame}"
    cv2.rectangle(tile, (6, 6), (6 + 8 * len(banner) + 16, 32), _BG, -1)
    _text(tile, banner, (12, 26), 0.6, TEXT_COLOR, 1)
    return tile


def _draw_id_list(
    canvas: Image,
    snapshots: Sequence[GlobalTrackSnapshot],
    origin: tuple[int, int],
    width: int,
    viewing_cached: bool,
    presentation: bool = False,
) -> None:
    x, y = origin
    if presentation:
        title = "SCHEMATIC GRID (cached)" if viewing_cached else "SCHEMATIC GRID"
    elif viewing_cached:
        title = "INFERENCE BEV (cached)"
    else:
        title = "LIVE INFERENCE BEV"
    _text(canvas, title, (x, y), 0.55, _FG, 1, width)
    y += 26
    _text(canvas, "global id    grid cells", (x, y), 0.48, _MUTED, 1, width)
    y += 24
    if not snapshots:
        _text(canvas, "no confirmed track on the map yet", (x, y), 0.48, _MUTED, 1, width)
        return
    shown = sorted(snapshots, key=lambda snap: snap.global_id)[:12]
    for snap in shown:
        wx, wy = (float(v) for v in np.asarray(snap.world_xy, dtype=np.float64).reshape(2))
        cams = ",".join(snap.supporting_cameras) if snap.supporting_cameras else "-"
        coast = "  coast" if snap.frames_since_measurement else ""
        line = f"ID {snap.global_id:<3}  {wx:6.1f}, {wy:6.1f}  {cams}{coast}"
        _text(canvas, line, (x, y), 0.48, id_color(snap.global_id), 1, width)
        y += 22
        if y > PREVIEW_H - 16:
            break
    extra = len(snapshots) - len(shown)
    if extra > 0 and y <= PREVIEW_H - 16:
        _text(canvas, f"+{extra} more", (x, y), 0.48, _MUTED, 1, width)


def _draw_header(canvas: Image, hud: LiveHud) -> None:
    if hud.inference_fps is None:
        if hud.last_ms is None:
            rate = "Inference: warmup"
        else:
            rate = f"Inference: warmup {hud.last_ms:.0f} ms"
    else:
        last = "" if hud.last_ms is None else f"   last {hud.last_ms:.0f} ms"
        rate = f"Inference: {hud.inference_fps:.1f} FPS{last}"
    video_t = hud.source_frame / hud.video_fps if hud.video_fps else 0.0
    state = "PAUSED" if hud.paused else "RUNNING"
    if hud.at_end:
        state = "END"
    if hud.viewing_cached:
        state = "CACHED"
    sync = "4 cameras synced" if hud.synced else "CAMERAS OUT OF SYNC"
    line1 = (
        f"EPFL live   {hud.weights_name}   {hud.device}   "
        f"frame {hud.source_frame}/{hud.frame_count}   t={video_t:.2f}s   {rate}"
    )
    coverage = "coverage on" if hud.show_frustums else "coverage hidden"
    line2 = (
        f"{state}   grid cells, not metres   {sync}   {coverage}   "
        "q quit  space pause  r restart  a/d step  m map  f coverage"
    )
    if hud.benchmark_detector:
        line3 = "published benchmark detector (yolo11x)   no frames skipped"
    else:
        line3 = "DEMO / PERFORMANCE CONFIG - not the published benchmark (yolo11x)"
    _text(canvas, line1, (16, 24), 0.55, _FG, 1, PREVIEW_W - 32)
    line2_color = _WARN if hud.paused or hud.at_end else _MUTED
    _text(canvas, line2, (16, 48), 0.5, line2_color, 1, PREVIEW_W - 32)
    third = _WARN if not hud.benchmark_detector else _MUTED
    if not hud.synced:
        third = _BAD
    _text(canvas, line3, (16, 72), 0.5, third, 1, PREVIEW_W - 32)


def compose_preview(
    frames: Mapping[str, Image],
    observations: Mapping[str, list[ViewObservation]],
    assignment: Mapping[tuple[str, int], int],
    bev: Image,
    snapshots: Sequence[GlobalTrackSnapshot],
    hud: LiveHud,
    camera_order: Sequence[str],
    units: str = "grid cells",
) -> Image:
    """Build one 1920x1080 window. The map image is pasted, not recomputed."""
    assert_grid_units(units)
    if tuple(camera_order) != tuple(frames):
        missing = [c for c in camera_order if c not in frames]
        if missing or len(camera_order) != 4:
            raise ValueError(f"need frames for {list(camera_order)}, missing {missing}")
    if len(camera_order) != 4:
        raise ValueError(f"EPFL live view expects 4 cameras, got {list(camera_order)}")

    shown = bev
    layout = preview_layout(int(shown.shape[1]), int(shown.shape[0]))
    if shown.shape[1] > layout.right_w or HEADER_H + shown.shape[0] > PREVIEW_H - 160:
        raise ValueError(
            f"BEV {shown.shape[1]}x{shown.shape[0]} does not fit the preview column"
        )

    canvas = np.full((PREVIEW_H, PREVIEW_W, 3), _BG, dtype=np.uint8)
    _draw_header(canvas, hud)
    for index, camera_id in enumerate(camera_order):
        tile = _draw_tile(
            frames[camera_id],
            list(observations.get(camera_id, ())),
            assignment,
            camera_id,
            index,
            hud.source_frame,
        )
        x, y, width, height = layout.tiles[index]
        canvas[y : y + height, x : x + width] = tile
    bx, by = layout.bev_origin
    canvas[by : by + shown.shape[0], bx : bx + shown.shape[1]] = shown
    _draw_id_list(
        canvas,
        snapshots,
        layout.list_origin,
        layout.right_w,
        hud.viewing_cached,
        hud.presentation,
    )
    return canvas


def _rolling_fps(samples: Sequence[float]) -> float | None:
    """Mean rate after the first step. The first step pays for MPS warmup."""
    body = list(samples[1:]) if len(samples) > 1 else []
    recent = body[-20:]
    if not recent:
        return None
    mean = sum(recent) / len(recent)
    if mean <= 0.0:
        return None
    return 1.0 / mean


class _Session:
    """Four synced captures plus one shared detector and four trackers."""

    def __init__(
        self,
        root: Path,
        sequence: str,
        rig: RigCalib,
        fusion_config: FusionConfig,
        image_size: tuple[int, int],
        weights: Path,
        embedder_name: str,
        imgsz: int,
        conf: float,
        device: str,
        start_frame: int,
        presentation: bool = False,
    ) -> None:
        self.root = root
        self.sequence = sequence
        self.rig = rig
        self.fusion_config = fusion_config
        self.image_size = image_size
        self.weights = weights
        self.embedder_name = embedder_name
        self.start_frame = start_frame
        self.presentation = presentation
        # Coverage polygons are the working-map default. The schematic plate
        # starts without them; `f` toggles either way.
        self.show_frustums = not presentation
        self.camera_order = list(rig.camera_ids)
        if self.camera_order != [f"cam{i}" for i in range(len(self.camera_order))]:
            raise ValueError(f"expected cam0.. in order, got {self.camera_order}")
        self.view_config = GpuViewConfig(
            weights=weights,
            imgsz=imgsz,
            conf_threshold=conf,
            embedder=embedder_name,
            device=device,
        )
        self.per_view = PerViewConfig()
        self.device_spec = resolve_device(device)
        self.captures: list[cv2.VideoCapture] = []
        self.detector: Any = None
        self.embedder: Any = None
        self.backends: dict[str, GpuPerViewBackend] = {}
        self.manager: GlobalIDManager | None = None
        self.bev: BevRenderer | None = None
        self.fps = 0.0
        self.dt = 0.0
        self.frame_count = 0
        self._next_index = start_frame

    def open(self) -> None:
        if len(self.camera_order) != 4:
            raise ValueError(f"EPFL live view expects 4 cameras, got {self.camera_order}")
        self.captures = []
        for index in range(4):
            path = self.root / f"{self.sequence}-c{index}.avi"
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                self.close()
                raise FileNotFoundError(f"could not open {path}")
            self.captures.append(capture)
        self.fps = float(self.captures[0].get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.captures[0].get(cv2.CAP_PROP_FRAME_COUNT))
        if not self.fps > 1.0:
            raise RuntimeError(f"video FPS is not usable: {self.fps}")
        if self.start_frame < 0 or self.start_frame >= self.frame_count:
            raise ValueError(
                f"--live-start {self.start_frame} is outside 0..{self.frame_count - 1}"
            )
        self.dt = 1.0 / self.fps
        from ultralytics import YOLO

        weights = Path(self.weights)
        if not weights.is_file():
            raise FileNotFoundError(f"detector weights not found: {weights}")
        self.detector = YOLO(str(weights))
        self.embedder = build_embedder(
            self.embedder_name,
            device=self.device_spec,
            batch_size=self.view_config.embed_batch,
            weights_dir=self.view_config.weights_dir,
        )
        self._new_trackers()
        self._seek_all(self.start_frame)

    def close(self) -> None:
        for capture in self.captures:
            capture.release()
        self.captures = []

    def _new_trackers(self) -> None:
        """Fresh per-view state. The loaded weights stay where they are."""
        self.backends = {
            camera_id: GpuPerViewBackend(
                camera_id,
                self.view_config,
                self.per_view,
                detector=self.detector,
                embedder=self.embedder,
            )
            for camera_id in self.camera_order
        }
        self.manager = GlobalIDManager(self.rig, self.fusion_config)
        self.bev = BevRenderer(
            self.rig,
            canvas_size=BEV_CANVAS,
            margin_m=1.0,
            trail_length=TRAIL_FRAMES,
            grid_step_m=4.0,
            units="grid cells",
        )
        self._next_index = self.start_frame

    def _seek_all(self, frame: int) -> None:
        for capture in self.captures:
            capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame))

    def reset(self) -> None:
        """Start the file again. Identity does not survive a backward jump."""
        self._new_trackers()
        self._seek_all(self.start_frame)

    def step(self) -> LiveStep | None:
        """Read the next quartet and run one pipeline step. Never skips ahead."""
        if SKIPS_SOURCE_FRAMES:
            raise RuntimeError("the live view is not allowed to drop source frames")
        if self.manager is None or self.bev is None:
            raise RuntimeError("session is not open")
        images: list[Image] = []
        reported: list[int] = []
        for capture in self.captures:
            ok, frame = capture.read()
            if not ok or frame is None:
                return None
            image = np.asarray(frame, dtype=np.uint8)
            width, height = self.image_size
            if image.shape[1] != width or image.shape[0] != height:
                raise ValueError(
                    f"frame is {image.shape[1]}x{image.shape[0]}, "
                    f"calibration expects {width}x{height}"
                )
            images.append(image.copy())
            reported.append(int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1)

        synced = len(set(reported)) == 1
        if synced and reported[0] >= self._next_index:
            index = reported[0]
        else:
            index = self._next_index
            synced = False
        if index < self._next_index:
            raise RuntimeError(f"source frame went backwards: {index} after {self._next_index - 1}")
        self._next_index = index + 1

        started = time.perf_counter()
        observations: dict[str, list[ViewObservation]] = {}
        views: list[ViewObservation] = []
        for camera_id, image in zip(self.camera_order, images, strict=True):
            seen = self.backends[camera_id].step(image, index)
            observations[camera_id] = list(seen)
            views.extend(seen)
        snapshots = self.manager.step(views, index, self.dt)
        elapsed = time.perf_counter() - started
        self.bev.appearance = "schematic" if self.presentation else "working"
        coverage = self.camera_order if self.show_frustums else None
        bev = self.bev.render(snapshots, index, camera_order=coverage)
        return LiveStep(
            source_frame=index,
            frames=dict(zip(self.camera_order, images, strict=True)),
            observations=observations,
            assignment=dict(self.manager.last_assignment),
            bev=bev,
            snapshots=list(snapshots),
            inference_s=elapsed,
            synced=synced,
        )


def _hud(step: LiveStep, session: _Session, samples: Sequence[float], **flags: bool) -> LiveHud:
    return LiveHud(
        weights_name=Path(session.weights).name,
        device=session.device_spec.torch_device,
        source_frame=step.source_frame,
        frame_count=session.frame_count,
        video_fps=session.fps,
        inference_fps=_rolling_fps(samples),
        last_ms=step.inference_s * 1000.0,
        synced=step.synced,
        paused=flags["paused"],
        at_end=flags["at_end"],
        benchmark_detector=is_benchmark_detector(session.weights),
        viewing_cached=flags["viewing_cached"],
        confirmed_ids=len(step.snapshots),
        presentation=session.presentation,
        show_frustums=session.show_frustums,
    )


def _bev_digest(bev: Image) -> str:
    """Hash the map below its frame-number line, so a counter tick is not motion."""
    body = bev[30:] if bev.shape[0] > 30 else bev
    return hashlib.sha256(np.ascontiguousarray(body).tobytes()).hexdigest()


def format_live_summary(summary: Mapping[str, Any]) -> str:
    """Plain-text report printed when the window closes."""
    fps = summary.get("inference_fps")
    rate = "n/a" if fps is None else f"{float(fps):.2f}"
    kind = (
        "published benchmark detector"
        if summary["benchmark_detector"]
        else "DEMO / PERFORMANCE CONFIG - not the published benchmark"
    )
    lines = [
        f"device: {summary['device_name']}",
        f"detector: {summary['weights']} ({kind})",
        f"embedder: {summary['embedder']}",
        f"world unit: {summary['units']}",
        (
            "presentation map: schematic grid, no floor-plan image in the dataset"
            if summary.get("presentation_map")
            else "presentation map: off"
        ),
        (
            f"position cap: {summary['position_sigma_cap_cells']} cells "
            f"(benchmark cap {summary['benchmark_position_sigma_cap_cells']})"
        ),
        (
            f"frames processed: {summary['frames']} consecutive from {summary['start_frame']}"
            f"  (video {summary['video_fps']:.2f} fps, dt={summary['dt_s']:.4f}s)"
        ),
        f"inference FPS: {rate}   warmup {summary['warmup_ms']:.0f} ms excluded",
        f"cameras synced on every step: {summary['cameras_synced']}",
        f"bev changed across steps: {summary['bev_changed']}",
        f"global ids on the map: {summary['global_ids']}",
        f"global ids assigned to boxes: {summary['assigned_ids']}",
        f"max observations in one frame: {summary['max_observations']}",
        f"ground-truth positions used as tracks: {summary['ground_truth_positions_used']}",
        "keys: q quit, space pause/resume, r restart, a or left step back, "
        "d or right step forward, m schematic map, f camera coverage",
    ]
    return "\n".join(lines)


def run_epfl_live_view(
    *,
    root: Path,
    sequence: str,
    rig: RigCalib,
    fusion_config: FusionConfig,
    image_size: tuple[int, int],
    weights: Path,
    embedder_name: str,
    imgsz: int,
    conf: float,
    device: str,
    start_frame: int = 0,
    max_frames: int = 0,
    annotated_from_frame: int | None = None,
    show: bool = True,
    snapshot_dir: Path | None = None,
    presentation_map: bool = False,
) -> dict[str, Any]:
    """Open the window and step the pipeline until quit, the end, or ``max_frames``."""
    assert_grid_units("grid cells")
    if GROUND_TRUTH_POSITIONS_USED:
        raise RuntimeError("ground truth must not be fed to the live tracker")
    if max_frames < 0:
        raise ValueError(f"max_frames must be >= 0, got {max_frames}")
    if not show and max_frames <= 0:
        raise ValueError("a headless live view needs max_frames so it can exit")

    benchmark = is_benchmark_detector(weights)
    published_sigma = fusion_config.max_position_sigma_m
    fusion_config = live_fusion_config(fusion_config)
    session = _Session(
        root=root,
        sequence=sequence,
        rig=rig,
        fusion_config=fusion_config,
        image_size=image_size,
        weights=weights,
        embedder_name=embedder_name,
        imgsz=imgsz,
        conf=conf,
        device=device,
        start_frame=start_frame,
        presentation=presentation_map,
    )
    window_open = False
    samples: list[float] = []
    digests: list[str] = []
    global_ids: set[int] = set()
    assigned_ids: set[int] = set()
    max_observations = 0
    cameras_synced = True
    trace: list[dict[str, Any]] = []
    try:
        session.open()
        logger.info("EPFL live view on %s", session.device_spec)
        logger.info(
            "detector %s imgsz %d conf %.2f — %s",
            weights,
            imgsz,
            conf,
            "published benchmark detector"
            if benchmark
            else "DEMO / PERFORMANCE CONFIG, not the published yolo11x benchmark",
        )
        logger.info(
            "consecutive frames, video %.3f fps, dt %.4fs. per-view n_init=%d "
            "(tracker default). fusion n_init=%d. units: grid cells. no frames skipped",
            session.fps,
            session.dt,
            session.per_view.n_init,
            fusion_config.n_init,
        )
        if fusion_config.max_position_sigma_m != published_sigma:
            logger.info(
                "live position cap %.1f cells; the benchmark cap stays %.1f. "
                "A border-touching box is down-weighted, not discarded, so the "
                "map keeps the measurement while the person fills the frame.",
                fusion_config.max_position_sigma_m,
                published_sigma,
            )
        if annotated_from_frame is not None:
            logger.info(
                "annotations begin at frame %d; the tracker does not read those positions",
                annotated_from_frame,
            )
        if session.presentation:
            x0, y0, x1, y1 = rig.floor_extent()
            logger.info(
                "presentation map: schematic %gx%g grid, camera coverage hidden. "
                "No floor-plan image ships with this dataset; the plate is the grid, not a room.",
                x1 - x0,
                y1 - y0,
            )
        logger.info(
            "keys: q quit, space pause/resume, r restart, a/left back, d/right forward, "
            "m schematic map, f camera coverage"
        )
        if show:
            try:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW_NAME, PREVIEW_W, PREVIEW_H)
                window_open = True
            except cv2.error as exc:
                logger.warning("no preview window (%s); measuring headless", exc)
                if max_frames <= 0:
                    raise RuntimeError("could not open the preview window") from exc

        history: deque[LiveStep] = deque(maxlen=HISTORY_LIMIT)
        view = 0
        playing = True
        at_end = False
        auto_exit = False
        if snapshot_dir is not None:
            snapshot_dir.mkdir(parents=True, exist_ok=True)

        last_confirmed = -1

        def _restyle_frontier() -> None:
            """Redraw the newest map. Older scrub frames keep the style they were drawn in."""
            if session.bev is None or not history or view != len(history) - 1:
                return
            session.bev.appearance = "schematic" if session.presentation else "working"
            order = session.camera_order if session.show_frustums else None
            painted = session.bev.repaint(order)
            if painted is None:
                return
            current = history.pop()
            history.append(replace(current, bev=painted))

        def accept(step: LiveStep) -> None:
            nonlocal view, max_observations, cameras_synced, last_confirmed
            history.append(step)
            view = len(history) - 1
            samples.append(step.inference_s)
            digests.append(_bev_digest(step.bev))
            global_ids.update(s.global_id for s in step.snapshots)
            assigned_ids.update(step.assignment.values())
            observed = sum(len(v) for v in step.observations.values())
            max_observations = max(max_observations, observed)
            cameras_synced = cameras_synced and step.synced
            trace.append(
                {
                    "frame": step.source_frame,
                    "ms": round(step.inference_s * 1000.0, 1),
                    "observations": observed,
                    "ids": [
                        (
                            s.global_id,
                            round(float(s.world_xy[0]), 2),
                            round(float(s.world_xy[1]), 2),
                        )
                        for s in step.snapshots
                    ],
                    "synced": step.synced,
                }
            )
            confirmed = len(step.snapshots)
            if len(samples) == 1 or len(samples) % 25 == 0 or confirmed != last_confirmed:
                logger.info(
                    "frame %d  %.0f ms  observations %d  confirmed ids %d",
                    step.source_frame,
                    step.inference_s * 1000.0,
                    observed,
                    confirmed,
                )
            last_confirmed = confirmed

        while True:
            if playing and at_end:
                playing = False
            if playing and not at_end:
                step = session.step()
                if step is None:
                    at_end = True
                    playing = False
                    logger.info("end of sequence at frame %d", session._next_index - 1)
                else:
                    accept(step)
                    if max_frames and len(samples) >= max_frames:
                        auto_exit = True

            if not history:
                break

            current = history[view]
            canvas = compose_preview(
                current.frames,
                current.observations,
                current.assignment,
                current.bev,
                current.snapshots,
                _hud(
                    current,
                    session,
                    samples,
                    paused=not playing,
                    at_end=at_end,
                    viewing_cached=view != len(history) - 1,
                ),
                session.camera_order,
            )
            if snapshot_dir is not None and max_frames:
                marks = {1, max(1, max_frames // 2), max_frames}
                if len(samples) in marks and view == len(history) - 1:
                    path = snapshot_dir / f"frame_{current.source_frame:05d}.png"
                    cv2.imwrite(str(path), canvas)

            key = -1
            if window_open:
                try:
                    cv2.imshow(WINDOW_NAME, canvas)
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) == 0:
                        break
                    key = int(cv2.waitKeyEx(1 if playing or auto_exit else 0))
                except cv2.error as exc:
                    logger.warning("preview window closed (%s)", exc)
                    window_open = False
                    if max_frames <= 0:
                        break
            if auto_exit:
                break

            action = action_for_key(key)
            if action == "quit":
                break
            if action == "pause":
                playing = not at_end and not playing
            elif action == "restart":
                session.reset()
                history.clear()
                samples.clear()
                digests.clear()
                global_ids.clear()
                assigned_ids.clear()
                trace.clear()
                view = 0
                last_confirmed = -1
                max_observations = 0
                cameras_synced = True
                at_end = False
                playing = True
                logger.info("restarted at frame %d with a fresh tracker", session.start_frame)
            elif action == "map":
                session.presentation = not session.presentation
                session.show_frustums = not session.presentation
                _restyle_frontier()
            elif action == "frusta":
                session.show_frustums = not session.show_frustums
                _restyle_frontier()
            elif action == "left" and view > 0:
                playing = False
                view -= 1
            elif action == "right" and not playing:
                if view < len(history) - 1:
                    view += 1
                elif not at_end:
                    step = session.step()
                    if step is None:
                        at_end = True
                    else:
                        accept(step)
    finally:
        session.close()
        if window_open:
            cv2.destroyWindow(WINDOW_NAME)

    summary: dict[str, Any] = {
        "device": session.device_spec.torch_device,
        "device_name": str(session.device_spec),
        "weights": str(weights),
        "benchmark_detector": benchmark,
        "embedder": embedder_name,
        "imgsz": imgsz,
        "units": "grid cells",
        "frames": len(samples),
        "start_frame": start_frame,
        "video_fps": session.fps,
        "dt_s": session.dt,
        "inference_fps": _rolling_fps(samples),
        "warmup_ms": (samples[0] * 1000.0) if samples else 0.0,
        "cameras_synced": cameras_synced and bool(samples),
        "bev_changed": len(set(digests)) > 1,
        "global_ids": sorted(global_ids),
        "assigned_ids": sorted(assigned_ids),
        "max_observations": max_observations,
        "ground_truth_positions_used": GROUND_TRUTH_POSITIONS_USED,
        "skips_source_frames": SKIPS_SOURCE_FRAMES,
        "per_view_n_init": session.per_view.n_init,
        "fusion_n_init": fusion_config.n_init,
        "presentation_map": session.presentation,
        "show_frustums": session.show_frustums,
        "position_sigma_cap_cells": fusion_config.max_position_sigma_m,
        "benchmark_position_sigma_cap_cells": published_sigma,
        "trace": trace,
    }
    return summary
