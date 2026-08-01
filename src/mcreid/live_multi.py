"""Multi-camera live tracking session, uncalibrated.

The single-camera `LiveSession` can ignore calibration because with one camera
there is no cross-view fusion to do — the pixel-plane stand-in only exists to
keep the Kalman filter's metre-denominated gates in a sane numeric range. With
two cameras that reasoning stops holding, and the difference matters enough to
state plainly:

**Two uncalibrated cameras do not share a floor.** Give each one a pixel-plane
homography and their "world" coordinates are scaled pixel coordinates in their
own frame — unrelated quantities that happen to share units. Geometry is then
not merely uninformative, it is *wrong*: two different people standing at the
same pixel in two views look co-located, and one person seen at opposite corners
of two views looks metres apart. Every geometric gate in the fusion stage would
be answering a question about a floor that does not exist.

So this path suspends geometry **between cameras, and only between cameras**.
Within one camera the pixel plane is self-consistent and the motion gate is real
evidence that must keep applying. The first version of this file did open every
gate globally, and an adversarial review measured the cost on real WILDTRACK
crops: a stranger walking into the same camera an identity had just coasted out
of took that identity **63 % of the time**, against 0 % for the shipped config,
with no second camera involved at all. See `appearance_only_fusion_config` and
`FusionConfig.cross_camera_geometry_open`.

Nothing downstream claims a metric position and there is no BEV panel — the same
posture `mcreid-live` takes without `--homography`, carried through to the one
place where it actually costs something.

**The honest limit, measured rather than predicted.** The earlier version of this
docstring said to expect *fragmentation* into per-camera identities. The
measurement says the opposite: the failure mode is **over-fusion**. With one
person per camera, two DIFFERENT people are merged into one identity 76.7 % of
the time on real crops. The cause is not a threshold — the merge tests EMA-to-EMA
vectors, whose different-person mean (0.456) already sits inside the strict 0.48
gate, because averaging pulls every identity toward the population centroid.
Fixing it needs a better embedder or a non-appearance cue, which is v2; until
then this path is valid for ONE occupant and says so in
`SINGLE_OCCUPANT_WARNING`.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import numpy.typing as npt

from mcreid.calib.schema import CameraCalib, RigCalib
from mcreid.diagnostics.shadow import ShadowProbe
from mcreid.fusion.global_id import FusionConfig, GlobalIDManager
from mcreid.fusion.types import GlobalTrackSnapshot, TrackState, ViewObservation
from mcreid.live import IdentityTimeline, pixel_plane_calibration
from mcreid.utils.logging import get_logger
from mcreid.viz.palette import TEXT_COLOR, id_color

logger = get_logger(__name__)

Image = npt.NDArray[np.uint8]

_FONT = cv2.FONT_HERSHEY_DUPLEX

# A separation no pair of observations in a pixel-plane pseudo-world can reach.
# Frame height maps to `span_m` and width to at most twice that, so the largest
# separation at the default span of 6 m is ~12.2 pseudo-metres and 1e4 clears it
# by ~800x. Used instead of `inf` because several gates feed sums and products
# that would turn inf into nan.
#
# It is applied to `birth_cluster_radius_m` ONLY, which is a plain metre
# distance. An earlier version also assigned it to `association.chi2_gate`,
# which compares a *squared* Mahalanobis distance — different units, and the
# review measured the margin there at 1.69x rather than the four orders the
# comment claimed. Worse, it inverted above `span_m ~ 8`: the "opened" gate
# started rejecting exactly the cross-corner pairs it existed to accept. The
# cross-camera exemption is now a flag rather than a magic distance, so no gate
# in unfamiliar units is set from this constant.
GEOMETRY_FREE_M = 1.0e4


SINGLE_OCCUPANT_WARNING = (
    "appearance-only fusion is valid for ONE occupant. With two people in view "
    "it fuses them into one identity 76.7% of the time — measured on real "
    "WILDTRACK crops with the shipped OSNet, one person per camera, against "
    "0.0% for the geometry-gated config. The cause is the embedder, not these "
    "thresholds: same-person cross-camera distance is 0.516 against "
    "different-person 0.623, and the merge tests EMA-to-EMA vectors whose "
    "different-person mean (0.456) already sits INSIDE the strict 0.48 gate. No "
    "threshold on this path fixes that; a better embedder or a non-appearance "
    "cue would."
)
"""Why the profile must not be read as general-purpose multi-camera fusion.

Kept as a constant so the CLI, the docstrings and the test that pins it cannot
drift apart, and so nobody has to rediscover the number by running the rig with
two people in it.
"""


class MultiViewStepper(Protocol):
    """Anything that turns a frame per camera into per-view observations."""

    def step(self, frames: Mapping[str, Image], frame: int) -> list[ViewObservation]: ...


def uncalibrated_rig(sizes: Mapping[str, tuple[int, int]], span_m: float = 6.0) -> RigCalib:
    """A pixel-plane stand-in calibration per camera.

    Each camera's frame height is mapped to `span_m` pseudo-metres, so all
    cameras share a *unit convention* even though they do not share a floor.
    That keeps observation covariances comparable between views, which is the
    only thing the fusion stage still uses geometry for once
    `appearance_only_fusion_config` has opened the gates.
    """
    if not sizes:
        raise ValueError("need at least one camera size")
    cameras: list[CameraCalib] = []
    for camera_id, (width, height) in sizes.items():
        base = pixel_plane_calibration(width, height, span_m)
        cameras.append(
            base.model_copy(
                update={
                    "camera_id": camera_id,
                    "notes": (
                        "pixel-plane stand-in: scaled pixels, NOT metres, and NOT a "
                        "frame shared with any other camera in this rig"
                    ),
                }
            )
        )
    return RigCalib(
        cameras=cameras,
        world_notes=(
            "UNCALIBRATED multi-camera live rig. Each camera has its own pixel plane; "
            "the coordinates are not comparable between cameras and no metric claim "
            "may be made from them."
        ),
    )


def appearance_only_fusion_config(base: FusionConfig | None = None) -> FusionConfig:
    """Suspend geometry **between cameras**, and only between cameras.

    The first version of this opened every radius globally, and an adversarial
    review measured what that cost on real WILDTRACK crops. Two numbers decided
    the current shape:

    * A stranger walking into the same camera an identity had just coasted out
      of took that identity **63 % of trials** (shipped config: 0 %).
      **No second camera was involved in that attack.** Geometry is invalid
      *across* cameras; within one camera the pixel plane is self-consistent and
      the motion gate is real evidence. Opening it globally threw away a sound
      constraint to repair an unsound one. Scoped: back to **0 %**.
    * Two different people, one per camera, fused into one identity **90 % of
      the time**. Scoped: **76.7 %** — better, and still far too high, which is
      what `SINGLE_OCCUPANT_WARNING` exists to say out loud rather than bury.
      Genuine cross-view fusion is 94.7 % over the same crops.

    So the geometric exemption now lives in `FusionConfig.cross_camera_geometry_
    open`, which scopes it to pairs that do not share one camera's pixel plane,
    and only three things are set here:

    ``association.weight_geometry -> 0`` / ``weight_appearance -> 1``
        Ranking must not be moved by a distance between two unrelated pixel
        planes. Same-camera pairs still *gate* on geometry; this only stops
        geometry from ordering the candidates.
    ``association.max_cost -> 1.0``
        Not cosmetic, and independently confirmed by the review. With geometry
        weighted to zero the cost *is* the normalised appearance distance, so
        the shipped 0.85 ceiling would silently tighten the appearance gate from
        0.56 to 0.476 (measured: 0.4760 accepts, 0.4770 rejects). Raising it
        makes `max_appearance_distance` the only appearance decision, at the
        value it was measured at.
    ``birth_cluster_radius_m -> GEOMETRY_FREE_M``
        `_cluster` is the one path that is safe to open globally, because it
        already refuses to put two observations from the *same* camera in one
        cluster. Its radius therefore only ever widens cross-camera grouping.

    ``cluster_appearance_distance`` drops 0.62 -> 0.56 for the same reason as
    before: the shipped value is justified by co-location evidence an
    uncalibrated rig does not have, so it must become a decision gate rather
    than a veto. Tied to `association.max_appearance_distance` rather than being
    a new number.

    Deliberately NOT touched: `max_appearance_distance`, the merge and revive
    gates, and the whole dormant config. Appearance is the only evidence left,
    and loosening its gates here would be inventing numbers on the path with the
    least corroboration.

    **This profile is still only safe for a single occupant.** See
    `SINGLE_OCCUPANT_WARNING` — the residual cross-camera stranger-merge rate is
    high enough that two people in view will fuse, and that is a property of the
    embedder, not of these thresholds.
    """
    base = base or FusionConfig()
    association = replace(
        base.association,
        weight_geometry=0.0,
        weight_appearance=1.0,
        max_cost=1.0,
    )
    return replace(
        base,
        association=association,
        cross_camera_geometry_open=True,
        birth_cluster_radius_m=GEOMETRY_FREE_M,
        cluster_appearance_distance=association.max_appearance_distance,
    )


@dataclass
class CrossViewLedger:
    """Which cameras each global identity has been seen by, and when together.

    This is the acceptance evidence for a two-camera run: an identity that only
    ever appears with one supporting camera has not been fused across views,
    however long it survives.
    """

    cameras_ever: dict[int, set[str]] = field(default_factory=dict)
    frames_multi: dict[int, int] = field(default_factory=dict)
    first_multi_frame: dict[int, int] = field(default_factory=dict)

    def observe(self, snapshots: Sequence[GlobalTrackSnapshot], frame: int) -> list[int]:
        """Record this frame. Returns IDs that became multi-camera just now."""
        newly: list[int] = []
        for snap in snapshots:
            if not snap.supporting_cameras:
                continue
            seen = self.cameras_ever.setdefault(snap.global_id, set())
            seen.update(snap.supporting_cameras)
            if len(snap.supporting_cameras) >= 2:
                self.frames_multi[snap.global_id] = self.frames_multi.get(snap.global_id, 0) + 1
                if snap.global_id not in self.first_multi_frame:
                    self.first_multi_frame[snap.global_id] = frame
                    newly.append(snap.global_id)
        return newly

    @property
    def multi_camera_ids(self) -> list[int]:
        return sorted(gid for gid, cams in self.cameras_ever.items() if len(cams) >= 2)


@dataclass(frozen=True)
class MultiLiveConfig:
    span_m: float = 6.0
    reacquire_gap_s: float = 1.0
    tile_height: int = 480
    """Height each camera tile is scaled to in the mosaic."""
    show_bev: bool = True
    """Draw the metric floor plan. Ignored unless the rig is calibrated — a BEV
    built on pixel-plane stand-ins would be a map of nothing, drawn to scale."""
    bev_size: int = 480


class MultiLiveSession:
    """Stateful N-camera tracking session, appearance-only fusion."""

    def __init__(
        self,
        backend: MultiViewStepper,
        rig: RigCalib,
        config: MultiLiveConfig | None = None,
        fusion_config: FusionConfig | None = None,
        shadow: ShadowProbe | None = None,
        metric: bool = False,
    ) -> None:
        self.backend = backend
        self.rig = rig
        self.config = config or MultiLiveConfig()
        self.metric = metric
        """True only when the cameras share a real floor plane. Gates the BEV
        panel and every metric claim, exactly as on the single-camera path."""
        default_config = (
            FusionConfig() if metric else appearance_only_fusion_config()
        )
        self.manager = GlobalIDManager(rig, fusion_config or default_config)
        self.shadow = shadow
        self.timeline = IdentityTimeline()
        self.ledger = CrossViewLedger()
        self.frame_index = -1
        self.last_now = 0.0
        self._fps: list[float] = []
        self._dt: list[float] = []
        self._window = 30
        self.frames_by_camera: dict[str, int] = dict.fromkeys(rig.camera_ids, 0)
        """How many steps each camera actually contributed a frame to. A camera
        whose count stalls is unplugged, not empty."""
        self._bev: Any | None = None
        if self.metric and self.config.show_bev:
            from mcreid.viz.bev import BevRenderer

            self._bev = BevRenderer(
                rig,
                canvas_size=(self.config.bev_size, self.config.bev_size),
                grid_step_m=1.0,
                trail_length=45,
            )

    # --- metrics ----------------------------------------------------------

    def _push(self, series: list[float], value: float) -> None:
        series.append(value)
        if len(series) > self._window:
            del series[0]

    @property
    def fps(self) -> float:
        """Processing throughput: detect + embed + track + fuse + render."""
        return float(np.mean(self._fps)) if self._fps else 0.0

    @property
    def wall_fps(self) -> float:
        """End-to-end loop rate, capture handoff and display included."""
        return 1.0 / float(np.mean(self._dt)) if self._dt else 0.0

    @property
    def reported_ids(self) -> list[int]:
        """Global IDs that ever reached CONFIRMED — the identities a viewer saw."""
        return sorted(self.timeline.first_seen)

    # --- frame loop -------------------------------------------------------

    def process(
        self, frames: Mapping[str, Image], now: float, dt: float
    ) -> tuple[Image, dict[str, Any]]:
        """Track one step. `frames` holds only the cameras with a fresh frame."""
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        if not frames:
            raise ValueError("process needs at least one camera's frame")
        self.frame_index += 1
        self.last_now = now
        self._push(self._dt, dt)
        started = time.perf_counter()

        for camera_id in frames:
            self.frames_by_camera[camera_id] = self.frames_by_camera.get(camera_id, 0) + 1

        observations = self.backend.step(frames, self.frame_index)
        snapshots = self.manager.step(observations, self.frame_index, dt)
        if self.shadow is not None:
            self.shadow.observe(self.manager, self.frame_index, now)

        for snap in snapshots:
            if snap.state is not TrackState.COASTING:
                self.timeline.observe(snap.global_id, now, self.config.reacquire_gap_s)
        for gid in self.ledger.observe(snapshots, self.frame_index):
            logger.info(
                "frame %d: global id %d is now held ACROSS VIEWS by cameras %s",
                self.frame_index,
                gid,
                tuple(sorted(self.ledger.cameras_ever[gid])),
            )

        mosaic = self._render(frames, observations, snapshots, now)
        self._push(self._fps, 1.0 / max(time.perf_counter() - started, 1e-6))

        return mosaic, {
            "frame": self.frame_index,
            "cameras": sorted(frames),
            "observations": len(observations),
            "tracks": len(snapshots),
            "coasting": sum(1 for s in snapshots if s.state is TrackState.COASTING),
            "multi_camera_tracks": sum(1 for s in snapshots if len(s.supporting_cameras) >= 2),
            "dormant": len(self.manager.dormant),
            "resurrected": self.manager.dormant.n_resurrected,
            "reported_ids": len(self.timeline.first_seen),
            "fps": self.fps,
            "wall_fps": self.wall_fps,
        }

    # --- rendering --------------------------------------------------------

    def _render(
        self,
        frames: Mapping[str, Image],
        observations: Sequence[ViewObservation],
        snapshots: Sequence[GlobalTrackSnapshot],
        now: float,
    ) -> Image:
        states = {s.global_id: s for s in snapshots}
        by_camera: dict[str, list[ViewObservation]] = {}
        for obs in observations:
            by_camera.setdefault(obs.camera_id, []).append(obs)

        tiles = []
        for camera_id in self.rig.camera_ids:
            frame = frames.get(camera_id)
            tile = (
                self._annotate(
                    camera_id, frame, by_camera.get(camera_id, ()), states, now
                )
                if frame is not None
                else self._stale_tile(camera_id)
            )
            tiles.append(tile)
        mosaic = self._mosaic(tiles)
        if self._bev is not None:
            mosaic = self._attach_bev(mosaic, snapshots)
        return self._draw_banner(mosaic, snapshots, now)

    def _attach_bev(self, canvas: Image, snapshots: Sequence[GlobalTrackSnapshot]) -> Image:
        assert self._bev is not None
        panel = self._bev.render(list(snapshots), self.frame_index)
        # Scaled to the mosaic's height, keeping the plan's aspect ratio: a
        # metric map stretched to fit is worse than no map, which is a bug this
        # project already shipped once in the HPC demo.
        scale = canvas.shape[0] / panel.shape[0]
        resized = cv2.resize(
            panel,
            (max(int(panel.shape[1] * scale), 1), canvas.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        return np.asarray(np.hstack([canvas, np.asarray(resized, dtype=np.uint8)]), dtype=np.uint8)

    def _annotate(
        self,
        camera_id: str,
        frame: Image,
        observations: Sequence[ViewObservation],
        states: Mapping[int, GlobalTrackSnapshot],
        now: float,
    ) -> Image:
        canvas = frame.copy()
        assignment = self.manager.last_assignment
        for obs in observations:
            gid = assignment.get((camera_id, obs.local_track_id))
            box = np.asarray(obs.bbox_xyxy, dtype=np.float64)
            p0 = (int(box[0]), int(box[1]))
            p1 = (int(box[2]), int(box[3]))
            if gid is None:
                # Either not yet associated, or dropped before the fusion stage.
                # Grey rather than invisible: an unlabelled person on screen is
                # the difference between "nobody there" and "not fused yet".
                cv2.rectangle(canvas, p0, p1, (120, 120, 120), 1)
                continue

            snap = states.get(gid)
            coasting = snap is not None and snap.state is TrackState.COASTING
            cross = snap is not None and len(snap.supporting_cameras) >= 2
            colour = id_color(gid)
            cv2.rectangle(canvas, p0, p1, colour, 2 if coasting else 3)

            state_name = snap.state.value.upper() if snap else "TENTATIVE"
            held = self.timeline.held_seconds(gid, now)
            label = f"ID {gid} {state_name} {held:.0f}s" + ("  BOTH VIEWS" if cross else "")
            (tw, th), _ = cv2.getTextSize(label, _FONT, 0.55, 2)
            top = max(p0[1] - th - 12, 0)
            left = max(min(p0[0], canvas.shape[1] - tw - 14), 0)
            cv2.rectangle(canvas, (left, top), (left + tw + 12, top + th + 10), colour, -1)
            cv2.putText(
                canvas, label, (left + 6, top + th + 3), _FONT, 0.55, (0, 0, 0), 2, cv2.LINE_AA
            )

        cv2.putText(canvas, camera_id, (10, 26), _FONT, 0.7, TEXT_COLOR, 2, cv2.LINE_AA)
        return canvas

    def _stale_tile(self, camera_id: str) -> Image:
        calib = self.rig.get(camera_id)
        width, height = calib.intrinsics.image_size
        tile = np.full((height, width, 3), 24, dtype=np.uint8)
        cv2.putText(
            tile, f"{camera_id}: no new frame", (10, 40), _FONT, 0.7, (90, 90, 90), 2, cv2.LINE_AA
        )
        return tile

    def _mosaic(self, tiles: Sequence[Image]) -> Image:
        """Tiles side by side, scaled to a common height.

        Cameras on this rig have different native resolutions (720p laptop, VGA
        USB), so a plain hstack would fail. Equal displayed height, not equal
        pixels — the smaller sensor should not be shown as the smaller view.
        """
        target = self.config.tile_height
        scaled = []
        for tile in tiles:
            height, width = tile.shape[:2]
            new_w = max(int(round(width * target / height)), 1)
            scaled.append(cv2.resize(tile, (new_w, target), interpolation=cv2.INTER_AREA))
        if len(scaled) <= 2:
            return np.asarray(np.hstack(scaled), dtype=np.uint8)
        # 2 x ceil(n/2): a single row of four VGA tiles is unreadably small.
        half = (len(scaled) + 1) // 2
        rows = [scaled[:half], scaled[half:]]
        widths = [sum(t.shape[1] for t in row) for row in rows]
        padded = []
        for row, width in zip(rows, widths, strict=True):
            strip = np.asarray(np.hstack(row), dtype=np.uint8)
            if width < max(widths):
                pad = np.full((target, max(widths) - width, 3), 24, dtype=np.uint8)
                strip = np.asarray(np.hstack([strip, pad]), dtype=np.uint8)
            padded.append(strip)
        return np.asarray(np.vstack(padded), dtype=np.uint8)

    def _draw_banner(
        self, canvas: Image, snapshots: Sequence[GlobalTrackSnapshot], now: float
    ) -> Image:
        live = [s for s in snapshots if s.state is not TrackState.COASTING]
        parts = [f"{self.wall_fps:4.1f} FPS", f"tracks {len(snapshots)}"]
        if live:
            held_id = max(live, key=lambda s: self.timeline.held_seconds(s.global_id, now))
            parts.append(
                f"ID {held_id.global_id} held "
                f"{self.timeline.held_seconds(held_id.global_id, now):.0f}s"
            )
        cross = [s for s in snapshots if len(s.supporting_cameras) >= 2]
        if cross:
            parts.append(
                "ACROSS VIEWS: "
                + ", ".join(f"{s.global_id}{tuple(sorted(s.supporting_cameras))}" for s in cross)
            )
        elif self.ledger.multi_camera_ids:
            parts.append(f"cross-view seen: {self.ledger.multi_camera_ids}")
        if self.manager.dormant.n_resurrected:
            parts.append(f"resurrections {self.manager.dormant.n_resurrected}")
        parts.append(
            "calibrated / metric" if self.metric else "uncalibrated / appearance-only (no BEV)"
        )

        strip = np.full((40, canvas.shape[1], 3), 18, dtype=np.uint8)
        text = "   |   ".join(parts)
        # The banner is one line and the mosaic is only as wide as the tiles;
        # shrink rather than run text off the edge.
        scale = 0.58
        while scale > 0.34 and cv2.getTextSize(text, _FONT, scale, 1)[0][0] > canvas.shape[1] - 130:
            scale -= 0.04
        cv2.putText(strip, text, (12, 27), _FONT, scale, TEXT_COLOR, 1, cv2.LINE_AA)
        hint = "q quit   s snapshot"
        (tw, _), _ = cv2.getTextSize(hint, _FONT, 0.5, 1)
        cv2.putText(
            strip, hint, (strip.shape[1] - tw - 12, 27), _FONT, 0.5, (140, 140, 140), 1,
            cv2.LINE_AA,
        )
        return np.asarray(np.vstack([canvas, strip]), dtype=np.uint8)

    # --- reporting --------------------------------------------------------

    def save_snapshot(self, out_dir: Path, mosaic: Image) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"live_multi_{int(time.time())}.png"
        cv2.imwrite(str(path), mosaic)
        logger.info("saved annotated mosaic -> %s", path)
        return path

    def cross_view_report(self) -> list[str]:
        """The acceptance evidence, in the form the run is judged on."""
        lines: list[str] = []
        if not self.ledger.cameras_ever:
            return ["no identity was ever supported by a camera — nothing was tracked"]
        for gid in sorted(self.ledger.cameras_ever):
            cams = tuple(sorted(self.ledger.cameras_ever[gid]))
            multi = self.ledger.frames_multi.get(gid, 0)
            held = self.timeline.held_seconds(gid, self.last_now)
            verdict = "CROSS-VIEW" if len(cams) >= 2 else "single-view"
            first = self.ledger.first_multi_frame.get(gid)
            when = f", first together at frame {first}" if first is not None else ""
            lines.append(
                f"  id {gid}: {verdict} cameras {cams}, {multi} frames with >=2 cameras "
                f"at once{when}, held {held:.1f} s"
            )
        return lines
