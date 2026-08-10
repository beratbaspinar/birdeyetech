"""Composite demo frames: every camera view that produced the result, beside the BEV.

A BEV on its own is not legible. It shows dots on a floor plan and asks the
viewer to take on faith that the dots are people, that the people are the ones in
the footage, and that the number over a dot is the same number that was over that
person in each camera. **The composite is the evidence for all three**, because
the claim of this project is a cross-panel one: the same integer, in the same
colour, over the same human, in every view that can see them, and on the map.

## One renderer, four stages, on purpose

`OverlayStage` is cumulative — each stage is the previous one plus one layer:

    raw        the camera views, untouched
    boxes      + detection/per-view boxes, in the CAMERA's colour, no identity claim
    ids        + the global ID label, in the ID's colour, consistent across panels
    composite  + the BEV floor plan as the final panel

They come out of the same code path rather than four pipelines, so a single-stage
video cannot drift from the composite it is supposed to be an excerpt of. The
stage boundary is also the argument: `boxes` is what a per-camera detector alone
buys you, and `ids` is the only place a cross-camera claim is made.

## The licence guard is in this module, not in a convention

Every frame this module produces contains dataset pixels — that is the entire
point of it. `CLAUDE.md`: anything rendered from the dataset **is** the dataset,
and a `!docs/assets/*.gif` whitelist once let 5.9 MB of real WILDTRACK frames
into git history. So `write_stage_video` **refuses** to write under `docs/`,
which is the only tracked asset tree in this repo. That is a structural refusal
in the same spirit as `BevRenderer.render` taking no image argument: the BEV
cannot contain a dataset pixel because there is no way to hand it one, and a
composite cannot land in a tracked directory because the writer will not do it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from mcreid.fusion.types import ViewObservation
from mcreid.viz.palette import BGR, TEXT_COLOR, camera_color, id_color

Image = npt.NDArray[np.uint8]

_FONT = cv2.FONT_HERSHEY_DUPLEX
_PANEL_BG: BGR = (16, 16, 18)
_STRIP_BG: BGR = (18, 18, 20)
_UNFUSED: BGR = (110, 110, 110)
_STAMP_COLOR: BGR = (150, 150, 150)


class OverlayStage(str, Enum):
    """Cumulative overlay stages. Order is meaning, not presentation."""

    RAW = "raw"
    BOXES = "boxes"
    IDS = "ids"
    COMPOSITE = "composite"

    @property
    def _rank(self) -> int:
        return _STAGE_ORDER.index(self)

    @property
    def draws_boxes(self) -> bool:
        return self._rank >= OverlayStage.BOXES._rank

    @property
    def draws_ids(self) -> bool:
        return self._rank >= OverlayStage.IDS._rank

    @property
    def draws_bev(self) -> bool:
        return self._rank >= OverlayStage.COMPOSITE._rank


_STAGE_ORDER: tuple[OverlayStage, ...] = (
    OverlayStage.RAW,
    OverlayStage.BOXES,
    OverlayStage.IDS,
    OverlayStage.COMPOSITE,
)

ALL_STAGES: tuple[OverlayStage, ...] = _STAGE_ORDER

STAGE_CAPTIONS: dict[OverlayStage, str] = {
    OverlayStage.RAW: "raw camera views - no detection, no tracking, no claim",
    OverlayStage.BOXES: (
        "per-camera boxes only - each camera's own colour, no identity claim across views"
    ),
    OverlayStage.IDS: (
        "global IDs - one integer, one colour per person, in every view that sees them"
    ),
    OverlayStage.COMPOSITE: "global IDs + the fused floor plan - the same identities, one map",
}


@dataclass(frozen=True)
class PanelRecord:
    """What was actually drawn in one camera panel, for machine checking.

    The composite's whole claim is cross-panel agreement, so the renderer reports
    what it drew rather than leaving a reader to trust the pixels. ``drawn`` is
    ``(global_id, colour)`` pairs — the colour is recorded, not re-derived, so a
    renderer that coloured by camera instead of by identity would be caught by
    the same aggregate that reports success.
    """

    camera_id: str
    boxes: int
    drawn: tuple[tuple[int, BGR], ...]
    withheld: int = 0
    """Boxes the fusion stage HAD assigned a global ID, drawn unlabelled because
    that ID is not on the floor plan this frame.

    A number over a person in a camera panel and no dot for it on the map is the
    composite contradicting itself, and the contradiction is not cosmetic: the
    claim being made is that these are one identity space. So a track the manager
    has not put on the map is shown as a considered-but-uncommitted detection, in
    grey, and the count is reported rather than quietly dropped."""

    @property
    def ids(self) -> frozenset[int]:
        return frozenset(gid for gid, _ in self.drawn)


@dataclass(frozen=True)
class CompositeRecord:
    """What one composite frame asserts, without the pixels.

    Split from the image deliberately: a run is streamed to a video writer frame
    by frame, so holding 40 x 1080p composites in a list to summarise them later
    costs hundreds of megabytes to compute two integers. The records are kept,
    the images are not.
    """

    frame: int
    stage: OverlayStage
    panels: tuple[PanelRecord, ...]

    @property
    def ids_in_multiple_panels(self) -> frozenset[int]:
        """Global IDs drawn in two or more camera panels this frame.

        This is the demonstration itself. If it is empty for the whole run the
        composite shows four independent trackers, not one fused identity space,
        and no amount of legible layout fixes that.
        """
        counts: dict[int, int] = {}
        for panel in self.panels:
            for gid in panel.ids:
                counts[gid] = counts.get(gid, 0) + 1
        return frozenset(gid for gid, n in counts.items() if n >= 2)

    @property
    def colour_conflicts(self) -> frozenset[int]:
        """Global IDs drawn in more than one colour across this frame's panels."""
        seen: dict[int, set[BGR]] = {}
        for panel in self.panels:
            for gid, colour in panel.drawn:
                seen.setdefault(gid, set()).add(colour)
        return frozenset(gid for gid, colours in seen.items() if len(colours) > 1)


@dataclass(frozen=True, eq=False)
class CompositeFrame:
    """One rendered composite frame: the pixels, plus what they assert."""

    image: Image
    record: CompositeRecord

    @property
    def frame(self) -> int:
        return self.record.frame

    @property
    def stage(self) -> OverlayStage:
        return self.record.stage


def grid_shape(
    n_tiles: int, tile_size: tuple[int, int], target_aspect: float = 16.0 / 9.0
) -> tuple[int, int]:
    """Pick ``(rows, cols)`` for ``n_tiles`` tiles of ``tile_size``.

    Scored on how close the resulting grid lands to ``target_aspect``, measured as
    a **log ratio** so that twice-too-wide and twice-too-tall cost the same — a
    linear difference silently prefers tall layouts, because aspect is bounded
    below by 0 and unbounded above. Ties break toward fewer empty slots.

    Concretely: 4 cameras of 5:4 frames give 2x2, and 7 cameras of 16:9 frames
    give 3x3 with two blanks rather than 4x2, because 4x2 of 16:9 tiles is a
    3.6:1 letterbox nobody can read.
    """
    if n_tiles < 1:
        raise ValueError(f"n_tiles must be >= 1, got {n_tiles}")
    tile_w, tile_h = tile_size
    if tile_w <= 0 or tile_h <= 0:
        raise ValueError(f"degenerate tile_size {tile_size}")

    best: tuple[float, int, int, int] | None = None
    for rows in range(1, n_tiles + 1):
        cols = math.ceil(n_tiles / rows)
        aspect = (cols * tile_w) / (rows * tile_h)
        score = (abs(math.log(aspect / target_aspect)), rows * cols - n_tiles, rows, cols)
        if best is None or score < best:
            best = score
    assert best is not None
    return best[2], best[3]


def assert_generated_only(path: Path) -> Path:
    """Refuse a composite destination inside the repo's tracked asset tree.

    Composites contain dataset pixels by construction. `docs/` is the only
    directory in this repo whose contents are tracked and shipped, and
    `.gitignore` whitelists assets there **by exact filename** precisely because
    a directory glob once let real WILDTRACK frames into history. Rather than
    trust the next caller to remember that, the writer will not write there.
    """
    resolved = path.expanduser().resolve()
    for parent in (resolved, *resolved.parents):
        if parent.name == "docs":
            raise ValueError(
                f"refusing to write a composite to {path}: it contains dataset pixels, and "
                "docs/ is the tracked asset tree (CLAUDE.md — anything rendered from the "
                "dataset IS the dataset). Write it under reports/, which is gitignored."
            )
    return resolved


class StageWriter:
    """Streaming mp4 writer for one overlay stage.

    Streaming rather than collecting: four stages x 40 frames of a 7-camera
    1080p composite is roughly a gigabyte of RAM held only so it can be handed to
    a writer at the end. The destination is checked at construction, before a
    single frame is rendered, so a bad path fails in the first second of a run
    rather than after the GPU work.
    """

    def __init__(self, path: Path, fps: float) -> None:
        # Checked on the RESOLVED path (so `reports/../docs/x.mp4` cannot sneak
        # through) but KEPT as given. The path is written into a committed JSON,
        # and an absolute one would put `C:\Users\<name>\...` into a public repo —
        # something this project's pre-ship audit checks for by name.
        assert_generated_only(path)
        self.path = path
        self.fps = fps
        self.frames = 0
        self._writer: cv2.VideoWriter | None = None

    def write(self, image: Image) -> None:
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            height, width = image.shape[:2]
            self._writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter.fourcc(*"mp4v"), self.fps, (width, height)
            )
            if not self._writer.isOpened():
                raise RuntimeError(f"could not open video writer for {self.path}")
        self._writer.write(image)
        self.frames += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> StageWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def write_stage_video(frames: list[Image], path: Path, fps: float) -> Path:
    """Write a composite video in one call, refusing any tracked destination."""
    if not frames:
        raise ValueError("no frames to write")
    with StageWriter(path, fps) as writer:
        for frame in frames:
            writer.write(frame)
    return writer.path


def _fit(image: Image, size: tuple[int, int]) -> Image:
    """Resize preserving aspect ratio, letterboxed onto a canvas of ``size``."""
    target_w, target_h = size
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    resized = cv2.resize(
        image, (max(int(w * scale), 1), max(int(h * scale), 1)), interpolation=cv2.INTER_AREA
    )
    canvas = np.full((target_h, target_w, 3), _PANEL_BG[0], dtype=np.uint8)
    canvas[:] = _PANEL_BG
    y = (target_h - resized.shape[0]) // 2
    x = (target_w - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


class CompositeRenderer:
    """N camera panels beside the BEV floor plan, at one overlay stage."""

    def __init__(
        self,
        camera_order: list[str],
        frame_size: tuple[int, int],
        *,
        stage: OverlayStage = OverlayStage.COMPOSITE,
        tile_width: int = 480,
        target_aspect: float = 16.0 / 9.0,
    ) -> None:
        if not camera_order:
            raise ValueError("camera_order must be non-empty")
        width, height = frame_size
        if width <= 0 or height <= 0:
            raise ValueError(f"degenerate frame_size {frame_size}")

        self.camera_order = list(camera_order)
        self.stage = stage
        # Even tile height: an odd dimension makes some H.264/mp4v encoders pad
        # silently, and a padded frame is a different frame than the one hashed.
        tile_height = max(2 * round(tile_width * height / width / 2), 2)
        self.tile_size = (tile_width, tile_height)
        self.rows, self.cols = grid_shape(len(camera_order), self.tile_size, target_aspect)
        self.grid_size = (self.cols * tile_width, self.rows * tile_height)

    @property
    def bev_size(self) -> tuple[int, int]:
        """The BEV panel is square at the camera grid's full height."""
        return (self.grid_size[1], self.grid_size[1])

    def _draw_panel(
        self,
        frame: Image,
        observations: list[ViewObservation],
        assignment: dict[tuple[str, int], int],
        camera_id: str,
        camera_index: int,
        live_ids: frozenset[int] | None,
    ) -> tuple[Image, PanelRecord]:
        """One camera tile. Boxes are drawn at source resolution, then scaled, so
        line weight and text stay proportional instead of turning to mush."""
        canvas = frame.copy()
        tile_w, tile_h = self.tile_size
        scale = tile_w / canvas.shape[1]
        thickness = max(int(round(3 / scale)), 2)
        cam_colour = camera_color(camera_index)
        drawn: list[tuple[int, BGR]] = []
        boxes = 0
        withheld = 0

        if self.stage.draws_boxes:
            # Far boxes first, so near ones and their labels land on top.
            for obs in sorted(observations, key=lambda o: float(o.bbox_xyxy[3])):
                box = np.asarray(obs.bbox_xyxy, dtype=np.float64)
                p0 = (int(box[0]), int(box[1]))
                p1 = (int(box[2]), int(box[3]))
                gid = assignment.get((camera_id, obs.local_track_id))
                boxes += 1

                if not self.stage.draws_ids:
                    # Deliberately the CAMERA's colour: at this stage no
                    # cross-camera claim has been made, and colouring by identity
                    # here would show the viewer the answer one stage early.
                    cv2.rectangle(canvas, p0, p1, cam_colour, thickness)
                    continue
                on_map = live_ids is None or gid in live_ids
                if gid is None or not on_map:
                    # Drawn thin and grey rather than hidden: the viewer should
                    # see a detection the fusion stage considered and did not
                    # commit to. `not on_map` lands here for the same reason —
                    # labelling it would put a number on screen that the floor
                    # plan beside it does not carry.
                    cv2.rectangle(canvas, p0, p1, _UNFUSED, max(thickness // 2, 1))
                    withheld += gid is not None
                    continue

                colour = id_color(gid)
                cv2.rectangle(canvas, p0, p1, colour, thickness)
                drawn.append((gid, colour))
                self._label(canvas, p0, float(box[3] - box[1]), gid, colour, scale, thickness)

        panel: Image = np.asarray(
            cv2.resize(canvas, self.tile_size, interpolation=cv2.INTER_AREA), dtype=np.uint8
        )
        cv2.rectangle(panel, (0, 0), (tile_w - 1, tile_h - 1), cam_colour, 2)
        _banner(panel, camera_id, cam_colour)
        return panel, PanelRecord(
            camera_id=camera_id, boxes=boxes, drawn=tuple(drawn), withheld=withheld
        )

    def _label(
        self,
        canvas: Image,
        p0: tuple[int, int],
        box_h: float,
        gid: int,
        colour: BGR,
        scale: float,
        thickness: int,
    ) -> None:
        """Label size follows the box. A fixed size turns a distant group into a
        solid band of overlapping chips that hides the people it is labelling."""
        font_scale = float(np.clip(box_h / (150 / scale), 0.5, 1.3)) / scale
        text = str(gid)
        (tw, th), _ = cv2.getTextSize(text, _FONT, font_scale, thickness)
        pad = int(6 / scale)
        top = max(p0[1] - th - 2 * pad, 0)
        cv2.rectangle(canvas, (p0[0], top), (p0[0] + tw + 2 * pad, top + th + 2 * pad), colour, -1)
        cv2.putText(
            canvas, text, (p0[0] + pad, top + th + pad), _FONT, font_scale, (0, 0, 0), thickness,
            cv2.LINE_AA,
        )

    def render(
        self,
        views: dict[str, Image],
        observations: dict[str, list[ViewObservation]],
        assignment: dict[tuple[str, int], int],
        bev: Image | None,
        frame: int,
        *,
        live_ids: frozenset[int] | None = None,
        caption: str = "",
        subcaption: str = "",
    ) -> CompositeFrame:
        """Build one composite frame at this renderer's stage.

        ``live_ids`` are the global IDs the fusion stage put on the floor plan
        this frame. Passing them keeps the panels and the map saying the same
        thing; omitting them labels every assigned box, which is what you want
        when there is no map to contradict.
        """
        if self.stage.draws_bev and bev is None:
            raise ValueError("stage 'composite' needs a BEV panel, got None")

        tile_w, tile_h = self.tile_size
        grid = np.full((self.rows * tile_h, self.cols * tile_w, 3), _PANEL_BG[0], dtype=np.uint8)
        grid[:] = _PANEL_BG
        panels: list[PanelRecord] = []

        for index, camera_id in enumerate(self.camera_order):
            source = views.get(camera_id)
            if source is None:
                continue
            panel, record = self._draw_panel(
                source, observations.get(camera_id, []), assignment, camera_id, index, live_ids
            )
            row, col = divmod(index, self.cols)
            grid[row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w] = panel
            panels.append(record)

        composed = grid
        if self.stage.draws_bev and bev is not None:
            bev_panel = _fit(bev, self.bev_size)
            cv2.rectangle(
                bev_panel, (0, 0), (bev_panel.shape[1] - 1, bev_panel.shape[0] - 1), (70, 70, 70), 2
            )
            composed = np.hstack([grid, bev_panel])

        strip = self._strip(
            composed.shape[1],
            caption or STAGE_CAPTIONS[self.stage],
            subcaption,
            frame,
        )
        image: Image = np.asarray(np.vstack([composed, strip]), dtype=np.uint8)
        return CompositeFrame(
            image=image,
            record=CompositeRecord(frame=frame, stage=self.stage, panels=tuple(panels)),
        )

    def _strip(self, width: int, caption: str, subcaption: str, frame: int) -> Image:
        height = 76 if subcaption else 50
        strip = np.full((height, width, 3), _STRIP_BG[0], dtype=np.uint8)
        strip[:] = _STRIP_BG
        cv2.putText(strip, caption, (16, 32), _FONT, 0.62, TEXT_COLOR, 1, cv2.LINE_AA)
        if subcaption:
            cv2.putText(strip, subcaption, (16, 61), _FONT, 0.55, (150, 220, 255), 1, cv2.LINE_AA)
        stamp = f"frame {frame}"
        (tw, _), _ = cv2.getTextSize(stamp, _FONT, 0.55, 1)
        cv2.putText(
            strip, stamp, (width - tw - 16, 32), _FONT, 0.55, _STAMP_COLOR, 1, cv2.LINE_AA
        )
        return strip


def _banner(panel: Image, text: str, colour: BGR) -> None:
    (tw, th), _ = cv2.getTextSize(text, _FONT, 0.66, 2)
    cv2.rectangle(panel, (0, 0), (tw + 18, th + 16), _STRIP_BG, -1)
    cv2.putText(panel, text, (9, th + 7), _FONT, 0.66, colour, 2, cv2.LINE_AA)


def summarise(frames: list[CompositeRecord]) -> dict[str, object]:
    """Aggregate the cross-panel claim over a run, for the results JSON.

    Reports the two things a gate can actually check about a composite: that the
    same identity was drawn in more than one camera panel (otherwise the video
    shows N independent trackers side by side), and that no identity was ever
    drawn in two different colours (otherwise "follow the colour" is a lie).
    """
    if not frames:
        raise ValueError("no frames to summarise")
    multi = [len(f.ids_in_multiple_panels) for f in frames]
    conflicts = sorted({gid for f in frames for gid in f.colour_conflicts})
    labelled = sum(len(p.drawn) for f in frames for p in f.panels)
    withheld = sum(p.withheld for f in frames for p in f.panels)
    return {
        "frames": len(frames),
        "stage": frames[0].stage.value,
        "camera_panels": len(frames[0].panels),
        "labelled_boxes": labelled,
        "assigned_boxes_withheld_because_not_on_the_map": withheld,
        "frames_with_an_id_in_multiple_panels": int(sum(1 for m in multi if m > 0)),
        "max_ids_in_multiple_panels_in_a_frame": int(max(multi)),
        "id_colour_conflicts": conflicts,
        "id_colour_consistent": not conflicts,
    }
