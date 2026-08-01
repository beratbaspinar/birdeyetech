"""Click-to-mark helper for floor markers, with a sub-pixel loupe.

Marking accuracy is the binding constraint on whether a rig passes its gate:
measured pass rate over 12 seeds is 100 % at 0.5 px of click error, 75 % at
1.0 px and **8 % at 2.0 px**. A mouse click on a full-frame view is worth about
one screen pixel, and on a 1280x720 image shown scaled-to-fit that is already
worse than 1.0 image px — so clicking on the main view alone cannot reliably
produce a rig that passes.

Hence the loupe. A coarse click puts the cursor in the neighbourhood; a second
click inside a magnified window refines it, and a click in a 16x loupe is worth
**1/16 px**. That is the whole design: two clicks per marker, and the second one
is what the calibration actually uses.

The geometry and the state machine live here as pure functions so they can be
tested without a display; only `run_marker_ui` touches a window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from mcreid.utils.logging import get_logger

logger = get_logger(__name__)

Image = npt.NDArray[np.uint8]

DEFAULT_ZOOM = 16
"""A click in a 16x loupe resolves 1/16 px — an order below the 0.5 px that
measured 100 % pass. Higher zooms show fewer real pixels and stop helping."""
DEFAULT_LOUPE_PX = 480


def view_to_image(click: tuple[float, float], view_scale: float) -> tuple[float, float]:
    """Main-view click -> image coordinates. `view_scale` is displayed/actual."""
    if view_scale <= 0:
        raise ValueError(f"view_scale must be positive, got {view_scale}")
    return (click[0] / view_scale, click[1] / view_scale)


def loupe_to_image(
    click: tuple[float, float],
    centre: tuple[float, float],
    zoom: int = DEFAULT_ZOOM,
    loupe_px: int = DEFAULT_LOUPE_PX,
) -> tuple[float, float]:
    """Loupe click -> **sub-pixel** image coordinates.

    The loupe shows a `loupe_px / zoom` wide region of the image centred on
    `centre`, magnified `zoom` times. A click therefore resolves `1 / zoom` of an
    image pixel, which is the entire reason this helper exists.
    """
    if zoom <= 0:
        raise ValueError(f"zoom must be positive, got {zoom}")
    if loupe_px <= 0:
        raise ValueError(f"loupe_px must be positive, got {loupe_px}")
    half = loupe_px / 2.0
    return (
        centre[0] + (click[0] - half) / zoom,
        centre[1] + (click[1] - half) / zoom,
    )


def render_loupe(
    image: Image,
    centre: tuple[float, float],
    zoom: int = DEFAULT_ZOOM,
    loupe_px: int = DEFAULT_LOUPE_PX,
) -> Image:
    """Magnified patch around `centre`, with a crosshair on the exact centre.

    `getRectSubPix` is used rather than slicing because the centre is
    sub-pixel — the loupe has to be able to show the difference between 412.0
    and 412.5, which is the difference the gate cares about. INTER_NEAREST keeps
    real pixel boundaries visible so the eye can land on a corner.
    """
    import cv2

    patch_size = max(int(round(loupe_px / zoom)), 2)
    patch = cv2.getRectSubPix(image, (patch_size, patch_size), centre)
    view = cv2.resize(patch, (loupe_px, loupe_px), interpolation=cv2.INTER_NEAREST)

    half = loupe_px // 2
    cv2.line(view, (half, 0), (half, loupe_px), (0, 255, 255), 1)
    cv2.line(view, (0, half), (loupe_px, half), (0, 255, 255), 1)
    cv2.circle(view, (half, half), max(zoom // 2, 3), (0, 0, 255), 1)
    return np.asarray(view, dtype=np.uint8)


@dataclass
class MarkerState:
    """Which markers are placed, and which one is being placed now."""

    camera_id: str
    image_size: tuple[int, int]
    n_markers: int
    points: list[tuple[float, float] | None] = field(default_factory=list)
    index: int = 0

    def __post_init__(self) -> None:
        if self.n_markers < 1:
            raise ValueError(f"n_markers must be >= 1, got {self.n_markers}")
        if not self.points:
            self.points = [None] * self.n_markers
        if len(self.points) != self.n_markers:
            raise ValueError(
                f"{len(self.points)} points for {self.n_markers} markers"
            )

    def set_point(self, xy: tuple[float, float]) -> None:
        """Place the current marker, clamped to the image.

        Clamped rather than rejected: a click one pixel outside the frame is a
        hand tremor, not a decision to skip a marker, and silently dropping it
        would leave a None that only surfaces at save time.
        """
        width, height = self.image_size
        self.points[self.index] = (
            float(np.clip(xy[0], 0.0, width - 1)),
            float(np.clip(xy[1], 0.0, height - 1)),
        )

    def nudge(self, dx: float, dy: float) -> None:
        current = self.points[self.index]
        if current is not None:
            self.set_point((current[0] + dx, current[1] + dy))

    def advance(self, step: int = 1) -> None:
        self.index = int(np.clip(self.index + step, 0, self.n_markers - 1))

    def next_unplaced(self) -> None:
        """Jump to the first marker with no point — how you finish a session."""
        for i, point in enumerate(self.points):
            if point is None:
                self.index = i
                return

    def clear_current(self) -> None:
        self.points[self.index] = None

    @property
    def placed(self) -> int:
        return sum(1 for p in self.points if p is not None)

    @property
    def complete(self) -> bool:
        return self.placed == self.n_markers


def merge_markers(
    document: dict[str, Any],
    camera_id: str,
    points: list[tuple[float, float] | None],
    image_size: tuple[int, int],
) -> dict[str, Any]:
    """Write one camera's points into a floor-marker document.

    Other cameras and the world list are left exactly as they were, so marking
    cam0 and cam1 in separate sittings composes into one file — which is how
    anyone with two cameras and one tripod will actually do it.
    """
    missing = [i for i, p in enumerate(points) if p is None]
    if missing:
        raise ValueError(f"{camera_id}: markers {missing} were never placed")

    merged = dict(document)
    cameras = dict(merged.get("cameras") or {})
    cameras[camera_id] = {
        "image_size": [int(image_size[0]), int(image_size[1])],
        # Six decimals: the loupe resolves 1/16 px and rounding to whole pixels
        # here would throw away exactly the precision it exists to capture.
        "image_points": [[round(float(x), 6), round(float(y), 6)] for x, y in points],  # type: ignore[misc]
    }
    merged["cameras"] = cameras
    return merged


def load_document(path: Path) -> dict[str, Any]:
    """Existing floor-marker YAML, or a skeleton if there is none yet."""
    import yaml

    if not path.is_file():
        return {"world_points": [], "cameras": {}}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    loaded.setdefault("world_points", [])
    loaded.setdefault("cameras", {})
    return dict(loaded)


def save_document(path: Path, document: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(document, sort_keys=False, default_flow_style=None),
        encoding="utf-8",
    )


HELP = [
    "left-click MAIN view  : coarse position for this marker",
    "left-click LOUPE      : refine to sub-pixel  <- this is what gets saved",
    "arrow keys            : nudge 0.1 px         SHIFT+arrows: 1 px",
    "n / p                 : next / previous marker",
    "c                     : clear this marker",
    "s                     : save and quit        q: quit without saving",
]


def run_marker_ui(
    image: Image,
    camera_id: str,
    n_markers: int,
    existing: list[tuple[float, float] | None] | None = None,
    zoom: int = DEFAULT_ZOOM,
    loupe_px: int = DEFAULT_LOUPE_PX,
    max_view_px: int = 1100,
) -> MarkerState | None:
    """Interactive marking. Returns the state, or None if the user quit."""
    import cv2

    height, width = image.shape[:2]
    state = MarkerState(
        camera_id=camera_id,
        image_size=(width, height),
        n_markers=n_markers,
        points=list(existing) if existing else [],
    )
    view_scale = min(max_view_px / width, 1.0)
    view_size = (int(width * view_scale), int(height * view_scale))
    centre: list[float] = [width / 2.0, height / 2.0]
    window = f"mark floor markers — {camera_id}"

    def on_main(event: int, x: int, y: int, flags: int, _: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            point = view_to_image((x, y), view_scale)
            centre[0], centre[1] = point
            state.set_point(point)

    def on_loupe(event: int, x: int, y: int, flags: int, _: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            point = loupe_to_image((x, y), (centre[0], centre[1]), zoom, loupe_px)
            state.set_point(point)

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_main)
    loupe_window = f"loupe {zoom}x — click here to refine"
    cv2.namedWindow(loupe_window)
    cv2.setMouseCallback(loupe_window, on_loupe)

    try:
        while True:
            view = cv2.resize(image, view_size, interpolation=cv2.INTER_AREA)
            for i, point in enumerate(state.points):
                if point is None:
                    continue
                px = (int(point[0] * view_scale), int(point[1] * view_scale))
                colour = (0, 255, 0) if i != state.index else (0, 165, 255)
                cv2.drawMarker(view, px, colour, cv2.MARKER_CROSS, 14, 1)
                cv2.putText(
                    view, str(i), (px[0] + 6, px[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA,
                )
            current = state.points[state.index]
            status = (
                f"{camera_id}  marker {state.index}/{n_markers - 1}  "
                f"placed {state.placed}/{n_markers}  "
                + (f"at ({current[0]:.2f}, {current[1]:.2f})" if current else "UNPLACED")
            )
            bar = np.full((26, view_size[0], 3), 20, dtype=np.uint8)
            cv2.putText(
                bar, status, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (240, 240, 240), 1, cv2.LINE_AA,
            )
            cv2.imshow(window, np.vstack([view, bar]))
            cv2.imshow(loupe_window, render_loupe(image, (centre[0], centre[1]), zoom, loupe_px))

            key = cv2.waitKey(20) & 0xFFFF
            if key in (ord("q"), 27):
                return None
            if key == ord("s"):
                if not state.complete:
                    logger.warning(
                        "%d marker(s) still unplaced — jumping to the first",
                        n_markers - state.placed,
                    )
                    state.next_unplaced()
                    continue
                return state
            if key == ord("n"):
                state.advance(1)
            elif key == ord("p"):
                state.advance(-1)
            elif key == ord("c"):
                state.clear_current()
            elif key in (81, 2424832):  # left
                state.nudge(-0.1, 0.0)
            elif key in (83, 2555904):  # right
                state.nudge(0.1, 0.0)
            elif key in (82, 2490368):  # up
                state.nudge(0.0, -0.1)
            elif key in (84, 2621440):  # down
                state.nudge(0.0, 0.1)
            focus = state.points[state.index]
            if focus is not None:
                centre[0], centre[1] = focus
    finally:
        cv2.destroyWindow(window)
        cv2.destroyWindow(loupe_window)
