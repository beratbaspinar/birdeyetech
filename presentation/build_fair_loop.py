"""Assemble the fair-loop video from the three existing renders.

Does not run inference and does not modify source files.
"""

from __future__ import annotations

import av
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "presentation"
SEG_DIR = OUT_DIR / "segments"
OFFICE = ROOT / "outputs/demo/recorded.mp4"
HPC = ROOT / "reports/hpc_demo.mp4"
EPFL = ROOT / "reports/epfl_demo/epfl_6p_composite.mp4"

W, H = 1920, 1080
FPS = 30
BG = (16, 17, 20)  # RGB
FG = (236, 237, 239)
MUTED = (168, 172, 178)
LINE = (72, 118, 132)
NOTE = (232, 196, 122)

FONT = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
FONT_BOLD = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT), size)


def blank() -> np.ndarray:
    return np.full((H, W, 3), BG, np.uint8)


def text_width(text: str, face: ImageFont.FreeTypeFont) -> int:
    return int(face.getlength(text))


def draw(
    frame: np.ndarray,
    text: str,
    xy: tuple[int, int],
    face: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
) -> None:
    image = Image.fromarray(frame)
    ImageDraw.Draw(image).text(xy, text, font=face, fill=fill)
    frame[:] = np.asarray(image)


def fit(frame: np.ndarray, max_w: int, max_h: int, allow_upscale: bool) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(max_w / width, max_h / height)
    if not allow_upscale:
        scale = min(scale, 1.0)
    if abs(scale - 1.0) < 0.004:
        return frame
    size = (max(2, int(round(width * scale))), max(2, int(round(height * scale))))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(frame, size, interpolation=interp)


def paste(canvas: np.ndarray, panel_bgr: np.ndarray, y: int) -> None:
    rgb = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    x = (W - width) // 2
    y = max(0, min(y, H - height))
    canvas[y : y + height, x : x + width] = rgb


def rule(frame: np.ndarray, y: int) -> None:
    frame[y : y + 2, 72 : W - 72] = LINE


def open_card() -> np.ndarray:
    frame = blank()
    title = "Multi-Camera Spatial Tracking"
    sub = "Detection  •  Persistent Identity  •  Ground-Plane Mapping"
    title_face = font(58, bold=True)
    sub_face = font(28)
    draw(frame, title, ((W - text_width(title, title_face)) // 2, 430), title_face, FG)
    rule(frame, 520)
    draw(frame, sub, ((W - text_width(sub, sub_face)) // 2, 548), sub_face, MUTED)
    return frame


def close_card() -> np.ndarray:
    frame = blank()
    title = "Camera  →  Global Identity  →  World Position  →  Map"
    sub = "Proof-of-concept"
    title_face = font(40, bold=True)
    sub_face = font(28)
    # Shrink the title if a future font metric runs long.
    while text_width(title, title_face) > W - 120 and title_face.size > 28:
        title_face = font(title_face.size - 2, bold=True)
    draw(frame, title, ((W - text_width(title, title_face)) // 2, 450), title_face, FG)
    rule(frame, 530)
    draw(frame, sub, ((W - text_width(sub, sub_face)) // 2, 560), sub_face, MUTED)
    return frame


def header(
    frame: np.ndarray,
    title: str,
    lines: list[str],
    badge: str | None = None,
) -> int:
    """Draw the section header. Returns the y where the picture may start."""
    draw(frame, title, (72, 36), font(40, bold=True), FG)
    y = 96
    for line in lines:
        draw(frame, line, (72, y), font(24), MUTED)
        y += 34
    rule(frame, y + 8)
    if badge:
        face = font(18, bold=True)
        pad_x, pad_y = 16, 8
        tw = text_width(badge, face)
        th = 22
        x1 = W - 72 - tw - pad_x * 2
        y1 = 40
        overlay = Image.fromarray(frame)
        ImageDraw.Draw(overlay).rounded_rectangle(
            (x1, y1, x1 + tw + pad_x * 2, y1 + th + pad_y * 2),
            radius=6,
            outline=NOTE,
            width=2,
        )
        frame[:] = np.asarray(overlay)
        draw(frame, badge, (x1 + pad_x, y1 + 6), face, NOTE)
    return y + 24


def footer_note(frame: np.ndarray, text: str) -> None:
    face = font(24, bold=True)
    draw(frame, text, ((W - text_width(text, face)) // 2, H - 52), face, NOTE)


def compose_picture(
    panel_bgr: np.ndarray,
    top: int,
    bottom: int,
    allow_upscale: bool,
    max_scale: float | None = None,
) -> tuple[np.ndarray, int]:
    avail_h = bottom - top
    avail_w = W - 144
    height, width = panel_bgr.shape[:2]
    scale = min(avail_w / width, avail_h / height)
    if not allow_upscale:
        scale = min(scale, 1.0)
    if max_scale is not None:
        scale = min(scale, max_scale)
    if abs(scale - 1.0) < 0.004:
        fitted = panel_bgr
    else:
        size = (max(2, int(round(width * scale)) // 2 * 2), max(2, int(round(height * scale)) // 2 * 2))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.CUBIC if False else cv2.INTER_CUBIC
        fitted = cv2.resize(panel_bgr, size, interpolation=interp)
    y = top + max(0, (avail_h - fitted.shape[0]) // 2)
    return fitted, y


def section_frame(
    panel_bgr: np.ndarray,
    title: str,
    lines: list[str],
    badge: str | None = None,
    note: str | None = None,
    allow_upscale: bool = False,
    max_scale: float | None = None,
) -> np.ndarray:
    frame = blank()
    top = header(frame, title, lines, badge)
    bottom = H - (78 if note else 28)
    fitted, y = compose_picture(panel_bgr, top, bottom, allow_upscale, max_scale)
    paste(frame, fitted, y)
    if note:
        footer_note(frame, note)
    return frame


def read_span(path: Path, t0: float, t1: float) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    f0 = int(round(t0 * src_fps))
    f1 = int(round(t1 * src_fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    frames = []
    for _ in range(f1 - f0):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames in {path} {t0}-{t1}")
    return frames


def read_index_span(path: Path, start: int, end: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(end - start):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if len(frames) != end - start:
        raise RuntimeError(f"{path} expected {end - start} frames from {start}, got {len(frames)}")
    return frames


def hold_source_frame(path: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"missing frame {index} in {path}")
    return frame


def resample(frames: list[np.ndarray], n_out: int) -> list[np.ndarray]:
    if n_out == len(frames):
        return frames
    picked = []
    last = len(frames) - 1
    for i in range(n_out):
        src = int(round(i * last / max(n_out - 1, 1)))
        picked.append(frames[src])
    return picked


class Writer:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream("libx264", rate=FPS)
        self.stream.width = W
        self.stream.height = H
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": "18", "preset": "medium"}
        self.index = 0

    def write(self, frame: np.ndarray) -> None:
        video = av.VideoFrame.from_ndarray(frame, format="rgb24")
        video.pts = self.index
        self.index += 1
        for packet in self.stream.encode(video):
            self.container.mux(packet)

    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def build() -> list[tuple[str, int]]:
    # Office: skip the short ID 254 burst near 15.5s and stop before ID 387 near 21.8s.
    # Drop the source status strip. At 540px it already collides with the
    # "quit" hint and a session FPS readout. The two camera views stay whole.
    office_a = [frame[:-40] for frame in read_span(OFFICE, 1.00, 15.20)]
    office_b = [frame[:-40] for frame in read_span(OFFICE, 16.10, 21.30)]
    office_frames = resample(office_a + office_b, int(round(19.4 * FPS)))
    del office_a, office_b

    ranges = {
        "handoff": (78, 480, "Camera handoff"),
        "occlusion": (539, 720, "Total occlusion"),
        "resurrection": (1124, 1334, "Identity resurrection"),
    }
    # EPFL composite caption strip claims one global ID per person. Drop that
    # strip so this card does not repeat the claim. Cameras and the BEV stay.
    epfl_indices = (25, 27, 30)
    epfl_stills = [hold_source_frame(EPFL, index)[:-76] for index in epfl_indices]

    master = Writer(OUT_DIR / "fair_demo_loop.mp4")
    writers = {
        "open": Writer(SEG_DIR / "01_open.mp4"),
        "office": Writer(SEG_DIR / "02_office.mp4"),
        "synthetic": Writer(SEG_DIR / "03_synthetic.mp4"),
        "handoff": Writer(SEG_DIR / "03a_handoff.mp4"),
        "occlusion": Writer(SEG_DIR / "03b_occlusion.mp4"),
        "resurrection": Writer(SEG_DIR / "03c_resurrection.mp4"),
        "epfl": Writer(SEG_DIR / "04_epfl.mp4"),
        "close": Writer(SEG_DIR / "05_close.mp4"),
    }
    counts: list[tuple[str, int]] = []

    def emit(name: str, frame: np.ndarray, also: list[str] | None = None) -> None:
        master.write(frame)
        writers[name].write(frame)
        for extra in also or []:
            writers[extra].write(frame)

    opening = open_card()
    for _ in range(4 * FPS):
        emit("open", opening)
    counts.append(("open", 4 * FPS))

    for panel in office_frames:
        emit(
            "office",
            section_frame(
                panel,
                "REAL-WORLD CROSS-CAMERA ID",
                ["2 recorded camera views", "Same person  •  Global ID 1"],
                allow_upscale=True,
                max_scale=1.6,
            ),
        )
    counts.append(("office", len(office_frames)))
    del office_frames

    for key, (start, end, event) in ranges.items():
        panels = read_index_span(HPC, start, end)
        for panel in panels:
            emit(
                "synthetic",
                section_frame(
                    panel,
                    "MULTI-CAMERA BEHAVIOR",
                    [event],
                    badge="SYNTHETIC SCENARIO",
                ),
                also=[key],
            )
        counts.append((key, len(panels)))
        del panels

    epfl_hold = (11 * FPS) // len(epfl_stills)
    epfl_count = 0
    for still in epfl_stills:
        card = section_frame(
            still,
            "PUBLIC DATASET VALIDATION",
            ["EPFL Laboratory", "4 calibrated camera views", "Ground-plane projection"],
            note="Prototype result — known ID-switch limitation",
        )
        for _ in range(epfl_hold):
            emit("epfl", card)
            epfl_count += 1
    while epfl_count < 11 * FPS:
        emit("epfl", card)
        epfl_count += 1
    counts.append(("epfl", epfl_count))

    closing = close_card()
    close_n = int(4.5 * FPS)
    for _ in range(close_n):
        emit("close", closing)
    counts.append(("close", close_n))

    master.close()
    for writer in writers.values():
        writer.close()
    return counts


if __name__ == "__main__":
    counts = build()
    cursor = 0
    for name, n in counts:
        start = cursor / FPS
        cursor += n
        print(f"{name}: {start:.3f}s .. {cursor / FPS:.3f}s  ({n / FPS:.3f}s, {n} frames)")
    print(f"total {cursor / FPS:.3f}s  {cursor} frames")
