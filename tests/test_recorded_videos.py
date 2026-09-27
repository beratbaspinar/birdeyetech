"""Local-video replay: frame pairing and the no-download weight gate.

No torch and no weights. The detector is not constructed.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from mcreid.cli.recorded import (
    camera_ids_for,
    iter_synced_frames,
    parse_video_args,
    require_local_weights,
)
from mcreid.track.reid_models import DEFAULT_EMBEDDER


def _write_clip(path: Path, color: tuple[int, int, int], n: int) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 10.0, (32, 24))
    assert writer.isOpened()
    frame = np.full((24, 32, 3), color, dtype=np.uint8)
    try:
        for _ in range(n):
            writer.write(frame)
    finally:
        writer.release()


def test_synced_frames_stop_at_the_shorter_clip(tmp_path: Path) -> None:
    left = tmp_path / "cam0.mp4"
    right = tmp_path / "cam1.mp4"
    _write_clip(left, (0, 0, 255), 4)
    _write_clip(right, (255, 0, 0), 2)
    frames = list(iter_synced_frames([left, right], ["cam0", "cam1"]))
    assert len(frames) == 2
    index, dt, pair = frames[0]
    assert index == 0
    assert dt == pytest.approx(0.1)
    assert set(pair) == {"cam0", "cam1"}
    assert pair["cam0"].shape == (24, 32, 3)


def test_camera_ids_come_from_stems(tmp_path: Path) -> None:
    left = tmp_path / "cam0.mp4"
    right = tmp_path / "cam1.mp4"
    left.write_bytes(b"x")
    right.write_bytes(b"x")
    assert parse_video_args(f"{left}, {right}") == [left, right]
    assert camera_ids_for([left, right]) == ["cam0", "cam1"]
    with pytest.raises(ValueError, match="duplicate"):
        camera_ids_for([left, tmp_path / "other" / "cam0.mp4"])


def test_missing_weights_are_not_fetched(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not download"):
        require_local_weights(tmp_path / "yolo11s.pt", DEFAULT_EMBEDDER, tmp_path)
    (tmp_path / "yolo11s.pt").write_bytes(b"not a model")
    with pytest.raises(FileNotFoundError, match="OSNet"):
        require_local_weights(tmp_path / "yolo11s.pt", DEFAULT_EMBEDDER, tmp_path)
