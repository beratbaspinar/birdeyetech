"""The EPFL live window is a view over the existing pipeline, not a new one."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from mcreid.cli import epfl_live
from mcreid.cli.epfl_live import (
    LIVE_POSITION_SIGMA_CELLS,
    PREVIEW_H,
    PREVIEW_W,
    action_for_key,
    assert_grid_units,
    compose_preview,
    format_live_summary,
    is_benchmark_detector,
    live_fusion_config,
    preview_layout,
)
from mcreid.cli.public_demo import epfl, epfl_fusion_config
from mcreid.fusion.types import GlobalTrackSnapshot, TrackState, ViewObservation
from mcreid.viz.palette import id_color


def _frames() -> dict[str, np.ndarray]:
    return {
        f"cam{i}": np.zeros((288, 360, 3), dtype=np.uint8) for i in range(4)
    }


def _hud() -> epfl_live.LiveHud:
    return epfl_live.LiveHud(
        weights_name="yolo11x.pt",
        device="mps",
        source_frame=120,
        frame_count=2955,
        video_fps=25.0,
        inference_fps=4.2,
        last_ms=230.0,
        synced=True,
        paused=False,
        at_end=False,
        benchmark_detector=True,
        viewing_cached=False,
        confirmed_ids=1,
    )


def _snap(gid: int, x: float, y: float) -> GlobalTrackSnapshot:
    return GlobalTrackSnapshot(
        global_id=gid,
        frame=120,
        world_xy=np.array([x, y], dtype=np.float64),
        velocity_mps=np.zeros(2, dtype=np.float64),
        covariance=np.eye(2, dtype=np.float64),
        state=TrackState.CONFIRMED,
        supporting_cameras=("cam0", "cam1"),
        frames_since_measurement=0,
        hits=5,
    )


def test_preview_is_1080p_with_the_map_on_the_right() -> None:
    bev = np.zeros((640, 640, 3), dtype=np.uint8)
    bev[:] = (200, 10, 220)
    canvas = compose_preview(
        _frames(),
        {},
        {},
        bev,
        [_snap(3, 12.0, 40.0)],
        _hud(),
        [f"cam{i}" for i in range(4)],
    )
    assert canvas.shape == (PREVIEW_H, PREVIEW_W, 3)
    layout = preview_layout(640, 640)
    bx, by = layout.bev_origin
    pasted = canvas[by : by + 640, bx : bx + 640]
    assert np.array_equal(pasted, bev)
    assert bx > PREVIEW_W // 2

    other = np.zeros_like(bev)
    other[:] = (10, 220, 40)
    moved = compose_preview(
        _frames(),
        {},
        {},
        other,
        [_snap(3, 30.0, 8.0)],
        _hud(),
        [f"cam{i}" for i in range(4)],
    )
    assert not np.array_equal(canvas, moved)


def test_box_shows_global_id_colour() -> None:
    colour = id_color(7)
    emb = np.zeros(8, dtype=np.float64)
    emb[0] = 1.0
    obs = ViewObservation(
        camera_id="cam0",
        frame=120,
        local_track_id=3,
        bbox_xyxy=np.array([40.0, 30.0, 120.0, 200.0]),
        embedding=emb,
        score=0.91,
    )
    canvas = compose_preview(
        _frames(),
        {"cam0": [obs]},
        {("cam0", 3): 7},
        np.zeros((640, 640, 3), dtype=np.uint8),
        [],
        _hud(),
        [f"cam{i}" for i in range(4)],
    )
    x, y, width, height = preview_layout(640, 640).tiles[0]
    tile = canvas[y : y + height, x : x + width]
    assert np.any(np.all(tile == np.array(colour, dtype=np.uint8), axis=2))


def test_grid_cells_are_the_only_accepted_unit() -> None:
    assert_grid_units("grid cells")
    try:
        assert_grid_units("metres")
    except ValueError:
        pass
    else:
        raise AssertionError("a metre label must be refused")


def test_keys() -> None:
    assert action_for_key(ord("q")) == "quit"
    assert action_for_key(ord("Q")) == "quit"
    assert action_for_key(27) == "quit"
    assert action_for_key(ord(" ")) == "pause"
    assert action_for_key(ord("r")) == "restart"
    assert action_for_key(ord("a")) == "left"
    assert action_for_key(ord("d")) == "right"
    assert action_for_key(63234) == "left"
    assert action_for_key(63235) == "right"
    assert action_for_key(ord("m")) == "map"
    assert action_for_key(ord("f")) == "frusta"
    assert action_for_key(-1) is None
    assert action_for_key(ord("S")) is None


def test_yolo11s_is_not_the_benchmark_detector() -> None:
    assert is_benchmark_detector(Path("weights/yolo11x.pt"))
    assert not is_benchmark_detector(Path("weights/yolo11s.pt"))
    text = format_live_summary(
        {
            "device": "mps",
            "device_name": "mps (Apple MPS, half=False)",
            "weights": "weights/yolo11s.pt",
            "benchmark_detector": False,
            "embedder": "osnet_x1_0_msmt17",
            "units": "grid cells",
            "position_sigma_cap_cells": 8.5,
            "benchmark_position_sigma_cap_cells": 4.5,
            "frames": 8,
            "start_frame": 100,
            "video_fps": 25.0,
            "dt_s": 0.04,
            "inference_fps": 6.5,
            "warmup_ms": 400.0,
            "cameras_synced": True,
            "bev_changed": True,
            "global_ids": [1, 2],
            "assigned_ids": [1, 2],
            "max_observations": 4,
            "ground_truth_positions_used": False,
        }
    )
    assert "DEMO / PERFORMANCE CONFIG" in text
    assert "grid cells" in text
    assert "ground-truth positions used as tracks: False" in text
    assert "metres" not in text.casefold()


def test_live_path_does_not_read_ground_truth_or_skip_frames() -> None:
    src = inspect.getsource(epfl_live)
    assert "parse_ground_truth" not in src
    assert "evaluate_id_consistency" not in src
    assert "grid_id_to_world" not in src
    step = inspect.getsource(epfl_live._Session.step)
    assert "grab(" not in step
    assert epfl_live.GROUND_TRUTH_POSITIONS_USED is False
    assert epfl_live.SKIPS_SOURCE_FRAMES is False


def test_live_cap_does_not_change_the_benchmark_config() -> None:
    published = epfl_fusion_config()
    widened = live_fusion_config(published)
    assert published.max_position_sigma_m == 4.5
    assert widened.max_position_sigma_m == LIVE_POSITION_SIGMA_CELLS
    assert widened is not published


def test_live_view_returns_before_any_benchmark_write() -> None:
    src = inspect.getsource(epfl)
    assert src.index("run_epfl_live_view") < src.index("_run_arm")
    assert src.index("run_epfl_live_view") < src.index("epfl_demo.json")
