"""EPFL reference-format regressions, without videos, models or a GPU."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mcreid.calib.epfl import (
    EpflCameraCalibration,
    build_rig,
    grid_id_to_world_m,
    parse_ground_truth,
)
from mcreid.calib.geometry import ground_to_image


def _write_gt(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "ground_truth.txt"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("position", [0, 55, 56, 56 * 56 - 1])
def test_projection_matches_epfl_grid_to_tv_reference(position: int) -> None:
    # CVLab defines the top-view point at the CELL CENTRE. Test through the
    # rig's image projection as well, so anisotropic top-view scaling is covered.
    column, row = position % 56, position // 56
    expected_top_view = np.array([[(column + 0.5) * 358 / 56, (row + 0.5) * 360 / 56]])
    rig = build_rig(
        [EpflCameraCalibration("cam0", np.eye(3), None)],
        cell_size_m=1.0,
        image_size=(360, 288),
        grid=(56, 56),
    )
    world = np.array([grid_id_to_world_m(position, 56, 1.0)])
    pixels, valid = ground_to_image(rig.cameras[0], world)
    assert valid.tolist() == [True]
    np.testing.assert_allclose(pixels, expected_top_view, rtol=0, atol=1e-10)


def test_grid_units_can_be_scaled_without_losing_the_centre() -> None:
    assert grid_id_to_world_m(57, 56, 0.2) == pytest.approx((0.3, 0.3))


@pytest.mark.parametrize(
    "position,width,scale", [(-1, 56, 1), (-2, 56, 1), (0, 0, 1), (0, 56, 0), (0, 56, np.nan)]
)
def test_invalid_grid_inputs_are_not_projected(position, width, scale) -> None:
    with pytest.raises(ValueError):
        grid_id_to_world_m(position, width, scale)


@pytest.mark.parametrize("version", ["1\n", ""])
def test_gt_rows_are_video_frames_and_step_size_is_not_fps(tmp_path: Path, version: str) -> None:
    path = _write_gt(tmp_path, version + "3 2 2 2 25 0 2\n-1 -2\n0 3\n-1 -1\n")
    positions, header = parse_ground_truth(path)
    assert positions == {1: {0: 0, 1: 3}}  # row 1, not video frame 25
    assert header == {
        "n_frames": 3,
        "n_people": 2,
        "grid_w": 2,
        "grid_h": 2,
        "step_size": 25,
        "first_frame": 0,
        "last_frame": 2,
    }
    assert "fps" not in header


@pytest.mark.parametrize(
    "text",
    [
        "",
        "1\n",
        "2\n1 1 2 2 25 0 0\n0\n",  # empty / unsupported version
        "1 1 2 2 25\n0\n",  # incomplete header
        "1 1 2 2 25 0 0\n",  # missing video-frame row
        "1 2 2 2 25 0 0\n0\n",  # missing person column
        "1 1 2 2 25 0 0\n4\n",  # cell outside the grid
        "1 1 2 2 25 0 0\n-3\n",  # invalid sentinel
        "1 1 2 2 0 0 0\n0\n",  # zero annotation step
        "1 1 2 2 25 0 1\n0\n",  # last frame outside the file
    ],
)
def test_malformed_ground_truth_fails_before_evaluation(tmp_path: Path, text: str) -> None:
    with pytest.raises(ValueError):
        parse_ground_truth(_write_gt(tmp_path, text))


def test_geometry_cli_needs_no_models_and_does_not_claim_full_instrument_pass(
    tmp_path: Path,
) -> None:
    (tmp_path / "gt_lab_6p.txt").write_text("1\n1 1 2 2 25 0 0\n0\n", encoding="utf-8")
    (tmp_path / "calibration-6p.txt").write_text(
        "# Camera 0\n1 0 0\n0 1 0\n0 0 1\n0\n" "# Camera 1\n1 0 0\n0 1 0\n0 0 1\n0\n",
        encoding="utf-8",
    )
    out = tmp_path / "geometry.json"
    run = subprocess.run(
        [
            sys.executable,
            "scripts/check_epfl_instrument.py",
            "--geometry-only",
            "--root",
            str(tmp_path),
            "--out",
            str(out),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["A_coverage"]["fraction_seen_by_enough_cameras"] == 1.0
    assert result["ground_truth_convention"] == "cell_center_v1"
    assert result["B_agreement"]["status"] == "not_run"
    assert result["evaluation_complete"] is False
    assert result["pass"] is None
    assert len(result["ground_truth_sha256"]) == 64
