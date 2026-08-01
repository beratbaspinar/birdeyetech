"""Tests for the click-to-mark helper's geometry and state machine.

No window: `run_marker_ui` is the only part that touches one, and everything it
relies on is a pure function tested here. The loupe arithmetic is the piece that
matters — a sign error or an off-by-half in it produces coordinates that look
entirely reasonable and calibrate the rig to the wrong place, which the residual
gate would then blame on the marking.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from mcreid.calib.marker_ui import (
    DEFAULT_LOUPE_PX,
    DEFAULT_ZOOM,
    MarkerState,
    load_document,
    loupe_to_image,
    merge_markers,
    render_loupe,
    save_document,
    view_to_image,
)

# --- loupe geometry: the reason the helper exists -----------------------------------------------


def test_a_click_in_the_middle_of_the_loupe_is_the_centre_itself():
    assert loupe_to_image((240.0, 240.0), (412.5, 300.25), zoom=16, loupe_px=480) == (
        412.5,
        300.25,
    )


def test_the_loupe_resolves_one_over_zoom_of_a_pixel():
    """The whole design claim. One loupe pixel right of centre at 16x must move
    the answer by exactly 1/16 of an image pixel — that is what makes a click
    worth 0.0625 px against a gate that needs better than 0.5 px."""
    centre = (100.0, 100.0)
    one_right = loupe_to_image((241.0, 240.0), centre, zoom=16, loupe_px=480)
    assert one_right[0] - centre[0] == pytest.approx(1.0 / 16.0)
    assert one_right[1] == pytest.approx(centre[1])


@pytest.mark.parametrize("zoom", [2, 4, 8, 16, 32])
def test_resolution_scales_with_zoom(zoom: int):
    centre = (50.0, 50.0)
    half = DEFAULT_LOUPE_PX / 2
    moved = loupe_to_image((half + 1.0, half), centre, zoom=zoom, loupe_px=DEFAULT_LOUPE_PX)
    assert moved[0] - centre[0] == pytest.approx(1.0 / zoom)


def test_the_loupe_maps_all_four_quadrants_the_right_way_round():
    """A sign error here is invisible on screen and fatal in the YAML."""
    centre = (200.0, 150.0)
    left = loupe_to_image((0.0, 240.0), centre, zoom=16, loupe_px=480)
    right = loupe_to_image((480.0, 240.0), centre, zoom=16, loupe_px=480)
    up = loupe_to_image((240.0, 0.0), centre, zoom=16, loupe_px=480)
    down = loupe_to_image((240.0, 480.0), centre, zoom=16, loupe_px=480)
    assert left[0] < centre[0] < right[0]
    assert up[1] < centre[1] < down[1]
    assert right[0] - centre[0] == pytest.approx(centre[0] - left[0])


def test_a_zero_or_negative_zoom_is_refused():
    with pytest.raises(ValueError, match="zoom must be positive"):
        loupe_to_image((0.0, 0.0), (0.0, 0.0), zoom=0)
    with pytest.raises(ValueError, match="loupe_px must be positive"):
        loupe_to_image((0.0, 0.0), (0.0, 0.0), loupe_px=0)


def test_the_main_view_click_undoes_the_display_scaling():
    """A 1280-wide frame shown at 640 means every click is worth two image px —
    which is already past the 2.0 px level that measured 8 % pass, and is
    exactly why a main-view click alone is not good enough."""
    assert view_to_image((320.0, 180.0), 0.5) == (640.0, 360.0)
    assert view_to_image((100.0, 100.0), 1.0) == (100.0, 100.0)


def test_a_zero_view_scale_is_refused():
    with pytest.raises(ValueError, match="view_scale must be positive"):
        view_to_image((1.0, 1.0), 0.0)


def test_the_loupe_renders_at_the_requested_size_around_a_sub_pixel_centre():
    image = np.random.default_rng(0).integers(0, 255, (200, 300, 3), dtype=np.uint8)
    view = render_loupe(image, (150.5, 100.25), zoom=16, loupe_px=480)
    assert view.shape == (480, 480, 3)
    assert view.dtype == np.uint8


def test_the_loupe_survives_a_centre_on_the_image_border():
    """Clamping happens at the state, not the render — a marker on the frame
    edge must still be markable rather than crashing the tool mid-session."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    for centre in [(0.0, 0.0), (99.0, 99.0), (0.0, 99.0)]:
        assert render_loupe(image, centre, zoom=8, loupe_px=160).shape == (160, 160, 3)


# --- the state machine --------------------------------------------------------------------------


def _state(n: int = 12) -> MarkerState:
    return MarkerState(camera_id="cam0", image_size=(1280, 720), n_markers=n)


def test_points_start_unplaced_and_are_counted():
    state = _state()
    assert state.placed == 0 and not state.complete
    assert state.points == [None] * 12


def test_placing_every_marker_completes_the_session():
    state = _state(3)
    for i in range(3):
        state.index = i
        state.set_point((10.0 * i, 20.0 * i))
    assert state.complete and state.placed == 3


def test_a_click_outside_the_frame_is_clamped_not_dropped():
    """A click one pixel outside is a hand tremor, not a decision to skip a
    marker; dropping it would leave a None that only surfaces at save time."""
    state = _state()
    state.set_point((-5.0, 900.0))
    assert state.points[0] == (0.0, 719.0)


def test_nudging_moves_by_a_tenth_of_a_pixel_and_survives_re_clamping():
    state = _state()
    state.set_point((100.0, 100.0))
    state.nudge(-0.1, 0.0)
    assert state.points[0][0] == pytest.approx(99.9)
    state.nudge(0.0, 0.1)
    assert state.points[0][1] == pytest.approx(100.1)


def test_nudging_an_unplaced_marker_does_nothing():
    state = _state()
    state.nudge(1.0, 1.0)
    assert state.points[0] is None


def test_advance_is_clamped_at_both_ends():
    state = _state(3)
    state.advance(-1)
    assert state.index == 0
    state.advance(10)
    assert state.index == 2


def test_next_unplaced_finds_the_first_gap():
    state = _state(4)
    for i in (0, 1, 3):
        state.index = i
        state.set_point((1.0, 1.0))
    state.index = 3
    state.next_unplaced()
    assert state.index == 2


def test_clearing_a_marker_makes_it_unplaced_again():
    state = _state()
    state.set_point((5.0, 5.0))
    state.clear_current()
    assert state.points[0] is None and state.placed == 0


def test_a_mismatched_point_list_is_refused():
    with pytest.raises(ValueError, match="2 points for 3 markers"):
        MarkerState("cam0", (100, 100), 3, points=[(1.0, 1.0), (2.0, 2.0)])


# --- writing the YAML the calibrator reads -----------------------------------------------------


def test_merging_preserves_the_world_points_and_the_other_camera():
    """cam0 and cam1 get marked in separate sittings — anyone with two cameras
    and one tripod will do it that way, and the file has to accumulate."""
    document = {
        "world_points": [[0.0, 0.0], [1.0, 0.0]],
        "cameras": {"cam1": {"image_size": [640, 480], "image_points": [[1, 2], [3, 4]]}},
    }
    merged = merge_markers(document, "cam0", [(10.5, 20.25), (30.0, 40.0)], (1280, 720))
    assert merged["world_points"] == [[0.0, 0.0], [1.0, 0.0]]
    assert merged["cameras"]["cam1"]["image_points"] == [[1, 2], [3, 4]]
    assert merged["cameras"]["cam0"]["image_size"] == [1280, 720]
    assert merged["cameras"]["cam0"]["image_points"] == [[10.5, 20.25], [30.0, 40.0]]


def test_sub_pixel_precision_survives_the_round_trip(tmp_path):
    """Rounding to whole pixels here would discard exactly the precision the
    loupe exists to capture, and the gate would then fail on a marking error the
    user never made."""
    path = tmp_path / "floor.yaml"
    merged = merge_markers({}, "cam0", [(412.0625, 300.1875)], (1280, 720))
    save_document(path, merged)
    reloaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert reloaded["cameras"]["cam0"]["image_points"][0] == [412.0625, 300.1875]


def test_an_unplaced_marker_is_refused_at_save_time_with_its_index():
    with pytest.raises(ValueError, match=r"cam0: markers \[1\] were never placed"):
        merge_markers({}, "cam0", [(1.0, 1.0), None, (3.0, 3.0)], (100, 100))


def test_loading_a_missing_file_gives_an_empty_skeleton(tmp_path):
    document = load_document(tmp_path / "nope.yaml")
    assert document == {"world_points": [], "cameras": {}}


def test_loading_a_partial_file_fills_in_the_missing_keys(tmp_path):
    path = tmp_path / "floor.yaml"
    path.write_text(yaml.safe_dump({"world_points": [[0.0, 0.0]]}), encoding="utf-8")
    document = load_document(path)
    assert document["cameras"] == {}
    assert document["world_points"] == [[0.0, 0.0]]


def test_the_written_file_is_what_the_calibrator_can_read(tmp_path):
    """End to end against the real loader, so the two halves cannot drift."""
    from test_calib_floor import _markers

    from mcreid.calib.floor import load_floor_markers, measure_agreement

    reference = _markers()
    document = {"world_points": reference.world_points.tolist(), "cameras": {}}
    for camera_id in reference.camera_ids:
        document = merge_markers(
            document,
            camera_id,
            [(float(x), float(y)) for x, y in reference.image_points[camera_id]],
            reference.image_sizes[camera_id],
        )
    path = tmp_path / "floor.yaml"
    save_document(path, document)

    loaded = load_floor_markers(path)
    assert loaded.camera_ids == ["cam0", "cam1"]
    assert measure_agreement(loaded).passes


def test_the_helps_text_names_the_click_that_actually_gets_saved():
    from mcreid.calib.marker_ui import HELP

    assert any("LOUPE" in line and "sub-pixel" in line for line in HELP)
    assert DEFAULT_ZOOM == 16
