"""Tests for the composite demo renderer.

The composite exists to make a cross-panel claim visible, so the tests are about
that claim rather than about pixels being produced: the same identity must be
drawn in more than one panel, always in one colour, and the licence guard must
refuse a tracked destination.

Two of these are **controls** rather than assertions of success —
`test_colour_conflicts_catches_a_renderer_that_colours_by_camera` and
`test_summarise_reports_zero_when_no_id_spans_two_panels`. Without them the
"consistent" and "cross-panel" numbers in the results JSON could be reported by
a checker that is incapable of returning anything else, which is exactly the
degenerate-gate failure this project has a paper trail of.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mcreid.fusion.types import ViewObservation
from mcreid.viz.composite import (
    ALL_STAGES,
    CompositeRecord,
    CompositeRenderer,
    OverlayStage,
    PanelRecord,
    StageWriter,
    assert_generated_only,
    grid_shape,
    summarise,
    write_stage_video,
)
from mcreid.viz.palette import id_color

FRAME_SIZE = (360, 288)


def _frame(shade: int = 90) -> np.ndarray:
    return np.full((FRAME_SIZE[1], FRAME_SIZE[0], 3), shade, dtype=np.uint8)


def _bev(size: int = 300) -> np.ndarray:
    return np.full((size, size, 3), 30, dtype=np.uint8)


def _obs(camera_id: str, local_id: int, x: float = 40.0) -> ViewObservation:
    embedding = np.zeros(8, dtype=np.float64)
    embedding[local_id % 8] = 1.0
    return ViewObservation(
        camera_id=camera_id,
        frame=0,
        local_track_id=local_id,
        bbox_xyxy=np.array([x, 60.0, x + 50.0, 220.0], dtype=np.float64),
        embedding=embedding,
        score=0.9,
    )


# --- layout -----------------------------------------------------------------


def test_grid_shape_gives_two_by_two_for_four_cameras() -> None:
    assert grid_shape(4, (480, 384)) == (2, 2)


def test_grid_shape_prefers_a_readable_square_over_a_letterbox_for_seven() -> None:
    """4x2 fits seven 16:9 tiles with one blank and is a 3.6:1 letterbox; 3x3
    wastes two slots and is readable. The scorer must choose readability."""
    assert grid_shape(7, (480, 270)) == (3, 3)


def test_grid_shape_rejects_degenerate_input() -> None:
    with pytest.raises(ValueError):
        grid_shape(0, (480, 270))
    with pytest.raises(ValueError):
        grid_shape(4, (0, 270))


# --- the stage ladder -------------------------------------------------------


def test_stage_ladder_is_cumulative() -> None:
    assert [s.draws_boxes for s in ALL_STAGES] == [False, True, True, True]
    assert [s.draws_ids for s in ALL_STAGES] == [False, False, True, True]
    assert [s.draws_bev for s in ALL_STAGES] == [False, False, False, True]


def test_raw_stage_records_no_boxes_and_labels_nothing() -> None:
    renderer = CompositeRenderer(["c0", "c1"], FRAME_SIZE, stage=OverlayStage.RAW)
    built = renderer.render(
        {"c0": _frame(), "c1": _frame()},
        {"c0": [_obs("c0", 1)], "c1": [_obs("c1", 7)]},
        {("c0", 1): 3, ("c1", 7): 3},
        None,
        frame=0,
    )
    assert [p.boxes for p in built.record.panels] == [0, 0]
    assert built.record.ids_in_multiple_panels == frozenset()


def test_boxes_stage_draws_boxes_but_makes_no_identity_claim() -> None:
    """The point of the middle stage: a viewer sees detection and per-view
    tracking, and no number yet asserts that two panels show one person."""
    renderer = CompositeRenderer(["c0", "c1"], FRAME_SIZE, stage=OverlayStage.BOXES)
    built = renderer.render(
        {"c0": _frame(), "c1": _frame()},
        {"c0": [_obs("c0", 1)], "c1": [_obs("c1", 7)]},
        {("c0", 1): 3, ("c1", 7): 3},
        None,
        frame=0,
    )
    assert [p.boxes for p in built.record.panels] == [1, 1]
    assert all(p.drawn == () for p in built.record.panels)
    assert built.record.ids_in_multiple_panels == frozenset()


def test_ids_stage_puts_one_id_in_both_panels_in_one_colour() -> None:
    renderer = CompositeRenderer(["c0", "c1"], FRAME_SIZE, stage=OverlayStage.IDS)
    built = renderer.render(
        {"c0": _frame(), "c1": _frame()},
        {"c0": [_obs("c0", 1)], "c1": [_obs("c1", 7)]},
        {("c0", 1): 3, ("c1", 7): 3},
        None,
        frame=12,
    )
    assert built.record.ids_in_multiple_panels == frozenset({3})
    assert built.record.colour_conflicts == frozenset()
    colours = {colour for p in built.record.panels for _gid, colour in p.drawn}
    assert colours == {id_color(3)}


def test_a_detection_the_fusion_stage_did_not_fuse_is_drawn_but_not_labelled() -> None:
    renderer = CompositeRenderer(["c0"], FRAME_SIZE, stage=OverlayStage.IDS)
    built = renderer.render(
        {"c0": _frame()}, {"c0": [_obs("c0", 1)]}, {}, None, frame=0
    )
    panel = built.record.panels[0]
    assert panel.boxes == 1
    assert panel.drawn == ()


def test_an_assigned_id_that_is_not_on_the_map_is_drawn_grey_and_unlabelled() -> None:
    """The panels and the floor plan have to say the same thing.

    The fusion stage assigns a global ID to a per-view observation before it
    commits that track to the map. Labelling it anyway puts a number over a
    person that the BEV beside it does not carry, which contradicts the exact
    claim the composite exists to make.
    """
    renderer = CompositeRenderer(["c0", "c1"], FRAME_SIZE, stage=OverlayStage.IDS)
    built = renderer.render(
        {"c0": _frame(), "c1": _frame()},
        {"c0": [_obs("c0", 1)], "c1": [_obs("c1", 7)]},
        {("c0", 1): 3, ("c1", 7): 4},
        None,
        frame=0,
        live_ids=frozenset({3}),
    )
    by_camera = {p.camera_id: p for p in built.record.panels}
    assert by_camera["c0"].drawn == ((3, id_color(3)),)
    assert by_camera["c1"].drawn == ()
    assert by_camera["c1"].withheld == 1
    assert summarise([built.record])["assigned_boxes_withheld_because_not_on_the_map"] == 1


def test_without_live_ids_every_assigned_box_is_labelled() -> None:
    """The filter is opt-in: with no map to contradict, nothing is withheld."""
    renderer = CompositeRenderer(["c0", "c1"], FRAME_SIZE, stage=OverlayStage.IDS)
    built = renderer.render(
        {"c0": _frame(), "c1": _frame()},
        {"c0": [_obs("c0", 1)], "c1": [_obs("c1", 7)]},
        {("c0", 1): 3, ("c1", 7): 4},
        None,
        frame=0,
    )
    assert sum(len(p.drawn) for p in built.record.panels) == 2
    assert sum(p.withheld for p in built.record.panels) == 0


def test_composite_stage_appends_the_bev_and_the_others_do_not() -> None:
    views = {"c0": _frame(), "c1": _frame()}
    observations = {"c0": [_obs("c0", 1)], "c1": [_obs("c1", 7)]}
    assignment = {("c0", 1): 3, ("c1", 7): 3}

    ids = CompositeRenderer(["c0", "c1"], FRAME_SIZE, stage=OverlayStage.IDS)
    full = CompositeRenderer(["c0", "c1"], FRAME_SIZE, stage=OverlayStage.COMPOSITE)
    without = ids.render(views, observations, assignment, None, frame=0).image
    with_bev = full.render(views, observations, assignment, _bev(), frame=0).image

    assert with_bev.shape[1] == without.shape[1] + full.bev_size[0]
    assert with_bev.shape[0] == without.shape[0]


def test_composite_stage_refuses_to_render_without_a_bev() -> None:
    renderer = CompositeRenderer(["c0"], FRAME_SIZE, stage=OverlayStage.COMPOSITE)
    with pytest.raises(ValueError, match="needs a BEV"):
        renderer.render({"c0": _frame()}, {"c0": []}, {}, None, frame=0)


def test_renderer_rejects_degenerate_construction() -> None:
    with pytest.raises(ValueError):
        CompositeRenderer([], FRAME_SIZE)
    with pytest.raises(ValueError):
        CompositeRenderer(["c0"], (0, 288))


# --- the two controls -------------------------------------------------------


def test_colour_conflicts_catches_a_renderer_that_colours_by_camera() -> None:
    """CONTROL. `id_colour_consistent` is only evidence if it can come out false.

    A renderer that keyed colour to the camera instead of the identity would put
    global ID 3 on screen in two different colours while still reporting it in
    both panels. This builds exactly that record and asserts the check fires.
    """
    record = CompositeRecord(
        frame=0,
        stage=OverlayStage.IDS,
        panels=(
            PanelRecord("c0", boxes=1, drawn=((3, (10, 20, 30)),)),
            PanelRecord("c1", boxes=1, drawn=((3, (200, 100, 0)),)),
        ),
    )
    assert record.ids_in_multiple_panels == frozenset({3})
    assert record.colour_conflicts == frozenset({3})
    assert summarise([record])["id_colour_consistent"] is False


def test_summarise_reports_zero_when_no_id_spans_two_panels() -> None:
    """CONTROL. `frames_with_an_id_in_multiple_panels` must be able to be zero.

    Four independent per-camera trackers rendered side by side produce a
    perfectly pretty video that demonstrates nothing, and this is the number that
    tells them apart from a fused identity space.
    """
    record = CompositeRecord(
        frame=0,
        stage=OverlayStage.IDS,
        panels=(
            PanelRecord("c0", boxes=1, drawn=((1, id_color(1)),)),
            PanelRecord("c1", boxes=1, drawn=((2, id_color(2)),)),
        ),
    )
    stats = summarise([record])
    assert stats["frames_with_an_id_in_multiple_panels"] == 0
    assert stats["max_ids_in_multiple_panels_in_a_frame"] == 0
    assert stats["id_colour_consistent"] is True


def test_summarise_refuses_an_empty_run() -> None:
    with pytest.raises(ValueError):
        summarise([])


# --- the licence guard ------------------------------------------------------


def test_the_writer_refuses_any_destination_under_docs(tmp_path) -> None:
    """Composites contain dataset pixels; docs/ is the tracked asset tree."""
    for candidate in (
        tmp_path / "docs" / "assets" / "composite.mp4",
        tmp_path / "docs" / "composite.mp4",
        tmp_path / "nested" / "docs" / "deep" / "composite.mp4",
    ):
        with pytest.raises(ValueError, match="refusing to write a composite"):
            assert_generated_only(candidate)


def test_the_writer_keeps_the_path_it_was_given(tmp_path) -> None:
    """The path is written into a COMMITTED json. Resolving it would put an
    absolute `C:\\Users\\<name>\\...` into a public repo — which this project's
    pre-ship audit checks for by name."""
    relative = Path("reports") / "demo" / "composite.mp4"
    assert StageWriter(relative, 6.0).path == relative


def test_the_guard_resolves_before_checking_so_traversal_cannot_sneak_past(
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match="refusing to write a composite"):
        assert_generated_only(tmp_path / "reports" / ".." / "docs" / "assets" / "x.mp4")


def test_the_writer_refuses_before_a_single_frame_is_rendered(tmp_path) -> None:
    """The check is in the constructor, so a bad path fails in the first second
    of a run rather than after the detector has already been over every frame."""
    with pytest.raises(ValueError, match="refusing to write a composite"):
        StageWriter(tmp_path / "docs" / "assets" / "x.mp4", 6.0)


def test_the_writer_accepts_a_reports_destination_and_writes_frames(tmp_path) -> None:
    target = tmp_path / "reports" / "demo" / "composite.mp4"
    frames = [_frame(shade) for shade in (40, 80, 120)]
    written = write_stage_video(frames, target, fps=6.0)
    assert written.is_file()
    assert written.stat().st_size > 0


def test_the_writer_refuses_an_empty_run(tmp_path) -> None:
    with pytest.raises(ValueError):
        write_stage_video([], tmp_path / "reports" / "empty.mp4", fps=6.0)
