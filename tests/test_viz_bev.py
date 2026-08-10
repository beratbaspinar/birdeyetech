"""BEV canvas tests — the licence guard and the unit label.

Two things about this canvas are load-bearing rather than cosmetic. It is the
only rendered artifact this repo commits, so it must be impossible to feed it a
dataset pixel; and it is the README's front page, so the unit it prints has to be
the unit the arm actually works in.
"""

from __future__ import annotations

import inspect

import numpy as np

from mcreid.calib.schema import RigCalib
from mcreid.sim.toy import bedroom_rig
from mcreid.viz.bev import BevRenderer

ROOM = (6.0, 5.0)


def _rig() -> RigCalib:
    return RigCalib(cameras=[c.to_calib(floor_extent_m=(0.0, 0.0, *ROOM)) for c in bedroom_rig()])


def test_render_takes_no_image_argument() -> None:
    """The licence guard, asserted structurally.

    `CLAUDE.md`: anything rendered from the dataset IS the dataset. The BEV is
    the one artifact that ships, and it is safe to ship because there is no way
    to hand it footage — not because anyone promised not to. If a future change
    adds an image parameter here, that promise silently becomes a convention and
    this test is what says so.
    """
    params = set(inspect.signature(BevRenderer.render).parameters)
    assert params == {"self", "snapshots", "frame", "camera_positions", "camera_order"}


def test_default_units_are_metres() -> None:
    canvas = BevRenderer(_rig(), canvas_size=(320, 320)).render([], frame=7)
    assert canvas.shape == (320, 320, 3)
    assert np.any(canvas)


def test_the_unit_label_is_settable_for_the_grid_metric_arm() -> None:
    """The EPFL arm's scale is unidentifiable from the dataset, so it works in
    grid cells. A canvas hard-coded to "metres" would print a false unit on the
    hero image while the prose beside it correctly says cells."""
    metric = BevRenderer(_rig(), canvas_size=(320, 320))
    grid = BevRenderer(_rig(), canvas_size=(320, 320), units="grid cells")
    assert metric.units == "metres"
    assert grid.units == "grid cells"
    # Different label text means different pixels in the header band.
    assert not np.array_equal(metric.render([], frame=1), grid.render([], frame=1))
