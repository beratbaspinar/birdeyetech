"""The pose pass, without a GPU.

The model is injected, so what these tests exercise is the part that actually
breaks: crop rectangles, and the arithmetic that maps a keypoint back out of a
crop into frame coordinates. That translation is invisible on a centred crop and
wrong everywhere else, which is the definition of a bug that ships.
"""

from __future__ import annotations

import numpy as np
import pytest

from mcreid.track.pose import (
    LEFT_ANKLE,
    RIGHT_ANKLE,
    PoseBackend,
    PoseConfig,
    crop_boxes,
)


class _Tensor:
    """Minimal stand-in for the torch tensors ultralytics hands back."""

    def __init__(self, array) -> None:
        self._array = np.asarray(array, dtype=np.float64)

    def cpu(self) -> _Tensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._array

    def __len__(self) -> int:
        return len(self._array)


class _Keypoints:
    def __init__(self, data) -> None:
        self.data = _Tensor(data)


class _Boxes:
    def __init__(self, conf) -> None:
        self.conf = _Tensor(conf)


class _Result:
    def __init__(self, keypoint_data=None, conf=None) -> None:
        self.keypoints = _Keypoints(keypoint_data) if keypoint_data is not None else None
        self.boxes = _Boxes(conf) if conf is not None else None


class _FakeModel:
    """Returns a queued result per input patch, and records what it was given."""

    def __init__(self, results) -> None:
        self._results = list(results)
        self.seen_patches: list[tuple[int, int]] = []
        self.calls = 0

    def predict(self, source, **kwargs: object) -> list:
        self.calls += 1
        self.seen_patches.extend((p.shape[0], p.shape[1]) for p in source)
        out, self._results = self._results[: len(source)], self._results[len(source) :]
        return out


def _person(left_xy, right_xy, conf=0.9) -> np.ndarray:
    """One COCO-17 keypoint set with the ankles placed where we want them."""
    kp = np.zeros((17, 3), dtype=np.float64)
    kp[LEFT_ANKLE] = (*left_xy, conf)
    kp[RIGHT_ANKLE] = (*right_xy, conf)
    return kp


def _backend(results, config=None) -> PoseBackend:
    return PoseBackend(config or PoseConfig(device="cpu", half=False), model=_FakeModel(results))


# --- crop rectangles --------------------------------------------------------


def test_padding_is_proportional_and_symmetric():
    crops = crop_boxes([[100.0, 100.0, 200.0, 400.0]], (720, 1280), pad_frac=0.1)
    assert np.allclose(crops[0], [90.0, 70.0, 210.0, 430.0])


def test_a_box_at_the_frame_edge_clamps_instead_of_going_negative():
    """The half-outside-the-view case, which is ordinary rather than exceptional."""
    crops = crop_boxes([[0.0, 0.0, 50.0, 100.0]], (720, 1280), pad_frac=0.2)
    assert crops[0, 0] >= 0.0 and crops[0, 1] >= 0.0
    assert crops[0, 2] <= 1280.0 and crops[0, 3] <= 720.0


def test_a_degenerate_box_still_yields_a_usable_crop():
    """Zero-area boxes reach here from real detectors; a zero-width crop makes
    ultralytics raise on an otherwise ordinary frame."""
    crops = crop_boxes([[10.0, 10.0, 10.0, 10.0]], (720, 1280))
    assert crops[0, 2] > crops[0, 0] and crops[0, 3] > crops[0, 1]


# --- the crop -> frame mapping ---------------------------------------------


def test_keypoints_come_back_in_frame_coordinates_not_crop_coordinates():
    """The whole point of the class.

    The box is far from the origin, so a missing translation would be off by
    hundreds of pixels — and would still look plausible.
    """
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    box = np.array([[500.0, 200.0, 600.0, 500.0]])
    crop = crop_boxes(box, frame.shape)[0]
    # Ankles 10 px in from the crop's top-left, in crop coordinates.
    backend = _backend([_Result(_person((10.0, 290.0), (20.0, 292.0))[None], conf=[0.9])])

    got = backend.ankles(frame, box)
    assert np.allclose(got[0, 0, :2], [crop[0] + 10.0, crop[1] + 290.0])
    assert np.allclose(got[0, 1, :2], [crop[0] + 20.0, crop[1] + 292.0])
    assert np.allclose(got[0, :, 2], 0.9)


def test_the_most_confident_person_in_the_crop_wins():
    """A padded crop in a crowd contains neighbours. The crop was cut around OUR
    detection, so the most confident person in it is the one it was cut for —
    taking any other silently attributes a neighbour's feet."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    box = np.array([[100.0, 100.0, 200.0, 400.0]])
    neighbour = _person((5.0, 5.0), (6.0, 6.0), conf=0.4)
    ours = _person((50.0, 300.0), (55.0, 302.0), conf=0.95)
    backend = _backend([_Result(np.stack([neighbour, ours]), conf=[0.3, 0.92])])

    got = backend.ankles(frame, box)
    crop = crop_boxes(box, frame.shape)[0]
    assert np.allclose(got[0, 0, :2], [crop[0] + 50.0, crop[1] + 300.0])


def test_no_detection_in_the_crop_is_nan_at_zero_confidence():
    """Zero rather than NaN confidence: the estimator compares against a
    threshold, and a NaN would fail that comparison by accident, not by intent."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    backend = _backend([_Result(None)])
    got = backend.ankles(frame, np.array([[10.0, 10.0, 60.0, 200.0]]))
    assert np.all(np.isnan(got[0, :, :2]))
    assert np.all(got[0, :, 2] == 0.0)


def test_a_non_coco_keypoint_set_is_refused_rather_than_misindexed():
    """A 5-keypoint model would make index 15 an IndexError or, worse, silently
    valid on some other joint."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    backend = _backend([_Result(np.zeros((1, 5, 3)), conf=[0.9])])
    got = backend.ankles(frame, np.array([[10.0, 10.0, 60.0, 200.0]]))
    assert np.all(got[0, :, 2] == 0.0)


def test_no_boxes_means_no_inference_at_all():
    backend = _backend([])
    got = backend.ankles(np.zeros((720, 1280, 3), dtype=np.uint8), np.zeros((0, 4)))
    assert got.shape == (0, 2, 3)
    assert backend.model.calls == 0


def test_batching_splits_the_work_and_covers_every_box():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    boxes = np.array([[i * 50.0, 10.0, i * 50.0 + 40.0, 200.0] for i in range(5)])
    results = [_Result(_person((1.0, 2.0), (3.0, 4.0))[None], conf=[0.9]) for _ in range(5)]
    backend = _backend(results, PoseConfig(device="cpu", half=False, batch=2))

    got = backend.ankles(frame, boxes)
    assert got.shape == (5, 2, 3)
    assert np.all(got[:, :, 2] == 0.9)
    assert backend.model.calls == 3  # 2 + 2 + 1
    assert len(backend.model.seen_patches) == 5


# --- config -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"imgsz": 100}, "multiple of 32"),
        ({"conf_threshold": 0.0}, "conf_threshold"),
        ({"batch": 0}, "batch"),
    ],
)
def test_bad_config_fails_at_construction(kwargs, match):
    with pytest.raises(ValueError, match=match):
        PoseConfig(**kwargs)


def test_ankle_indices_are_the_coco_ones():
    """Pinned because they are the one magic number in the file and a silent
    off-by-one lands on the knees."""
    assert (LEFT_ANKLE, RIGHT_ANKLE) == (15, 16)
