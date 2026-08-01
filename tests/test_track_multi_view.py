"""Tests for the shared-model multi-camera backend and batched embedding.

No GPU and no torch: a fake detector and a fake embedder stand in, which is what
lets the batching contract — one detector call and one embedder call per frame,
results split back to the right camera — be asserted at all. That split is the
part worth pinning: a batching bug that mixes two cameras' crops produces
plausible embeddings attached to the wrong people, and nothing downstream can
detect it.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from mcreid.fusion.types import ViewObservation
from mcreid.track.multi_view import MultiViewBackend
from mcreid.track.reid_models import embed_views

DIM = 8
FloatArray = npt.NDArray[np.float64]
Image = npt.NDArray[np.uint8]


class _FakeBoxes:
    def __init__(self, boxes: object, scores: object) -> None:
        self.xyxy = _FakeTensor(boxes)
        self.conf = _FakeTensor(scores)

    def __len__(self) -> int:
        return len(self.xyxy.data)


class _FakeTensor:
    def __init__(self, data: object) -> None:
        self.data = np.asarray(data, dtype=np.float64)

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> FloatArray:
        return self.data


class _FakeResult:
    def __init__(self, boxes: object, scores: object) -> None:
        self.boxes = _FakeBoxes(boxes, scores) if len(boxes) else None


class _FakeDetector:
    """Returns one box per image, its x-offset keyed to the image's mean value.

    Keying the output to the image content is what makes a swapped-batch bug
    visible: if the backend hands camera 1's result to camera 0, the box lands
    where camera 1's frame said it should.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.batch_sizes: list[int] = []

    def predict(self, source: list[Image], **kwargs: object) -> list[_FakeResult]:
        self.calls += 1
        self.batch_sizes.append(len(source))
        results = []
        for image in source:
            offset = float(image[0, 0, 0])
            results.append(_FakeResult([[offset, 10.0, offset + 50.0, 200.0]], [0.9]))
        return results


class _FakeEmbedder:
    """Per-view embedder. Encodes the crop's mean into the vector."""

    dim = DIM

    def __init__(self) -> None:
        self.calls = 0

    def _vector(self, value: float) -> FloatArray:
        vec = np.zeros(DIM, dtype=np.float64)
        vec[0] = 1.0
        vec[1] = value / 255.0
        return vec / np.linalg.norm(vec)

    def __call__(self, image: Image, boxes: FloatArray) -> FloatArray:
        self.calls += 1
        boxes = np.asarray(boxes, dtype=np.float64)
        if boxes.shape[0] == 0:
            return np.zeros((0, DIM), dtype=np.float64)
        return np.stack([self._vector(float(image[0, 0, 0])) for _ in range(boxes.shape[0])])


class _BatchingEmbedder(_FakeEmbedder):
    """Same numbers, one call. Mirrors what `_CropEmbedder.embed_multi` does."""

    def __init__(self) -> None:
        super().__init__()
        self.multi_calls = 0

    def embed_multi(
        self, images: list[Image], boxes: list[FloatArray]
    ) -> list[FloatArray]:
        self.multi_calls += 1
        return [self(image, box) for image, box in zip(images, boxes, strict=True)]


def _frame(value: int, size: tuple[int, int] = (120, 160)) -> Image:
    return np.full((*size, 3), value, dtype=np.uint8)


def _backend(
    camera_ids: tuple[str, ...] = ("cam0", "cam1"), embedder: object = None
) -> MultiViewBackend:
    return MultiViewBackend(
        camera_ids=camera_ids,
        detector=_FakeDetector(),
        embedder=embedder or _FakeEmbedder(),
    )


def _run(
    backend: MultiViewBackend, frames: dict[str, Image], steps: int = 3
) -> list[ViewObservation]:
    """`PerViewTracker` confirms at n_init=2, so nothing is emitted on frame 0.
    Every routing assertion has to run past that or it asserts on an empty list
    and passes for the wrong reason."""
    observations: list[ViewObservation] = []
    for frame in range(steps):
        observations = backend.step(frames, frame=frame)
    return observations


# --- construction ------------------------------------------------------------------------------


def test_duplicate_camera_ids_are_refused():
    with pytest.raises(ValueError, match="unique"):
        MultiViewBackend(["cam0", "cam0"], detector=_FakeDetector(), embedder=_FakeEmbedder())


def test_an_empty_camera_list_is_refused():
    with pytest.raises(ValueError, match="at least one camera_id"):
        MultiViewBackend([], detector=_FakeDetector(), embedder=_FakeEmbedder())


def test_one_tracker_per_camera_and_one_model_for_all_of_them():
    backend = _backend()
    assert set(backend.trackers) == {"cam0", "cam1"}
    assert backend.trackers["cam0"] is not backend.trackers["cam1"]
    # The whole point: two cameras, one detector object and one embedder object.
    assert backend.detector is not None and backend.embedder is not None


# --- batching ----------------------------------------------------------------------------------


def test_two_cameras_cost_one_detector_call_not_two():
    backend = _backend()
    backend.step({"cam0": _frame(30), "cam1": _frame(60)}, frame=0)
    assert backend.detector.calls == 1
    assert backend.detector.batch_sizes == [2]


def test_a_batching_embedder_is_called_once_for_all_views():
    embedder = _BatchingEmbedder()
    backend = _backend(embedder=embedder)
    backend.step({"cam0": _frame(30), "cam1": _frame(60)}, frame=0)
    assert embedder.multi_calls == 1


def test_an_embedder_without_batching_still_works_per_view():
    """The toy generator and every test double implement the single-image call
    only. Dispatching on `embed_multi` must not make them a hard error."""
    embedder = _FakeEmbedder()
    backend = _backend(embedder=embedder)
    observations = _run(backend, {"cam0": _frame(30), "cam1": _frame(60)})
    assert embedder.calls == 6  # two views x three frames, no batching
    assert len(observations) == 2


def test_batched_and_per_view_embedding_agree_exactly():
    images = [_frame(30), _frame(60), _frame(90)]
    boxes = [np.array([[0.0, 0.0, 10.0, 20.0]]) for _ in images]
    per_view = embed_views(_FakeEmbedder(), images, boxes)
    batched = embed_views(_BatchingEmbedder(), images, boxes)
    for a, b in zip(per_view, batched, strict=True):
        np.testing.assert_allclose(a, b)


def test_embed_views_splits_uneven_box_counts_back_to_the_right_view():
    """Two boxes in one view and none in another is the normal case, and the
    obvious off-by-one in a concatenate-then-slice implementation."""

    class _Counting(_BatchingEmbedder):
        def embed_multi(
            self, images: list[Image], boxes: list[FloatArray]
        ) -> list[FloatArray]:
            self.multi_calls += 1
            counts = [np.asarray(b).shape[0] for b in boxes]
            flat = (
                np.stack(
                    [self._vector(float(i)) for i, c in enumerate(counts) for _ in range(c)]
                )
                if sum(counts)
                else np.zeros((0, DIM))
            )
            out: list[FloatArray] = []
            start = 0
            for count in counts:
                out.append(flat[start : start + count])
                start += count
            return out

    boxes = [np.zeros((2, 4)), np.zeros((0, 4)), np.zeros((3, 4))]
    result = embed_views(_Counting(), [_frame(1)] * 3, boxes)
    assert [r.shape[0] for r in result] == [2, 0, 3]


def test_an_embedder_returning_the_wrong_number_of_views_is_caught():
    class _Broken(_FakeEmbedder):
        def embed_multi(
            self, images: list[Image], boxes: list[FloatArray]
        ) -> list[FloatArray]:
            return [np.zeros((1, DIM))]

    with pytest.raises(RuntimeError, match="returned 1 arrays for 2 views"):
        embed_views(_Broken(), [_frame(1), _frame(2)], [np.zeros((1, 4))] * 2)


def test_a_detector_returning_the_wrong_number_of_results_is_caught():
    class _Broken(_FakeDetector):
        def predict(self, source: list[Image], **kwargs: object) -> list[_FakeResult]:
            return [_FakeResult([[0.0, 0.0, 1.0, 1.0]], [0.5])]

    backend = MultiViewBackend(["cam0", "cam1"], detector=_Broken(), embedder=_FakeEmbedder())
    with pytest.raises(RuntimeError, match="returned 1 results for 2 images"):
        backend.step({"cam0": _frame(30), "cam1": _frame(60)}, frame=0)


# --- per-camera routing ------------------------------------------------------------------------


def test_each_observation_carries_its_own_camera_id():
    backend = _backend()
    observations = _run(backend, {"cam0": _frame(30), "cam1": _frame(60)})
    assert sorted(o.camera_id for o in observations) == ["cam0", "cam1"]


def test_a_detection_lands_on_the_camera_whose_frame_produced_it():
    """The swapped-batch bug: plausible boxes on the wrong camera."""
    backend = _backend()
    observations = _run(backend, {"cam0": _frame(30), "cam1": _frame(60)})
    by_camera = {o.camera_id: o for o in observations}
    assert by_camera["cam0"].bbox_xyxy[0] == pytest.approx(30.0)
    assert by_camera["cam1"].bbox_xyxy[0] == pytest.approx(60.0)


def test_embeddings_land_on_the_camera_whose_frame_produced_them():
    backend = _backend(embedder=_BatchingEmbedder())
    observations = _run(backend, {"cam0": _frame(30), "cam1": _frame(60)})
    reference = _FakeEmbedder()
    by_camera = {o.camera_id: o for o in observations}
    np.testing.assert_allclose(by_camera["cam0"].embedding, reference._vector(30.0))
    np.testing.assert_allclose(by_camera["cam1"].embedding, reference._vector(60.0))


def test_local_track_ids_are_per_camera_and_may_collide_across_cameras():
    """They are only meaningful inside one view — the fusion stage keys on
    (camera_id, local_track_id), which is why a collision is harmless here and
    would be a defect if the trackers were shared."""
    backend = _backend()
    observations = _run(backend, {"cam0": _frame(30), "cam1": _frame(60)}, steps=4)
    ids = {o.camera_id: o.local_track_id for o in observations}
    assert ids["cam0"] == ids["cam1"] == 1


def test_a_camera_absent_from_frames_is_not_advanced():
    """Feeding the previous frame again would manufacture a measurement no
    sensor produced, and the per-view tracker would count it as a hit."""
    backend = _backend()
    _run(backend, {"cam0": _frame(30), "cam1": _frame(60)})
    before = backend.trackers["cam1"].tracks[0].hits

    backend.step({"cam0": _frame(30)}, frame=3)
    assert backend.trackers["cam1"].tracks[0].hits == before
    assert backend.trackers["cam0"].tracks[0].hits == before + 1


def test_only_the_present_cameras_reach_the_detector():
    backend = _backend()
    backend.step({"cam0": _frame(30)}, frame=0)
    assert backend.detector.batch_sizes == [1]


def test_an_unknown_camera_id_is_refused_rather_than_silently_dropped():
    backend = _backend()
    with pytest.raises(KeyError, match="cam9"):
        backend.step({"cam9": _frame(30)}, frame=0)


def test_no_frames_at_all_is_a_no_op():
    backend = _backend()
    assert backend.step({}, frame=0) == []
    assert backend.detector.calls == 0


def test_a_camera_with_no_detections_yields_no_observations():
    class _Empty(_FakeDetector):
        def predict(self, source: list[Image], **kwargs: object) -> list[_FakeResult]:
            self.calls += 1
            self.batch_sizes.append(len(source))
            return [_FakeResult([], []) for _ in source]

    backend = MultiViewBackend(["cam0", "cam1"], detector=_Empty(), embedder=_FakeEmbedder())
    assert backend.step({"cam0": _frame(30), "cam1": _frame(60)}, frame=0) == []


# --- warmup ------------------------------------------------------------------------------------


def test_warmup_exercises_the_detector_and_the_embedder_on_every_view():
    """Measured cost of skipping it: the first real step takes 5.4 s against a
    20 ms steady state, because CUDA init happens inside the first forward
    pass. That lands as a five-second freeze exactly when someone walks in."""
    embedder = _BatchingEmbedder()
    backend = _backend(embedder=embedder)
    backend.warmup({"cam0": (160, 120), "cam1": (64, 48)}, rounds=2)
    assert backend.detector.batch_sizes == [2, 2]
    assert embedder.multi_calls == 2


def test_warmup_does_not_advance_the_trackers():
    """A warmup that produced tracks would put a phantom identity in the scene
    before the first real frame."""
    backend = _backend()
    backend.warmup({"cam0": (160, 120), "cam1": (64, 48)})
    assert all(not tracker.tracks for tracker in backend.trackers.values())


def test_warmup_ignores_sizes_for_cameras_this_backend_does_not_own():
    backend = _backend(camera_ids=("cam0",))
    backend.warmup({"cam0": (160, 120), "cam9": (64, 48)})
    assert backend.detector.batch_sizes == [1, 1]


def test_warmup_with_no_matching_camera_is_a_no_op():
    backend = _backend(camera_ids=("cam0",))
    assert backend.warmup({"cam9": (64, 48)}) == 0.0
    assert backend.detector.calls == 0
