"""N cameras, ONE detector and ONE embedder, batched across views.

`GpuPerViewBackend` owns a detector, an embedder and a tracker per camera. That
is right for the dataset path, where views are processed one at a time anyway,
and wrong for a live rig: two `GpuPerViewBackend`s hold two copies of YOLO11 and
two of OSNet on an 8 GB card, and each frame pays two sequential forward passes
over a handful of crops while the GPU idles between them.

This backend keeps exactly one of each model and one `PerViewTracker` per
camera. Detection runs as a single batched `predict` over the list of frames;
embedding runs as a single forward pass over the concatenated crops. Tracking
stays per camera, because a local track ID is only meaningful inside one view.

The output contract is unchanged — a flat `list[ViewObservation]` with distinct
`camera_id`s — so the fusion stage cannot tell this apart from N single-camera
backends, and the WILDTRACK path's N-camera evidence carries over.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from mcreid.track.gpu_view import PERSON_CLASS, GpuViewConfig, precision_kwargs
from mcreid.track.per_view import Detection, PerViewConfig, PerViewTracker
from mcreid.track.reid_models import Embedder, build_embedder, embed_views
from mcreid.utils.device import resolve_device
from mcreid.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - import-time only for type checkers
    from mcreid.fusion.types import ViewObservation

logger = get_logger(__name__)

FloatArray = npt.NDArray[np.float64]
Image = npt.NDArray[np.uint8]


class MultiViewBackend:
    """One detector + one embedder shared across `camera_ids`."""

    def __init__(
        self,
        camera_ids: Sequence[str],
        config: GpuViewConfig | None = None,
        per_view_config: PerViewConfig | None = None,
        detector: Any | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        ids = list(camera_ids)
        if not ids:
            raise ValueError("need at least one camera_id")
        if len(set(ids)) != len(ids):
            raise ValueError(f"camera_ids must be unique, got {ids}")

        self.camera_ids = tuple(ids)
        self.config = config or GpuViewConfig()
        self.device = resolve_device(self.config.device, allow_half=self.config.half)
        self.trackers = {cid: PerViewTracker(cid, per_view_config) for cid in self.camera_ids}

        if detector is None:
            from ultralytics import YOLO

            weights = Path(self.config.weights)
            if not weights.is_file():
                raise FileNotFoundError(
                    f"detector weights not found: {weights}. Ultralytics will download "
                    "them on first use, or copy them in manually."
                )
            detector = YOLO(str(weights))
        self.detector = detector
        self.embedder = embedder or build_embedder(
            self.config.embedder,
            device=self.device,
            batch_size=self.config.embed_batch,
            weights_dir=self.config.weights_dir,
        )
        self._predict_kwargs = precision_kwargs(self.device)

    def warmup(self, sizes: Mapping[str, tuple[int, int]], rounds: int = 2) -> float:
        """Run the models once on blank frames. Returns seconds spent.

        Measured cost of not doing this: the first real step takes **5.4 s**
        against a 20 ms steady state, because CUDA context creation, cuDNN
        autotuning and the fp16 weight upload all happen lazily inside the first
        forward pass. Two things go wrong if that lands mid-session — the window
        freezes for five seconds exactly when someone walks in, and the number
        the session reports as its frame rate is an average over a 270x outlier.

        Blank frames, not real ones: this must not depend on a camera having
        delivered anything yet, and a detector finding nothing is fine —
        cuDNN autotunes on the input shape, which is all that is being paid for
        here. The embedder is warmed separately with a synthetic box, since a
        blank frame yields no detections and would leave OSNet cold.
        """
        started = time.perf_counter()
        frames = {
            camera_id: np.zeros((height, width, 3), dtype=np.uint8)
            for camera_id, (width, height) in sizes.items()
            if camera_id in self.camera_ids
        }
        if not frames:
            return 0.0
        images = list(frames.values())
        boxes = [
            np.array([[0.0, 0.0, image.shape[1] / 4.0, image.shape[0] / 2.0]])
            for image in images
        ]
        for _ in range(max(rounds, 1)):
            self.detect_batch(images)
            embed_views(self.embedder, images, boxes)
        elapsed = time.perf_counter() - started
        logger.info("warmed detector + embedder on %d view(s) in %.1f s", len(images), elapsed)
        return elapsed

    def detect_batch(self, images: Sequence[Image]) -> list[tuple[FloatArray, FloatArray]]:
        """One batched forward pass. Returns (boxes (N,4) xyxy, scores (N,)) per image."""
        if not images:
            return []
        predictions: Any = self.detector.predict(
            source=list(images),
            device=self.device.torch_device,
            classes=[PERSON_CLASS],
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.imgsz,
            max_det=self.config.max_detections,
            verbose=False,
            **self._predict_kwargs,
        )
        if len(predictions) != len(images):
            raise RuntimeError(
                f"detector returned {len(predictions)} results for {len(images)} images; "
                "batched prediction is not behaving as this backend assumes"
            )
        out: list[tuple[FloatArray, FloatArray]] = []
        for result in predictions:
            if result.boxes is None or len(result.boxes) == 0:
                out.append((np.zeros((0, 4), dtype=np.float64), np.zeros(0, dtype=np.float64)))
                continue
            out.append(
                (
                    result.boxes.xyxy.cpu().numpy().astype(np.float64),
                    result.boxes.conf.cpu().numpy().astype(np.float64),
                )
            )
        return out

    def step(self, frames: Mapping[str, Image], frame: int) -> list[ViewObservation]:
        """Advance every camera that supplied a frame.

        A camera absent from `frames` is NOT advanced: its local tracks keep
        their age, and the fusion stage simply sees no observation from it this
        step. That is the honest behaviour for a rig whose cameras run at
        different rates — feeding the previous frame again would manufacture a
        measurement that no sensor produced, and the per-view tracker would
        count it as a hit.
        """
        unknown = set(frames) - set(self.camera_ids)
        if unknown:
            raise KeyError(f"frames for unknown camera(s): {sorted(unknown)}")

        active = [cid for cid in self.camera_ids if cid in frames]
        if not active:
            return []

        images = [frames[cid] for cid in active]
        detections = self.detect_batch(images)
        boxes = [box for box, _ in detections]
        embeddings = embed_views(self.embedder, images, boxes)

        observations: list[ViewObservation] = []
        for camera_id, (view_boxes, scores), vectors in zip(
            active, detections, embeddings, strict=True
        ):
            per_camera = [
                Detection(
                    bbox_xyxy=view_boxes[i],
                    score=float(np.clip(scores[i], 0.0, 1.0)),
                    embedding=vectors[i],
                )
                for i in range(view_boxes.shape[0])
            ]
            observations.extend(self.trackers[camera_id].update(per_camera, frame))
        return observations
