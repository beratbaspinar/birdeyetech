"""Ankle keypoints for boxes we already have.

The only file in the foot-point work that needs a GPU. Everything it produces is
plain numpy, so `mcreid.calib.ground_contact` — where the actual estimator lives
— stays torch-free and testable on an analytic rig.

**Top-down on purpose.** YOLO11-pose is a one-stage detector-plus-pose model and
could be run on the whole frame, but then its person boxes are not our person
boxes and the comparison stops being a comparison: the D-004 measurement is only
attributable if the boxes are held fixed and *only* the foot-point rule changes.
So each of our boxes becomes a crop, and pose runs inside it (plan-footpoint.md
§8, decision 2).

**The crop is the box.** It is padded by a fixed fraction for context, because
top-down pose models expect some margin, and that padding is applied identically
to every box. It is *not* extended downward to go looking for feet the box cut
off — that would be repairing the box, which is a different intervention with a
different attribution, and the honest answer when the feet are genuinely behind
someone else is a low confidence and a fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from mcreid.track.gpu_view import precision_kwargs
from mcreid.utils.device import DeviceSpec, resolve_device
from mcreid.utils.logging import get_logger

logger = get_logger(__name__)

Image = npt.NDArray[np.uint8]
FloatArray = npt.NDArray[np.float64]

# COCO-17 keypoint order. Ankles are the last two, 0-indexed.
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

DEFAULT_POSE_WEIGHTS = Path("weights/yolo11x-pose.pt")


@dataclass(frozen=True)
class PoseConfig:
    """Everything the pose pass needs, in one place."""

    weights: Path = DEFAULT_POSE_WEIGHTS
    imgsz: int = 192
    """Crop input size. Small on purpose — a person crop is not a scene, and this
    pass runs once per detection rather than once per frame, so it is the term
    that decides whether G_FP3's 20% runtime budget survives."""
    conf_threshold: float = 0.25
    """Person confidence INSIDE the crop. Low, because we already know a person
    is in there — the detector said so. This only rejects crops where the pose
    model finds nothing at all."""
    batch: int = 32
    device: str = "auto"
    half: bool = True

    def __post_init__(self) -> None:
        if self.imgsz % 32 != 0:
            raise ValueError(f"imgsz must be a multiple of 32, got {self.imgsz}")
        if not 0.0 < self.conf_threshold < 1.0:
            raise ValueError(f"conf_threshold must be in (0, 1), got {self.conf_threshold}")
        if self.batch < 1:
            raise ValueError(f"batch must be >= 1, got {self.batch}")


def crop_boxes(
    boxes_xyxy: npt.ArrayLike, image_shape: tuple[int, ...], pad_frac: float = 0.1
) -> FloatArray:
    """Padded, image-clamped crop rectangles — one per box.

    Padding is symmetric and proportional. Clamping matters more than it looks:
    a box that touches the frame edge is exactly the case where a person is half
    outside the view, which is common and must not produce a negative-width crop.
    """
    boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    height, width = float(image_shape[0]), float(image_shape[1])
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    pad_x, pad_y = w * pad_frac, h * pad_frac
    out = np.stack(
        [
            np.clip(boxes[:, 0] - pad_x, 0.0, width - 1.0),
            np.clip(boxes[:, 1] - pad_y, 0.0, height - 1.0),
            np.clip(boxes[:, 2] + pad_x, 1.0, width),
            np.clip(boxes[:, 3] + pad_y, 1.0, height),
        ],
        axis=1,
    )
    # A degenerate crop (zero width or height) would make ultralytics raise on a
    # perfectly ordinary detection at the frame edge. Widen rather than drop.
    out[:, 2] = np.maximum(out[:, 2], out[:, 0] + 1.0)
    out[:, 3] = np.maximum(out[:, 3], out[:, 1] + 1.0)
    return out


def _empty_ankles(n: int) -> FloatArray:
    """(N, 2, 3) of NaN position and zero confidence — 'nothing was seen'.

    Zero confidence rather than NaN confidence: the estimator compares against a
    threshold, and NaN would make that comparison False by accident rather than
    by intent.
    """
    out = np.full((n, 2, 3), np.nan, dtype=np.float64)
    out[..., 2] = 0.0
    return out


class PoseBackend:
    """YOLO11-pose over person crops, returning full-frame ankle keypoints.

    `model` is injectable so the whole class is testable without a GPU or a
    checkpoint — the mapping from crop coordinates back to frame coordinates is
    where the bugs live, and it is pure arithmetic.
    """

    def __init__(self, config: PoseConfig | None = None, model: Any | None = None) -> None:
        self.config = config or PoseConfig()
        self.device: DeviceSpec = resolve_device(self.config.device, allow_half=self.config.half)

        if model is None:
            from ultralytics import YOLO

            weights = Path(self.config.weights)
            if not weights.is_file():
                # Ultralytics fetches from its own release assets on construction;
                # C0 verified that URL resolves unauthenticated (A0 gate).
                logger.info("pose weights %s not on disk — ultralytics will fetch them", weights)
            model = YOLO(str(weights))
        self.model = model
        self._predict_kwargs = precision_kwargs(self.device)

    def ankles(self, image: Image, boxes_xyxy: npt.ArrayLike) -> FloatArray:
        """Left/right ankle per box, in FULL-FRAME pixels.

        Returns:
            (N, 2, 3) as ``[[left_x, left_y, left_conf], [right_x, ...]], ...``.
            A box the model found nothing in comes back NaN at confidence 0,
            which the estimator reads as "fall back to the box bottom".
        """
        boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
        n = boxes.shape[0]
        if n == 0:
            return _empty_ankles(0)

        crops = crop_boxes(boxes, image.shape)
        patches = [
            image[int(y1) : int(np.ceil(y2)), int(x1) : int(np.ceil(x2))]
            for x1, y1, x2, y2 in crops
        ]

        out = _empty_ankles(n)
        for start in range(0, n, self.config.batch):
            stop = min(start + self.config.batch, n)
            predictions: Any = self.model.predict(
                source=patches[start:stop],
                device=self.device.torch_device,
                conf=self.config.conf_threshold,
                imgsz=self.config.imgsz,
                verbose=False,
                **self._predict_kwargs,
            )
            for offset, result in enumerate(predictions):
                row = start + offset
                pair = self._best_ankles(result)
                if pair is None:
                    continue
                # Crop -> frame. The pose model reports pixels inside the patch;
                # everything downstream is in frame coordinates, and getting this
                # translation wrong is invisible on a centred crop and wrong
                # everywhere else.
                pair[:, 0] += crops[row, 0]
                pair[:, 1] += crops[row, 1]
                out[row] = pair
        return out

    @staticmethod
    def _best_ankles(result: Any) -> FloatArray | None:
        """Ankles of the highest-scoring person in one crop, or None.

        More than one person can fall inside a padded crop — that is exactly what
        happens in the crowded frames this work is about. The crop was cut around
        *our* detection, so the most confident person in it is the one it was cut
        for; taking any other would silently attribute a neighbour's feet.
        """
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None or keypoints.data is None or len(keypoints.data) == 0:
            return None

        data = np.asarray(keypoints.data.cpu().numpy(), dtype=np.float64)
        if data.ndim != 3 or data.shape[1] <= RIGHT_ANKLE or data.shape[2] < 3:
            # A pose model that is not COCO-17, or a results object with no
            # confidence channel. Refuse rather than index into the wrong joint.
            logger.warning("unexpected keypoint tensor shape %s - skipping crop", data.shape)
            return None

        boxes = getattr(result, "boxes", None)
        if boxes is not None and getattr(boxes, "conf", None) is not None and len(boxes.conf):
            best = int(np.argmax(np.asarray(boxes.conf.cpu().numpy(), dtype=np.float64)))
        else:
            best = int(np.argmax(data[:, :, 2].mean(axis=1)))
        return data[best, [LEFT_ANKLE, RIGHT_ANKLE], :3].copy()
