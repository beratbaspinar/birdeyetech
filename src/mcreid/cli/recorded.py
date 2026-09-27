"""Replay local video files through the existing multi-camera pipeline.

No new tracker. Frames go to `MultiViewBackend` (YOLO + embedder + per-view
tracker) and `MultiLiveSession` (the same fusion stage as live multi-camera).
Videos are assumed to start together; this does not read `sync.json`.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import json
import time

import cv2
import numpy as np
import numpy.typing as npt

from mcreid.calib.schema import RigCalib
from mcreid.live_multi import (
    SINGLE_OCCUPANT_WARNING,
    MultiLiveConfig,
    MultiLiveSession,
    uncalibrated_rig,
)
from mcreid.track.gpu_view import GpuViewConfig
from mcreid.track.multi_view import MultiViewBackend
from mcreid.track.reid_models import DEFAULT_EMBEDDER, IMAGENET_RESNET18, OSNET_MSMT17
from mcreid.utils.device import probe_compute_device
from mcreid.utils.logging import get_logger

logger = get_logger(__name__)

Image = npt.NDArray[np.uint8]
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi"}


def parse_video_args(videos: str) -> list[Path]:
    """`"cam0.mp4,cam1.mp4"` -> paths. Order is camera order."""
    paths = [Path(part.strip()) for part in videos.split(",") if part.strip()]
    if len(paths) < 1:
        raise ValueError("--videos needs at least one path")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"video file(s) not found: {missing}")
    return paths


def videos_in_dir(footage: Path) -> list[Path]:
    """Sorted video files in a directory. Stems become camera ids."""
    if not footage.is_dir():
        raise FileNotFoundError(f"footage directory not found: {footage}")
    paths = sorted(
        path for path in footage.iterdir() if path.suffix.lower() in VIDEO_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"no video files in {footage}")
    return paths


def camera_ids_for(paths: list[Path]) -> list[str]:
    """File stem is the camera id (`cam0.mp4` -> `cam0`)."""
    ids = [path.stem for path in paths]
    dupes = sorted({name for name in ids if ids.count(name) > 1})
    if dupes:
        raise ValueError(f"duplicate camera id(s) from filenames: {dupes}")
    return ids


def require_local_weights(weights: Path, embedder: str, weights_dir: Path) -> None:
    """Refuse to start if a weight file would have to be downloaded."""
    if not weights.is_file():
        raise FileNotFoundError(
            f"detector weights not found: {weights}. Place the file there manually. "
            "This command does not download it. Published sizes: yolo11s ~19 MB, "
            "yolo11x ~110 MB."
        )
    if embedder == OSNET_MSMT17.name:
        path = weights_dir / (OSNET_MSMT17.weights_file or "")
        if not path.is_file():
            raise FileNotFoundError(
                f"OSNet weights not found: {path}. Place the MSMT17 checkpoint there "
                "manually. This command does not download it (tens of MB; the repo "
                "pins the URL and SHA-256, not the byte size)."
            )
        return
    if embedder == IMAGENET_RESNET18.name:
        raise FileNotFoundError(
            "imagenet_resnet18 would download torchvision ImageNet weights "
            "(~45 MB) on first use. This command does not do that. Use "
            f"{DEFAULT_EMBEDDER} with a local checkpoint instead."
        )
    raise ValueError(f"unknown embedder {embedder!r}")


def _fps_of(capture: cv2.VideoCapture) -> float:
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(fps) or fps < 1.0:
        return 30.0
    return fps


def iter_synced_frames(
    paths: list[Path], camera_ids: list[str], max_frames: int = 0
) -> Iterator[tuple[int, float, dict[str, Image]]]:
    """Yield `(index, dt, frames)` until the shortest video ends.

    Frame i of every file is treated as the same instant. No clap alignment.
    """
    captures = [cv2.VideoCapture(str(path)) for path in paths]
    try:
        closed = [not cap.isOpened() for cap in captures]
        if any(closed):
            failed = [str(path) for path, bad in zip(paths, closed, strict=True) if bad]
            raise OSError(f"could not open video(s): {failed}")
        rates = [_fps_of(cap) for cap in captures]
        if max(rates) - min(rates) > 1.0:
            logger.warning(
                "video frame rates differ (%s); stepping frame-by-frame anyway",
                ", ".join(f"{cid}={rate:.2f}" for cid, rate in zip(camera_ids, rates, strict=True)),
            )
        dt = 1.0 / rates[0]
        index = 0
        while max_frames <= 0 or index < max_frames:
            frames: dict[str, Image] = {}
            for camera_id, capture in zip(camera_ids, captures, strict=True):
                ok, frame = capture.read()
                if not ok or frame is None:
                    return
                frames[camera_id] = np.asarray(frame, dtype=np.uint8)
            yield index, dt, frames
            index += 1
    finally:
        for capture in captures:
            capture.release()


def _rig_for(
    calib: Path | None, camera_ids: list[str], first: dict[str, Image], span_m: float
) -> tuple[RigCalib, bool]:
    if calib is None:
        sizes = {cid: (int(frame.shape[1]), int(frame.shape[0])) for cid, frame in first.items()}
        return uncalibrated_rig(sizes, span_m), False
    rig = RigCalib.load(calib)
    missing = [cid for cid in camera_ids if cid not in rig.camera_ids]
    if missing:
        raise ValueError(
            f"video camera id(s) {missing} are not in {calib} (have {rig.camera_ids}). "
            "Name each file after its camera id, e.g. cam0.mp4."
        )
    subset = rig.model_copy(update={"cameras": [rig.get(cid) for cid in camera_ids]})
    for cid, frame in first.items():
        expected = subset.get(cid).intrinsics.image_size
        actual = (int(frame.shape[1]), int(frame.shape[0]))
        if actual != expected:
            raise ValueError(
                f"{cid} frame is {actual[0]}x{actual[1]} but {calib} says "
                f"{expected[0]}x{expected[1]}. World-coordinate fusion needs matching sizes."
            )
    return subset, True


def run_recorded(
    paths: list[Path],
    *,
    calib: Path | None,
    out: Path,
    weights: Path,
    embedder: str,
    weights_dir: Path,
    imgsz: int,
    conf: float,
    device: str,
    allow_cpu: bool,
    max_frames: int,
    span_m: float,
) -> Path:
    """Detect, embed, track and fuse `paths`. Returns the written mp4."""
    probe_compute_device(device, "mcreid-demo recorded", allow_cpu=allow_cpu)
    require_local_weights(weights, embedder, weights_dir)
    camera_ids = camera_ids_for(paths)
    stream = iter_synced_frames(paths, camera_ids, max_frames=max_frames)
    try:
        index, dt, first = next(stream)
    except StopIteration as exc:
        raise OSError(f"no frames decoded from {paths}") from exc

    rig, metric = _rig_for(calib, camera_ids, first, span_m)
    if metric:
        logger.info("calibrated rig %s — geometry fusion and BEV enabled", camera_ids)
    else:
        logger.warning("no --calib: %s", SINGLE_OCCUPANT_WARNING)

    backend = MultiViewBackend(
        camera_ids,
        GpuViewConfig(
            weights=weights,
            imgsz=imgsz,
            conf_threshold=conf,
            embedder=embedder,
            device=device,
            weights_dir=weights_dir,
        ),
    )
    session = MultiLiveSession(
        backend, rig, MultiLiveConfig(span_m=span_m, show_bev=metric), metric=metric
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    writer: cv2.VideoWriter | None = None
    written = 0
    now = 0.0
    det_counts = {cid: 0 for cid in camera_ids}
    frames_with_det = {cid: 0 for cid in camera_ids}
    local_global: dict[tuple[str, int], list[int]] = {}
    both_views = 0
    gid_frames: dict[int, int] = {}
    cross_sims: list[float] = []
    started = time.perf_counter()

    def _consume(step_index: int, step_dt: float, frames: dict[str, Image]) -> None:
        nonlocal writer, written, now, both_views
        now += step_dt
        mosaic, stats = session.process(frames, now, step_dt)
        for cid, count in getattr(backend, "last_detection_counts", {}).items():
            det_counts[cid] = det_counts.get(cid, 0) + int(count)
            if count:
                frames_with_det[cid] = frames_with_det.get(cid, 0) + 1
        by_gid: dict[int, list[np.ndarray]] = {}
        for cid, tracker in backend.trackers.items():
            for track in tracker.tracks:
                if not track.confirmed or track.time_since_update != 0:
                    continue
                gid = session.manager.last_assignment.get((cid, track.track_id))
                if gid is None:
                    continue
                history = local_global.setdefault((cid, track.track_id), [])
                if not history or history[-1] != gid:
                    history.append(gid)
                by_gid.setdefault(gid, []).append(np.asarray(track.embedding, dtype=np.float64))
        live_ids = [s.global_id for s in session.manager.active_snapshots()]
        for gid in live_ids:
            gid_frames[gid] = gid_frames.get(gid, 0) + 1
        if stats["multi_camera_tracks"]:
            both_views += int(stats["multi_camera_tracks"])
        for vectors in by_gid.values():
            if len(vectors) < 2:
                continue
            a = vectors[0] / max(float(np.linalg.norm(vectors[0])), 1e-12)
            b = vectors[1] / max(float(np.linalg.norm(vectors[1])), 1e-12)
            cross_sims.append(float(a @ b))
        if writer is None:
            height, width = mosaic.shape[:2]
            writer = cv2.VideoWriter(
                str(out), cv2.VideoWriter.fourcc(*"mp4v"), 1.0 / step_dt, (width, height)
            )
            if not writer.isOpened():
                raise RuntimeError(f"could not open video writer for {out}")
        writer.write(mosaic)
        written += 1
        if step_index % 30 == 0:
            logger.info(
                "frame %d  tracks %s  fps %.1f",
                step_index,
                stats["tracks"],
                stats["fps"],
            )

    try:
        _consume(index, dt, first)
        for index, dt, frames in stream:
            _consume(index, dt, frames)
    finally:
        if writer is not None:
            writer.release()
    if written == 0:
        raise OSError("recorded run produced no frames")
    switches = {
        f"{cid}:{local_id}": gids
        for (cid, local_id), gids in local_global.items()
        if len(gids) > 1
    }
    summary = {
        "device": str(backend.device),
        "device_kind": backend.device.kind,
        "calibrated": metric,
        "frames": written,
        "elapsed_s": round(time.perf_counter() - started, 2),
        "processing_fps": round(session.fps, 2),
        "detections": det_counts,
        "frames_with_detection": frames_with_det,
        "local_tracks_issued": {cid: tracker._next_id for cid, tracker in backend.trackers.items()},
        "global_ids_issued": session.manager.n_ids_issued,
        "global_ids_shown": session.reported_ids,
        "frames_per_global_id": gid_frames,
        "frames_with_cross_view_track": both_views,
        "multi_camera_ids": session.ledger.multi_camera_ids,
        "cameras_ever": {str(gid): sorted(cams) for gid, cams in session.ledger.cameras_ever.items()},
        "local_id_switches": switches,
        "resurrections": session.manager.dormant.n_resurrected,
        "cross_view_cosine_similarity": {
            "n": len(cross_sims),
            "mean": None if not cross_sims else round(float(np.mean(cross_sims)), 3),
            "min": None if not cross_sims else round(float(np.min(cross_sims)), 3),
            "max": None if not cross_sims else round(float(np.max(cross_sims)), 3),
        },
    }
    stats_path = out.with_suffix(".json")
    stats_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        "wrote %s (%d frames, ids shown %s, calibrated=%s, device=%s)",
        out,
        written,
        session.reported_ids,
        metric,
        backend.device,
    )
    logger.info("wrote %s", stats_path)
    return out
