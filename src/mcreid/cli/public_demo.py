"""`mcreid-public-demo` — the shippable demo, from public data only.

`plan-public-demo.md`. The live capture session was cancelled permanently by the
operator on 2026-08-10, so there is no room rig and never will be: the demo runs
on a public dataset, using **that dataset's own calibration**, and nothing here
touches a webcam or a marker session.

## Two things about this command that are deliberate and easy to undo by accident

**The segment is pinned in code, not defaulted in a flag.** `SHIP_SEGMENT` is the
sparsest 40-frame window WILDTRACK contains, and the artifact, the gate and the
README all have to describe the *same* frames or the comparison is between two
different things. A flag default drifts; a constant does not.

**The hero artifact is the BEV, and that is a licence constraint, not a taste.**
`CLAUDE.md`: anything rendered from the dataset **is** the dataset — a
`!docs/assets/*.gif` whitelist once let 5.9 MB of real WILDTRACK frames into
history. The BEV canvas is procedural: floor grid, per-identity dots, camera
frusta, no dataset pixels anywhere in it. Per-camera overlays are still rendered,
because they are the honest way to look at a run, and they stay in gitignored
`reports/`. If you are about to whitelist something out of `reports/`, stop.

**The COMPOSITE is the legible artifact and it is generated, never committed.**
A BEV alone asks the viewer to take on faith that the dots are the people in the
footage; the composite shows the camera views that produced it beside the map,
with the same integer in the same colour over the same person in every panel.
It therefore contains dataset pixels by construction, so it is written under
gitignored `reports/` and `mcreid.viz.composite.assert_generated_only` refuses
any destination inside `docs/`. Its hash and a `contains_dataset_pixels: true`
flag go into the committed results JSON — the metric record ships, the pixels
never do. Gate G_D4.

## What it honestly is

WILDTRACK's sparsest window still holds **15.6 people/frame** (floor 13). That is
not the one-to-a-handful regime `context.md` scopes this project to, and this
command does not pretend otherwise — it prints the count, writes it into the
artifact, and the README says it. The sparse arm is EPFL Laboratory.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import typer

from mcreid.calib.epfl import (
    build_rig,
    grid_id_to_world_m,
    parse_calibration,
    parse_ground_truth,
)
from mcreid.cli.epfl_live import format_live_summary, run_epfl_live_view
from mcreid.eval.id_metrics import evaluate_id_consistency
from mcreid.eval.wildtrack import load_annotations, load_rig
from mcreid.fusion.associate import AssociationConfig
from mcreid.fusion.global_id import FusionConfig, GlobalIDManager
from mcreid.live_multi import appearance_only_fusion_config
from mcreid.track.gpu_view import GpuPerViewBackend, GpuViewConfig
from mcreid.track.per_view import PerViewConfig
from mcreid.track.reid_models import DEFAULT_EMBEDDER
from mcreid.utils.device import probe_compute_device
from mcreid.utils.logging import get_logger, setup_logging
from mcreid.utils.seed import DEFAULT_SEED, seed_everything
from mcreid.viz.bev import BevRenderer
from mcreid.viz.composite import (
    ALL_STAGES,
    CompositeRecord,
    CompositeRenderer,
    OverlayStage,
    StageWriter,
    summarise,
)

logger = get_logger(__name__)
app = typer.Typer(add_completion=False, help="The public-data demo (plan-public-demo.md).")


@app.callback()
def _main() -> None:
    """Public-data demo. One subcommand per dataset arm.

    The callback exists so typer keeps subcommand dispatch with only one command
    registered — otherwise `mcreid-public-demo wildtrack` collapses to a bare
    command and the documented invocation breaks the moment a second arm lands.
    """

Image = npt.NDArray[np.uint8]
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Segment:
    """A pinned stretch of a public dataset. Constants, not flag defaults."""

    name: str
    start_slot: int
    n_frames: int
    why: str


# Chosen by measurement over all 400 annotated frames, not by eye: this is the
# sparsest 40-frame window in WILDTRACK. It is still a crowd.
SHIP_SEGMENT = Segment(
    name="wildtrack-sparsest-40",
    start_slot=311,
    n_frames=40,
    why=(
        "sparsest 40-frame window in WILDTRACK: mean 15.6 people/frame, min 13, max 19, "
        "every one of them visible in >=2 cameras. The dataset's floor over all 400 "
        "annotated frames is 13 people, so no sparser segment exists to pick."
    ),
)

ARTIFACTS = Path("docs/artifacts")
BEV_CANVAS = (900, 900)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_video(frames: list[Image], path: Path, fps: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {path}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    return path


def _write_gif(frames: list[Image], path: Path, fps: float, width: int = 720) -> Path:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    scale = width / frames[0].shape[1]
    resized: list[Any] = [
        cv2.cvtColor(
            cv2.resize(f, (width, int(f.shape[0] * scale)), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB,
        )
        for f in frames
    ]
    imageio.mimsave(path, resized, duration=1.0 / fps, loop=0)
    return path


@dataclass(frozen=True)
class CompositeSpec:
    """What the composite renderer should emit for one arm.

    ``stages`` is a list because the four overlay stages come out of ONE renderer
    (`mcreid.viz.composite`), so a stage video is an excerpt of the composite
    rather than a second pipeline that can drift from it.
    """

    stages: tuple[OverlayStage, ...]
    out_dir: Path
    prefix: str
    fps: float
    tile_width: int
    caption: str
    subcaption: str


def _run_arm(
    *,
    label: str,
    rig: Any,
    fusion_config: FusionConfig,
    frames_by_camera: dict[str, list[Image]],
    frame_indices: list[int],
    backends: dict[str, GpuPerViewBackend],
    dt: float,
    render: bool,
    bev_units: str = "metres",
    composite: CompositeSpec | None = None,
) -> dict[str, Any]:
    """One pass of the pipeline over pre-loaded frames.

    Frames are loaded once by the caller and handed to both arms, so both see
    byte-identical input and the only thing differing between their numbers is the
    fusion configuration. The caller hands each arm FRESH backends, because
    `PerViewTracker` carries state across frames and a second arm inheriting the
    first arm's tracks would be measuring the first arm.

    ``composite`` renders the camera views beside the BEV, streamed straight to
    disk. It needs ``render``, because the composite's final panel *is* the BEV,
    and its cost is reported separately from the pipeline's: a demo renderer that
    quietly inflated the pipeline's own frame time would corrupt the FPS claim
    this project makes elsewhere.
    """
    if composite is not None and not render:
        raise ValueError("a composite needs the BEV, so it needs render=True")

    manager = GlobalIDManager(rig, fusion_config)
    bev = BevRenderer(rig, canvas_size=BEV_CANVAS, grid_step_m=2.0, units=bev_units)
    camera_order = list(frames_by_camera)

    snapshots_per_frame: list[list[Any]] = []
    bev_frames: list[Any] = []
    timings: list[float] = []

    renderers: dict[OverlayStage, CompositeRenderer] = {}
    writers: dict[OverlayStage, StageWriter] = {}
    records: dict[OverlayStage, list[CompositeRecord]] = {}
    composite_seconds = 0.0
    if composite is not None:
        first = frames_by_camera[camera_order[0]][0]
        frame_size = (int(first.shape[1]), int(first.shape[0]))
        for stage in composite.stages:
            renderers[stage] = CompositeRenderer(
                camera_order, frame_size, stage=stage, tile_width=composite.tile_width
            )
            # Constructed before any GPU work: a refused destination should fail
            # in the first second of the run, not after the detector has run.
            writers[stage] = StageWriter(
                composite.out_dir / f"{composite.prefix}_{stage.value}.mp4", composite.fps
            )
            records[stage] = []

    try:
        for slot, index in enumerate(frame_indices):
            started = time.perf_counter()
            observations: dict[str, list[Any]] = {}
            views = []
            for camera_id, images in frames_by_camera.items():
                seen = backends[camera_id].step(images[slot], index)
                observations[camera_id] = list(seen)
                views.extend(seen)
            snapshots = manager.step(views, index, dt)
            timings.append(time.perf_counter() - started)
            snapshots_per_frame.append(snapshots)

            if render:
                bev_frame = bev.render(snapshots, index)
                bev_frames.append(bev_frame)
                if composite is not None:
                    assignment = dict(manager.last_assignment)
                    started_render = time.perf_counter()
                    for stage, renderer in renderers.items():
                        built = renderer.render(
                            {c: frames_by_camera[c][slot] for c in camera_order},
                            observations,
                            assignment,
                            bev_frame,
                            index,
                            live_ids=frozenset(s.global_id for s in snapshots),
                            caption=composite.caption,
                            subcaption=composite.subcaption,
                        )
                        writers[stage].write(built.image)
                        records[stage].append(built.record)
                    composite_seconds += time.perf_counter() - started_render
    finally:
        for writer in writers.values():
            writer.close()

    visible_ids = {s.global_id for snaps in snapshots_per_frame for s in snaps}
    result: dict[str, Any] = {
        "label": label,
        "snapshots_per_frame": snapshots_per_frame,
        "bev_frames": bev_frames,
        "n_ids_issued": manager.n_ids_issued,
        "ids_shown": len(visible_ids),
        "mean_live_ids_per_frame": float(np.mean([len(s) for s in snapshots_per_frame])),
        "median_ms_per_frame": float(np.median(timings)) * 1000.0,
    }
    if composite is not None:
        result["composite"] = _composite_artifact(writers, records, composite_seconds)
    return result


def _composite_artifact(
    writers: dict[OverlayStage, StageWriter],
    records: dict[OverlayStage, list[CompositeRecord]],
    seconds: float,
) -> dict[str, Any]:
    """The composite's committed record: hashes, the licence flag, and the two
    numbers that say whether the video demonstrates anything.

    `contains_dataset_pixels` is **true** here and that is not a failure — it is
    why the file is not committed. The BEV artifact carries the same key with
    `false`, and the two together are the whole licence position in machine-
    readable form: the procedural render ships, the one with footage in it does not.
    """
    total_frames = sum(len(r) for r in records.values())
    stages: dict[str, Any] = {}
    for stage, writer in writers.items():
        path = writer.path
        stats = summarise(records[stage])
        stages[stage.value] = {
            "path": path.as_posix(),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "frames": writer.frames,
            **stats,
        }
    return {
        "contains_dataset_pixels": True,
        "committed": False,
        "why_not_committed": (
            "every frame is rendered from dataset footage, and CLAUDE.md's rule is that "
            "anything rendered from the dataset IS the dataset. Written under gitignored "
            "reports/; mcreid.viz.composite.assert_generated_only refuses any path under "
            "docs/. Regenerate it with the command in the README."
        ),
        "render_seconds": round(seconds, 3),
        "median_ms_per_composite_frame": (
            round(seconds * 1000.0 / total_frames, 2) if total_frames else 0.0
        ),
        "stages": stages,
    }


def _echo_composite(artifact: dict[str, Any] | None) -> None:
    """Print the composite's licence position rather than only its filename."""
    if not artifact:
        return
    for name, stage in artifact["stages"].items():
        typer.echo(
            f"wrote {stage['path']} ({stage['bytes'] / 1e6:.1f} MB, {stage['frames']} frames, "
            f"stage '{name}')"
        )
    full = artifact["stages"].get("composite")
    if full is not None:
        typer.echo(
            f"  same ID in >=2 camera panels on {full['frames_with_an_id_in_multiple_panels']}"
            f"/{full['frames']} frames; colour conflicts: {len(full['id_colour_conflicts'])}"
        )
    typer.echo(
        "  NOT COMMITTED — contains dataset pixels. Its hash is in the results JSON; "
        "the video stays local."
    )


def _parse_stages(value: str) -> tuple[OverlayStage, ...]:
    """`--stages` -> stages. 'all' is the whole cumulative ladder."""
    if value.strip().lower() == "all":
        return ALL_STAGES
    chosen = []
    for name in value.split(","):
        token = name.strip().lower()
        if not token:
            continue
        try:
            chosen.append(OverlayStage(token))
        except ValueError as exc:
            valid = ", ".join(s.value for s in ALL_STAGES)
            raise typer.BadParameter(f"unknown stage {token!r}; valid: {valid}, all") from exc
    if not chosen:
        raise typer.BadParameter("--stages must name at least one stage")
    # Emit in ladder order regardless of how they were typed, so the artifact's
    # stage keys read as the progression they are.
    return tuple(s for s in ALL_STAGES if s in chosen)


@app.command("wildtrack")
def wildtrack(
    root: Path = typer.Option(Path("data/wildtrack_full"), help="WILDTRACK root."),
    out_dir: Path = typer.Option(Path("docs/assets"), help="Where the BEV artifacts land."),
    reports_dir: Path = typer.Option(
        Path("reports/public_demo"), help="Per-camera overlays — gitignored, never whitelisted."
    ),
    composite: bool = typer.Option(
        True, help="Render the composite (camera views + BEV) into --reports-dir."
    ),
    stages: str = typer.Option(
        "composite",
        help="Overlay stages: raw | boxes | ids | composite, comma-separated, or 'all'.",
    ),
    tile_width: int = typer.Option(480, help="Width of one camera tile in the composite."),
    weights: Path = typer.Option(Path("weights/yolo11x.pt"), help="Detector weights."),
    embedder: str = typer.Option(DEFAULT_EMBEDDER),
    imgsz: int = typer.Option(1280, help="Detector input size."),
    conf: float = typer.Option(0.25, help="Detection confidence floor."),
    fps: float = typer.Option(2.0, help="WILDTRACK's annotated-frame rate."),
    playback_fps: float = typer.Option(6.0, help="Playback rate of the exported artifacts."),
    match_radius_m: float = typer.Option(1.0, help="GT<->prediction match radius."),
    device: str = typer.Option("auto", help="Compute device: auto | cpu | cuda | cuda:N."),
    allow_cpu: bool = typer.Option(False, help="Run even if the probe resolves to CPU (§5)."),
    seed: int = typer.Option(DEFAULT_SEED),
    log_level: str = typer.Option("INFO"),
) -> None:
    """Run the demo on the pinned WILDTRACK segment, both arms, and write the artifacts.

    G_D1 (the run and its BEV artifact) and G_D2 (calibrated vs uncalibrated) come
    out of this one command, on the same frames, because a demo whose numbers were
    measured on a different segment than it shows is not evidence about the demo.
    """
    setup_logging(log_level)
    seed_everything(seed)
    probe_compute_device(device, "mcreid-public-demo wildtrack", allow_cpu=allow_cpu)
    if not root.is_dir():
        raise typer.BadParameter(
            f"{root} not found. Run: python scripts/download_wildtrack.py fetch"
        )

    segment = SHIP_SEGMENT
    rig = load_rig(root / "calibrations")
    annotations = load_annotations(root / "annotations_positions", camera_ids=rig.camera_ids)

    per_camera_paths = {
        cam.camera_id: sorted((root / "Image_subsets" / f"C{i + 1}").glob("*.png"))
        for i, cam in enumerate(rig.cameras)
    }
    slots = range(segment.start_slot, segment.start_slot + segment.n_frames)
    frame_indices = [int(per_camera_paths[rig.camera_ids[0]][s].stem) for s in slots]

    typer.echo(f"segment {segment.name}: frames {frame_indices[0]}..{frame_indices[-1]}")
    typer.echo(f"  {segment.why}")

    frames_by_camera: dict[str, list[Image]] = {}
    for camera_id, paths in per_camera_paths.items():
        images = []
        for s in slots:
            raw = cv2.imread(str(paths[s]), cv2.IMREAD_COLOR)
            if raw is None:
                raise OSError(f"could not read {paths[s]}")
            images.append(np.asarray(raw, dtype=np.uint8))
        frames_by_camera[camera_id] = images

    view_config = GpuViewConfig(
        weights=weights, imgsz=imgsz, conf_threshold=conf, embedder=embedder, device=device
    )

    def fresh_backends() -> dict[str, GpuPerViewBackend]:
        """New trackers per arm. `PerViewTracker` is stateful across frames, so a
        second arm reusing them would inherit the first arm's tracks and measure it."""
        return {c.camera_id: GpuPerViewBackend(c.camera_id, view_config) for c in rig.cameras}

    # Ground truth over exactly these frames, for both arms.
    person_ids = sorted({r.person_id for f in frame_indices for r in annotations.get(f, [])})
    gt_world = {
        pid: np.full((segment.n_frames, 2), np.nan, dtype=np.float64) for pid in person_ids
    }
    gt_visible = {
        pid: np.zeros((segment.n_frames, len(rig.cameras)), dtype=bool) for pid in person_ids
    }
    people_per_frame = []
    for slot, frame in enumerate(frame_indices):
        records = annotations.get(frame, [])
        people_per_frame.append(len(records))
        for record in records:
            gt_world[record.person_id][slot] = record.world_xy
            for cam_index, camera_id in enumerate(rig.camera_ids):
                if record.bboxes.get(camera_id) is not None:
                    gt_visible[record.person_id][slot, cam_index] = True

    dt = 1.0 / fps
    # The composite is rendered for the CALIBRATED arm only — it is the arm that
    # ships, and it is the only one with a fused floor plan to show. Rendering the
    # appearance-only arm too would double the cost to illustrate a collapse that
    # the numbers already state.
    spec = (
        CompositeSpec(
            stages=_parse_stages(stages),
            out_dir=reports_dir,
            prefix="wildtrack",
            fps=playback_fps,
            tile_width=tile_width,
            caption=(
                "WILDTRACK - 7 cameras, the dataset's own calibration, "
                "one global ID per person"
            ),
            subcaption=(
                "CROWD REGIME, not the regime this project claims: "
                "15.6 people/frame in the sparsest window the dataset contains"
            ),
        )
        if composite
        else None
    )
    arms = {}
    for label, config, render in (
        ("calibrated", FusionConfig(), True),
        ("uncalibrated", appearance_only_fusion_config(FusionConfig()), False),
    ):
        typer.echo(f"running arm: {label} ...")
        arms[label] = _run_arm(
            label=label,
            rig=rig,
            fusion_config=config,
            frames_by_camera=frames_by_camera,
            frame_indices=frame_indices,
            backends=fresh_backends(),
            dt=dt,
            render=render,
            composite=spec if render else None,
        )

    reports = {}
    for label, arm in arms.items():
        reports[label] = evaluate_id_consistency(
            gt_world=gt_world,
            gt_visible=gt_visible,
            results=arm["snapshots_per_frame"],
            n_ids_issued=arm["n_ids_issued"],
            match_radius_m=match_radius_m,
        )

    # --- artifacts. BEV only; no dataset pixels leave reports/. ---------------
    out_dir.mkdir(parents=True, exist_ok=True)
    bev_frames = arms["calibrated"]["bev_frames"]
    mp4 = _write_video(bev_frames, out_dir / "public_demo_bev.mp4", playback_fps)
    gif = _write_gif(bev_frames, out_dir / "public_demo_bev.gif", playback_fps)
    reports_dir.mkdir(parents=True, exist_ok=True)

    def _switches(report: Any) -> int:
        return int(sum(report.id_switches.values()))

    demo: dict[str, Any] = {
        "what_this_is": (
            "the shipped demo run: WILDTRACK's own calibration, the sparsest segment it "
            "contains, calibrated geometry fusion end to end. Gate evidence for G_D1."
        ),
        "honest_scope": (
            "This is the CROWD regime, not the sparse hero regime. WILDTRACK's sparsest "
            "40-frame window averages 15.6 people/frame and its floor over all 400 "
            "annotated frames is 13. context.md scopes 102 to one-to-a-handful of people "
            "in a room; no segment of this dataset is that."
        ),
        "segment": {
            "name": segment.name,
            "start_slot": segment.start_slot,
            "n_frames": segment.n_frames,
            "first_frame": frame_indices[0],
            "last_frame": frame_indices[-1],
            "why": segment.why,
        },
        "calibration_source": (
            "WILDTRACK's own intrinsics/extrinsics, via mcreid.eval.wildtrack.load_rig"
        ),
        "cameras": len(rig.cameras),
        "people_per_frame": {
            "mean": float(np.mean(people_per_frame)),
            "min": int(np.min(people_per_frame)),
            "max": int(np.max(people_per_frame)),
        },
        "gt_identities_in_segment": len(person_ids),
        "detector": {"weights": str(weights), "imgsz": imgsz, "conf": conf},
        "embedder": embedder,
        "seed": seed,
        "median_ms_per_frame_all_cameras": arms["calibrated"]["median_ms_per_frame"],
        "artifacts": {
            "bev_mp4": {"path": str(mp4), "sha256": _sha256(mp4), "bytes": mp4.stat().st_size},
            "bev_gif": {"path": str(gif), "sha256": _sha256(gif), "bytes": gif.stat().st_size},
            "contains_dataset_pixels": False,
            "note": (
                "BEV canvas only — procedural floor grid, identity dots, camera frusta. "
                "The composite below DOES render WILDTRACK frames and is therefore "
                "generated locally and never committed (CLAUDE.md)."
            ),
        },
    }
    if "composite" in arms["calibrated"]:
        demo["composite"] = arms["calibrated"]["composite"]
    (ARTIFACTS / "public_demo_wildtrack.json").write_text(
        json.dumps(demo, indent=2), encoding="utf-8"
    )

    comparison: dict[str, Any] = {
        "what_this_measures": (
            "D-014's pattern re-run on the segment that actually ships: identity "
            "persistence with the dataset's calibrated geometry versus appearance-only "
            "fusion. Identical frames, identical detections, identical embedder; the only "
            "difference is whether geometry is allowed to speak. Gate evidence for G_D2."
        ),
        "segment": demo["segment"],
        "gt_identities_in_segment": len(person_ids),
        "arms": {
            label: {
                "ids_issued": arms[label]["n_ids_issued"],
                "ids_shown": arms[label]["ids_shown"],
                "mean_live_ids_per_frame": arms[label]["mean_live_ids_per_frame"],
                "id_switches": _switches(reports[label]),
                "mean_position_error_m": reports[label].mean_position_error_m,
                "false_positive_tracks": reports[label].false_positive_tracks,
            }
            for label in arms
        },
    }
    (ARTIFACTS / "public_demo_arms.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )

    typer.echo(f"\nwrote {mp4} ({mp4.stat().st_size / 1e6:.1f} MB) and {gif}")
    _echo_composite(demo.get("composite"))
    typer.echo(f"wrote {ARTIFACTS / 'public_demo_wildtrack.json'}")
    typer.echo(f"wrote {ARTIFACTS / 'public_demo_arms.json'}")
    typer.echo("\n| arm | ids shown | ids/frame | switches | pos err |")
    typer.echo("|---|---|---|---|---|")
    for label in ("calibrated", "uncalibrated"):
        a = comparison["arms"][label]
        typer.echo(
            f"| {label} | {a['ids_shown']} | {a['mean_live_ids_per_frame']:.1f} | "
            f"{a['id_switches']} | {a['mean_position_error_m']:.3f} m |"
        )
    typer.echo(f"\nground truth in this segment: {len(person_ids)} identities")


# --- EPFL Laboratory, arm 2 -------------------------------------------------
#
# GRID-METRIC, NOT METRIC. The scale derivation dead-ended on evidence
# (`plan-public-demo.md` §10, `reports/deviation-log.md` row 3): EPFL ships no
# intrinsics, only two of four cameras carry the head-plane homography that is the
# metric ruler, and the ground homography's two Zhang constraints are mutually
# inconsistent under a centred principal point — under fx != fy no camera has a
# positive solution at all. A vertical ruler cannot be transferred to a horizontal
# ground distance without the camera's internals, so the cell size is
# unidentifiable from this data.
#
# So this arm works in GRID CELLS and says so everywhere. The radii below are
# RE-DERIVED, not converted: a conversion would need the scale we just said we do
# not have. A cluster radius has to sit above same-person cross-camera
# disagreement and below the separation of distinct people, or it fuses two people
# by construction — and EPFL's own ground truth puts distinct people at p05 6.1
# cells apart, so 3.0 cells is the same comfortably-inside-the-bracket position
# 1.0 m held on a metric rig.
EPFL_CELL_UNIT = 1.0  # world unit == one grid cell, by construction
EPFL_BIRTH_CLUSTER_CELLS = 3.0
EPFL_MERGE_CELLS = 2.25
EPFL_IMAGE_SIZE = (360, 288)
EPFL_GT_STRIDE = 25  # the ground truth is annotated once a second at 25 fps


# EVERY metric constant in the fusion path, re-derived — not just the two obvious
# radii. Missing one is not a small error: with `ground_model_sigma` left at its
# metre value the Mahalanobis gate is ~13x too tight in cell units, rejects every
# association, and the calibrated arm reports ZERO identities while the
# appearance-only arm (which opens the geometric radii) still works. That is
# exactly the failure this cost a run to find, and it is why the list is explicit.
EPFL_ASSOCIATION_MAX_CELLS = 7.5
"""Association gate. Kept at 2.5x the birth radius, its original relationship."""
EPFL_REVIVE_SPEED_MARGIN_CELLS = 4.5
"""Revival speed margin. Kept at 1.5x the birth radius."""
EPFL_GROUND_MODEL_SIGMA_CELLS = 2.0
"""World-space error floor. NOT a ratio — this one is MEASURED. It exists because
a box's bottom edge is not the ground-contact point, and `check_epfl_instrument.py`
measures precisely that quantity on this rig: median 2.02 cells from a detected
foot to the nearest annotated person."""
EPFL_MAX_POSITION_SIGMA_CELLS = 4.5
"""Cap on positional sigma. 1.5x the birth radius, and it must stay above the
sigma floor above or the cap would clamp the floor."""


def epfl_fusion_config() -> FusionConfig:
    """`FusionConfig` in grid cells. Deviation-log row 3."""
    return FusionConfig(
        birth_cluster_radius_m=EPFL_BIRTH_CLUSTER_CELLS,
        merge_radius_m=EPFL_MERGE_CELLS,
        revive_speed_margin_m=EPFL_REVIVE_SPEED_MARGIN_CELLS,
        ground_model_sigma_m=EPFL_GROUND_MODEL_SIGMA_CELLS,
        max_position_sigma_m=EPFL_MAX_POSITION_SIGMA_CELLS,
        association=AssociationConfig(max_distance_m=EPFL_ASSOCIATION_MAX_CELLS),
    )


@app.command("epfl")
def epfl(
    root: Path = typer.Option(Path("data/epfl_lab"), help="EPFL Laboratory sequence root."),
    sequence: str = typer.Option("6p", help="Which sequence: 4p | 6p."),
    out_dir: Path = typer.Option(Path("docs/assets"), help="Where the BEV artifacts land."),
    reports_dir: Path = typer.Option(
        Path("reports/epfl_demo"), help="Composite output — gitignored, never whitelisted."
    ),
    composite: bool = typer.Option(
        True, help="Render the composite (camera views + BEV) into --reports-dir."
    ),
    stages: str = typer.Option(
        "composite",
        help="Overlay stages: raw | boxes | ids | composite, comma-separated, or 'all'.",
    ),
    tile_width: int = typer.Option(480, help="Width of one camera tile in the composite."),
    n_frames: int = typer.Option(60, help="Annotated frames (1 s apart) to run."),
    start: int = typer.Option(0, help="First annotated frame."),
    weights: Path = typer.Option(Path("weights/yolo11x.pt"), help="Detector weights."),
    embedder: str = typer.Option(DEFAULT_EMBEDDER),
    imgsz: int = typer.Option(640, help="Detector input size. The frames are 360x288."),
    conf: float = typer.Option(0.25, help="Detection confidence floor."),
    playback_fps: float = typer.Option(6.0, help="Playback rate of the exported artifacts."),
    device: str = typer.Option("auto", help="Compute device: auto | cpu | cuda | cuda:N."),
    allow_cpu: bool = typer.Option(False, help="Run even if the probe resolves to CPU."),
    seed: int = typer.Option(DEFAULT_SEED),
    log_level: str = typer.Option("INFO"),
    live_view: bool = typer.Option(
        False,
        "--live-view",
        help=(
            "Open one window and run the pipeline on consecutive video frames. "
            "Does not write the benchmark mp4, gif, JSON, or composite."
        ),
    ),
    live_start: int = typer.Option(
        0,
        help="With --live-view, first video frame. Later frames are read in order.",
    ),
    live_frames: int = typer.Option(
        0,
        help="With --live-view, stop after this many frames. 0 runs until q or the end.",
    ),
    live_snapshot_dir: Path | None = typer.Option(
        None,
        help="With --live-view, save a few preview PNGs here. Omit to write nothing.",
    ),
    presentation_map: bool = typer.Option(
        False,
        "--presentation-map",
        help=(
            "With --live-view, draw a schematic plate on the dataset's own grid "
            "and hide camera coverage polygons. Not a measured floor plan."
        ),
    ),
) -> None:
    """The SPARSE demo: 4-6 people in a room, 4 cameras, the dataset's own calibration.

    This is the regime `context.md` §1 actually scopes 102 to, and the only public
    dataset on hand that contains it. Distances are in GRID CELLS, not metres —
    see the module note and deviation-log row 3.

    ``--live-view`` watches consecutive frames in one window. It does not read
    ground-truth positions into the tracker and it does not write the benchmark
    artifacts.
    """
    setup_logging(log_level)
    seed_everything(seed)
    probe_compute_device(device, "mcreid-public-demo epfl", allow_cpu=allow_cpu)
    if not root.is_dir():
        raise typer.BadParameter(f"{root} not found. See the README quickstart for the fetch.")

    calibs = parse_calibration(root / f"calibration-{sequence}.txt")
    positions, header = parse_ground_truth(root / f"gt_lab_{sequence}.txt")
    grid = (header["grid_w"], header["grid_h"])
    rig = build_rig(calibs, EPFL_CELL_UNIT, EPFL_IMAGE_SIZE, grid)
    typer.echo(
        f"{len(rig.cameras)} cameras, grid {grid[0]}x{grid[1]}, GRID-METRIC (1 unit = 1 cell)"
    )

    if live_view:
        if live_start < 0 or live_frames < 0:
            raise typer.BadParameter("--live-start and --live-frames must be >= 0")
        summary = run_epfl_live_view(
            root=root,
            sequence=sequence,
            rig=rig,
            fusion_config=epfl_fusion_config(),
            image_size=EPFL_IMAGE_SIZE,
            weights=weights,
            embedder_name=embedder,
            imgsz=imgsz,
            conf=conf,
            device=device,
            start_frame=live_start,
            max_frames=live_frames,
            annotated_from_frame=min(positions) if positions else None,
            snapshot_dir=live_snapshot_dir,
            presentation_map=presentation_map,
        )
        typer.echo(format_live_summary(summary))
        return

    captures = [
        cv2.VideoCapture(str(root / f"{sequence}-c{i}.avi")) for i in range(len(rig.cameras))
    ]
    for index, capture in enumerate(captures):
        if not capture.isOpened():
            raise typer.BadParameter(f"could not open {root}/{sequence}-c{index}.avi")

    annotated = sorted(positions)[start : start + n_frames]
    if not annotated:
        raise typer.BadParameter(f"no annotated frames from index {start}")

    frames_by_camera: dict[str, list[Any]] = {c.camera_id: [] for c in rig.cameras}
    kept: list[int] = []
    for row in annotated:
        images = []
        ok = True
        for capture in captures:
            capture.set(cv2.CAP_PROP_POS_FRAMES, row)
            good, frame = capture.read()
            if not good or frame is None:
                ok = False
                break
            images.append(np.asarray(frame, dtype=np.uint8))
        if not ok:
            continue
        for cam, image in zip(rig.cameras, images, strict=True):
            frames_by_camera[cam.camera_id].append(image)
        kept.append(row)
    for capture in captures:
        capture.release()
    stride_s = EPFL_GT_STRIDE / float(header["fps"])
    typer.echo(f"decoded {len(kept)} annotated frames ({stride_s:.1f} s apart)")

    view_config = GpuViewConfig(
        weights=weights, imgsz=imgsz, conf_threshold=conf, embedder=embedder, device=device
    )

    # n_init=1, for the reason `cli/eval_wildtrack.py` already documents: the
    # per-view tracker links frames by IoU continuity, and EPFL's ground truth is
    # annotated ONCE A SECOND. A person moves ~5 grid cells in that time, so
    # consecutive boxes do not overlap at all, nothing ever reaches the default
    # n_init=5, and the first run of this command reported ZERO identities with a
    # detector that was finding people at 0.86-0.92 confidence. The cross-camera
    # identity work is done by the fusion stage here, not the per-view stage.
    #
    # D-002's consequence applies and is accepted: n_init also sets the
    # dormant-adoption window (`hits in [2, n_init)`), which at n_init=1 is empty,
    # so dormant adoption is off on this arm. Over 60 frames one second apart that
    # mechanism was never the one on display.
    per_view = PerViewConfig(n_init=1)

    def fresh() -> dict[str, GpuPerViewBackend]:
        return {
            c.camera_id: GpuPerViewBackend(c.camera_id, view_config, per_view)
            for c in rig.cameras
        }

    people = sorted({p for row in kept for p in positions[row]})
    gt_world = {p: np.full((len(kept), 2), np.nan, dtype=np.float64) for p in people}
    gt_visible = {p: np.zeros((len(kept), len(rig.cameras)), dtype=bool) for p in people}
    occupancy = []
    for slot, row in enumerate(kept):
        occupancy.append(len(positions[row]))
        for person, position_id in positions[row].items():
            gt_world[person][slot] = grid_id_to_world_m(position_id, grid[0], EPFL_CELL_UNIT)
            # EPFL's GT records presence in the ROOM, not per camera. Marking all
            # cameras keeps the coverage metric honest about what is known rather
            # than inventing per-view visibility the dataset never claimed.
            gt_visible[person][slot, :] = True

    spec = (
        CompositeSpec(
            stages=_parse_stages(stages),
            out_dir=reports_dir,
            prefix=f"epfl_{sequence}",
            fps=playback_fps,
            tile_width=tile_width,
            caption=(
                "EPFL Laboratory - 4 cameras, 1-5 people in a room, "
                "one global ID per person"
            ),
            subcaption=(
                "the regime this project claims. GRID-METRIC: distances are in grid cells, "
                "not metres - the scale is unidentifiable from this dataset"
            ),
        )
        if composite
        else None
    )
    arms = {}
    for label, config, render in (
        ("calibrated", epfl_fusion_config(), True),
        ("uncalibrated", appearance_only_fusion_config(epfl_fusion_config()), False),
    ):
        typer.echo(f"running arm: {label} ...")
        arms[label] = _run_arm(
            label=label,
            rig=rig,
            fusion_config=config,
            frames_by_camera=frames_by_camera,
            frame_indices=kept,
            backends=fresh(),
            dt=stride_s,
            render=render,
            # NOT "metres". This arm is grid-metric and the BEV is the README's
            # hero image; a canvas hard-coded to print "metres" would put a unit
            # on the front page that the prose beside it correctly denies.
            bev_units="grid cells",
            composite=spec if render else None,
        )

    reports = {
        label: evaluate_id_consistency(
            gt_world=gt_world,
            gt_visible=gt_visible,
            results=arm["snapshots_per_frame"],
            n_ids_issued=arm["n_ids_issued"],
            match_radius_m=EPFL_BIRTH_CLUSTER_CELLS,
        )
        for label, arm in arms.items()
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    bev_frames = arms["calibrated"]["bev_frames"]
    mp4 = _write_video(bev_frames, out_dir / "epfl_demo_bev.mp4", playback_fps)
    gif = _write_gif(bev_frames, out_dir / "epfl_demo_bev.gif", playback_fps)

    summary: dict[str, Any] = {
        "what_this_is": (
            "the SPARSE demo: EPFL CVLab Laboratory, 4 cameras, the dataset's own "
            "calibration, the regime context.md scopes 102 to. Gate evidence for "
            "G_S (fallback taken), G_D1e and G_D2e."
        ),
        "units": "GRID CELLS, not metres",
        "why_not_metres": (
            "The metric scale is not recoverable from this dataset: no intrinsics are "
            "shipped, only 2 of 4 cameras carry the head-plane homography that is the "
            "metric ruler, and the ground homography's two Zhang constraints are "
            "mutually inconsistent under a centred principal point (fx=fy leaves the "
            "spare constraint at 0.30 against 1.0; fx!=fy has no positive solution on "
            "any camera). See plan-public-demo.md section 10 and deviation-log row 3. "
            "NO DISTANCE HERE IS COMPARABLE TO ANY METRE-DENOMINATED NUMBER IN THIS REPO."
        ),
        "scale_derivation": "ATTEMPTED AND DEAD-ENDED - fallback to grid units taken",
        "sequence": sequence,
        "annotated_frames": len(kept),
        "seconds_between_frames": stride_s,
        "cameras": len(rig.cameras),
        "grid": {"w": grid[0], "h": grid[1]},
        "occupancy": {
            "mean": float(np.mean(occupancy)),
            "min": int(np.min(occupancy)),
            "max": int(np.max(occupancy)),
        },
        "gt_identities_in_segment": len(people),
        "radii_cells": {
            "birth_cluster": EPFL_BIRTH_CLUSTER_CELLS,
            "merge": EPFL_MERGE_CELLS,
            "association_max": EPFL_ASSOCIATION_MAX_CELLS,
            "revive_speed_margin": EPFL_REVIVE_SPEED_MARGIN_CELLS,
            "ground_model_sigma": EPFL_GROUND_MODEL_SIGMA_CELLS,
            "max_position_sigma": EPFL_MAX_POSITION_SIGMA_CELLS,
            "derived_from": "distinct-person nearest-neighbour p05 = 6.1 cells (EPFL GT)",
        },
        "detector": {"weights": str(weights), "imgsz": imgsz, "conf": conf},
        "embedder": embedder,
        "seed": seed,
        "per_view_n_init": 1,
        "per_view_n_init_why": (
            "GT is annotated once a second; consecutive boxes do not overlap, so the "
            "IoU-continuity tracker never confirms at the default n_init=5. Same "
            "reasoning and same value as cli/eval_wildtrack.py. D-002 consequence "
            "accepted: the dormant-adoption window is empty at n_init=1."
        ),
        "artifacts": {
            "bev_mp4": {"path": str(mp4), "sha256": _sha256(mp4), "bytes": mp4.stat().st_size},
            "bev_gif": {"path": str(gif), "sha256": _sha256(gif), "bytes": gif.stat().st_size},
            "contains_dataset_pixels": False,
            "note": "BEV canvas only - procedural. Same licence rule as the WILDTRACK arm.",
        },
        **(
            {"composite": arms["calibrated"]["composite"]}
            if "composite" in arms["calibrated"]
            else {}
        ),
        "arms": {
            label: {
                "ids_issued": arms[label]["n_ids_issued"],
                "ids_shown": arms[label]["ids_shown"],
                "mean_live_ids_per_frame": arms[label]["mean_live_ids_per_frame"],
                "id_switches": int(sum(reports[label].id_switches.values())),
                "mean_position_error_cells": reports[label].mean_position_error_m,
                "false_positive_tracks": reports[label].false_positive_tracks,
            }
            for label in arms
        },
    }
    (ARTIFACTS / "epfl_demo.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    typer.echo(f"\nwrote {mp4} and {gif}")
    _echo_composite(summary.get("composite"))
    typer.echo(f"wrote {ARTIFACTS / 'epfl_demo.json'}")
    typer.echo("\n| arm | ids shown | ids/frame | switches | pos err (CELLS) |")
    typer.echo("|---|---|---|---|---|")
    for label in ("calibrated", "uncalibrated"):
        a = summary["arms"][label]
        typer.echo(
            f"| {label} | {a['ids_shown']} | {a['mean_live_ids_per_frame']:.1f} | "
            f"{a['id_switches']} | {a['mean_position_error_cells']:.2f} |"
        )
    typer.echo(
        f"\nground truth: {len(people)} identities, occupancy "
        f"{summary['occupancy']['min']}-{summary['occupancy']['max']} "
        f"(mean {summary['occupancy']['mean']:.1f}) -- THE SPARSE REGIME"
    )
