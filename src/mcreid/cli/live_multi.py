"""`mcreid-live-multi` — live N-camera persistent-ID tracking from N webcams.

Two subcommands:

``probe``
    Enumerate capture devices, report the mode each negotiates, and measure the
    rate of every camera alone and of all of them together. Run this first: the
    numbers it prints are what `run`'s device indices and `--width/--height`
    should be chosen from, and USB bandwidth limits show up here rather than as
    a mysterious stall ten seconds into a session.

``run``
    The live session. **Uncalibrated**: the cameras do not share a floor plane,
    so fusion is appearance-only and there is no BEV panel — see
    `mcreid.live_multi` for exactly which gates that opens and why. Raw
    per-camera video and per-frame timestamps are recorded unconditionally, so
    the session is replayable offline whatever the live rate turns out to be.

Hotkeys:  q  quit        s  save an annotated mosaic
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import typer

from mcreid.capture import (
    CAPTURE_BACKENDS,
    DEFAULT_BACKEND,
    CameraSpec,
    build_rig,
    check_disk_space,
    estimated_bytes_per_second,
)
from mcreid.cli.live import resolve_fusion_config
from mcreid.diagnostics.shadow import ShadowProbe, summarise
from mcreid.fusion.global_id import FusionConfig
from mcreid.live_multi import (
    SINGLE_OCCUPANT_WARNING,
    MultiLiveConfig,
    MultiLiveSession,
    appearance_only_fusion_config,
    uncalibrated_rig,
)
from mcreid.track.gpu_view import GpuViewConfig
from mcreid.track.multi_view import MultiViewBackend
from mcreid.track.reid_models import DEFAULT_EMBEDDER
from mcreid.utils.logging import get_logger, setup_logging
from mcreid.utils.seed import DEFAULT_SEED, seed_everything

logger = get_logger(__name__)
app = typer.Typer(add_completion=False, help="Live multi-camera tracking (uncalibrated).")

WINDOW = "mcreid live-multi"


def parse_devices(devices: str) -> list[int]:
    """`"0,1"` -> `[0, 1]`, rejecting duplicates."""
    try:
        indices = [int(part.strip()) for part in devices.split(",") if part.strip()]
    except ValueError as exc:
        raise typer.BadParameter(
            f"--devices must be a comma-separated list of ints: {exc}"
        ) from exc
    if len(indices) < 1:
        raise typer.BadParameter("--devices needs at least one index")
    if len(set(indices)) != len(indices):
        raise typer.BadParameter(f"duplicate device index in {indices}")
    if any(i < 0 for i in indices):
        raise typer.BadParameter(f"device indices must be >= 0, got {indices}")
    return indices


def broadcast(values: str, n: int, what: str) -> list[str]:
    """One value per camera: either a single value for all, or exactly n.

    Cameras on one machine do not necessarily want the same settings — the
    measured example on this rig is capture rate, where one device tops out at
    15 FPS and the other at 30, so a single `--nominal-fps` would write a header
    that lies about one of the two recordings.
    """
    parts = [part.strip() for part in values.split(",") if part.strip()]
    if len(parts) == 1:
        return parts * n
    if len(parts) != n:
        raise typer.BadParameter(
            f"--{what} needs 1 value or exactly {n} (one per camera), got {len(parts)}"
        )
    return parts


def build_specs(
    devices: list[int],
    width: int,
    height: int,
    fourcc: str,
    nominal_fps: str = "30",
    backend: str = "msmf",
) -> list[CameraSpec]:
    """One `CameraSpec` per device, named cam0..camN in the order given."""
    rates = broadcast(nominal_fps, len(devices), "nominal-fps")
    backends = broadcast(backend, len(devices), "backend")
    try:
        parsed_rates = [float(rate) for rate in rates]
    except ValueError as exc:
        raise typer.BadParameter(f"--nominal-fps must be numeric: {exc}") from exc
    return [
        CameraSpec(
            camera_id=f"cam{position}",
            device=device,
            width=width,
            height=height,
            fourcc=fourcc,
            nominal_fps=rate,
            backend=name,
        )
        for position, (device, rate, name) in enumerate(
            zip(devices, parsed_rates, backends, strict=True)
        )
    ]


def _fourcc_of(capture: cv2.VideoCapture) -> str:
    """The negotiated pixel format, or '?' when the device does not report one.

    Non-printable bytes are dropped rather than emitted. A device that reports a
    partial FOURCC puts NUL bytes in this string, and printing those turns the
    whole run's stdout into a binary stream — which is how this was found: `grep`
    refused to read the probe's own output.
    """
    raw = int(capture.get(cv2.CAP_PROP_FOURCC))
    if raw <= 0:
        return "?"
    text = "".join(
        chr(byte) for i in range(4) if (byte := (raw >> (8 * i)) & 0xFF) >= 0x20
    )
    return text or "?"


def _open(index: int, backend: str, width: int, height: int) -> cv2.VideoCapture | None:
    capture = cv2.VideoCapture(index, CAPTURE_BACKENDS[backend])
    if not capture.isOpened():
        capture.release()
        return None
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    ok, frame = capture.read()
    if not ok or frame is None:
        capture.release()
        return None
    return capture


def _threaded_rates(
    captures: list[tuple[str, cv2.VideoCapture]], seconds: float
) -> dict[str, float]:
    """FPS per label with every capture reading concurrently — the real shape.

    A sequential loop over N cameras runs at the slowest one's rate because
    `read()` blocks, so measuring that way would report a bandwidth ceiling that
    does not exist.
    """
    import threading

    counts = {label: 0 for label, _ in captures}
    stop = threading.Event()

    def pump(label: str, capture: cv2.VideoCapture) -> None:
        while not stop.is_set():
            ok, frame = capture.read()
            if ok and frame is not None:
                counts[label] += 1

    threads = [
        threading.Thread(target=pump, args=(label, capture), daemon=True)
        for label, capture in captures
    ]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    while time.perf_counter() - started < seconds:
        time.sleep(0.05)
    stop.set()
    for thread in threads:
        thread.join(timeout=2.0)
    elapsed = max(time.perf_counter() - started, 1e-6)
    return {label: count / elapsed for label, count in counts.items()}


@app.command()
def probe(
    max_index: int = typer.Option(4, help="Highest device index to try (inclusive)."),
    width: int = typer.Option(1280, help="Resolution to request from each device."),
    height: int = typer.Option(720),
    seconds: float = typer.Option(3.0, help="Measurement window per configuration."),
    backends: str = typer.Option(
        "msmf,dshow",
        help=(
            "Backends to compare, comma-separated. Both by default because they do "
            "not negotiate the same media type: on this rig one USB camera reads "
            "5.0 FPS on dshow and 14.9 on msmf, at every resolution."
        ),
    ),
) -> None:
    """Enumerate capture devices, compare backends, and measure real rates."""
    setup_logging("INFO")
    names = [name.strip().lower() for name in backends.split(",") if name.strip()]
    unknown = [name for name in names if name not in CAPTURE_BACKENDS]
    if unknown:
        raise typer.BadParameter(f"unknown backend(s) {unknown}; have {sorted(CAPTURE_BACKENDS)}")

    typer.echo(f"scanning device indices 0..{max_index} on {names}")
    available: dict[str, list[int]] = {}
    for name in names:
        found = []
        for index in range(max_index + 1):
            capture = cv2.VideoCapture(index, CAPTURE_BACKENDS[name])
            ok = False
            if capture.isOpened():
                ok, frame = capture.read()
                ok = bool(ok and frame is not None)
            capture.release()
            if ok:
                found.append(index)
        available[name] = found
        typer.echo(f"  {name}: indices {found}")
    alive = sorted({index for found in available.values() for index in found})
    if not alive:
        typer.secho("no capture devices found", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo("")
    typer.echo(f"per-device, one at a time, {seconds:.0f} s each:")
    best: dict[int, tuple[str, float]] = {}
    for index in alive:
        for name in names:
            device = _open(index, name, width, height)
            if device is None:
                typer.echo(f"  index {index} {name:<5}: unavailable")
                continue
            ok, frame = device.read()
            rate = _threaded_rates([(f"{index}", device)], seconds)[f"{index}"]
            typer.echo(
                f"  index {index} {name:<5}: {frame.shape[1]}x{frame.shape[0]} "
                f"fourcc {_fourcc_of(device):<4} -> {rate:5.1f} FPS"
            )
            device.release()
            if index not in best or rate > best[index][1]:
                best[index] = (name, rate)
            time.sleep(0.3)

    typer.echo("")
    typer.echo("best backend per device:")
    for index, (name, rate) in sorted(best.items()):
        typer.echo(f"  index {index}: {name} at {rate:.1f} FPS")

    if len(best) >= 2:
        typer.echo("")
        typer.echo(f"all devices open at once, threaded, {seconds:.0f} s:")
        opened: list[tuple[str, cv2.VideoCapture]] = []
        try:
            for index, (name, _) in sorted(best.items()):
                device = _open(index, name, width, height)
                if device is None:
                    typer.secho(
                        f"  index {index}: opened alone but FAILED with "
                        f"{len(opened)} other camera(s) held — bandwidth or driver limit",
                        fg=typer.colors.YELLOW,
                    )
                    continue
                opened.append((f"{index}", device))
            rates = _threaded_rates(opened, seconds)
            for label, rate in sorted(rates.items()):
                alone = best[int(label)][1]
                drop = "" if rate > alone * 0.9 else f"  (DOWN from {alone:.1f} alone)"
                typer.echo(f"  index {label}: {rate:5.1f} FPS{drop}")
        finally:
            for _, capture in opened:
                capture.release()

    chosen = sorted(best)
    typer.echo("")
    typer.echo("suggested run command:")
    typer.echo(
        f"  uv run mcreid-live-multi run --devices {','.join(str(i) for i in chosen)}"
        f" --backend {','.join(best[i][0] for i in chosen)}"
        f" --nominal-fps {','.join(f'{best[i][1]:.0f}' for i in chosen)}"
    )
    specs = build_specs(
        chosen,
        width,
        height,
        "MJPG",
        ",".join(f"{best[i][1]:.0f}" for i in chosen),
        ",".join(best[i][0] for i in chosen),
    )
    rate_mb = estimated_bytes_per_second(specs) / 1e6
    typer.echo(
        f"estimated recording rate at these settings: "
        f"{rate_mb:.2f} MB/s ({rate_mb * 60:.0f} MB/min)"
    )


@app.command()
def run(
    devices: str = typer.Option(
        "0,1", help="Comma-separated capture device indices, in order. Named cam0, cam1, ..."
    ),
    width: int = typer.Option(1280, help="Requested capture width per camera."),
    height: int = typer.Option(720, help="Requested capture height per camera."),
    fourcc: str = typer.Option(
        "MJPG",
        help=(
            "Requested pixel format. Cameras that only offer YUY2 ignore it; the "
            "negotiated format is reported at startup."
        ),
    ),
    nominal_fps: str = typer.Option(
        "30",
        help=(
            "Rate written into the recordings' container header, and the basis of "
            "the disk estimate. One value for all cameras, or one per camera "
            "('30,15'). Real per-frame times always go to the timestamp CSV — set "
            "this from `probe` so playback speed is roughly right too."
        ),
    ),
    backend: str = typer.Option(
        DEFAULT_BACKEND,
        help=(
            "Capture backend, one value or one per camera. Default msmf on measured "
            "evidence: this rig's USB camera negotiates 5.0 FPS on dshow and 14.9 on "
            "msmf. Run `probe` to check yours."
        ),
    ),
    weights: Path = typer.Option(
        Path("weights/yolo11s.pt"),
        help="Detector weights. yolo11s is the real-time default; yolo11x is slower.",
    ),
    imgsz: int = typer.Option(960, help="Detector input size (multiple of 32)."),
    conf: float = typer.Option(0.35, help="Detection confidence floor."),
    embedder: str = typer.Option(DEFAULT_EMBEDDER, help="Appearance model."),
    span_m: float = typer.Option(
        6.0, help="Assumed floor span of the frame height in the pixel-plane stand-in."
    ),
    occupancy: str = typer.Option(
        "single",
        help=(
            "How many people will be in view. This is an ASSERTION ABOUT THE ROOM, "
            "not a preference, and it decides whether cross-view fusion runs at all. "
            "'single': geometry abstains between cameras so appearance can fuse the "
            "two views — measured to fuse two DIFFERENT people 76.7% of the time, so "
            "it is only valid when you are alone. 'multi': shipped geometry-gated "
            "config; strangers are safe and cross-view fusion will NOT work on an "
            "uncalibrated rig (measured 0% of genuine cross-view pairs fuse)."
        ),
    ),
    dormant_gate: float = typer.Option(
        None, help="Override the dormant (long-gap) appearance gate. Unset uses the shipped 0.42."
    ),
    single_occupant: bool = typer.Option(
        False,
        "--single-occupant",
        help=(
            "Dormant-gallery duplicate suppression plus the scoped probe retry. Valid "
            "ONLY when you are the only person in frame."
        ),
    ),
    shadow_probe: Path = typer.Option(
        None,
        help=(
            "DIAGNOSTIC. Per-frame dormant-distance record to PATH (.jsonl + .csv). "
            "Camera-agnostic: it reads the fused manager, so it costs the same here "
            "as on the single-camera path."
        ),
    ),
    record_dir: Path = typer.Option(
        Path("reports/live_multi"),
        help="Where the raw per-camera recordings and timestamp CSVs are written.",
    ),
    no_record: bool = typer.Option(
        False,
        "--no-record",
        help=(
            "Skip the raw recording. NOT recommended: the recording is what makes a "
            "session reusable offline when the live rate disappoints."
        ),
    ),
    expect_minutes: float = typer.Option(
        10.0, help="Session length the pre-flight disk check budgets for."
    ),
    tile_height: int = typer.Option(480, help="Display height of each camera tile."),
    max_frames: int = typer.Option(0, help="Stop after N processed steps (0 = until 'q')."),
    seed: int = typer.Option(DEFAULT_SEED),
    log_level: str = typer.Option("INFO"),
) -> None:
    """Open every camera and track continuously until 'q'."""
    setup_logging(log_level)
    seed_everything(seed)

    indices = parse_devices(devices)
    specs = build_specs(indices, width, height, fourcc, nominal_fps, backend)
    session_name = time.strftime("%Y%m%d_%H%M%S")

    if not no_record:
        needed_mb, free_mb, ok = check_disk_space(record_dir, specs, expect_minutes)
        typer.echo(
            f"disk: ~{needed_mb:.0f} MB for {expect_minutes:.0f} min of raw recording, "
            f"{free_mb / 1000:.1f} GB free"
        )
        if not ok:
            raise typer.BadParameter(
                f"not enough free disk: need ~{needed_mb:.0f} MB plus 5 GB reserve, "
                f"have {free_mb:.0f} MB. Free space, shorten --expect-minutes, or lower "
                "the resolution."
            )
    else:
        typer.secho(
            "--no-record: nothing is being saved. A disappointing live rate will leave "
            "you with no session to analyse offline.",
            fg=typer.colors.YELLOW,
        )

    rig_streams = build_rig(specs, None if no_record else record_dir, session_name)
    rig_streams.start()
    typer.echo(
        f"opened {len(specs)} camera(s): "
        + ", ".join(f"{s.camera_id}=device {s.device} on {s.backend}" for s in specs)
    )

    try:
        # Every camera must deliver one frame before the pixel-plane calibration
        # can be built: a camera that silently downgrades 720p to VGA would
        # otherwise get a calibration describing a resolution it never produced,
        # and every border/truncation test against it would be wrong.
        sizes: dict[str, tuple[int, int]] = {}
        deadline = time.perf_counter() + 10.0
        while len(sizes) < len(specs) and time.perf_counter() < deadline:
            rig_streams.raise_for_errors()
            for camera_id, frame in rig_streams.latest().items():
                if camera_id not in sizes:
                    sizes[camera_id] = (frame.image.shape[1], frame.image.shape[0])
            time.sleep(0.02)
        missing = [s.camera_id for s in specs if s.camera_id not in sizes]
        if missing:
            raise typer.BadParameter(f"no frames from {missing} within 10 s")
        for camera_id, (actual_w, actual_h) in sizes.items():
            note = "" if (actual_w, actual_h) == (width, height) else "  (DOWNGRADED)"
            typer.echo(f"  {camera_id}: {actual_w}x{actual_h}{note}")

        if occupancy not in {"single", "multi"}:
            raise typer.BadParameter(f"--occupancy must be 'single' or 'multi', got {occupancy!r}")
        rig = uncalibrated_rig(sizes, span_m)
        flag_config = resolve_fusion_config(dormant_gate, single_occupant)
        appearance_only = occupancy == "single"
        fusion_config = (
            appearance_only_fusion_config(flag_config)
            if appearance_only
            else (flag_config or FusionConfig())
        )
        typer.echo("uncalibrated rig: no shared floor plane, no BEV, no metric claim.")
        if appearance_only:
            typer.secho(f"--occupancy single: {SINGLE_OCCUPANT_WARNING}", fg=typer.colors.YELLOW)
        else:
            typer.secho(
                "--occupancy multi: geometric gates stay ACTIVE on pixel-plane "
                "coordinates that two cameras do not share. Strangers are safe; "
                "cross-view fusion will not happen. Measured on real crops: 0% of "
                "genuine cross-view pairs fuse under this config.",
                fg=typer.colors.YELLOW,
            )
        if single_occupant:
            typer.echo("single-occupant mode: dormant duplicate suppression + scoped retry on.")
        if dormant_gate is not None:
            typer.echo(f"dormant appearance gate overridden: {dormant_gate:.2f}")

        dormant_cfg = fusion_config.dormant
        shadow = (
            ShadowProbe(
                shadow_probe, gate=dormant_cfg.appearance_distance, top_k=dormant_cfg.top_k
            )
            if shadow_probe is not None
            else None
        )
        if shadow is not None:
            typer.echo(f"shadow probe ON (diagnostic): recording to {shadow_probe}.jsonl/.csv")

        stepper = MultiViewBackend(
            camera_ids=rig.camera_ids,
            config=GpuViewConfig(
                weights=weights, imgsz=imgsz, conf_threshold=conf, embedder=embedder
            ),
        )
        session = MultiLiveSession(
            backend=stepper,
            rig=rig,
            config=MultiLiveConfig(span_m=span_m, tile_height=tile_height),
            fusion_config=fusion_config,
            shadow=shadow,
        )
        typer.echo(f"warming up the models ({stepper.warmup(sizes):.1f} s)")

        typer.echo("running — 'q' to quit, 's' to save an annotated mosaic")
        processed = 0
        idle = 0
        previous = time.perf_counter()
        mosaic = None

        while True:
            rig_streams.raise_for_errors()
            frames = {cid: f.image for cid, f in rig_streams.latest().items()}
            if not frames:
                # No camera produced a new frame yet. Spinning here burns a core
                # for nothing; the shortest useful sleep is well under one frame
                # interval at any rate these cameras reach.
                idle += 1
                time.sleep(0.002)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            now = time.perf_counter()
            dt = max(now - previous, 1e-3)
            previous = now
            mosaic, info = session.process(frames, now, dt)
            cv2.imshow(WINDOW, mosaic)
            processed += 1

            if processed % 30 == 0:
                # Per-camera *cumulative* contribution, not just this step's
                # cameras: at 30 vs 15 FPS the slower camera is absent from half
                # the steps, so a snapshot of one step reads as a dead camera.
                contribution = " ".join(
                    f"{cid}:{session.frames_by_camera.get(cid, 0)}" for cid in rig.camera_ids
                )
                logger.info(
                    "%.1f FPS (%.1f processing) | fused steps per cam %s | tracks %d "
                    "(%d cross-view now, %s ever) | ids %d | coasting %d | dormant %d "
                    "| resurrected %d",
                    info["wall_fps"],
                    info["fps"],
                    contribution,
                    info["tracks"],
                    info["multi_camera_tracks"],
                    session.ledger.multi_camera_ids or "none",
                    info["reported_ids"],
                    info["coasting"],
                    info["dormant"],
                    info["resurrected"],
                )

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                typer.echo(f"saved {session.save_snapshot(record_dir, mosaic)}")
            if max_frames and processed >= max_frames:
                break
    finally:
        rig_streams.stop()
        cv2.destroyAllWindows()

    typer.echo("")
    typer.echo(
        f"processed {processed} fused steps at {session.wall_fps:.1f} FPS end-to-end "
        f"({session.fps:.1f} FPS tracking throughput), {idle} idle polls"
    )
    for stream in rig_streams.streams:
        stats = stream.stats
        recorded = stream.recorder.n_written if stream.recorder is not None else 0
        typer.echo(
            f"  {stream.spec.camera_id}: captured {stats.frames_read} frames at "
            f"{stats.measured_fps:.1f} FPS, {stats.frames_failed} failed reads, "
            f"{session.frames_by_camera.get(stream.spec.camera_id, 0)} fused, "
            f"{recorded} recorded"
        )
        # A header claiming 30 fps on a 5 fps recording plays back 6x fast, and
        # nothing about the file says so. The CSV is authoritative either way,
        # but silence here would let the mp4 be trusted.
        if recorded and abs(stats.measured_fps - stream.spec.nominal_fps) > (
            0.2 * stream.spec.nominal_fps
        ):
            typer.secho(
                f"    NOTE: recorded at a nominal {stream.spec.nominal_fps:.0f} fps but "
                f"captured at {stats.measured_fps:.1f} — the mp4 plays back "
                f"{stream.spec.nominal_fps / max(stats.measured_fps, 1e-6):.1f}x off. Use "
                f"the timestamp CSV for anything time-based, or re-run with "
                f"--nominal-fps {stats.measured_fps:.0f}.",
                fg=typer.colors.YELLOW,
            )
    if not no_record:
        typer.echo(f"raw recordings and timestamps: {record_dir} (session {session_name})")

    reported = session.reported_ids
    typer.echo(
        f"{len(reported)} identities confirmed and shown "
        f"({session.manager.n_ids_issued} tracks minted incl. tentative), "
        f"{session.manager.dormant.n_resurrected} resurrected from the gallery"
    )
    typer.echo("cross-view ledger — the acceptance evidence:")
    for line in session.cross_view_report():
        typer.echo(line)
    if occupancy == "single" and len(session.ledger.multi_camera_ids) >= 1:
        typer.secho(
            "    read this only as a SINGLE-OCCUPANT result: a CROSS-VIEW verdict is "
            "produced by two different people 76.7% of the time under this config, so "
            "it is evidence only if you were alone.",
            fg=typer.colors.YELLOW,
        )
    if session.timeline.reacquired_gap:
        gid, gap = max(session.timeline.reacquired_gap.items(), key=lambda kv: kv[1])
        typer.echo(f"longest gap survived: ID {gid} reacquired after {gap:.1f} s")

    for line in session.manager.dormant.probe_report():
        typer.echo(line)
    if session.shadow is not None:
        jsonl, csv_path = session.shadow.write()
        for line in summarise(session.shadow.rows, session.shadow.gate):
            typer.echo(line)
        typer.echo(f"shadow record: {jsonl}  and  {csv_path}")
    if session.manager.dormant.n_suppressed_duplicates:
        typer.echo(
            f"{session.manager.dormant.n_suppressed_duplicates} identity/identities "
            f"not stored as duplicates of someone already in the gallery"
        )


if __name__ == "__main__":  # pragma: no cover
    app()
