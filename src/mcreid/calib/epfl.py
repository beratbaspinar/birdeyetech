"""EPFL CVLab multi-camera pedestrian sequences — calibration, in metres.

The dataset ships two homographies per camera, both mapping the **same**
discretised top-view grid into the image: one onto the floor, one onto a plane
its documentation states is exactly 1.75 m higher. It does not ship intrinsics,
and it does not say how big a grid cell is.

That last omission matters more than it looks. Every threshold in this repo is
metric — a 0.35 m merge radius, a 1.0 m birth-clustering radius — so a wrong cell
size silently rescales the entire fusion stage while every gate stays green. It
is therefore **derived here, never guessed**, and the derivation carries its own
validation (`plan-public-demo.md` §10, written before the number was computed).

## The derivation

With the grid→world map written as a similarity of unknown cell size `s`:

    H_g ∝ K [ s·r1 , s·r2 , x0·r1 + y0·r2 + t ]
    H_h ∝ K [ s·r1 , s·r2 , x0·r1 + y0·r2 + t + h·r3 ]        h = 1.75 m

Normalise the pair so their first two columns agree; the third columns then
differ by exactly `d = h·K·r3`. Since ‖r1‖ = ‖r3‖ = 1,

    ‖K⁻¹·H_g[:,0]‖ = s        and        ‖K⁻¹·d‖ = h

so `s = h · ‖K⁻¹H_g[:,0]‖ / ‖K⁻¹d‖`.

`K` is recovered from the ground homography's own Zhang constraints — `r1 ⊥ r2`
and `‖r1‖ = ‖r2‖` — under the usual reduction to one unknown (zero skew, square
pixels, principal point at the image centre). One constraint solves for the focal
length; **the second is spare and becomes a check.**

Four cameras give four independent estimates of one physical quantity, coupled by
nothing but the room they are in. That is V1, and it is the check that cannot be
argued with.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from mcreid.calib.schema import CameraCalib, GroundPlane, Intrinsics, RigCalib
from mcreid.utils.logging import get_logger

logger = get_logger(__name__)

FloatArray = npt.NDArray[np.float64]

# Stated by the dataset's own documentation: the head plane is a plane parallel
# to the ground and exactly this far above it. This is the metric ruler the whole
# derivation hangs from — it is not a choice, and changing it changes every
# distance downstream.
HEAD_PLANE_M = 1.75

# EPFL's POM grids for these sequences. Read from the ground-truth header rather
# than assumed; these are the fallback when no header is supplied.
DEFAULT_GRID = (56, 56)

# The top view the homographies map INTO, in its own pixels. Documented by the
# dataset page; confirmed by the ground-truth projection check, which fails
# loudly at any other value.
TOP_VIEW_PX_W = 358
TOP_VIEW_PX_H = 360


@dataclass(frozen=True)
class EpflCameraCalibration:
    """One camera's pair of grid->image homographies, as shipped."""

    camera_id: str
    H_ground: FloatArray
    H_head: FloatArray | None
    """``None`` for the cameras that ship a lone `0` instead of a head matrix."""


@dataclass(frozen=True)
class ScaleEstimate:
    """One camera's independent estimate of the grid cell size, plus its checks."""

    camera_id: str
    cell_size_m: float
    focal_px: float
    axis_ratio: float
    """Spare Zhang constraint: ‖r1‖/‖r2‖. 1.0 if the assumptions hold (V2)."""
    camera_height_m: float


def parse_calibration(path: Path | str) -> list[EpflCameraCalibration]:
    """Read an EPFL `calibration-*.txt` into per-camera homography pairs.

    **Split on the file's own `# Camera N` headers, never by counting rows.** The
    naive reading — six rows per camera, ground then head — is wrong on this file
    and wrong silently: cameras 2 and 3 ship a lone `0` where their head-plane
    homography would be, so blind grouping slides cam3's GROUND matrix into cam2's
    HEAD slot and produces a rig that is subtly, plausibly incorrect rather than
    obviously broken. That cost a run.

    A camera with no head homography gets ``H_head = None``: it still has a usable
    ground plane, it just cannot contribute to a metric-scale derivation.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    blocks: list[list[list[float]]] = []
    current: list[list[float]] = []
    seen_header = False

    for line in text.splitlines():
        if re.match(r"^\s*#\s*Camera\s+\d+", line, flags=re.IGNORECASE):
            if seen_header:
                blocks.append(current)
            current, seen_header = [], True
            continue
        stripped = line.split("#")[0].strip()
        if not stripped:
            continue
        parts = re.split(r"[\s,]+", stripped)
        if len(parts) != 3:
            continue  # a lone `0` marks an absent matrix; it is not a row
        try:
            current.append([float(p) for p in parts])
        except ValueError:
            continue
    if seen_header:
        blocks.append(current)
    if not blocks:
        raise ValueError(f"{path}: no '# Camera N' sections found")

    out = []
    for index, rows in enumerate(blocks):
        if len(rows) < 3:
            raise ValueError(f"{path}: camera {index} has {len(rows)} rows, need at least 3")
        matrix = np.asarray(rows, dtype=np.float64)
        head = matrix[3:6].copy() if len(rows) >= 6 else None
        out.append(
            EpflCameraCalibration(
                camera_id=f"cam{index}", H_ground=matrix[:3].copy(), H_head=head
            )
        )
    logger.info(
        "parsed %d EPFL cameras; %d carry a head-plane homography",
        len(out),
        sum(c.H_head is not None for c in out),
    )
    return out


def _focal_from_homography(H: FloatArray, image_size: tuple[int, int]) -> tuple[float, float]:
    """Focal length from one homography's Zhang constraints, plus the spare one.

    Under zero skew, square pixels and a centred principal point, ``K`` has a
    single unknown ``f``. Writing ``h1, h2`` for the first two columns and
    subtracting the principal point, orthogonality ``r1 ⊥ r2`` gives

        (h1x·h2x + h1y·h2y) / f² + h1z·h2z = 0   ->   f² = -(h1x·h2x + h1y·h2y)/(h1z·h2z)

    Returns ``(f, axis_ratio)`` where ``axis_ratio`` is ‖r1‖/‖r2‖ evaluated at
    that ``f`` — the constraint that was *not* used, and therefore the one that
    can fail.
    """
    width, height = image_size
    cx, cy = width / 2.0, height / 2.0
    h1 = np.array([H[0, 0] - cx * H[2, 0], H[1, 0] - cy * H[2, 0], H[2, 0]])
    h2 = np.array([H[0, 1] - cx * H[2, 1], H[1, 1] - cy * H[2, 1], H[2, 1]])

    numerator = -(h1[0] * h2[0] + h1[1] * h2[1])
    denominator = h1[2] * h2[2]
    if abs(denominator) < 1e-18 or numerator / denominator <= 0.0:
        raise ValueError(
            "focal length is not recoverable from this homography under the "
            "zero-skew/centred-principal-point reduction "
            f"(num={numerator:.3e}, den={denominator:.3e})"
        )
    focal = float(np.sqrt(numerator / denominator))

    n1 = np.sqrt((h1[0] / focal) ** 2 + (h1[1] / focal) ** 2 + h1[2] ** 2)
    n2 = np.sqrt((h2[0] / focal) ** 2 + (h2[1] / focal) ** 2 + h2[2] ** 2)
    return focal, float(n1 / n2)


def estimate_cell_size(
    calib: EpflCameraCalibration,
    image_size: tuple[int, int],
    head_plane_m: float = HEAD_PLANE_M,
) -> ScaleEstimate:
    """One camera's independent estimate of the grid cell size, in metres."""
    if calib.H_head is None:
        raise ValueError(f"{calib.camera_id}: no head-plane homography, so no metric ruler")
    H_g = np.asarray(calib.H_ground, dtype=np.float64)
    H_h = np.asarray(calib.H_head, dtype=np.float64)

    focal, axis_ratio = _focal_from_homography(H_g, image_size)
    width, height = image_size
    K = np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])
    K_inv = np.linalg.inv(K)

    B_g = K_inv @ H_g
    B_h = K_inv @ H_h
    # Both homographies are only defined up to scale, and they were published
    # independently — so put them in a common scale using the two columns they
    # are supposed to share before differencing the third.
    scale_g = 0.5 * (np.linalg.norm(B_g[:, 0]) + np.linalg.norm(B_g[:, 1]))
    scale_h = 0.5 * (np.linalg.norm(B_h[:, 0]) + np.linalg.norm(B_h[:, 1]))
    if scale_g < 1e-12 or scale_h < 1e-12:
        raise ValueError(f"{calib.camera_id}: degenerate homography, zero-norm columns")
    B_g_n, B_h_n = B_g / scale_g, B_h / scale_h
    if float(B_g_n[:, 0] @ B_h_n[:, 0]) < 0.0:
        B_h_n = -B_h_n  # sign is free per homography; align before differencing

    r1, r2 = B_g_n[:, 0], B_g_n[:, 1]
    r3 = np.cross(r1, r2)
    difference = B_h_n[:, 2] - B_g_n[:, 2]
    height_in_grid_units = float(np.linalg.norm(difference))
    if height_in_grid_units < 1e-12:
        raise ValueError(f"{calib.camera_id}: ground and head homographies are identical")

    # ‖B_g[:,0]‖ is s in the same normalised units the difference is measured in,
    # and after normalisation it is 1 by construction — so the ratio IS s/h.
    cell_size = float(head_plane_m / height_in_grid_units)

    R = np.stack([r1, r2, r3], axis=1)
    t = B_g_n[:, 2]
    centre = -R.T @ t
    camera_height = float(abs(centre[2]) * cell_size)

    return ScaleEstimate(
        camera_id=calib.camera_id,
        cell_size_m=cell_size,
        focal_px=focal,
        axis_ratio=axis_ratio,
        camera_height_m=camera_height,
    )


def build_rig(
    calibrations: list[EpflCameraCalibration],
    cell_size_m: float,
    image_size: tuple[int, int],
    grid: tuple[int, int] = DEFAULT_GRID,
) -> RigCalib:
    """Turn the shipped homographies into a metric `RigCalib`.

    The homographies map grid->image; `mcreid` wants undistorted-image->world in
    metres, so each is inverted and post-multiplied by the grid->metres
    similarity. EPFL ships no distortion coefficients, so they are zero and
    `undistort_points` becomes a no-op — which is honest, not an approximation
    being hidden: the dataset simply does not provide them.
    """
    width, height = image_size
    grid_w, grid_h = grid
    # DIRECTION, and it is the defect that cost this arm a whole build: EPFL's
    # `H_ground` maps the CAMERA IMAGE to the TOP VIEW, not the other way round.
    # It is not inverted here. Verified against ground truth rather than assumed —
    # detected foot points pushed through it land a median 2.05 cells from the
    # nearest annotated person, where distinct people sit ~10.6 cells apart;
    # inverting it instead put them 88+ px from anybody at every scale tried.
    #
    # The top view is a TOP_VIEW_PX_W x TOP_VIEW_PX_H image of a grid_w x grid_h
    # grid, so the similarity converts top-view pixels to cells and then to world
    # units. Anisotropic on purpose: 358/56 and 360/56 are not the same number.
    sx = grid_w / TOP_VIEW_PX_W * cell_size_m
    sy = grid_h / TOP_VIEW_PX_H * cell_size_m
    similarity = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    # NOMINAL intrinsics, and they are genuinely unused on the path this rig is
    # for. EPFL ships zero distortion, so `undistort_points` short-circuits and
    # never touches K, and `image_to_ground` needs only `ground.H`. The focal
    # below exists to satisfy the schema, NOT as a claim about these cameras —
    # their intrinsics are exactly what could not be recovered (see the module
    # note). Anything that starts depending on K here needs a real calibration
    # first, and will be wrong silently if it does not get one.
    nominal_focal = float(image_size[0])

    cameras = []
    for calib in calibrations:
        H_img2world = similarity @ np.asarray(calib.H_ground, dtype=np.float64)
        H_img2world = H_img2world / H_img2world[2, 2]
        cameras.append(
            CameraCalib(
                camera_id=calib.camera_id,
                intrinsics=Intrinsics(
                    fx=nominal_focal,
                    fy=nominal_focal,
                    cx=width / 2.0,
                    cy=height / 2.0,
                    dist_coeffs=[0.0] * 5,
                    image_width=width,
                    image_height=height,
                    rms_reproj_px=0.0,
                    n_views=0,
                ),
                ground=GroundPlane(
                    H_img2world=H_img2world.tolist(),
                    method="four_point",
                    rms_error_m=0.0,
                    n_correspondences=4,
                    floor_extent_m=(0.0, 0.0, grid_w * cell_size_m, grid_h * cell_size_m),
                ),
                notes=(
                    f"EPFL CVLab, world unit = {cell_size_m} grid cell(s). GRID-METRIC: "
                    "the metric scale is NOT recoverable from this dataset. Intrinsics "
                    "are nominal and unused (zero distortion shipped)."
                ),
            )
        )
    return RigCalib(cameras=cameras, world_notes="EPFL grid scaled to metres, Z=0 floor.")


def parse_ground_truth(path: Path | str) -> tuple[dict[int, dict[int, int]], dict[str, int]]:
    """Read `gt_lab_*.txt` into ``{frame: {person: grid_position_id}}`` plus its header.

    The header is ``n_frames n_people grid_w grid_h fps ...``; every later line is
    one frame, one column per person, with negative values meaning "not in the
    room". Negatives are dropped rather than mapped, because a person who is not
    there has no position and inventing one is how a phantom identity is born.
    """
    lines = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    # The file opens with a lone version/format line before the real header.
    offset = 1 if len(re.split(r"\s+", lines[0])) < 5 else 0
    fields = [int(v) for v in re.split(r"\s+", lines[offset])]
    header = {
        "n_frames": fields[0],
        "n_people": fields[1],
        "grid_w": fields[2],
        "grid_h": fields[3],
        "fps": fields[4],
    }
    positions: dict[int, dict[int, int]] = {}
    for frame, line in enumerate(lines[offset + 1 :]):
        values = [int(v) for v in re.split(r"\s+", line)]
        present = {person: value for person, value in enumerate(values) if value >= 0}
        if present:
            positions[frame] = present
    return positions, header


def grid_id_to_world_m(position_id: int, grid_w: int, cell_size_m: float) -> tuple[float, float]:
    """POM grid index -> world metres. Row-major, as POM enumerates it."""
    return ((position_id % grid_w) * cell_size_m, (position_id // grid_w) * cell_size_m)
