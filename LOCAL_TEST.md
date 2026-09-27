# Local office test (read this first)

This note is for the next person or agent. It says what was changed, which videos are in the repo, what the run actually did, and what was deliberately not downloaded.

Upstream project: [matus012/multicam_persistent_id](https://github.com/matus012/multicam_persistent_id), AGPL-3.0-only. This checkout keeps that code and adds a local-video path plus an Apple MPS device choice. Tracking, ReID, and fusion were not rewritten.

## Videos in this repo

These files are our own office recordings, not a public dataset.

| File | What it is |
|---|---|
| `cam0.mp4` | Camera 0. H.264, 1080×1920, 29.97 fps, 914 frames, 30.50 s. |
| `cam1.mp4` | Camera 1. H.264, 1080×1920, 29.97 fps, 907 frames, 30.26 s. |
| `outputs/demo/recorded.mp4` | Annotated mosaic from the run below. 907 frames, 12.6 MB. Do not delete it. |
| `outputs/demo/recorded.json` | Counts from that same run. |

Both clips show the same person walking for about 30 seconds, from two angles. OpenCV reads them. They start at nearly the same time: cam1 is 7 frames (about 0.23 s) shorter. The player stops at the shorter file. No clap sync was applied. The code assumes frame N of each file is the same moment.

Model weights are **not** in git. `weights/` and `*.pt` / `*.pth` stay ignored. Download only these two files before a real run:

- `weights/yolo11s.pt` — 19,313,732 bytes. Ultralytics asset `v8.3.0/yolo11s.pt`.
- `weights/osnet_x1_0_msmt17.pth` — 16.4 MB. URL and SHA-256 are in `src/mcreid/track/reid_models.py` (`OSNET_MSMT17`). The file we used matched that hash.

Do not download YOLO11x, EPFL, or WILDTRACK for this test.

## What changed

| File | Why |
|---|---|
| `src/mcreid/utils/device.py` | `auto` is CUDA, then Apple MPS, then CPU. MPS runs fp32. The long-job probe still stops on CPU unless `--allow-cpu`. MPS is accepted. |
| `src/mcreid/cli/recorded.py` | New. Reads the mp4s with OpenCV and feeds the existing `MultiViewBackend` and `MultiLiveSession`. |
| `src/mcreid/cli/demo.py` | `mcreid-demo recorded` used to validate paths and exit 2 without detecting anything. It now runs the pipeline. |
| `src/mcreid/track/multi_view.py` | Stores `last_detection_counts` so the summary can count boxes. No change to association. |
| `pyproject.toml`, `README.md` | macOS perception install does not use the CUDA 12.6 index. Linux/Windows `cu126` command is unchanged. |
| `tests/test_utils_device.py`, `tests/test_recorded_videos.py` | Device order and the “do not download weights” gate. |

`mcreid-demo recorded` previously exited 2 on purpose. The detector front-end had never been wired; the command only checked that a calib file and per-camera filenames existed.

## How to run

macOS, Apple Silicon. Do not pass the cu126 index.

```bash
uv venv --python 3.11
uv pip install -e ".[perception]"
uv run mcreid-demo recorded --videos cam0.mp4,cam1.mp4
```

`--videos` uses the filename stem as the camera id (`cam0.mp4` → `cam0`).

There is no `calib.json` for this room. Without `--calib` the run is appearance-only: cross-camera association uses ReID, there is no shared floor, and there is no metric bird’s-eye view. A real `calib.json` whose image size matches the frames turns geometry fusion and the BEV back on.

On the machine that produced `outputs/demo/recorded.mp4`:

- CUDA was not available.
- `torch.backends.mps.is_available()` was true.
- The probe and the JSON both say `mps (Apple MPS, half=False)`.
- A tensor allocated on that device reported `mps`.

## Result of the office run

Command: `uv run mcreid-demo recorded --videos cam0.mp4,cam1.mp4`

Device: MPS. Wall time 74.1 s. Processing rate 14.2 FPS. 907 frames written.

The same person kept **global ID 1** in both cameras. This is not only a log line. Frames 30, 200, 660, and 850 of `outputs/demo/recorded.mp4` show `ID 1` on both views, and the banner says `ACROSS VIEWS: 1 ('cam0','cam1')`. ID 1 is live for 902 of 907 frames and is the only id that was ever seen by both cameras (894 cross-view frames).

| | cam0 | cam1 |
|---|---|---|
| YOLO person boxes | 1005 | 1645 |
| Frames with at least one box | 907 | 907 |
| Local tracks issued | 5 | 6 |

Shown global ids: **1, 254, 387**. The mint counter is 595. That counter is not a person count. Most of those ids are tentative tracks that die before they are shown. The viewer-facing ids are the three above.

- ID 1: the person, both cameras, held for about 28 s.
- ID 254: 6 frames, cam0 only.
- ID 387: 28 frames, cam0 only, then that local track returned to ID 1.
- Resurrections: 0. The person did not leave long enough for the dormant gallery. That gallery was not what held the id.
- Cross-view cosine similarity of the two embeddings on the same global id: n=895, mean 0.585, min 0.42, max 0.956.

cam1 averages about 1.8 boxes per frame. The extra boxes are the source of the short extra ids, not a failure to match the main person across cameras. The clips are close enough in time that sync is not the failure mode.

Appearance-only fusion in this repo is measured as safe for one occupant. Two different people can be merged. This clip is one person, which is why ID 1 staying fused is the expected success, not a general multi-person result.

Numpy `matmul` warnings in `associate.py` and `dormant.py` also appear on the synthetic demo. They did not stop this run.

## Synthetic check

After the office run, `uv run mcreid-hpc-demo --out reports/hpc_demo.mp4` still finished. Hero global id 1, reported ids `[1, 2]`, and the three events still fire: cross-camera handoff, 2.5 s total occlusion with the id held, and resurrection after 13.2 s. That mp4 is not committed. Rebuild it with the command above. It needs no weights and no GPU.

## Next smallest change

Raise `--conf` above 0.35 and see whether the second box, and therefore ids 254 and 387, disappear, while ID 1 still spans both cameras. Do not start with a new tracker or a dataset download.
