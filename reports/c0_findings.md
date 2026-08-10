# C0 findings — 102_multicam_reid (v1, late-fusion multi-camera persistent ID)

date: 2026-08-09
searched by: **BACKFILLED, not searched.** No C0 pass was ever run for 102 — the project
predates `refactored_method.md`. Every row below is reconstructed from evidence already
committed in this repo (the source header that states the verdict, the pin in
`pyproject.toml`, the entry in `decisions.md`). Nothing here was researched on 2026-08-09
and nothing is claimed that was not already on record.

> `refactored_method.md` §2: **search before building.** Library, tool, dataset, existing
> implementation. **If it exists and the licence works, adopt and stop.** This file is the authority
> for every number that later appears in `plan*.md` — the plan cites it, never restates it
> (doctrine 12).

## Standing of this file

This is a **record**, not a licence. It documents adoption decisions 102 demonstrably made and
wrote down at the time. It is **not** evidence that the search behind them was exhaustive, because
that search was never structured as C0. Any *new* build task in this repo — the foot-point
estimator is the next one — owes its own C0 pass with its own surfaces named, and must not cite
this file as having covered it.

## Where the evidence comes from

Not "where I searched" — this section is deliberately renamed, because no surfaces were swept.

| claim | carrier in this repo |
|---|---|
| detector choice + why not its tracker | `src/mcreid/track/gpu_view.py:1-18` (module docstring) |
| OSNet vendoring rationale | `src/mcreid/track/vendor/osnet.py:1-16` (vendor header) |
| dataset choice + integrity | `scripts/download_wildtrack.py` (SHA-256 pinned) |
| every version pin | `pyproject.toml`, `uv.lock` |
| the measured rejections | `decisions.md` D-006..D-009, `context.md` §4 |

## A0 — access gate (evaluated FIRST, permanently)

Added to the template 2026-08-07a after MMPTrack burned a plan revision on an email gate.
Evaluated retrospectively here; 102's one dataset passes it, which is why 102 never hit 103's wall.

| candidate | link resolves? | gate |
|---|---|---|
| WILDTRACK (EPFL CVLab) | yes — direct archive download, no registration, no agreement | **pass** |
| WILDTRACK **raw video** (not the 2 fps subset) | no — README says "we can provide if you ask" | **fail** — never used; 102's 2 fps limitation is a consequence |

## Candidates

| # | candidate | licence | fitness for our shape | verdict |
|---|---|---|---|---|
| 1 | Ultralytics YOLO11 (detection) | AGPL-3.0 | person detection at 720p on an 8 GB 4060 | **adopt** — and the repo took AGPL-3.0 from it |
| 2 | Ultralytics BoT-SORT / ByteTrack (its bundled trackers) | AGPL-3.0 | per-view tracking | **reject: no stable per-tracklet appearance vector, and the ReID plumbing varies between releases.** The whole fusion stage is built on that vector. `gpu_view.py:7-10` |
| 3 | Torchreid (`deep-person-reid`) OSNet | MIT | appearance embedding | **adapt `osnet.py`** — vendored as one file. Rejected as a dependency because it is a *training* library and its `__init__` chain pulls gdown + tensorboard for what we use at inference only |
| 4 | torchvision ImageNet ResNet-18 trunk | BSD-3 | zero-training appearance baseline | **adopt** as the *declared-weak* default. Measured inferior on real crops (0.377 vs 0.408 separation, i.e. 0.03 where the synthetic generator assumed 0.26) and shipped anyway, because v1's claim IS "no ReID training" |
| 5 | OpenCV (`opencv-contrib-python`) — `calibrateCamera`, `findHomography`, `aruco` 36h11 | Apache-2.0 | intrinsics + ground homography + floor markers | **adopt** — `calib/` is a thin layer over it, and it is the part of 102 the 103 postmortem lists first under "Keep" |
| 6 | scipy `linear_sum_assignment` | BSD-3 | per-camera Hungarian association | **adopt** — no assignment solver written here |
| 7 | WILDTRACK | research use, no EULA in the archive | public multi-view stress test with ground-plane GT | **adopt as a stress test, never as a benchmark claim** — `context.md` §5 |
| 8 | A trained multi-view detector (MVDet family) | — | the crowd case | **reject: out of scope by construction.** v1 is zero-training (`context.md` §5); published MVDet numbers are cited as context, and the two constants in `cli/eval_wildtrack.py` are deliberately left `None` rather than fabricated |

## Null results — recorded explicitly

What does **not** exist, as far as 102's own record shows. Weaker than a real C0 null: these are
"nothing was found in the course of building", not "the surfaces were swept and returned nothing".
Labelled as such so a later plan does not over-read them.

- **No off-the-shelf tracker exposes a stable per-tracklet appearance vector** across releases.
  This is the one that licensed building `PerViewTracker` — the single largest build-vs-borrow call
  in the project, and it is argued in a source header rather than in a findings file. → building it
  is justified *on the evidence stated there*, and that evidence is a property of the 2026-07
  Ultralytics API, not a permanent fact.
- **No usable synthetic harness for the identity core.** One was built (`sim/toy.py`) and then
  fenced: D-009 — a synthetic harness may test ORDERING, never appearance thresholds, because
  isotropic random vectors have no population centroid and the EMA statistic the merge actually
  tests does not move. Every identity-core number now comes from real WILDTRACK crops through real
  OSNet. → the null is *about the harness class*, not about a library.
- **No foot-point / ground-contact estimator was ever searched for.** The measurement exists
  (`mcreid-wildtrack footpoint`) and the defect is quantified (below); the *fix* has never had a
  prior-art pass of any kind. **This is the gap the next task must close first.**

## Numbers the plan will cite

Already measured, already committed. A foot-point plan cites these from here; it does not restate
them (doctrine 12) and it does not re-derive them.

| number | value | source (file) | what it will gate |
|---|---|---|---|
| GT-box cross-camera foot-point disagreement | mean 0.123 m, p90 0.207 m, 0 % beyond the merge radius | `docs/artifacts/footpoint_iou0.5.json` (+0.3/0.1 arms) | the floor a new estimator must approach |
| detector-box disagreement — **a range, never one number** (D-004) | mean **0.62 m at IoU 0.5 → 2.17 m at IoU 0.1** | same three artifacts | the ceiling a new estimator must beat |
| the radii the disagreement is reported against | `MERGE_RADIUS_M` 0.35 m, `CLUSTER_RADIUS_M` 1.00 m | `mcreid/eval/footpoint.py:40-41` | what "close enough to fuse" means **when reporting** |
| the radii the shipped fusion actually uses | `birth_cluster_radius_m` 1.0 m, `merge_radius_m` 0.75 m, `merge_unconditional_radius_m` **0.0 (disabled)** | `mcreid/fusion/global_id.py:124,145,153` | what "close enough to fuse" means **at runtime** — see the naming finding in `method_compliance_2026-08-09.md`; these are not the same 0.35 m and a plan must say which it means |
| calibration held-out bound | 0.25 m leave-one-out, 0.35 m cross-camera, ≥ 6 markers | D-013, `calib/floor.py` | that the rig is not the confound |
| foot-point error → cross-view hold, on the calibrated arm | 0 % → 99 %; 10 % → 98 %; 20 % → 33 %; 30 % → 13 % | D-015 (harness validated by its 0 % row landing at 0.08 m vs GT's 0.12 m) | the payoff curve — this is what makes the estimator worth building |

## Verdict

**Borrow detection, embedding, calibration primitives and the assignment solver; build the
per-view tracker, the fusion stage and the evaluation layer** — which is what 102 did, and the
reasoning for each is on record even though the file recording it did not exist until now.

**For the next task, this verdict does not transfer.** A stature-based ground-contact estimator is
a distinct C0 question (pose/keypoint estimators, ground-contact heads, published foot-point
regressors, whatever exists), and §2 requires it to be asked before a line is written.
