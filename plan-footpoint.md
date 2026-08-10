# Plan — foot-point estimator (102_multicam_reid)

date: 2026-08-10
status: approved (operator order 2026-08-10, in the task brief; gates and kill criterion as written below)
owner: matus
tier: **demo-grade**

> **No autonomous loop launches without this artifact approved.** The most expensive single failure
> in the method is a long loop burned on a bad opening prompt.

`tier: demo-grade` is binding: instrument-proved gate + honest README + one number. Not
claim-grade — no ablation-over-everything obligation, no pre-registered escalation matrix beyond
§7 below. The *reason* it can be demo-grade is that the number this task produces is an
engineering improvement to an input, not a finding about the world. Upgrading to claim-grade
after the numbers exist would be a new plan, not an edit.

## 0. Prior art (C0)

`reports/c0_findings.md` § "C0 #2 — the foot-point estimator". Searched GitHub, PyPI, HuggingFace,
ultralytics docs and the literature; A0 resolved on four candidates by `curl`/API rather than by
reading a claim.

Verdict: **borrow the keypoints, build the estimator.** YOLO11-pose (AGPL-3.0, already pinned at
`ultralytics==8.4.104`, weights resolve HTTP 200 unauthenticated) supplies COCO-17 ankles at
indices 15/16 with per-keypoint confidence. No library takes our boxes and returns a better ground
point — FootTrackNet comes closest and brings its own detector, which would break the
one-thing-changes attribution the whole measurement rests on. Stature/head-foot homology is thirty
lines of geometry we already hold every input for.

This section is not empty, and every number below is cited from C0 rather than restated (doctrine 12).

## 1. Goal

A person's ground-contact point comes from where their **feet** are, or — when the feet are
occluded — from where their **head** is plus how tall people are, instead of from the bottom edge
of a box that whoever stands in front has truncated. When this is done, the same person seen from
two calibrated cameras lands in the same place on the floor often enough that the calibration
gate's approved ≥ 90 % cross-view hold survives on *detector* boxes, not just on ground truth.

This is the oldest unpaid debt in the workspace. It is what 102's own postmortem named as "the top
item for the next session", and 103 inherited the defect and died with it upstream of everything
it measured.

## 2. Non-goals

Explicit, and this section is what incoming instructions get checked against.

- **No training and no fine-tuning of anything.** Pretrained weights only. A failing gate is a
  finding, not a licence to train — that sentence is in §4 of the method and it is 102's recorded
  cause of death.
- **Not the crowd regime.** WILDTRACK is the measurement surface because it has multi-view ground
  truth, not because 102 is claiming the crowd case. `context.md` §1 scopes this project to one-to-
  a-handful of people in a room, and that does not change here.
- **No calibration-session work.** Floor markers, `mcreid-calibrate mark`, the rig — all of it is
  blockers row 1 and belongs to the operator's hands. Nothing in this task touches it.
- **No new inference runtime.** onnxruntime is not being added for this (C0 candidates 2 and 3).
- **No detector change.** `yolo11x.pt` stays; `ultralytics` stays pinned at 8.4.104. The boxes are
  held fixed *on purpose* — see §3's instrument proof.
- **No re-tuning of fusion thresholds** to accommodate whatever the estimator produces. If the
  estimator works, the existing radii do their existing job.
- **Not the mint-before-dormant defect** (D-012), not the id-73 classification, not anything else
  open in `status.txt`.

## 3. Acceptance gates — machine-checkable

Each gate is a command that exits 0/1. `scripts/check_footpoint_gates.py` runs all of them against
committed JSON and is the single arbiter.

### The surface, and an honest correction to the brief

The task brief specifies gating "against the existing D-015 harness with its validated 0.08 m GT
reference row". **That harness does not exist in this repo.** C0 searched `src/`, `tests/`,
`scripts/` and the git history: D-015's noise sweep was session-time analysis and only its
*result* was recorded, in `decisions.md`. Its 0.08 m row is a number in a decision entry, not a
fixture that can be re-run.

What *is* committed and reproducible is `mcreid.eval.footpoint` + `mcreid-wildtrack footpoint`,
which measures **the same statistic** — cross-camera disagreement, in metres, for one annotated
person projected independently from two views — on real WILDTRACK boxes, with three committed
artifacts already on disk. The gates therefore measure there, and D-015's threshold is **routed
across by its derivation** (method §3: route a threshold breach by the derivation of the
threshold, not by its name). The derivation is spelled out in G_FP2 so it can be checked rather
than trusted. A blockers row records the substitution.

### G_FP0 — instrument proof, and it does NOT gate the kill counter

| # | Gate | Command | Passing |
|---|---|---|---|
| G_FP0a | The refactor did not move the measurement. Routing the *existing* box-bottom rule through the new pluggable estimator reproduces the three committed artifacts **exactly**. | `uv run python scripts/check_footpoint_gates.py --gate g0a` | mean/p50/p90 identical to `docs/artifacts/footpoint_iou{0.5,0.3,0.1}.json` to 1e-9 |
| G_FP0b | **Design ceiling, measured before any verdict.** Each arm is run on **GT boxes**. If an arm cannot reach G_FP2's threshold even with perfect boxes, the arm is structurally incapable and its failure is **VOID — the kill counter does not move.** | `uv run python scripts/check_footpoint_gates.py --gate g0b` | reported, not thresholded; consumed by the kill rule |

G_FP0b exists because this workspace has already charged a kill counter for a design that could
not have passed (103, 2026-08-08c, operator ruling). Measuring the atom's ceiling and never the
design's is the specific mistake being pre-empted. **Instrument findings do not increment kill
counters — only design rejections do** (method §3).

### G_FP1 — strictly better than box-bottom

| # | Gate | Command | Passing |
|---|---|---|---|
| G_FP1 | On the same frames, the same detections and the same homographies, the arm's cross-camera disagreement is **strictly lower than the box-bottom baseline in both p50 and p90, at every IoU attribution column measured (0.5 / 0.3 / 0.1)** | `uv run python scripts/check_footpoint_gates.py --gate g_fp1` | 6 of 6 comparisons strictly lower; both numbers in `docs/artifacts/footpoint_estimators.json` |

"At every column" is not extra strictness, it is D-004: the statistic **is** the sweep, and an arm
that improves the lenient column while the strict one worsens has moved the reporting gate, not
the foot point.

### G_FP2 — lands in the band where the calibration gate survives

| # | Gate | Command | Passing |
|---|---|---|---|
| G_FP2 | Mean cross-camera disagreement **≤ 0.48 m at every IoU column**, IoU 0.1 binding | `uv run python scripts/check_footpoint_gates.py --gate g_fp2` | max over the three columns ≤ 0.48 m |

**Derivation of 0.48, written out so it can be audited rather than believed.** D-015 records a
measured curve of cross-camera foot-point gap against the one-person-two-cameras hold rate:

| gap | 2-stay-2 | 1-holds-1 |
|---|---|---|
| 0.08 m | 100 % | **99 %** |
| 0.48 m | 100 % | **98 %** |
| 1.02 m | 100 % | **33 %** |
| 1.48 m | 100 % | **13 %** |

D-014 records the *approved* acceptance criterion for that arm: **1-holds-1 ≥ 90 %**. Reading the
curve against its own approved criterion: 0.08 m passes, **0.48 m passes**, 1.02 m fails hard.
0.48 m is therefore **the largest measured gap at which the approved criterion still holds**.

The curve is unsampled between 0.48 and 1.02, and interpolating it would be inventing a threshold
by picking a point no one measured. So the gate takes the last measured passing point. This is a
threshold inherited **with its derivation**, not with its name: it is not "the merge radius", not
"the clustering radius", and not any of the three different 0.35 m values this repo has
(`method_compliance_2026-08-09.md`) — it is the point on D-015's curve where D-014's criterion
stops being met.

**0.48 m does not move.** Any change is a dated `reports/deviation-log.md` row with the authority
named, and if the change would make a failing gate pass it is a `blockers.md` proposal instead.

### G_FP3 — the pipeline still runs

| # | Gate | Command | Passing |
|---|---|---|---|
| G_FP3 | End-to-end demo runs with `--footpoint pose`, no crash, FPS within 20 % of the `bbox` baseline on the same clip | `uv run python scripts/check_footpoint_gates.py --gate g_fp3` | exit 0 on both arms; `fps_pose >= 0.8 * fps_bbox`; both in `docs/artifacts/footpoint_runtime.json` |

**Demo-first**: G_FP3 runs the thing end to end. Compiling and green unit tests are not evidence.

## 4. Not machine-testable

| Item | Why not testable | Who checks | When |
|---|---|---|---|
| Whether pose ankles are *visually* on the feet | needs eyes on overlays | matus | before the arm is believed |
| Behaviour on the real room rig | needs the calibration session (blockers row 1) | matus | after markers exist |
| Whether the anthropometric ankle height generalises past adults standing upright | no fixed target | matus | flagged, not gated |

## 5. Task graph

```
T1 contracts (estimator protocol, torch-free)  ──┬── T2 pose backend (GPU, ultralytics)
                                                 ├── T3 stature/Z=h homography (pure geometry)
                                                 └── T4 eval plumbing + G_FP0a instrument proof
T2,T3,T4 ── T5 eval run (GPU, serialised) ── T6 gates ── T7 demo integration (G_FP3)
```

| id | task | depends on | shape | model tier |
|---|---|---|---|---|
| T1 | `calib/ground_contact.py` — protocol + `bbox` arm | — | inline | opus |
| T2 | `track/pose.py` — YOLO11-pose backend, per-box ankles + confidence | T1 | inline | sonnet |
| T3 | `ground_contact.py` — `stature` arm via the Z=h homography | T1 | inline | opus |
| T4 | `eval/footpoint.py` + CLI: `--footpoint`, artifacts | T1 | inline | sonnet |
| T5 | eval run on WILDTRACK, 1 GPU, serialised | T2,T3,T4 | GPU | — |
| T6 | `scripts/check_footpoint_gates.py` | T5 | inline | opus |
| T7 | demo integration + runtime arm | T6 | inline | sonnet |

Subagent cap: 1 on the GPU, serialised (method §6). The GPU step is T5 and it runs alone.

## 6. Kill criterion — event-based, written before the first number exists

> **102's foot-point task is DEAD on the 2nd rejected design.**
>
> **Counter: 0 of 2.**

A *design* is rejected when it has been implemented, its ceiling measured by G_FP0b, and it then
**fails G_FP1 or G_FP2** on the committed eval. The two designs are, in order:

1. **pose-keypoint** (ankles from YOLO11-pose, projected at anthropometric ankle height)
2. **stature / head-foot homology** (box top-centre projected through the Z = stature homography)

**Rejections that do not count.** A rejection is **VOID** — the counter does not move — when
G_FP0b shows the arm could not have passed even on GT boxes. That is a structural finding about
the design's ceiling, not a measurement of the design, and charging the counter for it is the
error the operator ruled on in 103 on 2026-08-08c. Instrument findings (G_FP0a, harness defects,
device faults) never increment the counter either (method §3).

**When it fires:** it is honored. Never re-read, never re-scoped, never softened. A criterion
revisable by the result it is about is not a criterion. The route on death is `blockers.md` with
FootTrackNet (C0 candidate 3) named as the recorded escape hatch and its two costs stated — **and
it is an operator scope decision, not a third design.**

Copy this criterion verbatim into `status.txt`. Update the counter there, never here.

## 7. Escalation routes

Demo-grade owes fewer of these than claim-grade, but the two failure modes that are actually
likely get a route now, before data exists, so a bad number has somewhere to go that is not "edit
the threshold".

| anticipated failure | route | authority |
|---|---|---|
| Ankles are occluded exactly when the box bottom is — pose confidence low on the same detections that are truncated, so the arm degrades to `bbox` and G_FP1 is a wash | **Expected, and it is the reason the stature arm exists**: the head survives what the feet do not. Report the fallback rate per column, spend design 2. Do NOT lower the confidence floor to force keypoints through — a forced low-confidence ankle is worse than an honest fallback | lead |
| Pose keypoints are good but the ankle-height constant biases the projection | Ablate the constant (Z=0 vs Z=0.09 m) and report both. It is anthropometric, not tuned — if the ablation shows it fitting our data, drop it | lead |
| The eval is slower than budgeted (7 cams × N frames × 2 models) | Subsample frames, unchanged gate semantics, one line in `deviation-log.md` per the >5x rule | lead |
| CUDA unavailable on the eval run | STOP per method §5. `probe_compute_device()` already refuses. Blockers row, do not grind | — |
| G_FP2 fails but G_FP1 passes — better, but not enough | **This is a design rejection, and it counts** (subject to G_FP0b). "Better" was never the bar; the bar is the band where the calibration gate survives | lead |

Thresholds are never edited to make a gate pass. Any change is a dated row in
`reports/deviation-log.md` with the authority named.

## 8. Open decisions

| # | question | options | decided |
|---|---|---|---|
| 1 | Pose model size | `yolo11s-pose` (fast) vs `yolo11x-pose` (accurate) | **`yolo11x-pose`** — matches the detector already in use, and G_FP3's 20 % FPS budget is measured, not assumed. Revisit only if G_FP3 fails on runtime alone |
| 2 | Pose input | full frame + IoU-associate to our boxes, vs per-box crops | **per-box crops** — keeps our detections authoritative, so only the foot-point rule changes and the comparison stays attributable |
| 3 | Ankle height constant | 0 m vs anthropometric | **0.09 m, ablated** (§7 row 2) |
| 4 | Stature constant | fixed 1.7 m vs per-track estimate | **fixed 1.7 m for v1.** A per-track estimate is fitted from the same boxes we distrust; YAGNI until the fixed one is measured |
