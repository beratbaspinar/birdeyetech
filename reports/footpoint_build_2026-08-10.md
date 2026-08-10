# Foot-point estimator — build, measurement and verdict

date: 2026-08-10
plan: `plan-footpoint.md` (C1, approved 2026-08-10)
C0: `reports/c0_findings.md` § "C0 #2"
tier: demo-grade
evidence: `docs/artifacts/footpoint_estimators.json`, `docs/artifacts/footpoint_runtime.json`
arbiter: `uv run python scripts/check_footpoint_gates.py`

---

## Verdict

**Both arms were built, measured and REJECTED. Both rejections are VOID. The kill counter stays
0 of 2, and what happens next is an operator scope decision rather than a third estimator.**

| gate | result |
|---|---|
| G_FP0a instrument proof | **PASS — exactly.** 16 of 16 comparisons at delta `0.000e+00` |
| G_FP0b design ceiling | pose **0.847 m**, stature **0.484 m** on GT boxes, against a 0.48 m band → **both structurally incapable** |
| G_FP1 strictly better | **FAIL**, 12 of 12 comparisons. Neither arm beats box-bottom anywhere |
| G_FP2 the surviving band | **FAIL.** Best arm is pose at 0.741 m against 0.48 m |
| G_FP3 pipeline runs | **FAIL on runtime.** Both arms run end to end without crashing; pose is 40 % of baseline FPS against an 80 % floor |

No threshold was moved. 0.48 m is where `plan-footpoint.md` §3 put it before any number existed.

---

## The finding, which is not the one anyone expected

**The premise this whole task inherited is only half true, and the wrong half was load-bearing.**

D-004 and the 103 postmortem both say the same thing: the box's bottom edge is truncated by
whoever stands in front, so the projected foot point lands short. The implied fix is to find the
feet some other way. Both arms did exactly that, and both made it *worse*.

Measured directly — solving for the world height whose projection lands on each detector box's
bottom edge, at the person's annotated ground position:

| IoU attribution column | n | implied height of the box bottom | p90 abs |
|---|---|---|---|
| 0.5 | 149 | **−0.013 m ± 0.118** | 0.171 m |
| 0.3 | 211 | +0.014 m ± 0.228 | 0.284 m |
| 0.1 | 223 | +0.027 m ± 0.291 | 0.482 m |

**The bottom edge is not systematically lifted. It is unbiased and noisy**, and the noise grows as
the attribution gate loosens. There is no offset for a better foot point to remove — there is
variance, and every alternative rule has to read that same vertical image coordinate through the
same geometry.

And that geometry is the multiplier. A camera 2.89 m above the floor looking at someone 15 m away
converts a vertical error at the ground plane into a depth error by roughly
`depth / camera_height` ≈ **5×**. The 0.118 m spread at IoU 0.5 becomes ~0.6 m of cross-camera
disagreement, which is the 0.619 m the box-bottom baseline actually scores. **The baseline is
already at the noise floor its own optics allow.**

That is why this task could not be won by estimating the foot point better, and why the
diagnosis matters more than the two rejections.

## Arm 1 — pose keypoints. Rejected, VOID.

Ankle keypoints from YOLO11-pose, per detection crop, projected from the plane an ankle actually
sits on (0.09 m).

**The keypoints are good.** Solving for the implied world height of the predicted ankle, at the
person's true position: +0.10, +0.07, +0.18, +0.15, −0.06, −0.06, −0.11, +0.27 m. They land on
the floor, which is what they are supposed to do.

**They are just less precise than a box edge**, and the 5× multiplier is unforgiving. A 0.18 m
height error becomes ~0.94 m of depth error, which is the 0.847 m the arm scores on *perfect*
boxes. It also fell back to the box bottom on 26.8 % of GT boxes and 15.3 % of detector boxes —
honest behaviour, and it means a quarter of the "pose" column is the baseline wearing a different
label.

| | GT boxes | IoU 0.5 | IoU 0.3 | IoU 0.1 |
|---|---|---|---|---|
| bbox mean | 0.123 | 0.619 | 1.143 | 2.166 |
| **pose mean** | **0.847** | **0.741** | **1.276** | **2.336** |

Rejection is **VOID**: at 0.847 m on ground-truth boxes it could not have reached 0.48 m with
perfect input, so the measurement never tested the design against a bar it could clear.

## Arm 2 — stature / head-foot homology. Rejected, VOID.

The box top projected through the plane at a person's height. The premise was the failure mode
read backwards: heads stick out of a crowd, feet do not.

**One parameter defect was found and fixed before any verdict** — `reports/deviation-log.md`
row 2. The first run scored 5.43 m on GT boxes, which is impossible for exact geometry on exact
boxes, so it was diagnosed rather than recorded. WILDTRACK's GT boxes are rendered from a POM
cylinder: solving for the height at each box edge gives **bottom ↔ +0.01 m, top ↔ +1.82 m**,
identically across cameras and people. The constant was 1.70 m. Corrected to 1.82 m for this
surface, GT-box error fell 5.43 → 0.484 m.

**And it still fails, on a surface rigged in its favour.** Inverting a 1.82 m cylinder with
h = 1.82 recovers the annotation almost by construction, so 0.484 m is close to a tautology — and
it is still above the band.

On detector boxes it is far worse (3.46 m at IoU 0.5), and the reason is conditioning, not the
constant. Detector box tops imply **h = 1.77 ± 0.06 m** — real people, real height variance. But
the amplification for a head-plane estimate is `depth / (camera_height − person_height)`, and with
cameras at 2.0–2.9 m and people at 1.8 m, that denominator is about **1 m**: roughly **14×** at
15 m. The head ray is nearly parallel to the plane it is being intersected with. **The stature arm
is structurally unusable on a rig whose cameras are barely taller than the people** — it would be
well-conditioned on a 5 m mount, and WILDTRACK is not that.

Rejection is **VOID** for the same reason as arm 1, and more strongly.

## Why both rejections are VOID, stated carefully

`plan-footpoint.md` §6 pre-registered the rule: an arm whose G_FP0b ceiling cannot reach the band
was structurally incapable, and charging the kill counter for it is the error the operator ruled
on for 103 on 2026-08-08c.

Both arms clear that bar for voidness — and the reason they do is *the same reason for both*, which
is the point. The limit is not pose quality and not the stature constant. It is that every arm in
this class reads a vertical image coordinate and multiplies it by the rig's `depth / height`
factor. **The designs were never given a fair test on this surface, because the surface bounds all
of them at once.** This is 103's G_A shape: the premise failed, not the design.

Recording "2 of 2, dead" here would be the more dramatic answer and the less true one.

**Counter: 0 of 2.**

## What was built and kept

| file | what |
|---|---|
| `src/mcreid/calib/ground_contact.py` | Three estimators behind one protocol, torch-free. `plane_homography()` — the exact homography of the plane Z=h, by decomposing `K⁻¹·H_world2img`. Correct and tested against an analytic camera |
| `src/mcreid/track/pose.py` | YOLO11-pose over per-detection crops, ankles in frame coordinates. Injectable model, GPU-free tests |
| `mcreid-wildtrack footpoint-arms` | 3 arms × 2 box sources × 3 IoU columns in one GPU pass, identical detections throughout |
| `scripts/check_footpoint_gates.py` | The gate arbiter. Reads committed JSON, computes no metric |
| `--footpoint {bbox,pose,stature}` on `mcreid-wildtrack run` | Ships. **Default stays `bbox`**, because no gate said otherwise |
| 47 new tests | 34 geometry, 13 pose |

The geometry is the part worth keeping regardless of this verdict. `plane_homography()` is correct,
proved against a true 3×4 projection matrix rather than against another homography, and it is what
any future height-aware work needs.

## What this forecloses, and what it does not

**Foreclosed.** Estimating a better foot point *per view* and projecting it through the ground
homography. Three rules now sit on the same 5× multiplier and the best of them is the one already
shipping. A fourth rule in this class is not a plan.

**Not foreclosed, and not started.** Two routes the evidence actually points at, neither of which
is a foot-point estimator:

1. **Stop projecting per view and triangulate across views.** The disagreement being measured is
   between two independent per-view projections. Two rays from two cameras intersect; the
   ill-conditioning that kills a single grazing ray is exactly what a second ray fixes. This is a
   fusion-stage change, not an estimator, and it is out of this plan's scope.
2. **FootTrackNet** (C0 candidate 3), which predicts foot landmarks *and* an explicit visibility
   flag. Recorded in C0 as the escape hatch, at the price of onnxruntime and its own detector — and
   note that it would face the same 5× multiplier, so it should be expected to help with the
   *fallback rate*, not with the conditioning.

Both are operator scope decisions. `plan-footpoint.md` §6 says so, and the blockers row is filed.

## Honest limits of this measurement

- **40 frames, 7 cameras, one dataset.** WILDTRACK is a stress test, not 102's target regime
  (`context.md` §1: one to a handful of people in a room). The conditioning finding is *worse* here
  than it would be in the target regime, where people are closer to the camera and `depth/height`
  is small. **This result does not say the foot point is fine in a room — it says WILDTRACK cannot
  answer that question.** The room rig is blockers row 1 and has never been calibrated.
- The stature arm's GT-box column is near-tautological (deviation-log row 2's caveat).
- G_FP3's runtime number is from a 20-frame clip with video export off, on a 4060 at 121
  detections per frame. Pose cost scales with crowd size, so the 40 % figure is a crowd figure.
- The pose arm used one confidence floor (0.5) and one model size (`yolo11x-pose`). Neither was
  swept. A sweep would not change the conditioning argument, which is why it was not run.
