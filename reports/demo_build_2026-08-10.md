# The public-data demo — build, measurement and verdict

date: 2026-08-10
plan: `plan-public-demo.md` (C1, approved 2026-08-10)
tier: demo-grade
evidence: `docs/artifacts/public_demo_wildtrack.json`, `docs/artifacts/public_demo_arms.json`
arbiter: `uv run python scripts/check_demo_gates.py`

---

## Verdict

**The demo ships. It is real, it is reproducible, and it shows the wrong regime — and it says so
on the front page.**

| gate | result |
|---|---|
| G_D1 runs end to end, BEV artifact | **PASS** — 8/8, including the licence assertion |
| G_D2 calibrated beats uncalibrated on the shipped segment | **FAIL** — both conditions |
| G_D3 README rewritten, honest, stranger-runnable | **PASS** — 6/6 |

**Arm 1 (WILDTRACK) is REJECTED on G_D2. Kill counter 1 of 2.**
**Arm 2 (EPFL Laboratory) is PARKED on the time rule — not rejected, counter unmoved.**

The kill criterion has **not** fired: it requires *both* arms attempted, and parking is explicitly
not an attempt. `plan-public-demo.md` §7 said so before the clock started.

---

## What the operator's decisions changed, before any of this

Three things closed permanently on 2026-08-10 and they are recorded rather than absorbed:

- **The repo is public and everything is pushed** (`a82667c..c797a87`, 25 commits, and the rest
  since). The visibility check that held the previous session was answered: public is intended.
- **The capture session is cancelled forever.** `calib/` stays empty; no rig of this room was ever
  calibrated; id 73 stays unclassified; D-015's live forecast will never be tested and keeps being
  called a forecast.
- **The floor-calibration subsystem is not dead code.** It is the path by which a *dataset's* own
  calibration becomes a rig, and it is what this demo runs on. Its own gate has still never been
  exercised on a real room, and now never will be.

## The finding that reshaped the plan, measured before the plan was written

The brief asked for "sparse segments matching the hero regime" from WILDTRACK. **They do not
exist.** Over all 400 annotated frames:

| | people/frame |
|---|---|
| minimum, any frame | **13** |
| median | 23 |
| maximum | 40 |
| sparsest 40-frame window (slot 311) | **mean 15.6, min 13, max 19** |

Every one of those people is visible in ≥ 2 cameras. `context.md` §1 scopes 102 to *one to a
handful of people in a room*. This dataset has no such segment, at all, anywhere.

So arm 1 ships an honest crowd demo, and the hero regime moved to arm 2 — which is also why arm 2
was **substituted**: the brief named PhysicalAI-SmartSpaces, which 103 measured at 25–33 identities
per scene. Swapping one crowd for another buys nothing and costs 8 GB. EPFL Laboratory is four to
six people in a room, ~300 MB, with a ground-plane homography per camera.

## Arm 1 — WILDTRACK. G_D1 passes, G_D2 does not.

Pinned segment, frames 1555–1750, 7 cameras, 27 ground-truth identities.

| | calibrated | uncalibrated (appearance only) |
|---|---|---|
| identities shown | 95 | **1** |
| live identities/frame | 46.2 | 0.9 |
| ID switches | 11 | 0 |
| mean position error | **0.141 m** | 0.638 m |
| false-positive tracks | 68 | 0 |

**Both arms fail, in opposite directions.** The calibrated arm over-segments ~3.5× against 27 real
people. The appearance-only arm collapses the entire scene into one identity — which is not a
surprise and not a bug: it is exactly the mechanism D-008 measured, where averaging appearance into
an EMA drags identities toward the population centroid until the different-person mean sits inside
the gate. At fifteen people it goes all the way.

**The switch column is a trap and the gate says so.** An arm holding one identity has zero switches
by construction. `context.md` §4 already records this shape of error — *"a single-agent gate
measures nothing… a stateless 25-line stub passed all five seeds"* — so `check_demo_gates.py`
prints the warning next to the number rather than letting a reader collect it as a win. This is
also why the gate's primary condition, fixed in the plan before any number existed, is
*closeness to the ground-truth identity count* rather than switches. Calibrated is off by 68,
uncalibrated by 26. **Calibrated loses that comparison and the gate fails honestly.**

**What calibration is still worth here** is the one metric that is not degenerate: mean position
error **0.141 m vs 0.638 m**, a 4.5× difference. Geometry is doing real work on *where* people are;
it is the *how many* that the crowd breaks.

**None of this contradicts D-014.** That result — two people one per camera 31.0 % → 100.0 %, two
people both in both views 4.0 % → 100.0 %, one person two cameras 96.0 % → 99.0 % — was measured in
sparse scenes, and the README now prints it directly beneath the crowd table with one sentence
between them: both are true, neither generalises to the other. That is the scope lock working
rather than being quoted.

## Arm 2 — EPFL Laboratory. Parked, with the data on disk.

**A0 verified empirically, not from a page**: `curl` HTTP 200, no registration, no agreement, on
the videos *and* the calibration *and* the ground truth. Downloaded, 311 MB. Videos decode —
360×288, 25 fps, 2955 frames, 4 cameras, 6 people. Calibration parses: ground-plane and head-plane
homography per camera. GT header gives a 56×56 ground grid. The dataset's page documents the head
plane as exactly **1.75 m** above the floor.

**Blocked on one thing: the grid-to-metres scale.** The homographies map a discretised 56×56
top-view grid; the page gives the top view as "358 × 360" in unstated units. Every threshold in
this repo is metric — 0.35 m, 0.75 m, 1.0 m — so a guessed scale silently rescales the whole
fusion stage while every gate stays green.

`plan-public-demo.md` §8 forbade guessing it, in writing, before the download started. So it is
parked. The derivation is available and named in the blockers row: the ground/head homography pair
plus a documented 1.75 m fixes the vertical vanishing point and the metric ratio without needing
intrinsics. Perhaps an hour, and it must be verified against the GT before any number is read.

## The licence guard shaped the deliverable, and was not weakened

`CLAUDE.md`: anything rendered from the dataset **is** the dataset; a `!docs/assets/*.gif`
whitelist once let 5.9 MB of real WILDTRACK frames into history.

- **The hero is the BEV.** `BevRenderer.render()` takes snapshots and a frame number — **no image
  argument exists**, and `_blank()` fills a solid canvas. A dataset pixel cannot reach it. That is
  structural, not a promise, and G_D1 asserts the artifact's own `contains_dataset_pixels: false`.
- **Whitelisted by exact filename**, one deliberate line, with the reasoning written into
  `.gitignore` so the next person does not re-derive it.
- **The .mp4 stays out** — rebuildable in one command, and git is not a video host.
- **Per-camera overlays are not exported at all** by this command. Looking at real frames is right;
  shipping them is not.

The guard cost the demo its most legible view and it was not negotiated with.

## Honest limits

- **40 frames.** 20 seconds at WILDTRACK's 2 fps. Enough to demo, not enough to characterise.
- **The demo shows the regime this project explicitly does not claim**, and the only mitigation
  available was to say so loudly on the front page, which is what the README does.
- **The sparse claim in the README rests entirely on D-014**, measured in scripted scenes with real
  embeddings — not on any end-to-end run, because no end-to-end sparse footage is available now
  that capture is cancelled and arm 2 is parked.
- **G_D2's failure is a real result about the crowd regime, not about calibration.** Reading it as
  "calibration does not work" would be exactly the over-generalisation the scope lock exists to
  prevent.

---

# Addendum — arm 2 attempted on operator order (2026-08-10, later)

**Operator: proceed with arm 2, derive the scale, no guessing.** Done, and the answer is negative
in a way worth having.

## The scale derivation DEAD-ENDS, on evidence

Validation was pre-registered in `plan-public-demo.md` §10 **before the number was computed** — V1
four independent per-camera estimates within 5 %, V3 an independent physical check against walking
speed off the ground truth. Neither could be satisfied, for reasons that are properties of the
data:

| finding | consequence |
|---|---|
| **Only 2 of 4 cameras ship a head-plane homography.** Cameras 2 and 3 write a lone `0`. | The 1.75 m ruler exists for at most two cameras, so V1's "four independent estimates" is unobtainable |
| Under `fx = fy`: the **spare Zhang constraint gives ‖r1‖/‖r2‖ = 0.30** against a required 1.0, and cam0 has no positive solution at all | **V2 fails.** The assumption set is refuted, not merely unconfirmed |
| Under `fx ≠ fy` (exactly determined, no spare check left): **no camera has a positive solution** | The relaxation does not rescue it |

The underlying reason is an identifiability result rather than a missing effort: **the 1.75 m
ruler is vertical and the cell size is horizontal, and transferring one to the other requires the
camera's internals.** EPFL ships no intrinsics, and recovering them from a single plane homography
needs the principal point, which for 2008 DV cameras cropped to 360×288 is not at the image centre.
No amount of further work extracts a metric scale from this data without either intrinsics or a
known ground distance, and the dataset provides neither.

**Fallback taken, as the operator specified:** grid units, radii **re-derived not converted**
(`reports/deviation-log.md` row 3 — a conversion would need the scale we just established does not
exist), and the arm labelled **grid-metric**.

## And then the arm did not work, so nothing from it ships

The fallback is implemented and the run completes. It reports **0 identities in both arms**, so its
artifacts were **deleted rather than committed** and the README is untouched. A BEV of an empty
floor is a demo of nothing, and the sparse hero does not exist.

Three defects were found and fixed on the way; the first is the kind that ships silently:

1. **The calibration parser grouped rows six at a time**, which slid cam3's *ground* matrix into
   cam2's *head* slot, because cameras 2 and 3 write a lone `0` instead of a head matrix. That
   yields a rig that is subtly wrong rather than obviously broken. Now split on the file's own
   `# Camera N` headers — and the corrected parse is what revealed the 2-of-4 fact above.
2. The GT header sits on line 2, behind a lone version line.
3. `n_init = 5` can never confirm on ground truth annotated **once a second**: consecutive boxes do
   not overlap at all. It is 1 here, the same value and reasoning `cli/eval_wildtrack.py` already
   documents, with D-002's consequence stated rather than discovered later.

Detection was never the problem — 0.86–0.92 confidence, one detection per ground-truth person.

**What is still wrong is localised**: the grid ↔ top-view ↔ world convention chain. The
homographies' domain is the **358×360 top-view image**, not the 56×56 grid (verified: the top-view
centre maps inside the camera image; the grid centre does not). A detected foot at pixel (285, 165)
currently lands at world (331.8, 231.3) where the ground truth says cell (37, 17) — **off by a
non-constant factor, so a convention error and not a scale one.**

**And the fix has a gate available that needs no new data**, which is the actual lesson: project
every GT cell through every ground homography and require it to land inside the 360×288 image for
the cameras that should see it, across all 115 annotated frames. Unit-free, unambiguous, and it
fails loudly on exactly this class of error. The scale derivation had a pre-registered check and
was stopped correctly the moment it failed; the *rig construction* had none, and that is why it
produced a plausible-looking rig and an empty demo instead of an error. Blockers row filed.

## Standing after the addendum

- **Arm 1 (WILDTRACK)**: rejected on G_D2, ships anyway as the honest crowd demo. Unchanged.
- **Arm 2 (EPFL)**: attempted, scale derivation dead-ended, fallback taken, **build incomplete —
  0 identities, nothing shipped**. Not a G_D1e/G_D2e failure, because neither gate was ever
  reached: the run does not produce a result to gate.
- **Kill counter: 1 of 2, unchanged.** Arm 2 has still not been *rejected* — it has not produced a
  measurement to reject.
- The demo that ships is still the WILDTRACK one, with the README exactly as it was.

---

# Addendum 2 — arm 2 completed (2026-08-10, operator order: build the instrument check first)

**The sparse demo works and ships.** EPFL Laboratory, 4 cameras, 1–5 people in a room, the
dataset's own calibration, grid-metric. It is the first time this project has demonstrated its
own stated regime on real footage.

| gate | result |
|---|---|
| **G0e** instrument proof (built first, as ordered) | **PASS** — coverage 96.4 %, agreement 2.02 cells against a 3.0 bound |
| **G_D1e** demo runs end to end, BEV artifact | **PASS** — 8/8 |
| **G_D2e** calibrated beats appearance-only | **FAIL** — and one clause of it is unpassable by any working design. Recorded **VOID**, blockers row filed |
| **G_D3** README | **PASS** — 6/6 |

**Kill counter unchanged at 1 of 2.** The criterion does not fire.

## The instrument check found the bug rather than merely guarding against it

Built first, per the order. Two conditions:

- **A. Coverage** — every annotated position must project inside the 360×288 image for ≥ 2 of the
  four cameras. **96.4 % of 476 positions, mean 3.31 cameras.**
- **B. Agreement** — a detected foot point pushed through the chain must land near an annotated
  person, bounded by **3.0 cells, which *is* the birth-clustering radius**. A projection error
  above the radius that decides "same person" makes every downstream identity claim arbitrary, so
  the bound is derived rather than read off the result. **Measured 2.02 cells.**

Scored against B, the defect was unmistakable and, importantly, *not what I had assumed*. Every
candidate row/column ordering and axis flip scored equally badly — 70–105 px in a 360×288 image —
and sweeping px-per-cell from 1 to 20 never got below 88 px. **A flat curve with no minimum is the
signature of "uncorrelated with where people are", not of "mis-scaled".**

So the error was structural: **EPFL's `H_ground` maps the camera image to the top view**, and I had
been inverting it. Applied in the correct direction at the documented 358/56 px per cell, detected
feet land **2.02 cells** from the nearest annotated person, against distinct people sitting ~10.6
cells apart.

## And then the second half, which the first fix exposed

With the geometry correct the calibrated arm *still* reported zero identities while the
appearance-only arm worked. Cause: **I re-derived the two obvious radii and left four other metric
constants at their metre values.** `ground_model_sigma` at 0.15 makes the Mahalanobis gate roughly
13× too tight in cell units and rejects every association — silently, because a gate that rejects
everything looks like a scene with nobody in it.

All four are now re-derived explicitly, and the one that matters most is **measured, not scaled**:
the world-space error floor exists because a box bottom is not a ground-contact point, and G0e
measures precisely that quantity at 2.02 cells.

This is deviation-log row 3's own warning coming true in the concrete — "a wrong scale silently
rescales the entire fusion stage" — and the lesson is narrower than "convert your units": **the
constants that break you are the ones you did not notice were constants.**

## The result

On 5 ground-truth identities, occupancy 1–5 (mean 2.6), 60 frames one second apart:

| | calibrated | appearance-only |
|---|---|---|
| **live identities/frame** (truth ≈ 2.6) | **3.2** | 0.9 |
| distinct identities over the run | 9 | 1 |
| ID switches | **27** | 0 |
| mean position error | 1.72 cells | 1.68 cells |
| false-positive tracks | **0** | 0 |

**With geometry the system holds 3.2 identities against 2.6 real people and invents none. Without
it, the room collapses into a single identity** — D-008's mechanism, unchanged.

**And it churns.** 27 switches over 60 frames for 5 people is a lot, and the reason is structural:
this ground truth is annotated once a second, a person crosses several cells between frames, and
`n_init` had to drop to 1 for anything to confirm at all. This demo shows cross-camera *fusion*
working in the sparse regime; it does not show long-run identity *persistence*, and the README
does not claim it does.

## G_D2e fails, and one clause of it is my mistake rather than the design's

Both clauses fail. Clause 1 (identity count closer to truth) is a **tie** — 9 vs 5 and 1 vs 5, both
off by 4. Clause 2 (no worse on switches) fails 27 to 0.

**Clause 2 cannot be passed by any working design.** D-008 guarantees appearance-only fusion
collapses a multi-person scene into one identity, and one identity has zero switches *by
construction*. Any arm that actually tracks people loses automatically.

That is a defect in the gate I wrote, not a result about the arm — and the reason it can be called
that honestly is that **the degeneracy was written into the checker's own warning text in commit
`2cee680`, before this run produced any number.** It was foreseen and then not acted on, which is
the mistake; discovering it now and calling it a design failure would be the worse one.

Recorded **VOID**, counter unchanged at **1 of 2**, with a blockers row asking the operator to rule
on the rejection and to replace the clause. The recommended replacement is already measured and
printed beside it: **live identities per frame against mean occupancy** — 3.2 vs 2.6 versus 0.9 vs
2.6 — which measures what switches were meant to and cannot be won by collapsing.

## What ships

- **The sparse hero is the EPFL BEV**, first thing on the README, labelled grid-metric and
  captioned with the failing switch count.
- **WILDTRACK stays exactly where it was**: the crowd demo, framed as measured failure analysis.
- **Licence guard unchanged**: BEV canvas only, procedural, whitelisted by exact filename, `.mp4`
  and every frame of either dataset stay out.
- **The quickstart is three commands** and the middle one is the instrument proof, with a sentence
  telling a stranger not to believe the third without it.

---

# Addendum 3 — G_D2e re-evaluated under the operator's amendment (2026-08-10, session close)

**Operator ruling: clause 2 is defective as foreseen; replace it with live identities per frame
against mean occupancy.** Done, `reports/deviation-log.md` row 4, authority **operator**.

## The verdict did not change

| clause | | result |
|---|---|---|
| 1 — distinct identities over the run, closer to truth | *unchanged, not part of the ruling* | **FAIL — a tie.** calibrated 9 vs truth 5, uncalibrated 1 vs truth 5, both off by 4 |
| 2 — live identities per frame, closer to mean occupancy | **AMENDED** | **PASS.** calibrated 3.2 (off by 0.6) vs uncalibrated 0.9 (off by 1.7) |

**G_D2e still FAILS.** No README table changed, because no number changed; the caption was
corrected, because it named a clause that no longer exists.

**That the amendment did not rescue the gate is the point.** A clause rewritten by the party whose
result it judges, which then produces a pass, should be read as a gate fitted to its answer. This
one was proposed with its defect named, signed by the operator, and still fails — which is what
makes it a correction.

**Clause 1 was left alone deliberately.** The ruling covered clause 2. Widening an operator's
amendment unasked, after seeing which clause blocks the pass, is precisely how a threshold gets
moved to fit a result.

## The kill counter is an open operator question, not mine to take

Strictly, `plan-public-demo.md` §7 rejects an arm that fails G_D2e — which would put the counter at
**2 of 2 and fire the criterion**, killing a task whose demo works, ships, and is on the front page
of the README.

Against that: the failure is a **tie, not a loss**, and `ids_shown` counts distinct identities
across 60 seconds, so it folds in every person who entered or left. A collapsing arm scores 1 and is
"off by 4" for reasons that have nothing to do with tracking well.

I have **not** fired it and I have **not** voided it. It is a `blockers.md` row with three options
and a recommendation. Deciding it myself in either direction would be the wrong move: firing a
criterion on a tie is as much a distortion as softening one on a result I like.

**Counter stands at 1 of 2, pending that ruling.**

## Where 102 exits this session

| | |
|---|---|
| **Sparse demo** — EPFL Laboratory, 4 cameras, 1–5 people in a room | **ships.** G0e PASS, G_D1e PASS 8/8, hero on the README, grid-metric and labelled |
| **Crowd demo** — WILDTRACK, sparsest segment | **ships**, framed as measured failure analysis, unchanged |
| **Foot-point estimator** | both arms rejected, both VOID, counter 0 of 2 — the box bottom is at the noise floor its own optics allow, and a per-view rule cannot beat it |
| **EPFL metric scale** | **unidentifiable** from the data; grid-metric everywhere and said so |
| **Live capture** | cancelled permanently; `calib/` empty and will stay so |
| **Repo** | public by operator decision, everything pushed |
| **Gates open** | G_D2 (crowd, honest failure), G_D2e (sparse, one tied clause) |
| **Tests** | 560 green, ruff + mypy clean |

The honest one-line summary of the project's state: **the geometry works in the regime the project
claims, on real public footage, and it is the identity bookkeeping over long runs — not the
geometry — that is still weak.** 27 switches over 60 frames says so, and the README says it too.

---

# Addendum 4 — the composite demo, and the licence answer (2026-08-10, session close)

**Operator: "the demo is not legible without its inputs — every demo must show the RGB camera
views that produced it, not the BEV alone."** Built, run on both arms, gated. **G_D4 PASSES.**

## Why a BEV alone was never evidence

A floor plan shows dots and asks the reader to accept three things on faith: that the dots are
people, that they are the people in the footage, and that the number over a dot is the number that
was over that person in each camera. **Only the third is the project's actual claim, and the BEV is
the one view that cannot show it.** The composite can: N camera panels with boxes and global-ID
labels, the same integer in the same colour over the same human in every panel that sees them, and
the floor plan as the final panel.

## One renderer, four stages — not four pipelines

`src/mcreid/viz/composite.py`. The stages are cumulative and come out of one code path, so a
single-stage video is an *excerpt* of the composite rather than a second pipeline that can drift
from it:

| stage | adds | why it is its own stage |
|---|---|---|
| `raw` | nothing | what the system is given |
| `boxes` | boxes in **each camera's** colour | what one camera alone buys — and colouring by identity here would show the answer one stage early |
| `ids` | the global ID in that identity's colour | the only place a cross-camera claim is made |
| `composite` | + the BEV as the final panel | the claim and the map in one frame |

Layout is an aspect-scored grid packing, measured as a **log ratio** so twice-too-wide and
twice-too-tall cost the same: 4 cameras of 5:4 frames give 2x2, and 7 cameras of 16:9 frames give
3x3 with two blanks rather than the 4x2 letterbox a linear score would have chosen.

## The result, both arms, on the 4060

| | EPFL (sparse hero) | WILDTRACK (crowd) |
|---|---|---|
| panels | 4 cameras + BEV | 7 cameras + BEV |
| frames rendered | 60/60 | 40/40 |
| an ID drawn in >= 2 camera panels | **39 of 60 frames** | **36 of 40 frames** |
| ID colour conflicts | **0** | **0** |
| render cost | 12.4 ms/frame | 37.9 ms/frame |

**NO IDENTITY METRIC MOVED.** Both arms reproduce their previous numbers exactly — EPFL
9 / 3.2 / 27 / 1.72 vs 1 / 0.9 / 0 / 1.68, WILDTRACK 95 / 46.2 / 11 / 0.141 vs 1 / 0.9 / 0 / 0.638.
That is the point: the composite is rendered beside the pipeline and timed separately from it, so
it cannot inflate the FPS claim this project makes elsewhere.

## Two defects the composite found, which is the argument for building it

Neither was visible in the BEV alone, and both were on the **shipped front page**.

1. **The hero artifact printed the wrong unit.** `BevRenderer` hard-coded `BEV (metres)` and
   `2 m grid`. The EPFL arm is grid-metric — its scale is *unidentifiable* from the dataset
   (deviation-log row 3) — so the README's first image carried a unit the prose beside it
   correctly denied. `units` is now a parameter; the arm passes `"grid cells"`.
2. **The panels labelled identities the map did not carry.** The fusion stage assigns a global ID
   to a per-view observation before it commits that track to the floor plan, so a person could
   wear ID 16 in three camera panels with no dot for 16 on the map beside them — the composite
   contradicting the one thing it exists to demonstrate. Fixed by passing the frame's live IDs:
   an assigned-but-uncommitted box is drawn grey, unlabelled, **and counted**.

   **The count is itself a finding: 115 of 340 assigned boxes on the EPFL run belong to tracks
   that never reached the map.** It costs the demo something visible — cross-panel frames fell
   47 -> 39 — and that is the honest number rather than the flattering one.

## The gate, and why its licence clause runs backwards from G_D1's

G_D4 checks the two things a pretty video cannot fake — every camera that fed the run has a panel,
and an identity in two panels is in **one** colour — plus existence, hash, and a runtime bound
labelled a sanity check rather than an FPS claim.

Its licence clause asserts in the **opposite direction to G_D1's**, and the pair is the whole
position in machine-readable form:

- the BEV must declare `contains_dataset_pixels: false`, **because it ships**;
- the composite must declare `contains_dataset_pixels: true` **and** `committed: false`, because
  that is precisely why it does not.

Both aggregate numbers are non-vacuous, and `tests/test_viz_composite.py` carries a **control for
each** — one record that colours by camera (caught), one where no ID spans two panels (reported as
zero). Without them, "colour-consistent" could be reported by a checker incapable of returning
anything else, which is the degenerate-gate failure this project already has a paper trail of.

## The guard was not weakened, and it moved into code

Composites are dataset pixels by construction. `assert_generated_only` **raises** on any
destination under `docs/`, resolving the path first so a `..` cannot walk past it, and the check
runs in `StageWriter.__init__` — before a frame exists — so a bad path fails in the first second of
a run rather than after the detector has been over every frame. This is the same structural move as
`BevRenderer.render` taking no image argument, and there is now a test asserting that signature too.

One thing caught late and worth recording: the writer originally stored the **resolved** path,
which put an absolute `C:\Users\<name>\...` into a **committed** JSON in a **public** repo — exactly
what session 3K's audit checks for by name. It now keeps the path as given and checks the resolved
one.

## The licence answer: no hosting route exists

Both dataset pages were read for this, today, rather than assumed:

| dataset | what the page actually says | derived clips redistributable? |
|---|---|---|
| EPFL CVLab (Laboratory) | *"All videos, calibration and ground truth files available on this page are copyrighted by CVLab - EPFL. You can use them for research purposes."* Citation required; derivatives not addressed. | **No.** A grant to *use* is not a grant to *redistribute*, and silence on derivatives is not permission. |
| WILDTRACK | **No licence text at all** — no named licence, no copyright statement, no redistribution terms. Citation requested; the download path carries a manual request/consent step. | **No.** Absent terms are not permissive terms. |

So the conditional in the brief resolves to its second branch and **no blockers row was filed**:
there is nothing for the operator to approve, because there is no defensible hosting route to
propose — not a release asset, not a README-linked mirror. What ships is the procedural BEV, a
description of what the composite shows, and the one command that renders it. Findings are in the
README's Licence section with the wording quoted, so a reader can check the reasoning rather than
take the conclusion.

## What the crowd composite is for, stated plainly

It is worth watching *because it is ugly*. The fusion works — an ID spans panels on 36 of 40
frames — and the labels still congest into an unreadable mat, which is what 15.6 people per frame
looks like and is the honest picture of the over-segmentation the WILDTRACK table reports. **No
crowd-thinning rule was added to the panels**, deliberately: the BEV has one, and adding it here
would hide the exact congestion that is the finding.

## Standing at session close

| | |
|---|---|
| **G_D4** composite, both arms | **PASS** — 11 checks per arm |
| G_D1 / G_D1e / G_D3 | PASS, unchanged |
| G_D2 (crowd) | FAIL, unchanged — the honest crowd result |
| G_D2e (sparse) | FAIL on clause 1's tie, unchanged; counter **1 of 2** per the operator's ruling |
| Tests | **585 green** (was 560), ruff + mypy clean |
| Composite videos | 8 files, gitignored `reports/`, hashes in the committed JSON, **never committed** |

Two stale files were left in place rather than deleted: `reports/epfl_demo/epfl_6p_composite_*.mp4`
from the first run, before the filename prefix was shortened. They are gitignored and harmless, and
deleting files is an operator-approval item.
