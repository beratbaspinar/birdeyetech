# Plan — the shippable demo, on public data (102_multicam_reid)

date: 2026-08-10
status: approved (operator order 2026-08-10; gates and kill criterion as written below)
owner: matus
tier: **demo-grade**

> **No autonomous loop launches without this artifact approved.**

`tier: demo-grade` is binding: instrument-proved gate + honest README + one number. This task
ships a *demo*, not a finding. The one measured claim it makes (G_D2) is a re-run of an existing,
already-approved comparison on the exact segment that ships — not a new result.

## 0. Prior art (C0)

`reports/c0_findings.md` §§ "C0 #1" (backfilled) and "C0 #2" (foot-point). A third sub-question was
answered inline this session and is recorded in §1 below: **which public dataset actually contains
102's target regime.** A0 resolved by `curl` on every candidate.

The adapter question answers itself: **`mcreid.eval.wildtrack.load_rig` already converts
WILDTRACK's own calibration into `RigCalib`**, has done since session 3-something, and every
WILDTRACK number this repo has published went through it. No adapter is needed for arm 1. Nothing
is built that already exists (doctrine 14).

## 1. The finding that reshapes this plan, measured before it was written

The brief asks for "sparse segments (fewest concurrent people) matching the hero regime" from
WILDTRACK. **WILDTRACK does not contain that regime.** Measured over all 400 annotated frames:

| | people/frame |
|---|---|
| minimum, any single frame | **13** |
| 10th percentile | 16 |
| median | 23 |
| maximum | 40 |
| **sparsest 40-frame window** (slot 311, frames 1555–1750) | **mean 15.6, min 13, max 19** |

Every one of those people is visible in ≥ 2 cameras, so this is not an artefact of counting
off-screen annotations. `context.md` §1 scopes 102 to *"one to a handful of people in a room"*.
Thirteen people in an outdoor plaza is not a sparse hero regime and calling it one would be the
kind of framing this repo's own docs go out of their way to avoid.

**Consequence, and it is the whole shape of the plan:** arm 1 ships a real, honest,
end-to-end demo — but it is a *crowd* demo, presented as the scope lock requires, with the
calibrated-vs-uncalibrated comparison as its measured backbone. **The hero regime moves to arm 2,
which is the only arm that can carry it.**

### Arm 2 substitution, and why I am not building the one the brief named

The brief names PhysicalAI-SmartSpaces `MTMC_Tracking_2024`. **Recommending against it, and
substituting EPFL CVLab's Laboratory sequence.** The reason is the same finding as above: 103
measured MTMC_Tracking_2024 at **25–33 identities per scene** with each identity fragmenting into
15.6–19.1 tracks per camera. It is a synthetic warehouse crowd. Swapping one crowd dataset for
another does not buy the hero regime, and it costs an ~8 GB download and a parser port.

| | PhysicalAI MTMC_2024 (named) | **EPFL Laboratory (substituted)** |
|---|---|---|
| people | 25–33 per scene | **4 and 6** — two sequences, `4p-*` and `6p-*` |
| setting | synthetic warehouse | **indoors, a room** |
| cameras | 30 | 4 |
| size | ~8 GB | ~100 MB |
| calibration | homographies shipped | **ground-plane homography per camera, plain text** |
| A0 | passed (103, 2026-08-07a) | **passed — `curl -I` HTTP 200, no registration, no agreement, verified this session on the video, the calibration and the ground truth** |
| parser | port from 103 `src/mcgp` | ~40 lines; the calibration file is a header and 3×3 matrices |

EPFL Laboratory is *literally* 102's stated target regime: four to six people, walking around a
room, four cameras, 25 fps. If the demo is meant to show what this project claims, that is the
footage it should show.

**Cost caveat, stated up front:** the EPFL homographies map a discretised top-view grid, not
metres. Recovering the grid scale is the integration risk, and it is what the park rule below
exists for.

## 2. Non-goals

- **The crowd regime as the headline.** Scope lock, unchanged: WILDTRACK crowds are measured
  *failure analysis*. Arm 1 may show a crowd; it may not claim one.
- **Any training or fine-tuning.** Pretrained weights only.
- **Live capture, a room rig, marker sessions.** Cancelled permanently by the operator 2026-08-10.
  Nothing in this plan touches `mcreid-calibrate mark`, `capture.py`, or any webcam path.
- **Resurrecting 103.** Arm 2's parser is written here; 103 stays archived either way, and the
  substitution means nothing is ported from it at all.
- **Re-opening the foot-point question.** It has its own plan, its own verdict, and its own
  blockers row.
- **Committing one pixel of either dataset.** See §6.
- **New fusion mechanisms, new thresholds, new tunables.** This task assembles what exists.

## 3. Acceptance gates — machine-checkable

`scripts/check_demo_gates.py` is the single arbiter; each gate exits 0/1 off committed JSON.

| # | Gate | Command | Passing |
|---|---|---|---|
| G_D1 | The demo runs end to end on the shipped WILDTRACK segment, using the dataset's own calibration, and produces a BEV video artifact | `uv run mcreid-public-demo wildtrack --segment ship` | exit 0; video written; `docs/artifacts/public_demo_wildtrack.json` records frames, cameras, identities and the artifact's own sha256 |
| G_D2 | Calibrated vs uncalibrated identity persistence, **on the exact segment that ships** | `uv run python scripts/check_demo_gates.py --gate g_d2` | both arms present in `docs/artifacts/public_demo_arms.json`; calibrated arm strictly better on identity count *and* switches; ratio to GT identity count reported |
| G_D3 | README rewritten: real hero artifact from arm 1, honest framing, stranger-runnable quickstart | `uv run python scripts/check_demo_gates.py --gate g_d3` | README references the shipped artifact by name; contains the quickstart command verbatim; contains the crowd-scope sentence; the synthetic hero is either removed or explicitly labelled synthetic |

**Demo-first**: G_D1 *is* the end-to-end run. G_D2 re-uses its output rather than a separate run,
so the number describes the thing that ships.

### On G_D2's shape

This is the D-014 pattern re-run, not a new experiment. D-014's measured backbone — two people one
per camera **31.0 % → 100.0 %**, two people both in both views **4.0 % → 100.0 %**, one person two
cameras **96.0 % → 99.0 %** — was measured on real OSNet over real WILDTRACK crops with exact rig
geometry, and it is what the README table surfaces. G_D2 does not restate it: it shows the same
*direction* holds on the segment that actually ships, on whole-pipeline identity metrics rather
than scripted scenes. **If the two disagree, the README says so** — that is a finding about the
crowd regime, which is exactly what the scope lock already predicts.

## 4. Not machine-testable

| Item | Why | Who | When |
|---|---|---|---|
| Whether the BEV video is *watchable* — legible, right pace | no fixed target | matus | before ship |
| Whether the README reads honestly to a stranger | judgement | matus | before ship |
| Whether 15.6 people/frame reads as "not the hero regime" to a reader | judgement | matus | before ship |

## 5. Task graph

```
T1 segment selection + calibrated run (WILDTRACK)  ── T2 BEV artifact ── T3 G_D2 arms ── T4 README
T5 EPFL arm (parked if > ~2 h)  ────────────────────────────────────────────────────────┘
```

| id | task | depends on | tier |
|---|---|---|---|
| T1 | `mcreid-public-demo wildtrack`, segment pinned in code | — | opus |
| T2 | BEV video + license-safe artifact | T1 | sonnet |
| T3 | calibrated vs uncalibrated arms → JSON | T1 | sonnet |
| T4 | README rewrite + gate checker | T3 | opus |
| T5 | EPFL Laboratory arm | — | sonnet |

GPU: one worker, serialised (§6).

## 6. License guard — the constraint that shapes the deliverable

`CLAUDE.md`: *"Never commit datasets, weights, footage… This extends to ANYTHING RENDERED FROM
them — a GIF/MP4/PNG of dataset or room frames **is** the data. A `!docs/assets/*.gif` whitelist
once let 5.9 MB of real WILDTRACK frames into history."*

**So the hero artifact cannot contain WILDTRACK pixels, and the guard is not being weakened.**
The route the rules already provide:

- **The hero is the BEV.** `mcreid.viz.bev` draws a procedural canvas — floor grid, per-identity
  dots, ID labels, camera frusta. **No dataset pixels enter it.** It is the same class as
  `docs/assets/hpc_demo.gif`, which is whitelisted precisely because it is "procedurally generated
  … no third-party or room pixels anywhere in it".
- **Per-camera overlays still get rendered** — they are the honest way to look at the run — and
  they stay in `reports/`, which is gitignored, and are never whitelisted.
- **Whitelisting is by exact filename**, one deliberate line, never a glob. The `docs/artifacts/`
  JSON convention (metrics, no pixels) carries every number.

If the BEV render turns out to be unwatchable on its own, that is a **blockers row**, not a licence
to ship frames.

## 7. Kill criterion — event-based, written before the first number exists

> **The public-data demo is DEAD if, after both arms have been attempted, no arm produces a
> demo that passes G_D1 and G_D2.**
>
> **Counter: 0 of 2.** (arm 1 = WILDTRACK, arm 2 = EPFL Laboratory.)

An arm is *rejected* when it has been built and then fails G_D1 or G_D2. **Parking arm 2 on the
time rule is NOT a rejection** — it is a scheduling decision and it does not touch the counter;
that distinction is written here, before the clock starts, precisely so a tired hour cannot be
laundered into a design verdict later.

Instrument findings do not increment the counter (§3). A dataset that will not download does not
increment it either — that is an access finding and a blockers row.

**If it fires** it is honored, never re-read or re-scoped, and 102 ships with the synthetic demo it
already has plus an honest README saying the public-data demo was attempted and failed.

Copy this into `status.txt`. The counter lives there, never here.

## 8. Time rule for arm 2

Arm 2 is parked with **one line in `status.txt` and a blockers row** if its integration passes
**~2 hours**, per the brief. Arm 1 alone ships. The stated integration risk is the grid-to-metres
scale in the EPFL homographies; if that is not resolved from the dataset's own documentation
inside the budget, park it rather than guess a scale — a guessed scale silently rescales every
distance in the fusion stage, and every threshold in this repo is in metres.

## 9. Open decisions

| # | question | options | decided |
|---|---|---|---|
| 1 | Which WILDTRACK segment ships | sparsest / densest / middle | **Sparsest 40-frame window, slot 311 (frames 1555–1750), mean 15.6 people.** Pinned in code, not a flag default, so the shipped artifact and the gate always describe the same frames |
| 2 | Hero artifact format | GIF / MP4 / both | **BEV MP4 for watching + BEV GIF for the README.** Both procedural, both license-clean |
| 3 | Uncalibrated arm's definition | no geometry at all / geometry radii opened globally | **Appearance-only fusion (geometry disabled)** — it is what D-007/D-008 measured and what `--geometry-only`'s mirror image already supports; no new mechanism |
| 4 | Arm 2 dataset | PhysicalAI MTMC_2024 / EPFL Laboratory | **EPFL Laboratory**, §1 |
