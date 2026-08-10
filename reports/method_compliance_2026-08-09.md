# Method compliance audit + retrofit — 102_multicam_reid

date: 2026-08-09
against: `000_infra/refactored_method.md` v1
scope: 102-local only. No method-doc edits, no new strategy artifacts, no rewriting of
recorded history (`decisions.md` and every past report are immutable).
reason: 102 predates the method. The next build task — the foot-point estimator — must run
under it, so the structure has to be there first.

---

## GO / NO-GO

**GO — 102 is ready to run the foot-point task under the method, with one physical
dependency named and one operator ratification outstanding, neither of which blocks the
task's C0 or C1 phases.**

The foot-point task starts at C0 (nothing has ever searched for a ground-contact estimator)
and stops at a C1 plan artifact carrying its own kill criterion. Both can proceed today.
Only *live* validation of the result waits on floor markers — blockers.md row 1 — and that
dependency was always there; the retrofit just made it a row instead of a paragraph in a
2000-line log.

---

## Per-section verdict

| § | subject | verdict |
|---|---|---|
| §2 | C0 findings file | **retrofitted** — created, labelled BACKFILLED, honest that no C0 pass was ever run |
| §2 | plan-artifact convention | **gap — closed by convention, not by artifact.** No plan exists and none is being written retroactively |
| §2 | past C1-equivalent | **none.** `context.md` is a partial stand-in; the gap is named below |
| §3 | gates machine-checkable | **gap-needs-operator** — blockers.md row 4, with a recommendation to leave §7 historical |
| §3 | instrument proof (G0 pattern) | **compliant in practice, uncodified** — the pattern is there and was used correctly, four times |
| §3 | pre-registered kill criterion | **ABSENT, correctly, and deliberately left absent** |
| §3 | deviation log | **retrofitted** — created, one backfilled row |
| §4 | agents, `disallowed-tools` | **compliant** — verified, not redone |
| §4 | blockers.md | **retrofitted** — did not exist; the pre-HPC filing left no 102 rows anywhere |
| §5 | environment probe | **retrofitted in code** — four commands now probe before compute |
| §5 | >5x rule / time tripwire | **compliant in spirit**, no logging surface existed; the deviation log is now it |
| §6 | venv / uv / no system Python | **compliant** |
| §6 | no sandbox-era shims | **compliant** — nothing of the `env.sh` class exists here |
| §6 | deny-list assumptions | **compliant** — 102 carries no local settings and inherits the doubled list |
| §7 | status.txt current | **retrofitted** — the header was three sessions behind its own body |
| §7 | nav entry | **retrofitted** — 102 had no entry at all |

---

## §2 — the build loop

### C0 — `reports/c0_findings.md`: created, and it says so

Created from the template. **It is a record, not a licence**, and it opens by saying that no
C0 pass was ever run for 102 and that nothing in it was researched on 2026-08-09.

Every row is reconstructed from evidence already committed: the module docstring at
`src/mcreid/track/gpu_view.py:1-18` that rejects Ultralytics' bundled trackers and says why;
the vendor header at `src/mcreid/track/vendor/osnet.py:1-16` that explains adapting OSNet as
one file rather than depending on a training library; the SHA-256 pins in
`scripts/download_wildtrack.py`; the version pins in `pyproject.toml`; and the measured
rejections in `decisions.md` D-006..D-009. Eight candidates, each ending in adopt / adapt /
reject — the three verdicts the template allows.

**What the backfill deliberately does not do** is manufacture a search. The "Null results"
section is explicitly weaker than a real C0 null: it records "nothing was found in the course
of building", not "the surfaces were swept and returned nothing", and labels itself as such
so a later plan cannot over-read it. The single largest build-vs-borrow decision in the whole
project — writing `PerViewTracker` instead of using BoT-SORT — is argued in a source header
and nowhere else, and the finding is that this is a property of the 2026-07 Ultralytics API
rather than a permanent fact about the world.

**The row that matters for what comes next:** *no foot-point / ground-contact estimator was
ever searched for.* The measurement exists and the defect is quantified to 0.62–2.17 m; the
fix has never had a prior-art pass of any kind. The next task's C0 is not a formality.

### C1 — no plan artifact has ever existed, and none is being written now

There is no `plan*.md` in 102, and there never was. The nearest thing is `context.md`, which
carries a goal, a locked architecture, design decisions and a gate table — but has **no tier
declaration, no non-goals section, no task graph, no not-machine-testable list and no kill
criterion**. It is a good architecture document and a partial C1 artifact.

Writing one retroactively would be worse than the gap. A plan artifact's whole function is to
fix the target *before* the loop runs; 102's loop ran for a month. **The convention is
therefore established forward, not backward:** the foot-point task's first deliverable is
`plan-footpoint.md` from `000_infra/templates/plan.md`, and no autonomous loop launches
before it is approved.

### Past C1-equivalent

The closest thing 102 has to a plan gate is the pattern in `scripts/README_v2_synthetic_engine.md`
— a design note headed "Status: NOT BUILT. v2 only. Do not implement in v1." That is a real
scope fence and the 103 postmortem credits it as the evidence that 102 stopped by design
rather than stalling. It is one note, not a convention.

---

## §3 — measurement discipline

### Instrument proof: the pattern is there, and it was used correctly

This is 102's strongest section, and it is entirely uncodified — there is no gate named G0,
just the habit applied four times:

- **D-015** — the foot-point noise harness is validated by its own 0 % row landing at 0.08 m
  against WILDTRACK GT's 0.12 m *before* the noise sweep is read. This is the G0 pattern
  exactly: prove the instrument against a known answer, then trust its unknowns.
- **D-009** — the synthetic harness is *fenced* rather than trusted: it may test ORDERING,
  never appearance thresholds, because isotropic random vectors have no population centroid,
  so the EMA statistic the merge actually tests does not move. "The generator and the gate
  shared the assumption" is the §3 failure mode caught in the act.
- **The foot-point GT arm** reproduces exactly (0.123 m mean, 0.207 m p90, 0 % beyond radius)
  and is threshold-independent, which is what licenses reading the detector arm at all.
- **D-013** — a gate was rejected for being an algebraic identity before it was ever run.
  Four correspondences fit a homography to ~1e-15, so a four-point residual could not fail.

**Verdict: compliant in practice.** No retrofit is owed. The foot-point plan should name its
instrument proof as a numbered gate rather than leave it as a habit, because a habit does not
survive a handoff.

### Kill criterion: ABSENT, and staying absent

102 has no pre-registered kill criterion and never had one. **This is confirmed and it is not
being fixed.**

Writing one now for the whole project would be fake pre-registration in the most literal
sense: the numbers already exist, so any criterion chosen today is chosen by the results it
is about — which is the precise failure §3 exists to prevent, and the one the nav records as
102's own cause of death ("revising a criterion by its own result"). The absence is recorded
here as a fact about 102's history.

**The foot-point task carries its own**, written before its first number exists, per the
template's §6. That is the correct place for it and the only honest one.

### Deviation log: created, one row, backfilled

`reports/deviation-log.md`, from the template. Empty apart from one backfilled row, and it
says plainly that the emptiness for sessions 3a–3X is a fact about 102 predating the method,
not a claim that no threshold ever moved.

Row 1 is the floor gate's **4 → 6 marker minimum** (D-013, session 3X). It is entered with
"gate semantics changed? **yes**" and authority `lead`, which the template calls a defect —
and it is entered anyway, with that flagged, because the change made the gate *stricter* and
was stated in the session it was taken. Suppressing it to keep the log clean would be the
worse failure. The operator has a blockers row to ratify or reject it retroactively, so the
precedent is explicit rather than silent.

### A threshold-semantics finding, raised and NOT fixed

Auditing the numbers turned up a naming collision that is worth writing down before the
foot-point task inherits it. **Three different radii are all called some variant of "the
merge radius" or "the clustering radius":**

| name | value | where |
|---|---|---|
| `MERGE_RADIUS_M` | 0.35 m | `src/mcreid/eval/footpoint.py:40` — a **reporting** radius |
| `CLUSTER_RADIUS_M` | 1.00 m | `src/mcreid/eval/footpoint.py:41` — a **reporting** radius |
| `merge_unconditional_radius_m` | **0.0 — disabled** | `src/mcreid/fusion/global_id.py:153`; 0.35 m is the value it was measured at and rejected |
| `merge_radius_m` | 0.75 m | `src/mcreid/fusion/global_id.py:145` — what merging actually uses |
| `birth_cluster_radius_m` | 1.0 m | `src/mcreid/fusion/global_id.py:124` — what birth clustering actually uses |

`decisions.md` D-013 says the calibration bounds are "0.25 m held-out (below the 0.35 m
birth-clustering radius) … 0.35 m cross-camera, which IS the clustering radius". The
**numeric bounds are fine** — 0.25 m and 0.35 m are the gate, and the gate works. The
*justification label* is crossed: 0.35 m is the disabled unconditional-merge radius, and the
birth-clustering radius is 1.0 m.

**Not fixed, on purpose.** `decisions.md` is the immutable record and D-013's reasoning is
what makes the entry useful. The finding is recorded here and cross-referenced from
`c0_findings.md`, which now lists both sets of radii side by side so a plan citing "the
0.35 m radius" has to say which one it means. Nothing in the shipped behaviour changes.

---

## §4 — unbreakable constraints

**Agents: compliant, verified rather than redone.** All three definitions in
`000_infra/agents/` carry `disallowed-tools: AskUserQuestion` in frontmatter, with §6 model
tiering (`lead`/opus/high, `impl`/sonnet/medium, `mech`/haiku/low). 102 has no `.claude/`
directory, so it inherits them unmodified — there is no project-local override to drift.

**blockers.md: did not exist.** The pre-HPC audit filing left no 102 rows here or in
`000_infra/reports/blockers.md` (checked — none). Created from the template with five rows,
each naming its provenance:

1. **Floor markers not laid out** — blocks the calibrated live run and any live validation of
   the D-015 forecast. The path is built and verified on simulated loupe clicks; what is
   missing is physical. Recommendation: 12 markers, not the 6-marker refusal threshold —
   D-016 measures 6 markers passing 0 % of the time at 1.5 px click error against 12 at 92 %.
2. **id 73 unclassified** — `reports/console_dumps.txt` is 0 bytes, re-checked twice.
   Recommendation: re-run with a redirect, since the console buffer has now failed twice.
3. **3-cam device/backend mapping unconfirmed** — same 0-byte dump; fold into the same re-run.
4. **`context.md` §7 gates are prose, not commands** — gap-needs-operator (below).
5. **Ratification of deviation row 1.**

Rows 1–3 are backfilled from items 102 already recorded as owed in sessions 3W/3X. Rows 4–5
were raised by this retrofit. Nothing is invented.

**Git as the only rollback layer: compliant.** Three checkpoint commits this session.

---

## §5 — efficiency: the one place the retrofit touched code

**The gap.** `mcreid.utils.resolve_device` already fails fast on an explicit `cuda` request
and logs the resolved device — genuinely good, and better than most of the workspace. But
`"auto"` degrades to CPU silently, which is *correct* for a library whose core is
deliberately torch-free (`pyproject.toml` says so, and the CI suite depends on it) and
*wrong* for a command that runs a 1280 px detector over seven cameras. None of the
GPU-touching CLIs exposed a device option at all, so every one of them ran at
`GpuViewConfig.device = "auto"` and would have ground on CPU without saying so. That is the
canonical §5 violation, and it was one driver update away.

**The fix, and it was a small one.** `probe_compute_device(requested, task, allow_cpu=False)`
in `src/mcreid/utils/device.py`: resolve, log, and refuse to start on CPU unless told
otherwise. Wired into the four unattended GPU commands, each of which also gained `--device`
and `--allow-cpu` and now passes the device *down to `GpuViewConfig`* — previously they left
it at the default, so even an explicit choice would not have reached the detector.

| command | probes | why |
|---|---|---|
| `mcreid-wildtrack footpoint` | yes | the detector arm *is* the measurement |
| `mcreid-wildtrack run` | yes | reports timings, which are meaningless off the intended device |
| `mcreid-wildtrack-demo` | yes | same cost class, plus a video and GIF encode |
| `mcreid-eval` | n/a | no detector — GT boxes through fusion, CPU by design |
| `mcreid-live`, `mcreid-live-multi` | **deliberately not** | interactive; they print FPS every second to someone watching, so a CPU fallback announces itself in seconds. A hard refusal would also break the torch-free path the CI suite runs on. |

The probe runs **before** the dataset existence check, so a missing GPU is reported in the
first second rather than after the frames are found. Verified end to end: `--device cpu` on
this box raises with the task named and `--allow-cpu` offered.

Five tests in `tests/test_utils_device.py`, none of which touch a GPU — they patch
`resolve_device` and assert the policy. A check that catches a missing GPU must not require
one.

**The >5x rule and the time tripwire** have no violations on record — 102's runs are minutes,
not hours. There was no surface to log a >5x decision on; `deviation-log.md` is now it.

---

## §6 — environment

Compliant throughout, with nothing to fix.

- `.venv` in the repo root, `uv.lock` committed, every dependency pinned exactly in
  `pyproject.toml` including the `typer==0.27.0` / `click==8.4.2` pair whose comment records
  why they move together. No system Python anywhere.
- **No sandbox-era shims.** Nothing of the `env.sh` class exists — there is not a single
  `.sh`, `.bat` or `.ps1` file in the repo. 102 never acquired the containment layer that
  103 is still shedding, which is one benefit of predating it.
- **Deny-list assumptions consistent.** 102 carries no `.claude/settings.json` and no local
  permissions, so it inherits the canonical doubled `Bash(...)`/`PowerShell(...)` list
  without a second source to drift from (doctrine 12).
- Python 3.11 pinned `>=3.11,<3.12`, ruff + mypy configured and clean, and `pyproject.toml`
  already declares `gpu` and `footage` pytest markers — the "no GPU or real footage in
  CI-able tests" rule is enforced by the test runner, not by convention.

---

## §7 — session rituals

**`status.txt`: drift found and fixed.** The header read
`Last updated: 2026-07-31 (session 3U — M1(a)+(b) CLOSED; M1(c) capture is next, Matus's hands)`
while the body ran through session 3X — capture had shipped, calibration had been built, its
acceptance had passed, and the marking helper had shipped too. The header had been wrong
since 3V.

Replaced with a dated 3Y header plus a **one-paragraph current-state summary at the top**,
because a 2174-line append-only log whose header is the only summary will drift again. The
body is untouched — it is the record.

**Nav: 102 had no entry at all.** `000_infra/strejc-nav.md` mentions 102 only in the past
tense, as the source of a lesson ("102's cause of death"), in text written when the 2026-08-03
postmortem described it as a finished v1. That postmortem was written from the *pushed* state
of the GitHub repo; 102 had 12–14 unpushed commits at the time and has continued through
session 3X. **The nav has been describing 102 as a closed project while it was live.** A Log
entry has been added stating the current state and that the postmortem's scope was the pushed
tree, not the working tree.

---

## Edits made

| file | change |
|---|---|
| `.gitignore` | `/reports/` → `/reports/*` plus four exact-filename whitelists. Git never descends into an excluded *directory*, so a `!` line under `/reports/` would have been silently inert — the method's own artifacts could not have been tracked at all. Every calibration overlay, video and CSV stays ignored. Filenames, never a glob (`!docs/assets/*.gif` is why). |
| `reports/c0_findings.md` | **new** — backfilled C0 record, from the template |
| `reports/deviation-log.md` | **new** — from the template, one backfilled row |
| `reports/blockers.md` | **new** — from the template, five rows |
| `reports/method_compliance_2026-08-09.md` | **new** — this file |
| `src/mcreid/utils/device.py` | **new** `probe_compute_device()` — the §5 policy |
| `src/mcreid/utils/__init__.py` | export it |
| `src/mcreid/cli/wildtrack_run.py` | `--device` / `--allow-cpu` + probe on `footpoint` and `run`; device now reaches `GpuViewConfig` |
| `src/mcreid/cli/wildtrack_demo.py` | same |
| `tests/test_utils_device.py` | **new** — 5 tests, GPU-free |
| `status.txt` | header drift fixed; current-state paragraph added. Body untouched |
| `000_infra/strejc-nav.md` | Log entry — 102 is live, not closed |

**Not edited, on purpose:** `decisions.md` and every past report (immutable record);
`context.md` §7's gate table (blockers row 4); `refactored_method.md` and every template
(out of scope); the two interactive webcam CLIs (reasoned above).

Tests: **513 green** (508 before, +5). `ruff check` clean, `mypy` clean on 58 source files.
Commits: `952126b`, `5b69e62`, `885380e`, plus this report.

---

## What the foot-point task inherits

1. **Start at C0.** No ground-contact estimator has ever been searched for. `c0_findings.md`
   says so explicitly so the next agent cannot mistake the backfill for coverage.
2. **Stop at C1.** `plan-footpoint.md` from `000_infra/templates/plan.md`, with `tier:`
   declared before the first number exists. Almost certainly **demo-grade** — an instrument-
   proved gate, an honest README line and one number — but that is the plan's call to state,
   not this report's.
3. **Its own kill criterion**, event-based, written before the first number. 102 has none and
   is not getting one retroactively.
4. **Its numbers are already measured** and are listed in `c0_findings.md` — GT floor
   0.123 m, detector ceiling 0.62–2.17 m as a *range* (D-004: any doc collapsing it to one
   number is a regression), and D-015's payoff curve showing what a better foot point buys.
   Cite them from there; do not re-derive them.
5. **Name which radius it means.** See the threshold-semantics finding above.
6. **Live validation waits on floor markers** — blockers.md row 1. Everything up to it does
   not.
