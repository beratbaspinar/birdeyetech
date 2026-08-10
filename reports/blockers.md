# blockers — 102_multicam_reid

opened: 2026-08-09 (retrofit; the project predates `refactored_method.md`)

> `refactored_method.md` §4: **there is no `AskUserQuestion`.** A blocker is a row here, and then
> **work continues on everything not blocked.** The loop never stops to ask.
>
> A row is only for the five things that actually require the operator — force-push, deleting
> repos/branches, making a repo public, spending money, wiping an environment — or for a genuine
> external dependency (access, hardware, a threshold change). Ambiguity is **not** a blocker:
> implement the reading you can defend and state the assumption.

Every row carries a **recommendation**. A row that presents options without one is asking a
question with extra steps.

**Provenance.** No blockers file existed before 2026-08-09; the pre-HPC audit filing did not put
102 rows here or in `000_infra/reports/blockers.md` (checked — none). Rows 1–3 are **backfilled
from items 102 already recorded as owed** in `status.txt` sessions 3W/3X; rows 4–5 are raised by
the 2026-08-09 method-compliance retrofit. Nothing is invented: every row names where it was
already on record, or that the retrofit is what raised it.

| date | task | what is blocked | what was tried | what is needed | recommendation | state |
|---|---|---|---|---|---|---|
| 2026-08-09 | §3 — gates | **`context.md` §7's gate table is prose, not machine-checkable.** Five gates, four of which resolve to "see status.txt". §3 requires a gate to be a command that exits 0/1. | Nothing — flagged by the compliance retrofit, not attempted. Authoring a gate set for already-shipped work is a strategy act (it fixes thresholds retroactively), which the retrofit brief explicitly excludes. | An operator decision on scope: retrofit 102's existing gates, or leave them as the historical record and require machine-checkable gates only from the foot-point plan forward. | **(b) Leave §7 as the historical record; require machine-checkable gates from the foot-point C1 plan forward.** Retrofitting gates onto measurements that already exist is pre-registration theatre — the numbers are already known, so any threshold chosen now is chosen by its own result, which is the failure §3 exists to prevent. The next plan pays the cost properly. | open |
| 2026-08-10 | foot-point task / plan-footpoint.md §6 | **What comes after both foot-point arms were rejected.** Nothing is queued behind this — the flag ships, the default is unchanged, and the repo is in a working state. | Both designs built, measured and rejected on the pre-registered gates; both rejections **VOID** by G_FP0b (neither could reach 0.48 m even on GT boxes), so the **kill counter stays 0 of 2**. The diagnosis is the reason there is no third design to try: the detector's box bottom is *unbiased* (h = -0.013 +/- 0.118 m at IoU 0.5), so there is no offset to remove, and the rig converts vertical error to depth error by `depth / camera_height` ~ 5x. Every per-view foot-point rule reads that same coordinate through that same multiplier. Full evidence: `reports/footpoint_build_2026-08-10.md`. | An operator **scope decision**, which `plan-footpoint.md` §6 reserves rather than spending a third design on. | **(a) Triangulate across views instead of projecting per view.** The statistic being measured is disagreement between two independent per-view projections; two rays intersect, and the ill-conditioning that kills one grazing ray is what a second ray fixes. It is a fusion-stage change, not an estimator, so it needs its own plan. **(b)** FootTrackNet (C0 candidate 3) — but note it faces the same 5x multiplier, so expect it to help the 15-27 % fallback rate, not the conditioning. **(c)** Accept the box bottom and close the debt: it is already at the noise floor its own optics allow on this surface. | open |
| 2026-08-10 | foot-point task | Nothing. **Recording a scope limit so it is not mistaken for a result.** The foot-point measurement ran entirely on WILDTRACK, which is 102's *stress test*, not the sparse regime `context.md` §1 scopes the project to. | The conditioning that defeated both arms is `depth / camera_height`, far worse on a wide outdoor scene than at room distances. | **Nothing — and the way this was going to be settled is now gone.** The 2026-08-10 recommendation was *"re-run `footpoint-arms` on the room rig once it is calibrated"*. **The operator cancelled the capture session permanently on 2026-08-10**, so no room rig will ever exist and that check cannot be run. | **(a) Leave the conclusion scoped to WILDTRACK and say so wherever it is cited.** It stands as measured — the box bottom is at the noise floor its optics allow *on a wide outdoor rig* — and it must not be generalised to the sparse regime, in either direction, because the surface that would have tested it no longer exists. **(b)** If a public dataset with room-scale camera-to-subject distances turns up, re-run there; PhysicalAI-SmartSpaces is the nearest candidate. | open |
| 2026-08-10 | T5 / G_FP1, G_FP2 | Nothing — recorded because a **specified measurement surface does not exist** and a substitute was chosen without asking. The task brief gates the foot-point work "against the existing D-015 harness with its validated 0.08 m GT reference row". **That harness is not in this repo.** | Searched `src/`, `tests/`, `scripts/` and the git history. D-015's noise sweep (0 % → 0.08 m → 99 %, 10 % → 0.48 m → 98 %, 20 % → 1.02 m → 33 %, 30 % → 1.48 m → 13 %) was session-time analysis; only its **result** was committed, as a `decisions.md` entry. Its 0.08 m row is a number in a decision, not a fixture. | Nothing blocking. Recorded so the substitution is visible: the gates measure on `mcreid.eval.footpoint` + `mcreid-wildtrack footpoint`, which is committed, reproducible, and measures **the same statistic** (cross-camera disagreement in metres) on real WILDTRACK boxes with three artifacts already on disk. D-015's threshold is carried across **by its derivation**, written out in `plan-footpoint.md` §3 G_FP2. | **(a) Accept the substitute surface.** The alternative — rebuilding D-015's virtual-rig sweep from a table in a decision entry — would reconstruct a harness from its own published output, which is not a validation, and it would delay the build behind a re-derivation of a number that is already trusted. If the operator wants the sweep re-committed as a fixture, that is its own task. | open |
| 2026-08-09 | §3 — deviation log row 1 | Nothing is blocked. **Ratification** of the one backfilled deviation: the floor gate's 4 → 6 marker minimum, a gate-semantics change signed by `lead` because no operator channel existed at the time. | The deviation was taken, executed, and stated in the session it happened (D-013, session 3X). It made the gate stricter, not passable. | One line from the operator: ratify or reject. | **(a) Ratify.** The change made a gate that could not fail into one that can; rejecting it would restore an algebraic identity in place of a gate. Recorded so the precedent is explicit rather than silent. | open |

## Resolved

Keep them. A resolved row is the record of who decided what, and several have later turned out to
be the thing that explained a result.

### Resolved 2026-08-10 — publication

**RESOLVED: the operator confirmed the repo stays PUBLIC and approved the push. Done —
`a82667c..c797a87`, 25 commits.** The check was still worth making: the repo's own protocol
requires it (`context.md` §6, after session 3L), the visibility genuinely contradicted
`CLAUDE.md`'s private-until-approved wording, and a push to a public repo is not reversible the
way a private one is. The answer was "public is intended". Original row:

| 2026-08-10 | push / publication | **19 unpushed commits** (`dcacd2a..4582cd7`) — the entire N-camera capture stack, the floor-calibration subsystem, the marking helper and the method retrofit. `origin/main` is still `a82667c`, pushed 2026-07-31. The work exists on one laptop. | Nothing — **the push was ordered and I stopped before running it.** The order was premised on "push is not on the approval list", which is true of a private repo. `gh repo view` says otherwise: `"isPrivate": false, "visibility": "PUBLIC"`. This repo's own protocol requires exactly this check before any push (`context.md` §6, after session 3L, when a stated flip to private had silently not applied and the API was the only thing that caught it). It caught it again. | **One line from the operator**, and the two questions are separate: (1) is `multicam_persistent_id` *meant* to be public right now? `CLAUDE.md` says PRIVATE until explicit approval and `context.md` §6 records the flip as never made as of session 3M. (2) If yes, publish these 19 commits. | **(a) Confirm the visibility is intended, then push.** The content itself looks publishable — the four session logs are gitignored, `docs/artifacts/` is metric JSON with no pixels, and the pre-ship audit returned GO in 3M — but "looks publishable" is not the operator's signature, and a push to a public repo is irreversible in a way a private one is not. If the public state is **not** intended, flip to private first and the push stops being an approval question at all. | open |


### CANCELLED 2026-08-10 — every row that needed the operator's hands

**Operator decision, 2026-08-10: the live capture session is CANCELLED PERMANENTLY. No owner rig,
no marker session, no live feed.** 102's demo pivots to public datasets with dataset-provided
calibration. These three rows are closed as **cancelled — not resolved, not deferred**: the work
they describe will not happen and nothing should wait on it.

What that costs, stated plainly rather than buried:

- `calib/` stays empty. **No rig of this room was ever calibrated**, and the floor-calibration
  subsystem — `calib/floor.py`, the leave-one-out gate, `mcreid-calibrate mark` with its 16x loupe,
  and D-013/D-016/D-017 — has never touched a real room. It is verified end to end on *simulated*
  loupe clicks and against analytic rigs, and that is now the strongest evidence it will ever have.
  It is not dead code: it is what makes a dataset rig loadable, and the demo uses that path.
- **id 73 will never be classified**, and the 3-camera device/backend mapping will never be
  confirmed. `reports/console_dumps.txt` arrived 0 bytes three times; the only fix was another
  capture session. Both are recorded in `status.txt` session 3W as permanently open questions.
- D-015's forecast — *"two-person separation will hold live; one-person cross-view hold may not"* —
  **will never be tested live.** It stays a forecast on the record, and the honest thing is to keep
  calling it one.

The cancelled rows, kept verbatim:

| 2026-08-01 | M1(c) — live rig | **The calibrated live run**, and with it any live validation of the D-015 foot-point forecast. `mcreid-calibrate floor` refuses without ≥ 6 marked floor points per camera. | The whole path is built and verified end-to-end on *simulated* loupe clicks (two cameras, two sittings, `build_and_gate` → PASS at 0.000 m held-out). `mcreid-calibrate mark` exists precisely so this is 20 minutes of work, not an afternoon. | Matus lays out ≥ 6 (procedure asks 12) floor markers per camera, runs `mcreid-calibrate mark` once per camera, types the world coordinates, runs `floor`. Physical, cannot be delegated. | **(a) Do the 12-marker pass before the foot-point build starts.** D-016 measures the cost of skimping: at 1.5 px click error, 6 markers pass 0 % of the time and 12 pass 92 %. Six is the *refusal* threshold, not the target. | open |
| 2026-08-01 | 3-cam session — defect (b) | **Classification of id 73** (a 31 s cam1-only fragment) as cross-view fragmentation / duplicate split / static false positive. | `reports/console_dumps.txt` was created and is **0 bytes**; re-checked twice in session 3X, mtime unchanged. The transport workaround failed too. | The ledger line for id 73, plus the ledger lines for every id live between its first and last frame. Three candidate signatures and the exact lines needed are in the 3W block of `status.txt`. | **(a) Re-run the 3-cam session with output redirected to a file rather than recovered from a console buffer** — the buffer has now failed twice, and the fix is a redirect, not another attempt at the same transport. Low priority: it blocks nothing downstream. | open |
| 2026-08-01 | 3-cam session — rig record | Confirmation of the **3-camera device → backend mapping**. Same 0-byte dump. | As above. | The `mcreid-live-multi scan` output from the 3-cam rig. | **(a) Fold it into the re-run above** — one session discharges both rows. | open |

