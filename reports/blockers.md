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
| 2026-08-10 | push / publication | **19 unpushed commits** (`dcacd2a..4582cd7`) — the entire N-camera capture stack, the floor-calibration subsystem, the marking helper and the method retrofit. `origin/main` is still `a82667c`, pushed 2026-07-31. The work exists on one laptop. | Nothing — **the push was ordered and I stopped before running it.** The order was premised on "push is not on the approval list", which is true of a private repo. `gh repo view` says otherwise: `"isPrivate": false, "visibility": "PUBLIC"`. This repo's own protocol requires exactly this check before any push (`context.md` §6, after session 3L, when a stated flip to private had silently not applied and the API was the only thing that caught it). It caught it again. | **One line from the operator**, and the two questions are separate: (1) is `multicam_persistent_id` *meant* to be public right now? `CLAUDE.md` says PRIVATE until explicit approval and `context.md` §6 records the flip as never made as of session 3M. (2) If yes, publish these 19 commits. | **(a) Confirm the visibility is intended, then push.** The content itself looks publishable — the four session logs are gitignored, `docs/artifacts/` is metric JSON with no pixels, and the pre-ship audit returned GO in 3M — but "looks publishable" is not the operator's signature, and a push to a public repo is irreversible in a way a private one is not. If the public state is **not** intended, flip to private first and the push stops being an approval question at all. | open |
| 2026-08-01 | M1(c) — live rig | **The calibrated live run**, and with it any live validation of the D-015 foot-point forecast. `mcreid-calibrate floor` refuses without ≥ 6 marked floor points per camera. | The whole path is built and verified end-to-end on *simulated* loupe clicks (two cameras, two sittings, `build_and_gate` → PASS at 0.000 m held-out). `mcreid-calibrate mark` exists precisely so this is 20 minutes of work, not an afternoon. | Matus lays out ≥ 6 (procedure asks 12) floor markers per camera, runs `mcreid-calibrate mark` once per camera, types the world coordinates, runs `floor`. Physical, cannot be delegated. | **(a) Do the 12-marker pass before the foot-point build starts.** D-016 measures the cost of skimping: at 1.5 px click error, 6 markers pass 0 % of the time and 12 pass 92 %. Six is the *refusal* threshold, not the target. | open |
| 2026-08-01 | 3-cam session — defect (b) | **Classification of id 73** (a 31 s cam1-only fragment) as cross-view fragmentation / duplicate split / static false positive. | `reports/console_dumps.txt` was created and is **0 bytes**; re-checked twice in session 3X, mtime unchanged. The transport workaround failed too. | The ledger line for id 73, plus the ledger lines for every id live between its first and last frame. Three candidate signatures and the exact lines needed are in the 3W block of `status.txt`. | **(a) Re-run the 3-cam session with output redirected to a file rather than recovered from a console buffer** — the buffer has now failed twice, and the fix is a redirect, not another attempt at the same transport. Low priority: it blocks nothing downstream. | open |
| 2026-08-01 | 3-cam session — rig record | Confirmation of the **3-camera device → backend mapping**. Same 0-byte dump. | As above. | The `mcreid-live-multi scan` output from the 3-cam rig. | **(a) Fold it into the re-run above** — one session discharges both rows. | open |
| 2026-08-09 | §3 — gates | **`context.md` §7's gate table is prose, not machine-checkable.** Five gates, four of which resolve to "see status.txt". §3 requires a gate to be a command that exits 0/1. | Nothing — flagged by the compliance retrofit, not attempted. Authoring a gate set for already-shipped work is a strategy act (it fixes thresholds retroactively), which the retrofit brief explicitly excludes. | An operator decision on scope: retrofit 102's existing gates, or leave them as the historical record and require machine-checkable gates only from the foot-point plan forward. | **(b) Leave §7 as the historical record; require machine-checkable gates from the foot-point C1 plan forward.** Retrofitting gates onto measurements that already exist is pre-registration theatre — the numbers are already known, so any threshold chosen now is chosen by its own result, which is the failure §3 exists to prevent. The next plan pays the cost properly. | open |
| 2026-08-09 | §3 — deviation log row 1 | Nothing is blocked. **Ratification** of the one backfilled deviation: the floor gate's 4 → 6 marker minimum, a gate-semantics change signed by `lead` because no operator channel existed at the time. | The deviation was taken, executed, and stated in the session it happened (D-013, session 3X). It made the gate stricter, not passable. | One line from the operator: ratify or reject. | **(a) Ratify.** The change made a gate that could not fail into one that can; rejecting it would restore an algebraic identity in place of a gate. Recorded so the precedent is explicit rather than silent. | open |

## Resolved

Keep them. A resolved row is the record of who decided what, and several have later turned out to
be the thing that explained a result.

*(none yet — the file was opened 2026-08-09)*
