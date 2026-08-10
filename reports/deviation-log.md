# deviation log — 102_multicam_reid

opened: 2026-08-09 (retrofit; the project predates `refactored_method.md`)

> `refactored_method.md` §3: **thresholds are never edited to make a gate pass.** Any change to a
> threshold, a gate's semantics, or a plan clause is a **dated entry here with the authority
> named** — never a silent edit to a config default or a plan clause.
>
> If the change would make a failing gate pass, it is not a deviation, it is a `blockers.md`
> proposal awaiting the operator. The distinction is the whole point of the file.

Also the landing place for §5's **>5x rule**: a path ≥5x cheaper with *unchanged* gate semantics is
taken and logged as one line. If it would change a threshold, it is a blockers row instead.

**Row 1 is BACKFILLED** from a deviation 102 recorded at the time, in `decisions.md` D-013 and in
the session 3X block of `status.txt`. It is entered here because the record exists, not to
manufacture one. **The log is otherwise empty and starts now** — the absence of rows for sessions
3a–3X is a fact about 102 predating the method, not a claim that no threshold ever moved. Read
`decisions.md` for the pre-2026-08-09 record; it is the immutable one.

| # | date | what changed | from → to | why | authority | gate semantics changed? |
|---|---|---|---|---|---|---|
| 1 | 2026-08-01 | floor-calibration gate: minimum marker count | **4 → 6 markers** (procedure asks 12) | Four correspondences fit a homography EXACTLY (`getPerspectiveTransform` reproduces them to ~1e-15), so the in-sample residual is identically zero for every rig *including a badly mismarked one*, and cross-camera agreement on those same points is zero by construction. A four-point gate is not a weak gate, it is an algebraic identity. Leave-one-out over ≥ 6 markers measures generalisation to a point the fit never saw. | lead | **yes — and it made the gate STRICTER.** The approved scope said four points; the approved *requirement* was a gate that can fail, and four points cannot produce one. Recorded as a deviation from the scope, executed, and stated in the same session it was taken (`decisions.md` D-013). |

| 2 | 2026-08-10 | foot-point eval: stature constant used for the WILDTRACK measurement | **1.70 m → 1.82 m** (CLI argument only; `DEFAULT_STATURE_M` in code stays 1.70) | **Recovered from WILDTRACK's own annotation geometry, before the gates were read.** G_FP0b showed the stature arm at 5.43 m mean disagreement on GT boxes — an impossible number for exact geometry on exact boxes, so it was diagnosed rather than recorded. Solving for the world height whose projection lands on each GT box edge gives **box bottom ↔ h = +0.01 m and box top ↔ h = +1.82 m, identically across cameras and people** (CVLab1 +1.82/+1.82/+1.83/+1.82, CVLab3 +1.82). WILDTRACK's boxes are rendered from a POM cylinder ~1.8 m tall; 1.70 m was simply the wrong number for this surface. This is a **parameter defect found by an instrument check, not a threshold moved to pass a gate** — no gate threshold changed, and the arm may still fail. | lead | **no** — no gate, threshold or acceptance criterion was touched. G_FP2 remains 0.48 m |

| 3 | 2026-08-10 | **EPFL Laboratory arm only** — fusion radii re-expressed in grid cells, because the metric scale is not recoverable from this dataset | `birth_cluster_radius` 1.0 m → **3.0 cells**; `merge_radius` 0.75 m → **2.25 cells**; `merge_unconditional_radius` 0.0 → 0.0 (still disabled) | **The derivation dead-ended on evidence and the fallback is the operator's, authorised 2026-08-10.** The scale is not recoverable: EPFL ships no intrinsics, only 2 of 4 cameras carry the head-plane homography that is the metric ruler, and the ground homography's two Zhang constraints are **mutually inconsistent** under a centred principal point — under `fx = fy` the spare constraint gives ‖r1‖/‖r2‖ = 0.30 against a required 1.0, and under `fx ≠ fy` (exactly determined, no spare check left) **no camera has a positive solution at all**. Transferring a vertical 1.75 m ruler to a horizontal ground distance needs the camera's internals; without them the cell size is unidentifiable, which is a property of the data and not of the effort. So the arm runs in grid units. **The radii are RE-DERIVED, not converted** — a conversion would need the very scale we just said we do not have. Each is re-derived from what it was for: a cluster/merge radius must sit **above** same-person cross-camera disagreement and **below** the separation of distinct people, or it fuses two people by construction. Measured from EPFL's own ground truth: nearest-neighbour separation between distinct people is **p05 6.1 cells**, p50 10.6, p90 18.7. 3.0 cells is half the p05 — the same "comfortably inside the bracket" position the metric 1.0 m held, and `merge_radius` keeps its original 0.75 ratio to it. | lead | **yes, for this arm only** — and the gate is re-derived with them: G_D2e keeps G_D2's two conditions verbatim (identity count closer to truth, no worse on switches), which are **unit-free**, so nothing about what "passing" means has moved. The WILDTRACK arm and every shipped default are untouched. |

**Row 2's honest caveat, stated because it weakens a number in our favour.** Deriving the
constant from GT-box geometry makes the stature arm's **GT-box column near-tautological**: those
boxes are *generated* from a 1.8 m cylinder standing at the annotated position, so inverting them
with h = 1.82 recovers the annotation by construction rather than by estimation. The GT column is
therefore not a meaningful ceiling for this arm. **The detector column is, and it is the one the
gates read.** The same caveat applies in reverse to the `bbox` arm: on GT boxes the box bottom is
the cylinder's base at h ≈ 0.01 m, which is why it scores 0.123 m there and why "beat box-bottom
on GT boxes" is not a fair contest either.

A row with "gate semantics changed? **yes**" and authority "lead" is normally a defect — only the
operator can sign that. Row 1 is entered anyway, with its reasoning intact, because suppressing it
would be the worse failure: the deviation happened, the operator was told, and the direction was
toward a harder gate, not a passing one. **The operator has a `blockers.md` row to ratify or
reject it retroactively.** Do not treat this row as precedent for a lead signing a semantics change.

**Row 3's honest caveat.** "Grid-metric" is not metric. Every distance the EPFL arm reports is in
grid cells and **cannot be compared to any metre-denominated number in this repo** — not D-004's
0.62–2.17 m, not D-014's radii, not the 0.48 m band. The README labels the arm grid-metric for
exactly that reason. What survives the unit change is everything unit-free: identity counts,
switches, and whether calibrated beats appearance-only.
