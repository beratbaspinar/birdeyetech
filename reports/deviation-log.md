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

A row with "gate semantics changed? **yes**" and authority "lead" is normally a defect — only the
operator can sign that. Row 1 is entered anyway, with its reasoning intact, because suppressing it
would be the worse failure: the deviation happened, the operator was told, and the direction was
toward a harder gate, not a passing one. **The operator has a `blockers.md` row to ratify or
reject it retroactively.** Do not treat this row as precedent for a lead signing a semantics change.
