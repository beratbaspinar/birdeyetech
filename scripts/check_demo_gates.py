"""Arbiter for `plan-public-demo.md`'s gates.

Reads committed JSON and the README. Computes no metric of its own, so a gate
cannot be satisfied by the same code that produced the number it reads.

    uv run python scripts/check_demo_gates.py             # all gates
    uv run python scripts/check_demo_gates.py --gate g_d2

    G_D1  the demo runs and produces a BEV artifact, from the dataset's calibration
    G_D2  calibrated vs uncalibrated identity persistence, on the shipped segment
    G_D3  README: real artifact, honest framing, stranger-runnable quickstart
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
DEMO_JSON = REPO / "docs" / "artifacts" / "public_demo_wildtrack.json"
ARMS_JSON = REPO / "docs" / "artifacts" / "public_demo_arms.json"
README = REPO / "README.md"
QUICKSTART = "uv run mcreid-public-demo wildtrack"


class GateResult:
    def __init__(self, name: str) -> None:
        self.name = name
        self.lines: list[str] = []
        self.ok = True
        self.skipped = False

    def check(self, condition: bool, message: str) -> None:
        self.lines.append(f"    {'PASS' if condition else 'FAIL'}  {message}")
        self.ok = self.ok and condition

    def note(self, message: str) -> None:
        self.lines.append(f"    ....  {message}")

    def skip(self, message: str) -> None:
        self.skipped = True
        self.lines.append(f"    SKIP  {message}")

    def report(self) -> None:
        status = "SKIP" if self.skipped else ("PASS" if self.ok else "FAIL")
        print(f"[{status}] {self.name}")
        for line in self.lines:
            print(line)


def _load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def gate_d1() -> GateResult:
    r = GateResult("G_D1   demo runs end to end on the pinned segment, BEV artifact written")
    demo = _load(DEMO_JSON)
    if demo is None:
        r.skip(f"{DEMO_JSON.relative_to(REPO)} not written yet")
        return r

    r.check(demo["segment"]["n_frames"] > 0, f"ran {demo['segment']['n_frames']} frames")
    r.check(demo["cameras"] >= 2, f"{demo['cameras']} cameras (fusion needs >= 2)")
    r.check(
        "load_rig" in demo["calibration_source"],
        f"calibration came from the dataset: {demo['calibration_source']}",
    )
    for key in ("bev_mp4", "bev_gif"):
        art = demo["artifacts"][key]
        path = REPO / art["path"]
        r.check(path.is_file(), f"{key} exists on disk ({art['bytes'] / 1e6:.2f} MB)")
        r.check(len(art["sha256"]) == 64, f"{key} sha256 recorded")
    # The licence guard, asserted rather than trusted. A hero artifact containing
    # dataset pixels IS the dataset (CLAUDE.md), and the whole reason the hero is
    # a BEV is that the BEV is procedural.
    r.check(
        demo["artifacts"]["contains_dataset_pixels"] is False,
        "artifact declares no dataset pixels (BEV canvas is procedural)",
    )
    r.note(demo["honest_scope"])
    return r


def gate_d2() -> GateResult:
    """Calibrated vs uncalibrated, on the shipped segment.

    Two conditions, both fixed in the plan before any number existed: the
    calibrated arm must land CLOSER to the ground-truth identity count, and it
    must not pay for that with more switches.

    The switch column needs reading with care and the plan said so: an arm that
    collapses the whole scene into one identity has zero switches by construction,
    which is `context.md` §4's "a single-agent gate measures nothing" in a new
    costume. That is why closeness-to-truth is the primary condition.
    """
    r = GateResult("G_D2   calibrated beats uncalibrated on the segment that ships")
    arms = _load(ARMS_JSON)
    if arms is None:
        r.skip(f"{ARMS_JSON.relative_to(REPO)} not written yet")
        return r

    truth = int(arms["gt_identities_in_segment"])
    cal, unc = arms["arms"]["calibrated"], arms["arms"]["uncalibrated"]
    r.note(f"ground truth in the segment: {truth} identities")

    cal_err = abs(int(cal["ids_shown"]) - truth)
    unc_err = abs(int(unc["ids_shown"]) - truth)
    r.check(
        cal_err < unc_err,
        f"identity count closer to truth: calibrated {cal['ids_shown']} (off by {cal_err}) "
        f"vs uncalibrated {unc['ids_shown']} (off by {unc_err})",
    )
    r.check(
        int(cal["id_switches"]) <= int(unc["id_switches"]),
        f"no worse on switches: calibrated {cal['id_switches']} vs "
        f"uncalibrated {unc['id_switches']}",
    )
    r.note(
        f"position error: calibrated {cal['mean_position_error_m']:.3f} m vs "
        f"uncalibrated {unc['mean_position_error_m']:.3f} m"
    )
    if int(unc["ids_shown"]) <= 1:
        r.note(
            "READ THE SWITCH COLUMN WITH CARE: the uncalibrated arm collapsed the scene to "
            f"{unc['ids_shown']} identity, so its 0 switches are an artefact of having "
            "nothing to switch between, not evidence of stability."
        )
    return r


def gate_d3() -> GateResult:
    r = GateResult("G_D3   README rewritten: real artifact, honest framing, runnable quickstart")
    if not README.is_file():
        r.skip("README.md missing")
        return r
    text = README.read_text(encoding="utf-8")
    demo = _load(DEMO_JSON)

    if demo is not None:
        gif = Path(demo["artifacts"]["bev_gif"]["path"]).name
        r.check(gif in text, f"references the shipped artifact by name ({gif})")
    r.check(QUICKSTART in text, f"contains the quickstart verbatim: `{QUICKSTART}`")
    r.check(
        "download_wildtrack" in text,
        "tells a stranger how to get the data (scripts/download_wildtrack.py)",
    )
    # The scope lock, checked as text because it is the claim that is easiest to
    # lose in a rewrite and the most damaging to lose.
    crowd_words = ("failure analysis", "not a benchmark", "crowd")
    r.check(
        any(w in text.lower() for w in crowd_words),
        "keeps the crowd-scope framing (failure analysis, not a benchmark claim)",
    )
    r.check(
        "15.6" in text or "13 people" in text,
        "states the measured crowd density of the shipped segment",
    )
    synthetic_ok = "cardboard_demo.gif" not in text or "synthetic" in text.lower()
    r.check(synthetic_ok, "synthetic hero is removed, or present and labelled synthetic")
    return r


GATES = {"g_d1": gate_d1, "g_d2": gate_d2, "g_d3": gate_d3}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", choices=sorted(GATES))
    args = parser.parse_args()

    results = [GATES[n]() for n in ([args.gate] if args.gate else list(GATES))]
    for result in results:
        result.report()

    failed = [r.name for r in results if not r.ok and not r.skipped]
    skipped = [r.name for r in results if r.skipped]
    print()
    if failed:
        print(f"FAILED: {len(failed)} gate(s)")
    if skipped:
        print(f"SKIPPED (no evidence on disk): {len(skipped)} gate(s)")
    if not failed and not skipped:
        print("all gates PASS")
    return 1 if (failed or skipped) else 0


if __name__ == "__main__":
    sys.exit(main())
