"""Fetch EPFL CVLab's Laboratory sequence — the sparse demo's data.

Direct download, no registration, no signed agreement (A0 verified by resolving
every URL below, not by reading a page that claims it). ~310 MB.

The dataset is research-use and is **never committed**: it lands in `data/`,
which is gitignored, and nothing rendered from its pixels is shipped. The demo's
only visual artifact is the procedural BEV canvas.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

VIDEO = "https://documents.epfl.ch/groups/c/cv/cvlab-pom-video1/www"
DOCS = "https://www.epfl.ch/labs/cvlab/wp-content/uploads/2018/08"
ROOT = Path(__file__).resolve().parent.parent / "data" / "epfl_lab"

FILES = [(f"{VIDEO}/6p-c{i}.avi", f"6p-c{i}.avi") for i in range(4)] + [
    (f"{DOCS}/calibration-6p.txt", "calibration-6p.txt"),
    (f"{DOCS}/gt_lab_6p.txt", "gt_lab_6p.txt"),
]


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    for url, name in FILES:
        target = ROOT / name
        if target.is_file() and target.stat().st_size > 0:
            print(f"  have {name} ({target.stat().st_size / 1e6:.1f} MB)")
            continue
        print(f"  fetching {name} ...")
        urllib.request.urlretrieve(url, target)  # noqa: S310 - fixed https URLs above
        print(f"    {target.stat().st_size / 1e6:.1f} MB")
    print()
    print(f"EPFL Laboratory sequence in {ROOT}")
    print("Next: uv run python scripts/check_epfl_instrument.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
