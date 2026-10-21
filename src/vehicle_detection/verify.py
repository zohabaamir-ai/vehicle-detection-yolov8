"""Check that a dataset folder is exactly the one the split manifest describes. No torch needed.

Usage:
    python -m vehicle_detection.verify --root D:\\data\\clean --manifest results\\split_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import yaml

from vehicle_detection.audit import IMAGE_EXTS, scan_split
from vehicle_detection.constants import SPLIT_NAMES


def verify_dataset(root: Path | str, manifest_csv: Path | str, summary_json: Path | str) -> dict:
    """Compare images (by sha256) and per-split box counts with the committed manifest/summary."""
    root = Path(root)
    with Path(manifest_csv).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    expected = json.loads(Path(summary_json).read_text(encoding="utf-8"))

    missing, mismatched = [], []
    for r in rows:
        path = root / r["split"] / "images" / r["file"]
        if not path.exists():
            missing.append(r["file"])
        elif hashlib.sha256(path.read_bytes()).hexdigest() != r["sha256"]:
            mismatched.append(r["file"])
    listed = {(r["split"], r["file"]) for r in rows}
    unexpected = [
        f"{split}/{p.name}"
        for split in SPLIT_NAMES
        for p in sorted((root / split / "images").glob("*"))
        if p.suffix.lower() in IMAGE_EXTS and (split, p.name) not in listed
    ]

    count_problems = []
    class_names = expected["class_names"]
    for split in SPLIT_NAMES:
        got, _ = scan_split(root / split)
        want = expected["splits"][split]
        if got["n_images"] != want["n_images"]:
            count_problems.append(f"{split}: {got['n_images']} images, expected {want['n_images']}")
        if got["n_boxes"] != want["n_boxes"]:
            count_problems.append(f"{split}: {got['n_boxes']} boxes, expected {want['n_boxes']}")
        by_name = {class_names[int(k)]: v for k, v in got["boxes_per_class"].items()}
        for name in class_names:
            if by_name.get(name, 0) != want["boxes_per_class"][name]:
                count_problems.append(f"{split}: wrong number of {name} boxes")
    cfg_path = root / "data.yaml"
    if not cfg_path.exists():
        count_problems.append("data.yaml is missing")
    else:
        names = yaml.safe_load(cfg_path.read_text(encoding="utf-8")).get("names")
        got_names = [names[k] for k in sorted(names)] if isinstance(names, dict) else names
        if got_names != class_names:
            count_problems.append(f"data.yaml names {got_names} differ from {class_names}")
    ok = not (missing or mismatched or unexpected or count_problems)
    return {
        "ok": ok,
        "n_manifest_images": len(rows),
        "missing": missing[:10],
        "hash_mismatch": mismatched[:10],
        "unexpected": unexpected[:10],
        "count_problems": count_problems,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--manifest", type=Path, default=Path("results/split_manifest.csv"))
    ap.add_argument("--summary", type=Path, default=Path("results/split_summary.json"))
    args = ap.parse_args(argv)
    report = verify_dataset(args.root, args.manifest, args.summary)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
