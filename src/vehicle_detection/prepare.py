"""Build a clean YOLO dataset: grouped train/val/test split, box-only labels.

Usage:
    python -m vehicle_detection.prepare --root D:\\data\\Objects --out D:\\data\\clean
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from vehicle_detection.audit import roboflow_base_name
from vehicle_detection.constants import (
    CLASS_NAMES,
    DEDUP_THRESHOLD,
    ISOLATED_MIN_DISTANCE,
    SPLIT_NAMES,
    SPLIT_RATIOS,
    SPLIT_SEED_CANDIDATES,
)
from vehicle_detection.dedup import cluster, distance_matrix, hash_image, list_images
from vehicle_detection.labels import EPS, Box, parse_label_text
from vehicle_detection.split import best_split, cross_split_neighbours

RESIDUAL_THRESHOLDS = [DEDUP_THRESHOLD, 14, 16, 20]


def clip_box(box: Box) -> Box | None:
    """Clip a box to the image. Returns None if nothing meaningful is left."""
    x0, x1 = max(0.0, box.xc - box.w / 2), min(1.0, box.xc + box.w / 2)
    y0, y1 = max(0.0, box.yc - box.h / 2), min(1.0, box.yc + box.h / 2)
    if x1 - x0 <= EPS or y1 - y0 <= EPS:
        return None
    return Box(box.cls, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)


def format_label(boxes: list[Box]) -> str:
    return "".join(f"{b.cls} {b.xc:.6f} {b.yc:.6f} {b.w:.6f} {b.h:.6f}\n" for b in boxes)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root: Path, out: Path, n_seeds: int, threshold: int, ratios: tuple[float, ...]):
    """Returns (summary dict, manifest rows). Writes the dataset to `out`."""
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Output folder {out} already exists and is not empty. Remove it first.")
    items = list_images(root)
    if not items:
        raise SystemExit(f"No images found under {root}")
    paths = [p for p, _ in items]
    names = [p.name for p in paths]
    if len(set(names)) != len(names):
        raise SystemExit("Two images have the same file name in different splits; cannot merge.")

    keys = [roboflow_base_name(n) for n in names]
    hs, hf = zip(*(hash_image(p) for p in paths), strict=True)
    dist = distance_matrix(np.stack(hs), np.stack(hf))
    groups = cluster(dist, threshold, keys)

    boxes_of: list[list[Box]] = []
    n_polygon_lines = n_mixed = n_dropped = n_bad_lines = 0
    for p in paths:
        label = p.parent.parent / "labels" / f"{p.stem}.txt"
        if not label.exists():
            raise SystemExit(f"Missing label file for {p.name}")
        parsed = parse_label_text(label.read_text(encoding="utf-8", errors="replace"))
        n_polygon_lines += parsed.n_polygons
        n_mixed += int(parsed.is_mixed)
        n_bad_lines += sum(1 for m in parsed.problems if "outside" not in m)
        kept: list[Box] = []
        for b in parsed.boxes:
            if b.cls >= len(CLASS_NAMES):
                raise SystemExit(f"{p.name}: class id {b.cls} is not in {CLASS_NAMES}")
            clipped = clip_box(b)
            if clipped is None:
                n_dropped += 1
            else:
                kept.append(clipped)
        boxes_of.append(kept)

    n_cls = range(len(CLASS_NAMES))
    # balanced across splits: images, boxes per class, and images containing each class
    features = np.array(
        [
            [
                1,
                *[sum(b.cls == c for b in bx) for c in n_cls],
                *[int(any(b.cls == c for b in bx)) for c in n_cls],
            ]
            for bx in boxes_of
        ]
    )
    chosen = best_split(groups, features, ratios, n_seeds)
    split_of = chosen["split"]

    for name in SPLIT_NAMES:
        (out / name / "images").mkdir(parents=True, exist_ok=True)
        (out / name / "labels").mkdir(parents=True, exist_ok=True)
    train_idx = [i for i, s in enumerate(split_of) if s == 0]
    rows = []
    for i, (p, orig, g, s, bx) in enumerate(
        zip(paths, (o for _, o in items), groups, split_of, boxes_of, strict=True)
    ):
        split = SPLIT_NAMES[s]
        shutil.copy2(p, out / split / "images" / p.name)
        (out / split / "labels" / f"{p.stem}.txt").write_text(format_label(bx), encoding="utf-8")
        rows.append(
            {
                "file": p.name,
                "original_split": orig,
                "split": split,
                "group": g,
                "nearest_train_dist": int(dist[i, train_idx].min()) if s != 0 else "",
                "sha256": sha256_of(p),
            }
        )
    data_yaml = {
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": dict(enumerate(CLASS_NAMES)),
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    summary: dict = {
        "split_seed": chosen["seed"],
        "n_seed_candidates": n_seeds,
        "imbalance_chosen": round(chosen["imbalance"], 4),
        "imbalance_seed_0": round(chosen["imbalance_seed_0"], 4),
        "imbalance_median_candidate": round(chosen["imbalance_median"], 4),
        "dedup_threshold": threshold,
        "ratios": dict(zip(SPLIT_NAMES, ratios, strict=True)),
        "class_names": list(CLASS_NAMES),
        "n_images": len(paths),
        "n_groups": max(groups) + 1,
        "polygon_lines_converted_to_boxes": n_polygon_lines,
        "label_files_that_mixed_polygons_and_boxes": n_mixed,
        "label_lines_skipped_as_invalid": n_bad_lines,
        "boxes_dropped_after_clipping": n_dropped,
        "splits": {},
    }
    for si, name in enumerate(SPLIT_NAMES):
        idx = [i for i, s in enumerate(split_of) if s == si]
        per_class = Counter(b.cls for i in idx for b in boxes_of[i])
        imgs_per_class = Counter(c for i in idx for c in {b.cls for b in boxes_of[i]})
        summary["splits"][name] = {
            "n_images": len(idx),
            "n_groups": len({groups[i] for i in idx}),
            "n_boxes": sum(per_class.values()),
            "boxes_per_class": {
                CLASS_NAMES[c]: per_class.get(c, 0) for c in range(len(CLASS_NAMES))
            },
            "images_per_class": {
                CLASS_NAMES[c]: imgs_per_class.get(c, 0) for c in range(len(CLASS_NAMES))
            },
        }
    summary["isolated_images"] = {
        "definition": f"nearest train image is more than {ISOLATED_MIN_DISTANCE} hash bits away",
        **{
            name: sum(
                1
                for r in rows
                if r["split"] == name and r["nearest_train_dist"] > ISOLATED_MIN_DISTANCE
            )
            for name in SPLIT_NAMES[1:]
        },
    }
    summary["residual_cross_split_neighbours"] = cross_split_neighbours(
        dist, split_of, RESIDUAL_THRESHOLDS
    )
    return summary, rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path, help="original dataset (train/ test/ ...)")
    ap.add_argument("--out", required=True, type=Path, help="NEW folder for the clean dataset")
    ap.add_argument("--seeds", type=int, default=SPLIT_SEED_CANDIDATES, help="split seeds to try")
    ap.add_argument("--threshold", type=int, default=DEDUP_THRESHOLD)
    ap.add_argument("--summary", type=Path, default=Path("results/split_summary.json"))
    ap.add_argument("--manifest", type=Path, default=Path("results/split_manifest.csv"))
    args = ap.parse_args(argv)
    summary, rows = build(args.root, args.out, args.seeds, args.threshold, SPLIT_RATIOS)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with args.manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(
        f"split seed {summary['split_seed']} of {summary['n_seed_candidates']} candidates: "
        f"imbalance {summary['imbalance_chosen']} (seed 0: {summary['imbalance_seed_0']}, "
        f"median: {summary['imbalance_median_candidate']})"
    )
    for name, s in summary["splits"].items():
        print(f"{name:>5}: {s}")
    print("residual near-duplicates across splits:", summary["residual_cross_split_neighbours"])
    print(f"Dataset written to {args.out}")
    print(f"Summary: {args.summary}   Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
