"""Dataset audit: label quality, class balance and train/test overlap. No torch needed.

Usage:
    python -m vehicle_detection.audit --root D:\\path\\to\\ObjectsDataset
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image

from vehicle_detection.labels import parse_label_text

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TINY_AREA = 0.0025  # normalised box area; about the COCO "small" cut-off at 640x640
_RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]{32}$")


def roboflow_base_name(filename: str) -> str:
    """'abc_jpg.rf.<32 hex>.jpg' -> 'abc_jpg'. Copies of one source share this name."""
    return _RF_SUFFIX.sub("", Path(filename).stem)


def scan_split(split_dir: Path) -> tuple[dict, dict]:
    """Audit one split folder that has images/ and labels/. Returns (summary, details)."""
    img_dir, lbl_dir = split_dir / "images", split_dir / "labels"
    images = {p.stem: p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS}
    labels = {p.stem: p for p in sorted(lbl_dir.glob("*.txt"))}

    boxes_per_class: Counter = Counter()
    images_per_class: Counter = Counter()
    sizes: Counter = Counter()
    problem_files: dict[str, list[str]] = {}
    empty, polygon_files, polygon_lines, tiny, total_boxes = [], 0, 0, 0, 0
    mixed: list[str] = []
    mixed_plain_lines = 0
    unreadable: list[str] = []
    per_image_counts: list[int] = []

    for stem, img_path in images.items():
        try:
            with Image.open(img_path) as im:
                sizes[im.size] += 1
        except Exception:  # noqa: BLE001 - any decode failure counts as unreadable
            unreadable.append(img_path.name)
        if stem not in labels:
            continue
        parsed = parse_label_text(labels[stem].read_text(encoding="utf-8", errors="replace"))
        if parsed.problems:
            problem_files[labels[stem].name] = parsed.problems
        if parsed.n_polygons:
            polygon_files += 1
            polygon_lines += parsed.n_polygons
        if parsed.is_mixed:
            mixed.append(labels[stem].name)
            mixed_plain_lines += parsed.n_plain_boxes
        if not parsed.boxes:
            empty.append(labels[stem].name)
        per_image_counts.append(len(parsed.boxes))
        total_boxes += len(parsed.boxes)
        for cid in {b.cls for b in parsed.boxes}:
            images_per_class[cid] += 1
        for b in parsed.boxes:
            boxes_per_class[b.cls] += 1
            if b.area < TINY_AREA:
                tiny += 1

    base_counts = Counter(roboflow_base_name(p.name) for p in images.values())
    summary = {
        "n_images": len(images),
        "n_label_files": len(labels),
        "images_without_label": sorted(set(images) - set(labels))[:20],
        "n_images_without_label": len(set(images) - set(labels)),
        "n_labels_without_image": len(set(labels) - set(images)),
        "n_empty_label_files": len(empty),
        "n_unreadable_images": len(unreadable),
        "n_boxes": total_boxes,
        "boxes_per_class": {str(k): v for k, v in sorted(boxes_per_class.items())},
        "images_per_class": {str(k): v for k, v in sorted(images_per_class.items())},
        "n_label_files_with_polygons": polygon_files,
        "n_polygon_lines": polygon_lines,
        "n_label_files_mixed_polygon_and_box": len(mixed),
        "n_plain_box_lines_in_mixed_files": mixed_plain_lines,
        "n_label_files_with_problems": len(problem_files),
        "n_tiny_boxes": tiny,
        "tiny_box_fraction": round(tiny / total_boxes, 4) if total_boxes else 0.0,
        "boxes_per_image_mean": round(total_boxes / len(per_image_counts), 2)
        if per_image_counts
        else 0.0,
        "boxes_per_image_max": max(per_image_counts, default=0),
        "n_unique_source_names": len(base_counts),
        "n_source_names_with_copies": sum(1 for c in base_counts.values() if c > 1),
        "top_image_sizes": [{"size": list(s), "count": c} for s, c in sizes.most_common(8)],
    }
    details = {
        "problem_files": problem_files,
        "empty_label_files": empty,
        "mixed_polygon_box_files": mixed,
        "unreadable_images": unreadable,
        "source_names": {p.name: roboflow_base_name(p.name) for p in images.values()},
    }
    return summary, details


def find_splits(root: Path) -> list[Path]:
    """Sub-folders of root that look like a split (have images/ and labels/)."""
    return [
        d
        for d in sorted(root.iterdir())
        if d.is_dir() and (d / "images").is_dir() and (d / "labels").is_dir()
    ]


def audit_dataset(root: Path) -> tuple[dict, dict]:
    splits = find_splits(root)
    if not splits:
        raise SystemExit(f"No split folders with images/ and labels/ found in {root}")
    summary: dict = {"dataset_folder_name": root.name, "splits": {}, "overlap": {}}
    details: dict = {"splits": {}}
    names: dict[str, set[str]] = {}
    for sp in splits:
        s, d = scan_split(sp)
        summary["splits"][sp.name] = s
        details["splits"][sp.name] = d
        names[sp.name] = set(d["source_names"].values())
    split_names = list(names)
    for i, a in enumerate(split_names):
        for b in split_names[i + 1 :]:
            shared = sorted(names[a] & names[b])
            summary["overlap"][f"{a}__{b}"] = {
                "n_shared_source_names": len(shared),
                "examples": shared[:10],
            }
    return summary, details


def print_report(summary: dict) -> None:
    for name, s in summary["splits"].items():
        print(f"\n== split: {name} ==")
        for key, value in s.items():
            if key in ("images_without_label",):
                continue
            print(f"  {key}: {value}")
    print("\n== overlap between splits (same source image name) ==")
    for key, value in summary["overlap"].items():
        print(f"  {key}: {value['n_shared_source_names']} shared  e.g. {value['examples'][:3]}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path, help="folder holding train/ test/ ...")
    ap.add_argument("--out", type=Path, default=Path("results/audit_summary.json"))
    ap.add_argument("--details", type=Path, default=Path("data/audit/details.json"))
    args = ap.parse_args(argv)
    summary, details = audit_dataset(args.root)
    for path, payload in ((args.out, summary), (args.details, details)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print_report(summary)
    print(f"\nSummary written to {args.out}")
    print(f"Per-file details written to {args.details} (not for committing)")


if __name__ == "__main__":
    main()
