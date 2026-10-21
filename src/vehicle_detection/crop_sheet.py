"""Make a contact sheet of cropped objects of one class, to eyeball what a class means.

Usage:
    python -m vehicle_detection.crop_sheet --root D:\\path\\to\\ObjectsDataset --cls 1
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image

from vehicle_detection.audit import IMAGE_EXTS, find_splits
from vehicle_detection.labels import Box, parse_label_text


def box_to_pixels(box: Box, width: int, height: int, pad: float = 0.1) -> tuple[int, int, int, int]:
    """Normalised box -> (left, top, right, bottom) pixels with padding, clipped to the image."""
    bw, bh = box.w * width, box.h * height
    cx, cy = box.xc * width, box.yc * height
    left = max(0, int(cx - bw / 2 - pad * bw))
    top = max(0, int(cy - bh / 2 - pad * bh))
    right = min(width, int(cx + bw / 2 + pad * bw) + 1)
    bottom = min(height, int(cy + bh / 2 + pad * bh) + 1)
    return left, top, right, bottom


def collect_boxes(root: Path, cls_id: int) -> list[tuple[Path, Box]]:
    items: list[tuple[Path, Box]] = []
    for split in find_splits(root):
        labels = {p.stem: p for p in (split / "labels").glob("*.txt")}
        for img in sorted((split / "images").iterdir()):
            if img.suffix.lower() not in IMAGE_EXTS or img.stem not in labels:
                continue
            parsed = parse_label_text(labels[img.stem].read_text(errors="replace"))
            items.extend((img, b) for b in parsed.boxes if b.cls == cls_id)
    return items


def make_sheet(items: list[tuple[Path, Box]], n: int, seed: int, cell: int = 160) -> Image.Image:
    rng = random.Random(seed)
    chosen = rng.sample(items, min(n, len(items)))
    cols = 6
    rows = max(1, (len(chosen) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * cell, rows * cell), "white")
    for k, (path, box) in enumerate(chosen):
        with Image.open(path) as im:
            crop = im.convert("RGB").crop(box_to_pixels(box, *im.size))
        crop.thumbnail((cell - 4, cell - 4))
        sheet.paste(crop, ((k % cols) * cell + 2, (k // cols) * cell + 2))
    return sheet


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--cls", required=True, type=int, help="class id to show")
    ap.add_argument("--n", type=int, default=36)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    items = collect_boxes(args.root, args.cls)
    if not items:
        raise SystemExit(f"No boxes with class id {args.cls} found.")
    out = args.out or Path(f"data/audit/class_{args.cls}.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    make_sheet(items, args.n, args.seed).save(out, quality=90)
    print(
        f"{len(items)} boxes of class {args.cls}; sheet of {min(args.n, len(items))} saved to {out}"
    )


if __name__ == "__main__":
    main()
