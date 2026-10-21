"""Error analysis on one split: what goes wrong, by type and by object size. Needs Pillow only.

Usage:
    python -m vehicle_detection.errors --preds p_val.json --split-dir D:\\data\\clean\\val ^
        --conf 0.29 --out-dir data\\audit\\errors_val --out-json results\\error_analysis_val.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from vehicle_detection.audit import TINY_AREA
from vehicle_detection.constants import CLASS_NAMES
from vehicle_detection.evaluate import build_images, load_predictions
from vehicle_detection.metrics import ImageData, box_iou

SIZE_BINS = (("tiny", TINY_AREA), ("small", 0.01), ("medium", 0.05), ("large", 1.01))
COLORS = {
    "gt": (0, 200, 0),
    "correct": (30, 110, 255),
    "wrong_class": (255, 140, 0),
    "miss": (255, 230, 0),
    "false_alarm": (230, 20, 20),
}


def pair_boxes(im: ImageData, conf: float, iou_threshold: float = 0.5) -> dict:
    """Pair predictions (at or above conf) with ground truth one-to-one by IoU, ignoring class.

    Returns indices: correct and wrong_class are (gt_index, pred_index) pairs into the filtered
    prediction list `keep`; miss lists unpaired ground truth; false_alarm unpaired predictions.
    """
    keep = np.flatnonzero(np.asarray(im.pred_conf) >= conf)
    pb = np.asarray(im.pred_boxes, dtype=float).reshape(-1, 4)[keep]
    pc = np.asarray(im.pred_cls, dtype=int)[keep]
    gb = np.asarray(im.gt_boxes, dtype=float).reshape(-1, 4)
    gc = np.asarray(im.gt_cls, dtype=int)
    used_g, used_p = set(), set()
    correct, wrong = [], []
    if len(pb) and len(gb):
        iou = box_iou(gb, pb)
        pairs = np.argwhere(iou >= iou_threshold)
        pairs = pairs[np.argsort(-iou[pairs[:, 0], pairs[:, 1]], kind="stable")]
        for g, p in pairs:
            if g in used_g or p in used_p:
                continue
            used_g.add(int(g))
            used_p.add(int(p))
            (correct if gc[g] == pc[p] else wrong).append((int(g), int(p)))
    return {
        "keep": keep,
        "correct": correct,
        "wrong_class": wrong,
        "miss": [g for g in range(len(gb)) if g not in used_g],
        "false_alarm": [p for p in range(len(pb)) if p not in used_p],
    }


def area_fraction(box: np.ndarray, width: int, height: int) -> float:
    """Share of the image covered by an xyxy pixel box."""
    return float((box[2] - box[0]) * (box[3] - box[1]) / (width * height))


def size_bin(fraction: float) -> str:
    for name, upper in SIZE_BINS:
        if fraction < upper:
            return name
    return SIZE_BINS[-1][0]


def analyse(images: list[ImageData], sizes: dict[str, tuple[int, int]], conf: float) -> dict:
    """Totals by error type, detection rate by object size, class confusions, per-image ranking."""
    totals: Counter = Counter()
    by_size: dict[str, Counter] = {name: Counter() for name, _ in SIZE_BINS}
    by_class = {n: Counter() for n in CLASS_NAMES}
    confusions: Counter = Counter()
    ranking = []
    for im in images:
        w, h = sizes[im.name]
        p = pair_boxes(im, conf)
        gc, pc = np.asarray(im.gt_cls, dtype=int), np.asarray(im.pred_cls, dtype=int)[p["keep"]]
        for g, _ in p["correct"]:
            totals["correct"] += 1
            by_size[size_bin(area_fraction(im.gt_boxes[g], w, h))]["correct"] += 1
            by_class[CLASS_NAMES[gc[g]]]["correct"] += 1
        for g, q in p["wrong_class"]:
            totals["wrong_class"] += 1
            by_size[size_bin(area_fraction(im.gt_boxes[g], w, h))]["wrong_class"] += 1
            by_class[CLASS_NAMES[gc[g]]]["wrong_class"] += 1
            confusions[f"{CLASS_NAMES[gc[g]]} -> {CLASS_NAMES[pc[q]]}"] += 1
        for g in p["miss"]:
            totals["miss"] += 1
            by_size[size_bin(area_fraction(im.gt_boxes[g], w, h))]["miss"] += 1
            by_class[CLASS_NAMES[gc[g]]]["miss"] += 1
        for q in p["false_alarm"]:
            totals["false_alarm"] += 1
            by_class[CLASS_NAMES[pc[q]]]["false_alarm"] += 1
        n_err = len(p["wrong_class"]) + len(p["miss"]) + len(p["false_alarm"])
        ranking.append((n_err, im.name, len(gc)))
    ranking.sort(key=lambda r: (-r[0], r[1]))
    n_gt = totals["correct"] + totals["wrong_class"] + totals["miss"]
    size_rows = {}
    for name, c in by_size.items():
        n = sum(c.values())
        size_rows[name] = {
            "n_objects": n,
            "detected_correctly": round(c["correct"] / n, 3) if n else None,
            "wrong_class": c["wrong_class"],
            "missed": c["miss"],
        }
    return {
        "conf_threshold": conf,
        "n_images": len(images),
        "n_objects": n_gt,
        "totals": dict(totals),
        "detection_rate_by_object_size": size_rows,
        "per_class": {n: dict(c) for n, c in by_class.items()},
        "class_confusions": dict(confusions.most_common()),
        "worst_images": [{"image": n, "errors": e, "objects": o} for e, n, o in ranking if e > 0],
    }


def _box(b) -> tuple[float, float, float, float]:
    """Corners in drawing order, so a malformed (inverted) box cannot crash the drawing."""
    return (min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3]))


def draw_errors(path: Path, im: ImageData, conf: float, title: str = "") -> Image.Image:
    """Image with ground truth (green), correct (blue), wrong class (orange), misses (yellow)
    and false alarms (red) drawn on it."""
    img = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(img)
    p = pair_boxes(im, conf)
    pb = np.asarray(im.pred_boxes, dtype=float).reshape(-1, 4)[p["keep"]]
    pc = np.asarray(im.pred_cls, dtype=int)[p["keep"]]
    ps = np.asarray(im.pred_conf, dtype=float)[p["keep"]]
    gb, gc = np.asarray(im.gt_boxes, dtype=float).reshape(-1, 4), np.asarray(im.gt_cls, dtype=int)
    for b in gb:
        d.rectangle(_box(b), outline=COLORS["gt"], width=1)
    for _g, q in p["correct"]:
        d.rectangle(_box(pb[q]), outline=COLORS["correct"], width=2)
    for g, q in p["wrong_class"]:
        d.rectangle(_box(pb[q]), outline=COLORS["wrong_class"], width=3)
        d.text(
            (pb[q][0] + 2, pb[q][1] + 2),
            f"{CLASS_NAMES[gc[g]]}>{CLASS_NAMES[pc[q]]}",
            fill=COLORS["wrong_class"],
        )
    for g in p["miss"]:
        d.rectangle(_box(gb[g]), outline=COLORS["miss"], width=3)
        d.text((gb[g][0] + 2, gb[g][1] + 2), f"MISS {CLASS_NAMES[gc[g]]}", fill=COLORS["miss"])
    for q in p["false_alarm"]:
        d.rectangle(_box(pb[q]), outline=COLORS["false_alarm"], width=3)
        d.text(
            (pb[q][0] + 2, pb[q][1] + 2),
            f"FP {CLASS_NAMES[pc[q]]} {ps[q]:.2f}",
            fill=COLORS["false_alarm"],
        )
    if title:
        d.rectangle((0, 0, 8 * len(title) + 6, 14), fill=(0, 0, 0))
        d.text((3, 1), title, fill=(255, 255, 255))
    return img


def make_sheets(items: list[Image.Image], out_dir: Path, per_sheet: int = 6, cell: int = 560):
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for s in range(0, len(items), per_sheet):
        chunk = items[s : s + per_sheet]
        cols = 3
        rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cell, rows * cell), "white")
        for k, im in enumerate(chunk):
            thumb = im.copy()
            thumb.thumbnail((cell - 4, cell - 4))
            sheet.paste(thumb, ((k % cols) * cell + 2, (k // cols) * cell + 2))
        path = out_dir / f"errors_sheet_{s // per_sheet + 1}.jpg"
        sheet.save(path, quality=88)
        files.append(path.name)
    return files


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", required=True, type=Path)
    ap.add_argument("--split-dir", required=True, type=Path)
    ap.add_argument("--conf", required=True, type=float)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--n", type=int, default=20, help="how many worst images to draw")
    ap.add_argument("--per-sheet", type=int, default=6)
    args = ap.parse_args(argv)
    preds = load_predictions(args.preds)
    images = build_images(args.split_dir, preds)
    sizes = {n: (r["width"], r["height"]) for n, r in preds["images"].items()}
    result = analyse(images, sizes, args.conf)
    by_name = {im.name: im for im in images}
    drawn = []
    for k, row in enumerate(result["worst_images"][: args.n], start=1):
        im = by_name[row["image"]]
        title = f"#{k} errors={row['errors']} objects={row['objects']}"
        drawn.append(draw_errors(args.split_dir / "images" / im.name, im, args.conf, title))
    result["sheets"] = make_sheets(drawn, args.out_dir, args.per_sheet)
    result["legend"] = {k: "RGB" + str(v) for k, v in COLORS.items()}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    t = result["totals"]
    print(
        f"objects: {result['n_objects']}  correct: {t.get('correct', 0)}  "
        f"wrong class: {t.get('wrong_class', 0)}  missed: {t.get('miss', 0)}  "
        f"false alarms: {t.get('false_alarm', 0)}"
    )
    for name, row in result["detection_rate_by_object_size"].items():
        rate = row["detected_correctly"]
        print(f"  {name:7s} objects {row['n_objects']:4d}  detected correctly {rate}")
    print("  confusions:", result["class_confusions"])
    print(f"Sheets in {args.out_dir}: {result['sheets']}")


if __name__ == "__main__":
    main()
