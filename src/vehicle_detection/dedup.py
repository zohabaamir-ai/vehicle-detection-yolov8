"""Near-duplicate image detection with a perceptual hash. No torch needed.

Usage:
    python -m vehicle_detection.dedup --root D:\\path\\to\\ObjectsDataset
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from vehicle_detection.audit import IMAGE_EXTS, find_splits, roboflow_base_name

HASH_SIDE = 8  # 8x8 differences -> 64-bit hash


def dhash_bits(img: Image.Image, side: int = HASH_SIDE) -> np.ndarray:
    """Difference hash: is each pixel brighter than its left neighbour (tiny grayscale copy)?"""
    small = img.convert("L").resize((side + 1, side), Image.Resampling.LANCZOS)
    a = np.asarray(small, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).reshape(-1)


def hash_image(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Hash of the image and of its horizontal mirror (augmentation often flips images)."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return dhash_bits(im), dhash_bits(ImageOps.mirror(im))


def hamming_matrix(a: np.ndarray, b: np.ndarray, chunk: int = 256) -> np.ndarray:
    """All pairwise Hamming distances between rows of a and rows of b."""
    out = np.empty((len(a), len(b)), dtype=np.int16)
    for i in range(0, len(a), chunk):
        out[i : i + chunk] = (a[i : i + chunk, None, :] != b[None, :, :]).sum(axis=2)
    return out


def distance_matrix(hashes: np.ndarray, flipped: np.ndarray) -> np.ndarray:
    """Distance between images, taking the best of 'as is' and 'mirrored'."""
    return np.minimum(hamming_matrix(hashes, hashes), hamming_matrix(hashes, flipped))


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def cluster(dist: np.ndarray, threshold: int, keys: list[str] | None = None) -> list[int]:
    """Group images: close in hash distance, or sharing an exact key (same source name).

    Returns one group id per image, numbered 0.. in order of first appearance.
    """
    n = len(dist)
    uf = UnionFind(n)
    ii, jj = np.nonzero(np.triu(dist <= threshold, k=1))
    for i, j in zip(ii.tolist(), jj.tolist(), strict=True):
        uf.union(i, j)
    if keys is not None:
        first: dict[str, int] = {}
        for i, key in enumerate(keys):
            if key in first:
                uf.union(first[key], i)
            else:
                first[key] = i
    ids: dict[int, int] = {}
    return [ids.setdefault(uf.find(i), len(ids)) for i in range(n)]


def list_images(root: Path) -> list[tuple[Path, str]]:
    """(image path, original split name) for every image under root, in a stable order."""
    items = []
    for split in find_splits(root):
        for p in sorted((split / "images").iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                items.append((p, split.name))
    return items


def select_pairs(
    dist: np.ndarray, keys: list[str], lo: int, hi: int, n: int, seed: int
) -> list[tuple[int, int, int]]:
    """Random (i, j, distance) pairs with lo <= distance <= hi and different source names."""
    ii, jj = np.nonzero(np.triu((dist >= lo) & (dist <= hi), k=1))
    pairs = [
        (i, j, int(dist[i, j]))
        for i, j in zip(ii.tolist(), jj.tolist(), strict=True)
        if keys[i] != keys[j]
    ]
    return random.Random(seed).sample(pairs, min(n, len(pairs)))


def pairs_sheet(items: list[Path], pairs: list[tuple[int, int, int]], cell: int = 160):
    cols = 3
    rows = max(1, (len(pairs) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * (2 * cell + 10), rows * (cell + 4)), "white")
    draw = ImageDraw.Draw(sheet)
    for k, (i, j, d) in enumerate(pairs):
        x0, y0 = (k % cols) * (2 * cell + 10), (k // cols) * (cell + 4)
        for m, idx in enumerate((i, j)):
            with Image.open(items[idx]) as im:
                im = im.convert("RGB")
                im.thumbnail((cell - 2, cell - 2))
                sheet.paste(im, (x0 + m * cell, y0))
        draw.rectangle((x0, y0, x0 + 44, y0 + 12), fill="black")
        draw.text((x0 + 2, y0), f"d={d}", fill="white")
    return sheet


def analyse(
    root: Path, thresholds: list[int]
) -> tuple[dict, np.ndarray, list[Path], list[str], list[str]]:
    items = list_images(root)
    paths = [p for p, _ in items]
    splits = [s for _, s in items]
    keys = [roboflow_base_name(p.name) for p in paths]
    hs, hf = zip(*(hash_image(p) for p in paths), strict=True)
    dist = distance_matrix(np.stack(hs), np.stack(hf))
    summary: dict = {"n_images": len(paths), "hash": "dhash 8x8 + mirror", "by_threshold": {}}
    split_arr = np.array(splits)
    for t in thresholds:
        groups = np.array(cluster(dist, t, keys))
        sizes = np.bincount(groups)
        cross = sum(1 for g in range(len(sizes)) if len(set(split_arr[groups == g])) > 1)
        row = {
            "n_groups": int(len(sizes)),
            "largest_group": int(sizes.max()),
            "n_groups_spanning_original_splits": cross,
        }
        if {"train", "test"} <= set(splits):
            close = dist <= t
            tr, te = split_arr == "train", split_arr == "test"
            same_key = np.array([[a == b for b in keys] for a in keys])
            row["test_images_with_a_train_near_duplicate"] = int(
                ((close | same_key)[np.ix_(te, tr)]).any(axis=1).sum()
            )
        summary["by_threshold"][str(t)] = row
    return summary, dist, paths, keys, splits


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--thresholds", type=int, nargs="+", default=[0, 2, 4, 6, 8, 10, 12])
    ap.add_argument("--pairs-range", type=int, nargs=2, default=[6, 12], metavar=("LO", "HI"))
    ap.add_argument("--pairs-out", type=Path, default=Path("data/audit/pairs.jpg"))
    ap.add_argument("--out", type=Path, default=Path("results/dedup_summary.json"))
    args = ap.parse_args(argv)
    summary, dist, paths, keys, _ = analyse(args.root, args.thresholds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{summary['n_images']} images, hash = {summary['hash']}")
    for t, row in summary["by_threshold"].items():
        print(f"  threshold {t:>2}: {row}")
    pairs = select_pairs(dist, keys, args.pairs_range[0], args.pairs_range[1], 12, seed=0)
    if pairs:
        args.pairs_out.parent.mkdir(parents=True, exist_ok=True)
        pairs_sheet(paths, pairs).save(args.pairs_out, quality=90)
        print(f"{len(pairs)} example pairs (distance {args.pairs_range}) saved to {args.pairs_out}")
    print(f"Summary written to {args.out}")


if __name__ == "__main__":
    main()
