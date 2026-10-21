"""Group-aware train/val/test split. No torch needed.

Whole groups of near-duplicate images go to one split, so no scene appears on both sides.
"""

from __future__ import annotations

import numpy as np

from vehicle_detection.constants import SEED, SPLIT_NAMES, SPLIT_RATIOS

SIZE_BANDS = (2, 3, 5, 9, 17)  # group sizes 1 | 2 | 3-4 | 5-8 | 9-16 | 17+


def assign_groups(
    groups: list[int],
    features: np.ndarray,
    ratios: tuple[float, ...] = SPLIT_RATIOS,
    seed: int = SEED,
) -> list[int]:
    """Give every image a split index (0, 1, 2 ...), keeping each group together.

    groups[i] is the group id of image i (ids must be 0..k-1 with none missing).
    features[i] is a row of numbers to balance across splits (for example 1 for the image
    itself, then the number of boxes of each class), so classes stay spread out.

    Groups are dealt out separately within bands of similar size (singletons, pairs, small
    groups, big groups), largest band first. Inside a band, groups are shuffled (seeded) and
    each goes to the split that is furthest behind its target share of that band. This way
    every split receives its share of every kind of group, instead of one split taking all
    the big groups. The last band (the smallest groups) steers toward the overall target, which
    corrects the rounding left over from the big groups.
    """
    feats = np.asarray(features, dtype=float)
    if len(groups) != len(feats):
        raise ValueError("groups and features must have one row per image")
    n_groups = max(groups) + 1
    if set(groups) != set(range(n_groups)):
        raise ValueError("group ids must be 0..k-1 with none missing")
    if abs(sum(ratios) - 1) > 1e-9 or min(ratios) <= 0:
        raise ValueError("ratios must be positive and sum to 1")

    gfeat = np.zeros((n_groups, feats.shape[1]))
    for g, row in zip(groups, feats, strict=True):
        gfeat[g] += row
    gsize = np.bincount(groups, minlength=n_groups)
    band_of = np.digitize(gsize, SIZE_BANDS)

    rng = np.random.default_rng(seed)
    group_split = np.zeros(n_groups, dtype=int)
    overall = np.zeros((len(ratios), feats.shape[1]))
    overall_target = np.outer(ratios, np.where(gfeat.sum(axis=0) == 0, 1.0, gfeat.sum(axis=0)))
    bands = sorted(set(band_of.tolist()), reverse=True)
    for band in bands:
        members = np.flatnonzero(band_of == band)
        members = members[rng.permutation(len(members))]
        members = members[np.argsort(-gsize[members], kind="stable")]  # big first, ties shuffled
        total = gfeat[members].sum(axis=0)
        total[total == 0] = 1.0
        target = np.outer(ratios, total)
        current = np.zeros_like(target)
        last_band = band == bands[-1]
        for g in members:
            if last_band:
                behind = ((overall_target - overall) / overall_target).sum(axis=1)
            else:
                behind = ((target - current) / target).sum(axis=1)
            s = int(np.argmax(behind))
            current[s] += gfeat[g]
            overall[s] += gfeat[g]
            group_split[g] = s
    return [int(group_split[g]) for g in groups]


def split_imbalance(
    split_of: list[int], features: np.ndarray, ratios: tuple[float, ...] = SPLIT_RATIOS
) -> float:
    """Worst gap between a split's share of any feature total and its target ratio (0 = perfect)."""
    feats = np.asarray(features, dtype=float)
    arr = np.asarray(split_of)
    total = feats.sum(axis=0)
    total[total == 0] = 1.0
    worst = 0.0
    for s, ratio in enumerate(ratios):
        share = feats[arr == s].sum(axis=0) / total
        worst = max(worst, float(np.abs(share - ratio).max()))
    return worst


def best_split(
    groups: list[int],
    features: np.ndarray,
    ratios: tuple[float, ...] = SPLIT_RATIOS,
    n_seeds: int = 50,
) -> dict:
    """Try seeds 0..n_seeds-1 and keep the split with the smallest imbalance (ties: lowest seed).

    The score uses labels only, never model results, so choosing among seeds is not tuning on
    the test set.
    """
    if n_seeds < 1:
        raise ValueError("n_seeds must be at least 1")
    results = []
    for seed in range(n_seeds):
        split = assign_groups(groups, features, ratios, seed)
        results.append((split_imbalance(split, features, ratios), seed, split))
    score, seed, split = min(results, key=lambda r: (r[0], r[1]))
    scores = sorted(r[0] for r in results)
    return {
        "seed": seed,
        "split": split,
        "imbalance": score,
        "imbalance_seed_0": results[0][0],
        "imbalance_median": scores[len(scores) // 2],
    }


def cross_split_neighbours(
    dist: np.ndarray,
    split_of: list[int],
    thresholds: list[int],
    names: tuple[str, ...] = SPLIT_NAMES,
) -> dict[str, dict[str, int]]:
    """For each threshold: how many val/test images still have a close image in an earlier split."""
    arr = np.asarray(split_of)
    out: dict[str, dict[str, int]] = {}
    for t in thresholds:
        close = dist <= t
        row: dict[str, int] = {}
        for a in range(1, len(names)):
            for b in range(a):
                sub = close[np.ix_(arr == a, arr == b)]
                row[f"{names[a]}_images_near_{names[b]}"] = int(sub.any(axis=1).sum())
        out[str(t)] = row
    return out
