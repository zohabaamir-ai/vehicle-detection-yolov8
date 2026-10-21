import numpy as np
import pytest

from vehicle_detection.split import (
    assign_groups,
    best_split,
    cross_split_neighbours,
    split_imbalance,
)


def make_problem(n_groups=150, seed=0):
    rng = np.random.default_rng(seed)
    groups, feats = [], []
    for g in range(n_groups):
        for _ in range(int(rng.integers(1, 9))):
            groups.append(g)
            counts = rng.integers(0, 6, size=3) * (rng.random(3) < [0.8, 0.5, 0.3])
            feats.append([1, *counts])
    return groups, np.array(feats)


def test_groups_are_never_split_across_splits():
    groups, feats = make_problem()
    split = assign_groups(groups, feats, seed=1)
    seen = {}
    for g, s in zip(groups, split, strict=True):
        assert seen.setdefault(g, s) == s


def test_split_is_close_to_target_ratios_for_images_and_classes():
    groups, feats = make_problem()
    split = np.array(assign_groups(groups, feats, (0.7, 0.15, 0.15), seed=1))
    for col in range(feats.shape[1]):
        total = feats[:, col].sum()
        for s, ratio in enumerate((0.7, 0.15, 0.15)):
            share = feats[split == s, col].sum() / total
            assert abs(share - ratio) < 0.04, (col, s, share)


def test_every_split_contains_every_class():
    groups, feats = make_problem()
    split = np.array(assign_groups(groups, feats, seed=3))
    for s in range(3):
        assert (feats[split == s, 1:].sum(axis=0) > 0).all()


def test_deterministic_and_seed_dependent():
    groups, feats = make_problem()
    a = assign_groups(groups, feats, seed=5)
    assert a == assign_groups(groups, feats, seed=5)
    assert a != assign_groups(groups, feats, seed=6)


def test_bad_inputs_are_rejected():
    with pytest.raises(ValueError):
        assign_groups([0, 2], np.ones((2, 1)))  # group 1 missing
    with pytest.raises(ValueError):
        assign_groups([0, 1], np.ones((3, 1)))  # length mismatch
    with pytest.raises(ValueError):
        assign_groups([0, 1], np.ones((2, 1)), ratios=(0.5, 0.2, 0.2))


def test_cross_split_neighbours_counts_images_with_a_close_earlier_image():
    dist = np.full((4, 4), 50, dtype=np.int16)
    np.fill_diagonal(dist, 0)
    dist[0, 2] = dist[2, 0] = 5  # image 2 (val) is close to image 0 (train)
    dist[1, 3] = dist[3, 1] = 15  # image 3 (test) is close to image 1 (train) only at 16+
    out = cross_split_neighbours(dist, [0, 0, 1, 2], [12, 16])
    assert out["12"]["val_images_near_train"] == 1
    assert out["12"]["test_images_near_train"] == 0
    assert out["16"]["test_images_near_train"] == 1
    assert out["16"]["test_images_near_val"] == 0


def make_heavy_tail(seed=0):
    """Like the real data: mostly single images, plus a few big groups (up to 42 images)."""
    rng = np.random.default_rng(seed)
    sizes = [1] * 500 + [2] * 50 + [3] * 20 + [5] * 14 + [8] * 8 + [13] * 5 + [20] * 3 + [42]
    groups, feats = [], []
    for g, size in enumerate(sizes):
        for _ in range(size):
            groups.append(g)
            counts = rng.integers(0, 5, size=3) * (rng.random(3) < [0.8, 0.5, 0.4])
            feats.append([1, *counts])
    return sizes, np.array(groups), np.array(feats)


def test_big_groups_are_shared_out_not_all_given_to_train():
    sizes, groups, feats = make_heavy_tail()
    split = np.array(assign_groups(groups.tolist(), feats, seed=42))
    in_multi = np.array([sizes[g] > 1 for g in groups])
    for s, (low, high) in enumerate([(0.55, 0.80), (0.08, 0.25), (0.08, 0.25)]):
        share = (in_multi & (split == s)).sum() / in_multi.sum()
        assert low < share < high, (s, share)
    for s, ratio in enumerate((0.70, 0.15, 0.15)):
        assert abs((split == s).mean() - ratio) < 0.02


def test_split_imbalance_is_zero_for_a_perfect_split_and_large_for_a_bad_one():
    feats = np.ones((20, 1))
    perfect = [0] * 14 + [1] * 3 + [2] * 3
    assert split_imbalance(perfect, feats, (0.7, 0.15, 0.15)) < 1e-9
    assert split_imbalance([0] * 20, feats, (0.7, 0.15, 0.15)) == pytest.approx(0.3)


def test_best_split_is_no_worse_than_seed_0_and_is_deterministic():
    sizes, groups, feats = make_heavy_tail()
    first = best_split(groups.tolist(), feats, n_seeds=20)
    assert first["imbalance"] <= first["imbalance_seed_0"] + 1e-12
    assert first["imbalance"] <= first["imbalance_median"] + 1e-12
    assert best_split(groups.tolist(), feats, n_seeds=20)["seed"] == first["seed"]
    assert split_imbalance(first["split"], feats) == pytest.approx(first["imbalance"])
    assert 0 <= first["seed"] < 20


def test_best_split_with_one_seed_equals_assign_groups_seed_0():
    sizes, groups, feats = make_heavy_tail()
    one = best_split(groups.tolist(), feats, n_seeds=1)
    assert one["split"] == assign_groups(groups.tolist(), feats, seed=0)
    with pytest.raises(ValueError):
        best_split(groups.tolist(), feats, n_seeds=0)
