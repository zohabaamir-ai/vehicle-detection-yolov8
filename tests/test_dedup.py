import json

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from vehicle_detection.dedup import (
    cluster,
    dhash_bits,
    distance_matrix,
    hash_image,
    main,
    select_pairs,
)


def noise_image(seed: int, size=(96, 64)) -> Image.Image:
    rng = np.random.RandomState(seed)
    small = rng.randint(0, 256, (8, 12, 3), dtype=np.uint8)
    return Image.fromarray(small).resize(size, Image.Resampling.BICUBIC)


def _dist(images):
    hs, hf = zip(*(hash_image_of(im) for im in images), strict=True)
    return distance_matrix(np.stack(hs), np.stack(hf))


def hash_image_of(im):
    return dhash_bits(im), dhash_bits(ImageOps.mirror(im))


def test_hash_has_64_bits_and_is_stable():
    h = dhash_bits(noise_image(1))
    assert h.shape == (64,)
    assert (h == dhash_bits(noise_image(1))).all()


def test_brightened_and_mirrored_copies_are_close_and_other_images_are_far():
    a = noise_image(1)
    bright = ImageEnhance.Brightness(a).enhance(1.15)
    mirrored = ImageOps.mirror(a)
    others = [noise_image(s) for s in range(10, 16)]
    d = _dist([a, bright, mirrored, *others])
    assert d[0, 1] <= 4
    assert d[0, 2] == 0  # the mirror is caught by the flipped hash
    assert d[0, 3:].min() >= 12


def test_cluster_groups_by_distance_and_by_key_and_is_transitive():
    dist = np.full((5, 5), 40, dtype=np.int16)
    np.fill_diagonal(dist, 0)
    dist[0, 1] = dist[1, 0] = 3  # 0~1
    dist[1, 2] = dist[2, 1] = 5  # 1~2 so 0,1,2 chain together
    groups = cluster(dist, threshold=6, keys=["a", "b", "c", "d", "d"])  # 3 and 4 share a key
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] == groups[4]
    assert groups[0] != groups[3]
    assert groups == [0, 0, 0, 1, 1]
    assert cluster(dist, threshold=2, keys=None) == [0, 1, 2, 3, 4]


def test_select_pairs_is_deterministic_and_skips_same_key():
    dist = np.full((4, 4), 40, dtype=np.int16)
    dist[0, 1] = dist[1, 0] = 8
    dist[2, 3] = dist[3, 2] = 9
    keys = ["x", "y", "z", "z"]  # pair (2, 3) shares a key and is skipped
    pairs = select_pairs(dist, keys, lo=6, hi=12, n=5, seed=0)
    assert pairs == [(0, 1, 8)]
    assert select_pairs(dist, keys, 6, 12, 5, seed=0) == pairs


def test_cli_finds_planted_train_test_duplicate(tmp_path, write_sample, capsys):
    root = tmp_path / "ds"
    a = noise_image(1)
    for i, im in enumerate([a, noise_image(2), noise_image(3)]):
        write_sample(root / "train", f"train{i}", "0 0.5 0.5 0.2 0.2\n")
        im.save(root / "train" / "images" / f"train{i}.jpg", quality=98)
    for name, im in [
        ("testdup", ImageEnhance.Brightness(a).enhance(1.1)),
        ("testnew", noise_image(9)),
    ]:
        write_sample(root / "test", name, "0 0.5 0.5 0.2 0.2\n")
        im.save(root / "test" / "images" / f"{name}.jpg", quality=98)
    out = tmp_path / "s.json"
    main(
        [
            "--root",
            str(root),
            "--out",
            str(out),
            "--pairs-out",
            str(tmp_path / "p.jpg"),
            "--thresholds",
            "0",
            "6",
        ]
    )
    data = json.loads(out.read_text())
    assert data["n_images"] == 5
    assert data["by_threshold"]["6"]["test_images_with_a_train_near_duplicate"] == 1
    assert data["by_threshold"]["6"]["n_groups"] == 4
    assert "threshold" in capsys.readouterr().out


def test_hash_image_reads_a_file(tmp_path):
    path = tmp_path / "x.jpg"
    noise_image(4).save(path, quality=98)
    h, hf = hash_image(path)
    assert h.shape == hf.shape == (64,)
