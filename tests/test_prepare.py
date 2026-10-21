import csv
import json
from collections import defaultdict

import numpy as np
import pytest
import yaml
from PIL import Image, ImageEnhance

from vehicle_detection.labels import Box, parse_label_text
from vehicle_detection.prepare import clip_box, format_label, main


def noise_image(seed: int) -> Image.Image:
    rng = np.random.RandomState(seed)
    small = rng.randint(0, 256, (8, 12, 3), dtype=np.uint8)
    return Image.fromarray(small).resize((96, 64), Image.Resampling.BICUBIC)


def label_for(i: int) -> str:
    cls = i % 3
    return f"{cls} 0.5 0.5 0.2 0.2\n{(cls + 1) % 3} 0.3 0.3 0.1 0.1\n"


@pytest.fixture
def messy_dataset(tmp_path, write_sample):
    """60 distinct images (20 of them have a brightened near-duplicate) plus bad labels."""
    root = tmp_path / "raw"
    for i in range(60):
        split = "test" if i % 6 == 0 else "train"
        write_sample(root / split, f"img{i}", label_for(i))
        noise_image(i).save(root / split / "images" / f"img{i}.jpg", quality=98)
    for i in range(20):  # near-duplicates, placed in the OTHER original split
        split = "train" if i % 6 == 0 else "test"
        write_sample(root / split, f"dup{i}", label_for(i))
        dup = ImageEnhance.Brightness(noise_image(i)).enhance(1.1)
        dup.save(root / split / "images" / f"dup{i}.jpg", quality=98)
    # label oddities
    (root / "train" / "labels" / "img1.txt").write_text("2 0.2 0.2 0.6 0.2 0.6 0.8 0.2 0.8\n")
    (root / "train" / "labels" / "img2.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n1 0.2 0.2 0.6 0.2 0.6 0.8 0.2 0.8\n"
    )  # mixed
    (root / "train" / "labels" / "img4.txt").write_text("0 0.95 0.5 0.2 0.2\n")  # sticks out
    return root


def run(root, out, tmp_path):
    main(
        [
            "--root",
            str(root),
            "--out",
            str(out),
            "--summary",
            str(tmp_path / "s.json"),
            "--manifest",
            str(tmp_path / "m.csv"),
        ]
    )


def test_clip_box_and_format_label():
    c = clip_box(Box(0, 0.95, 0.5, 0.2, 0.2))
    assert c is not None and abs(c.xc + c.w / 2 - 1.0) < 1e-9
    assert clip_box(Box(0, 1.5, 0.5, 0.1, 0.1)) is None
    assert format_label([Box(1, 0.5, 0.5, 0.25, 0.5)]) == "1 0.500000 0.500000 0.250000 0.500000\n"


def test_prepare_builds_a_clean_leak_free_dataset(messy_dataset, tmp_path):
    out = tmp_path / "clean"
    run(messy_dataset, out, tmp_path)

    cfg = yaml.safe_load((out / "data.yaml").read_text())
    assert cfg["names"] == {0: "car", 1: "motorcycle", 2: "truck"}
    assert "path" not in cfg and cfg["val"] == "val/images"

    n_out = 0
    for split in ("train", "val", "test"):
        imgs = sorted((out / split / "images").glob("*.jpg"))
        lbls = sorted((out / split / "labels").glob("*.txt"))
        assert [p.stem for p in imgs] == [p.stem for p in lbls]
        n_out += len(imgs)
        for lbl in lbls:
            text = lbl.read_text()
            parsed = parse_label_text(text)
            assert parsed.problems == [] and parsed.n_polygons == 0 and not parsed.is_mixed
            assert all(len(line.split()) == 5 for line in text.splitlines())
    assert n_out == 80  # nothing lost, nothing duplicated

    rows = list(csv.DictReader((tmp_path / "m.csv").open()))
    assert len(rows) == 80 and {len(r["sha256"]) for r in rows} == {64}
    splits_of_group = defaultdict(set)
    for r in rows:
        splits_of_group[r["group"]].add(r["split"])
    assert all(len(s) == 1 for s in splits_of_group.values())  # no scene in two splits
    by_file = {r["file"]: r for r in rows}
    for i in range(20):  # every planted duplicate sits in the same split as its original
        assert by_file[f"dup{i}.jpg"]["split"] == by_file[f"img{i}.jpg"]["split"]

    for r in rows:  # distance to the nearest train image is recorded for val/test only
        if r["split"] == "train":
            assert r["nearest_train_dist"] == ""
        else:
            assert int(r["nearest_train_dist"]) > 12  # leak-free at the grouping threshold

    summary = json.loads((tmp_path / "s.json").read_text())
    assert set(summary["isolated_images"]) == {"definition", "val", "test"}
    assert summary["n_seed_candidates"] == 50
    assert 0 <= summary["split_seed"] < 50
    assert summary["imbalance_chosen"] <= summary["imbalance_seed_0"] + 1e-9
    assert summary["polygon_lines_converted_to_boxes"] == 2
    assert summary["label_files_that_mixed_polygons_and_boxes"] == 1
    assert summary["residual_cross_split_neighbours"]["12"]["test_images_near_train"] == 0
    assert str(tmp_path) not in (tmp_path / "s.json").read_text()


def test_prepare_refuses_non_empty_output_folder(messy_dataset, tmp_path):
    out = tmp_path / "clean"
    out.mkdir()
    (out / "x.txt").write_text("keep me")
    with pytest.raises(SystemExit):
        run(messy_dataset, out, tmp_path)
    assert (out / "x.txt").read_text() == "keep me"


def test_prepare_is_deterministic(messy_dataset, tmp_path):
    run(messy_dataset, tmp_path / "a", tmp_path)
    first = (tmp_path / "m.csv").read_text()
    run(messy_dataset, tmp_path / "b", tmp_path)
    assert (tmp_path / "m.csv").read_text() == first
