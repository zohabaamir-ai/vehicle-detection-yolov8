import json

import numpy as np
from PIL import Image

from vehicle_detection.errors import (
    analyse,
    area_fraction,
    draw_errors,
    main,
    pair_boxes,
    size_bin,
)
from vehicle_detection.metrics import ImageData


def im(name, gt, preds):
    """gt: list of (cls, box); preds: list of (cls, conf, box). Boxes are xyxy pixels."""
    return ImageData(
        name,
        np.array([b for _, b in gt], dtype=float).reshape(-1, 4),
        np.array([c for c, _ in gt], dtype=int),
        np.array([b for _, _, b in preds], dtype=float).reshape(-1, 4),
        np.array([s for _, s, _ in preds], dtype=float),
        np.array([c for c, _, _ in preds], dtype=int),
    )


def toy():
    a = im("a.jpg", [(0, [10, 10, 60, 60])], [(0, 0.9, [10, 10, 60, 60])])  # correct
    b = im("b.jpg", [(2, [10, 10, 90, 90])], [(0, 0.9, [10, 10, 90, 90])])  # truck seen as car
    c = im("c.jpg", [(1, [20, 20, 40, 40])], [(2, 0.8, [60, 60, 90, 90])])  # miss + false alarm
    return [a, b, c], {n: (100, 100) for n in ("a.jpg", "b.jpg", "c.jpg")}


def test_pairing_finds_correct_wrong_class_misses_and_false_alarms():
    images, _ = toy()
    assert len(pair_boxes(images[0], 0.5)["correct"]) == 1
    assert len(pair_boxes(images[1], 0.5)["wrong_class"]) == 1
    p = pair_boxes(images[2], 0.5)
    assert p["miss"] == [0] and p["false_alarm"] == [0]
    assert pair_boxes(images[0], 0.95)["miss"] == [0]  # prediction below the threshold is ignored


def test_summary_counts_and_ranking():
    images, sizes = toy()
    res = analyse(images, sizes, 0.5)
    assert res["totals"] == {"correct": 1, "wrong_class": 1, "miss": 1, "false_alarm": 1}
    assert res["n_objects"] == 3
    assert res["class_confusions"] == {"truck -> car": 1}
    assert [w["image"] for w in res["worst_images"]] == ["c.jpg", "b.jpg"]  # a.jpg has no errors
    assert res["worst_images"][0]["errors"] == 2
    sizes_row = res["detection_rate_by_object_size"]
    assert sizes_row["large"]["n_objects"] == 2 and sizes_row["medium"]["n_objects"] == 1
    assert res["per_class"]["motorcycle"]["miss"] == 1
    assert res["per_class"]["truck"]["false_alarm"] == 1


def test_size_bins():
    assert area_fraction(np.array([0, 0, 10, 10]), 100, 100) == 0.01
    assert [size_bin(x) for x in (0.001, 0.005, 0.02, 0.5)] == ["tiny", "small", "medium", "large"]


def test_drawing_returns_an_image_of_the_same_size(tmp_path):
    images, _ = toy()
    path = tmp_path / "c.jpg"
    Image.new("RGB", (100, 100), "gray").save(path)
    out = draw_errors(path, images[2], 0.5, "title")
    assert out.size == (100, 100)
    assert out.getpixel((90, 90)) != (128, 128, 128)  # something was drawn


def test_cli_writes_json_and_sheets(tmp_path, capsys):
    images, sizes = toy()
    split = tmp_path / "val"
    for sub in ("images", "labels"):
        (split / sub).mkdir(parents=True)
    records = []
    for x in images:
        Image.new("RGB", (100, 100), "gray").save(split / "images" / x.name)
        lines = [
            f"{c} {(b[0] + b[2]) / 200} {(b[1] + b[3]) / 200} {(b[2] - b[0]) / 100} "
            f"{(b[3] - b[1]) / 100}"
            for c, b in zip(x.gt_cls, x.gt_boxes, strict=True)
        ]
        (split / "labels" / f"{x.name[:-4]}.txt").write_text("\n".join(lines) + "\n")
        records.append(
            {
                "file": x.name,
                "width": 100,
                "height": 100,
                "boxes": x.pred_boxes.tolist(),
                "scores": x.pred_conf.tolist(),
                "classes": x.pred_cls.tolist(),
            }
        )
    preds = tmp_path / "p.json"
    preds.write_text(json.dumps({"meta": {}, "images": records}))
    out_json, out_dir = tmp_path / "e.json", tmp_path / "sheets"
    main(
        ["--preds", str(preds), "--split-dir", str(split), "--conf", "0.5"]
        + ["--out-dir", str(out_dir), "--out-json", str(out_json), "--n", "5", "--per-sheet", "2"]
    )
    saved = json.loads(out_json.read_text())
    assert saved["sheets"] == ["errors_sheet_1.jpg"] and (out_dir / "errors_sheet_1.jpg").exists()
    assert saved["totals"]["miss"] == 1 and str(tmp_path) not in out_json.read_text()
    assert "wrong class: 1" in capsys.readouterr().out


def test_drawing_survives_a_box_with_inverted_corners(tmp_path):
    x = im("d.jpg", [(0, [10, 10, 40, 40])], [(0, 0.9, [60, 60, 50, 50])])  # x1 < x0, y1 < y0
    path = tmp_path / "d.jpg"
    Image.new("RGB", (100, 100), "gray").save(path)
    assert draw_errors(path, x, 0.5).size == (100, 100)
