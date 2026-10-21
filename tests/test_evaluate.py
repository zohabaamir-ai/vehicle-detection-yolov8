import json

import numpy as np
import pytest
from PIL import Image

from vehicle_detection.evaluate import evaluate, main
from vehicle_detection.labels import parse_label_text
from vehicle_detection.prepare import main as prepare_main


def noise_image(seed):
    rng = np.random.RandomState(seed)
    small = rng.randint(0, 256, (8, 12, 3), dtype=np.uint8)
    return Image.fromarray(small).resize((96, 64), Image.Resampling.BICUBIC)


@pytest.fixture
def scored(tmp_path, write_sample):
    """A small prepared dataset plus predictions that are right except for planted mistakes."""
    root = tmp_path / "raw"
    for i in range(40):
        split = "test" if i % 5 == 0 else "train"
        write_sample(
            root / split,
            f"img{i}",
            f"{i % 3} 0.5 0.5 0.3 0.3\n{(i + 1) % 3} 0.2 0.2 0.1 0.1\n",
        )
        noise_image(i).save(root / split / "images" / f"img{i}.jpg", quality=98)
    clean, manifest, summary = tmp_path / "clean", tmp_path / "m.csv", tmp_path / "s.json"
    prepare_main(
        ["--root", str(root), "--out", str(clean)]
        + ["--summary", str(summary), "--manifest", str(manifest)]
    )
    val = clean / "val"
    records = []
    for img in sorted((val / "images").glob("*.jpg")):
        parsed = parse_label_text((val / "labels" / f"{img.stem}.txt").read_text())
        boxes, scores, classes = [], [], []
        for b in parsed.boxes:
            boxes.append(
                [(b.xc - b.w / 2) * 96, (b.yc - b.h / 2) * 64]
                + [(b.xc + b.w / 2) * 96, (b.yc + b.h / 2) * 64]
            )
            scores.append(0.9)
            classes.append(b.cls)
        boxes.append([0, 0, 5, 5])  # a low-confidence false alarm on every image
        scores.append(0.05)
        classes.append(0)
        records.append(
            {
                "file": img.name,
                "width": 96,
                "height": 64,
                "boxes": boxes,
                "scores": scores,
                "classes": classes,
            }
        )
    preds = tmp_path / "p_val.json"
    preds.write_text(json.dumps({"meta": {"split": "val"}, "images": records}))
    return preds, val, manifest, records


def test_choosing_the_threshold_on_validation(scored):
    preds, val, manifest, records = scored
    res = evaluate(preds, val, manifest, choose_conf=True, n_boot=30)
    assert res["n_images"] == len(records) >= 2
    assert res["conf_threshold_was_chosen_here"] is True
    assert 0.06 <= res["conf_threshold"] <= 0.9  # between the false alarms and the real boxes
    assert res["ranking_metrics"]["map50"] > 0.9
    assert res["at_threshold"]["macro_f1"] == pytest.approx(1.0)
    assert res["counting"]["all_vehicles"]["mean_abs_error_per_image"] == 0.0
    assert set(res["at_threshold"]["precision"]) == {"car", "motorcycle", "truck"}
    cm = np.array(res["confusion_matrix"]["counts"])
    assert cm.shape == (4, 4) and np.trace(cm[:3, :3]) == cm.sum() - cm[:, 3].sum() - cm[3, :].sum()
    assert {"map50", "ap50_motorcycle", "macro_f1"} <= set(res["bootstrap"]["by_image"])
    assert res["bootstrap"]["by_group"]["map50"]["lo"] <= res["ranking_metrics"]["map50"] + 1e-9
    assert res["isolated_subset"]["n_images"] == len(records)  # random images: all isolated
    json.dumps(res)  # everything must be JSON-serialisable


def test_a_fixed_threshold_is_used_as_given(scored):
    preds, val, manifest, _ = scored
    res = evaluate(preds, val, manifest, conf=0.5, n_boot=10)
    assert res["conf_threshold"] == 0.5 and res["conf_threshold_was_chosen_here"] is False
    assert res["threshold_choice"] is None


def test_exactly_one_of_conf_or_choose_conf(scored):
    preds, val, manifest, _ = scored
    with pytest.raises(SystemExit):
        evaluate(preds, val, manifest, n_boot=5)
    with pytest.raises(SystemExit):
        evaluate(preds, val, manifest, conf=0.5, choose_conf=True, n_boot=5)


def test_predictions_must_cover_exactly_the_images_in_the_split(scored, tmp_path):
    preds, val, manifest, records = scored
    short = tmp_path / "short.json"
    short.write_text(json.dumps({"meta": {}, "images": records[1:]}))
    with pytest.raises(SystemExit):
        evaluate(short, val, manifest, conf=0.5, n_boot=5)


def test_cli_writes_results_and_prints_the_reference_check(scored, tmp_path, capsys):
    preds, val, manifest, _ = scored
    out = tmp_path / "res" / "val_eval.json"
    main(
        ["--preds", str(preds), "--split-dir", str(val), "--manifest", str(manifest)]
        + ["--out", str(out), "--choose-conf", "--boot", "20", "--reference-map50", "0.99"]
    )
    text = capsys.readouterr().out
    assert "check vs Ultralytics map50" in text and "counting:" in text
    saved = json.loads(out.read_text())
    assert saved["reference_check"]["map50"]["reference"] == 0.99
    assert str(tmp_path) not in out.read_text()
