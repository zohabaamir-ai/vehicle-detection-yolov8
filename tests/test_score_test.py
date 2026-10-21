import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from vehicle_detection.labels import parse_label_text
from vehicle_detection.prepare import main as prepare_main
from vehicle_detection.score_test import main, score_final


def noise_image(seed):
    rng = np.random.RandomState(seed)
    small = rng.randint(0, 256, (8, 12, 3), dtype=np.uint8)
    return Image.fromarray(small).resize((96, 64), Image.Resampling.BICUBIC)


@pytest.fixture
def setup(tmp_path, write_sample):
    """A prepared dataset, a weights file, a validation result, and predictions for both splits."""
    root = tmp_path / "raw"
    for i in range(60):
        split = "test" if i % 5 == 0 else "train"
        write_sample(
            root / split, f"img{i}", f"{i % 3} 0.5 0.5 0.3 0.3\n{(i + 1) % 3} 0.2 0.2 0.1 0.1\n"
        )
        noise_image(i).save(root / split / "images" / f"img{i}.jpg", quality=98)
    clean, manifest = tmp_path / "clean", tmp_path / "m.csv"
    prepare_main(
        ["--root", str(root), "--out", str(clean)]
        + ["--summary", str(tmp_path / "s.json"), "--manifest", str(manifest)]
    )

    def predictions(split):
        d = clean / split
        records = []
        for img in sorted((d / "images").glob("*.jpg")):
            parsed = parse_label_text((d / "labels" / f"{img.stem}.txt").read_text())
            boxes = [
                [(b.xc - b.w / 2) * 96, (b.yc - b.h / 2) * 64]
                + [(b.xc + b.w / 2) * 96, (b.yc + b.h / 2) * 64]
                for b in parsed.boxes
            ]
            records.append(
                {
                    "file": img.name,
                    "width": 96,
                    "height": 64,
                    "boxes": boxes,
                    "scores": [0.9] * len(boxes),
                    "classes": [b.cls for b in parsed.boxes],
                }
            )
        path = tmp_path / f"p_{split}.json"
        path.write_text(
            json.dumps({"meta": {"split": split, "ultralytics": "t"}, "images": records})
        )
        return path

    weights = tmp_path / "best.pt"
    weights.write_bytes(b"pretend these are weights")
    val_result = tmp_path / "val_result.json"
    val_result.write_text(
        json.dumps(
            {
                "split": "val",
                "conf_threshold": 0.29,
                "conf_threshold_was_chosen_here": True,
                "n_images": 9,
            }
        )
    )
    return {
        "clean": clean,
        "manifest": manifest,
        "weights": weights,
        "val_result": val_result,
        "preds": {s: predictions(s) for s in ("val", "test")},
        "tmp": tmp_path,
    }


def real(setup, out, **kw):
    args = dict(
        preds=setup["preds"]["test"],
        split_dir=setup["clean"] / "test",
        manifest=setup["manifest"],
        weights=setup["weights"],
        conf=0.29,
        val_result=setup["val_result"],
        out=out,
        n_boot=10,
    )
    args.update(kw)
    return score_final(**args)


def test_the_final_run_writes_a_result_that_records_the_exact_weights(setup):
    out = setup["tmp"] / "results" / "test_result.json"
    res = real(setup, out)
    saved = json.loads(out.read_text())
    assert saved["split"] == "test" and saved["conf_threshold"] == 0.29
    assert (
        saved["final_model"]["weights_sha256"]
        == hashlib.sha256(b"pretend these are weights").hexdigest()
    )
    assert saved["protocol"]["rehearsal"] is False and "scored_at_utc" in saved["protocol"]
    assert res["ranking_metrics"]["map50"] > 0.9
    assert str(setup["tmp"]) not in out.read_text()


def test_the_test_split_is_scored_only_once(setup):
    out = setup["tmp"] / "results" / "test_result.json"
    real(setup, out)
    first = out.read_text()
    with pytest.raises(SystemExit):
        real(setup, out)
    assert out.read_text() == first


def test_the_threshold_must_be_the_one_chosen_on_validation(setup):
    out = setup["tmp"] / "t.json"
    with pytest.raises(SystemExit):
        real(setup, out, conf=0.5)
    assert not out.exists()
    setup["val_result"].write_text(
        json.dumps(
            {"split": "val", "conf_threshold": 0.29, "conf_threshold_was_chosen_here": False}
        )
    )
    with pytest.raises(SystemExit):
        real(setup, out)
    assert not out.exists()


def test_a_real_run_refuses_the_validation_split_and_wrong_predictions(setup):
    out = setup["tmp"] / "t.json"
    with pytest.raises(SystemExit):
        real(setup, out, split_dir=setup["clean"] / "val", preds=setup["preds"]["val"])
    with pytest.raises(SystemExit):  # predictions made from another split
        real(setup, out, preds=setup["preds"]["val"])
    assert not out.exists()


def test_rehearsal_works_on_validation_repeatedly_and_never_on_test(setup):
    out = setup["tmp"] / "rehearsal.json"
    kw = dict(split_dir=setup["clean"] / "val", preds=setup["preds"]["val"], rehearsal=True)
    real(setup, out, **kw)
    real(setup, out, **kw)  # can be repeated
    assert json.loads(out.read_text())["protocol"]["rehearsal"] is True
    with pytest.raises(SystemExit):
        real(setup, out, rehearsal=True)  # test split with --rehearsal
    with pytest.raises(SystemExit):
        real(setup, setup["tmp"] / "test_result.json", **kw)  # must not write the final name


def test_cli_prints_the_banner_and_the_weights_hash(setup, capsys):
    out = setup["tmp"] / "r.json"
    main(
        ["--preds", str(setup["preds"]["val"]), "--split-dir", str(setup["clean"] / "val")]
        + [
            "--weights",
            str(setup["weights"]),
            "--conf",
            "0.29",
            "--manifest",
            str(setup["manifest"]),
        ]
        + [
            "--val-result",
            str(setup["val_result"]),
            "--out",
            str(out),
            "--rehearsal",
            "--boot",
            "5",
        ]
    )
    text = capsys.readouterr().out
    assert "REHEARSAL on validation" in text and "weights sha256:" in text
