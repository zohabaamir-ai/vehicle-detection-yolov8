import json

import numpy as np
import pytest
import yaml
from PIL import Image

from vehicle_detection.prepare import main as prepare_main
from vehicle_detection.verify import main, verify_dataset


def noise_image(seed):
    rng = np.random.RandomState(seed)
    small = rng.randint(0, 256, (8, 12, 3), dtype=np.uint8)
    return Image.fromarray(small).resize((96, 64), Image.Resampling.BICUBIC)


@pytest.fixture
def built(tmp_path, write_sample):
    root = tmp_path / "raw"
    for i in range(30):
        split = "test" if i % 5 == 0 else "train"
        write_sample(
            root / split, f"img{i}", f"{i % 3} 0.5 0.5 0.2 0.2\n{(i + 1) % 3} 0.3 0.3 0.1 0.1\n"
        )
        noise_image(i).save(root / split / "images" / f"img{i}.jpg", quality=98)
    out = tmp_path / "clean"
    manifest, summary = tmp_path / "m.csv", tmp_path / "s.json"
    prepare_main(
        [
            "--root",
            str(root),
            "--out",
            str(out),
            "--summary",
            str(summary),
            "--manifest",
            str(manifest),
        ]
    )
    return out, manifest, summary


def test_a_correct_dataset_verifies(built):
    out, manifest, summary = built
    report = verify_dataset(out, manifest, summary)
    assert report["ok"] and report["n_manifest_images"] == 30


def test_changed_missing_and_extra_files_are_caught(built):
    out, manifest, summary = built
    images = sorted((out / "train" / "images").glob("*.jpg"))
    images[0].write_bytes(images[0].read_bytes() + b"x")  # changed content
    images[1].unlink()  # missing image
    noise_image(99).save(out / "val" / "images" / "extra.jpg")  # not in the manifest
    report = verify_dataset(out, manifest, summary)
    assert not report["ok"]
    assert report["hash_mismatch"] == [images[0].name]
    assert report["missing"] == [images[1].name]
    assert report["unexpected"] == ["val/extra.jpg"]


def test_wrong_box_counts_are_caught(built):
    out, manifest, summary = built
    label = sorted((out / "train" / "labels").glob("*.txt"))[0]
    label.write_text(label.read_text() + "0 0.5 0.5 0.1 0.1\n")
    report = verify_dataset(out, manifest, summary)
    assert not report["ok"] and report["count_problems"]


def test_cli_exit_code(built, capsys):
    out, manifest, summary = built
    with pytest.raises(SystemExit) as exc:
        main(["--root", str(out), "--manifest", str(manifest), "--summary", str(summary)])
    assert exc.value.code == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_stale_or_missing_data_yaml_is_caught(built):
    out, manifest, summary = built
    cfg = yaml.safe_load((out / "data.yaml").read_text())
    cfg["names"] = {0: "Car", 1: "Motorbike", 2: "Truck"}  # what an old dataset copy would say
    (out / "data.yaml").write_text(yaml.safe_dump(cfg))
    report = verify_dataset(out, manifest, summary)
    assert not report["ok"] and "data.yaml names" in report["count_problems"][0]
    (out / "data.yaml").unlink()
    assert "data.yaml is missing" in verify_dataset(out, manifest, summary)["count_problems"]
