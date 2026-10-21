import json
import sys
import types

import numpy as np
import pytest
from PIL import Image

from vehicle_detection.predict import list_images, predict_split


class Arr:
    """Mimics a torch tensor just enough for .cpu().numpy()."""

    def __init__(self, a):
        self.a = np.array(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


@pytest.fixture
def fake_ultralytics(monkeypatch):
    calls = {}

    class Result:
        def __init__(self, path):
            self.path = path
            self.orig_shape = (64, 96)  # height, width
            self.boxes = types.SimpleNamespace(
                xyxy=Arr([[1.234, 2.0, 50.0, 60.0], [5, 5, 10, 10]]),
                conf=Arr([0.912345678, 0.0021]),
                cls=Arr([1.0, 2.0]),
            )

    class YOLO:
        def __init__(self, weights):
            calls["weights"] = weights

        def predict(self, **kwargs):
            calls["kwargs"] = kwargs
            return (Result(p) for p in kwargs["source"])

    module = types.ModuleType("ultralytics")
    module.YOLO = YOLO
    module.__version__ = "8.4.174-test"
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    return calls


def make_images(root, split, names=("b.jpg", "a.jpg")):
    d = root / split / "images"
    d.mkdir(parents=True)
    for n in names:
        Image.new("RGB", (96, 64)).save(d / n)
    (d / "notes.txt").write_text("not an image")
    return d


def test_list_images_is_sorted_and_ignores_other_files(tmp_path):
    d = make_images(tmp_path, "val")
    assert [p.name for p in list_images(d)] == ["a.jpg", "b.jpg"]


def test_predictions_are_saved_with_the_documented_settings(tmp_path, fake_ultralytics):
    d = make_images(tmp_path, "val")
    out = tmp_path / "out" / "p.json"
    meta = predict_split(tmp_path / "w" / "best.pt", d, out)
    kw = fake_ultralytics["kwargs"]
    assert kw["conf"] == 0.001 and kw["iou"] == 0.7 and kw["imgsz"] == 672
    assert kw["max_det"] == 300 and kw["stream"] is True
    assert [p.split("/")[-1].split("\\")[-1] for p in kw["source"]] == ["a.jpg", "b.jpg"]
    data = json.loads(out.read_text())
    first = data["images"][0]
    assert first["file"] == "a.jpg" and (first["width"], first["height"]) == (96, 64)
    assert first["boxes"][0] == [1.23, 2.0, 50.0, 60.0] and first["classes"] == [1, 2]
    assert first["scores"][0] == 0.91235
    assert meta["n_images"] == 2 and meta["n_predictions"] == 4 and meta["split"] == "val"
    assert meta["weights"] == "best.pt" and str(tmp_path) not in out.read_text()


def test_validation_predictions_can_be_redone(tmp_path, fake_ultralytics):
    d = make_images(tmp_path, "val")
    out = tmp_path / "p.json"
    predict_split("w.pt", d, out)
    predict_split("w.pt", d, out)  # overwriting validation output is fine


def test_the_test_split_is_protected(tmp_path, fake_ultralytics):
    d = make_images(tmp_path, "test")
    out = tmp_path / "t.json"
    with pytest.raises(SystemExit):
        predict_split("w.pt", d, out)  # needs allow_test=True
    assert not out.exists()
    predict_split("w.pt", d, out, allow_test=True)
    with pytest.raises(SystemExit):
        predict_split("w.pt", d, out, allow_test=True)  # never twice


def test_an_empty_folder_is_an_error(tmp_path, fake_ultralytics):
    d = tmp_path / "val" / "images"
    d.mkdir(parents=True)
    with pytest.raises(SystemExit):
        predict_split("w.pt", d, tmp_path / "p.json")
