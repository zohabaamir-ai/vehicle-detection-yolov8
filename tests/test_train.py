import sys
import types

import pytest

from vehicle_detection import train
from vehicle_detection.train import RunConfig, run_status, run_training, train_kwargs


@pytest.fixture
def cfg(tmp_path):
    return RunConfig(
        name="v8n_seed0",
        model="yolov8n.pt",
        data="/content/data/clean/data.yaml",
        project=str(tmp_path / "runs"),
        seed=3,
    )


@pytest.fixture
def fake_yolo(monkeypatch):
    """Replace the ultralytics module with a recorder so no torch is needed."""
    calls = []

    class FakeYOLO:
        def __init__(self, weights):
            calls.append(("init", weights))

        def train(self, **kwargs):
            calls.append(("train", kwargs))

    module = types.ModuleType("ultralytics")
    module.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    return calls


def make_run(cfg, epochs_done=0, last=False, best=False):
    d = train.run_dir(cfg)
    (d / "weights").mkdir(parents=True)
    if epochs_done:
        lines = ["epoch,metrics/mAP50-95(B)"] + [
            f"{i},0.{i % 9 + 1}" for i in range(1, epochs_done + 1)
        ]
        (d / "results.csv").write_text("\n".join(lines) + "\n")
    if last:
        (d / "weights" / "last.pt").write_bytes(b"x")
    if best:
        (d / "weights" / "best.pt").write_bytes(b"x")
    return d


def test_train_kwargs_are_the_documented_settings(cfg):
    kw = train_kwargs(cfg)
    assert kw["data"] == cfg.data and kw["seed"] == 3 and kw["imgsz"] == 672
    assert kw["epochs"] == 100 and kw["batch"] == 16 and kw["pretrained"] is True
    assert kw["deterministic"] is True and kw["exist_ok"] is True
    assert kw["project"] == cfg.project and kw["name"] == "v8n_seed0"
    assert "resume" not in kw and "split" not in kw


def test_run_status_new_resume_finished(cfg):
    assert run_status(cfg) == "new"
    make_run(cfg, epochs_done=40, last=True)
    assert run_status(cfg) == "resume"


def test_run_status_finished_needs_all_epochs_and_best_weights(tmp_path):
    cfg = RunConfig("a", "yolov8n.pt", "d.yaml", str(tmp_path / "r"), epochs=5)
    make_run(cfg, epochs_done=5, last=True, best=True)
    assert run_status(cfg) == "finished"
    cfg2 = RunConfig("b", "yolov8n.pt", "d.yaml", str(tmp_path / "r"), epochs=5)
    make_run(cfg2, epochs_done=5, last=True, best=False)  # crashed before best.pt
    assert run_status(cfg2) == "resume"


def test_new_run_loads_pretrained_weights_and_trains(cfg, fake_yolo):
    run_training(cfg)
    assert fake_yolo[0] == ("init", "yolov8n.pt")
    assert fake_yolo[1] == ("train", train_kwargs(cfg))


def test_interrupted_run_resumes_from_last_pt_with_the_data_path(cfg, fake_yolo):
    d = make_run(cfg, epochs_done=40, last=True)
    run_training(cfg)
    assert fake_yolo[0] == ("init", str(d / "weights" / "last.pt"))
    assert fake_yolo[1] == ("train", {"resume": True, "data": cfg.data})


def test_finished_run_does_nothing_and_does_not_need_ultralytics(tmp_path, monkeypatch):
    cfg = RunConfig("a", "yolov8n.pt", "d.yaml", str(tmp_path / "r"), epochs=5)
    make_run(cfg, epochs_done=5, last=True, best=True)
    monkeypatch.setitem(sys.modules, "ultralytics", None)  # importing it would now fail
    assert run_training(cfg) == train.run_dir(cfg)
