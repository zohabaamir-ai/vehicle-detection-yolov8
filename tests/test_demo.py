import types

import numpy as np
import pytest
from PIL import Image

from vehicle_detection import demo
from vehicle_detection.demo import (
    Detection,
    count_by_class,
    counts_table,
    draw_detections,
    filter_detections,
    render,
    run_model,
)

DETS = [
    Detection((10, 10, 60, 60), 0.9, 0),
    Detection((70, 20, 120, 90), 0.5, 1),
    Detection((5, 70, 50, 120), 0.1, 2),
]


class Arr:
    def __init__(self, a):
        self.a = np.array(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class FakeModel:
    """Mimics model.predict: honours conf and counts how often it is called."""

    def __init__(self):
        self.calls = 0

    def predict(self, source, conf, iou, imgsz, device, verbose):
        self.calls += 1
        raw = [
            ([10, 10, 60, 60], 0.9, 0),
            ([70, 20, 120, 90], 0.5, 1),
            ([5, 70, 50, 120], 0.1, 2),
            ([1, 1, 4, 4], 0.02, 0),  # below the floor: the real model never returns it
        ]
        keep = [r for r in raw if r[1] >= conf]
        boxes = types.SimpleNamespace(
            xyxy=Arr([r[0] for r in keep]),
            conf=Arr([r[1] for r in keep]),
            cls=Arr([float(r[2]) for r in keep]),
        )
        return [types.SimpleNamespace(boxes=boxes)]


def test_filtering_and_counting():
    assert len(filter_detections(DETS, 0.29)) == 2
    assert len(filter_detections(DETS, 0.9)) == 1  # a detection equal to the threshold stays
    assert count_by_class(filter_detections(DETS, 0.29)) == {"car": 1, "motorcycle": 1, "truck": 0}
    assert count_by_class([]) == {"car": 0, "motorcycle": 0, "truck": 0}


def test_counts_table_has_a_total_row():
    assert counts_table({"car": 2, "motorcycle": 1, "truck": 0}) == [
        ["car", 2],
        ["motorcycle", 1],
        ["truck", 0],
        ["total", 3],
    ]


def test_drawing_marks_the_image_and_leaves_the_original_alone():
    img = Image.new("RGB", (160, 140), (128, 128, 128))
    out = draw_detections(img, DETS[:1])
    assert out.size == img.size
    assert out.getpixel((10, 30)) == demo.COLORS["car"]  # left edge of the car box
    assert img.getpixel((10, 30)) == (128, 128, 128)
    assert draw_detections(img, []).tobytes() == img.tobytes()


def test_render_handles_no_image_and_applies_the_threshold():
    shown, table = render(None, None, 0.29)
    assert shown is None and table[-1] == ["total", 0]
    img = Image.new("RGB", (160, 140))
    shown, table = render(img, DETS, 0.29)
    assert shown is not None and table[-1] == ["total", 2]
    assert render(img, DETS, 0.05)[1][-1] == ["total", 3]


def test_run_model_returns_detections_above_the_floor_only():
    model = FakeModel()
    dets = run_model(model, Image.new("RGB", (160, 140)))
    assert [d.cls for d in dets] == [0, 1, 2]
    assert dets[0] == Detection((10.0, 10.0, 60.0, 60.0), pytest.approx(0.9), 0)
    assert model.calls == 1


def test_the_app_detects_once_and_the_slider_only_refilters(tmp_path, monkeypatch):
    pytest.importorskip("gradio")
    gradio_client = pytest.importorskip("gradio_client")
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    path = tmp_path / "road.png"
    Image.new("RGB", (160, 140), (90, 90, 90)).save(path)

    model = FakeModel()
    app = demo.build_app(model)
    app.launch(prevent_thread_lock=True, quiet=True)
    try:
        client = gradio_client.Client(app.local_url, verbose=False)
        shown, table, *_ = client.predict(
            gradio_client.handle_file(str(path)), 0.29, api_name="/detect"
        )
        rows = {r[0]: r[1] for r in table["data"]}
        assert rows == {"car": 1, "motorcycle": 1, "truck": 0, "total": 2}
        assert model.calls == 1
        shown, table = client.predict(
            gradio_client.handle_file(str(path)), 0.05, api_name="/refilter"
        )
        assert {r[0]: r[1] for r in table["data"]}["total"] == 3
        shown, table = client.predict(
            gradio_client.handle_file(str(path)), 0.95, api_name="/refilter"
        )
        assert {r[0]: r[1] for r in table["data"]}["total"] == 0
        assert model.calls == 1  # moving the slider never ran the model again
    finally:
        app.close()
