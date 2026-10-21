"""Gradio demo: upload an image, move the confidence slider, see boxes and counts per class.

Runs on CPU. Needs the [app] extra (gradio and ultralytics).

Usage:
    python -m vehicle_detection.demo --weights weights\\vehicle_detector_yolov8n.pt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from vehicle_detection.constants import CLASS_NAMES

DEFAULT_CONF = 0.29  # chosen on the validation split, see the README
MIN_CONF = 0.05  # detections below this are never computed, so the slider cannot go lower
COLORS = {"car": (30, 110, 255), "motorcycle": (255, 140, 0), "truck": (200, 30, 200)}
NOTICE = (
    "Not for enforcement, safety or any decision about people. The model misses many small, "
    "distant vehicles (see the README). Uploaded images are processed on this machine; Gradio "
    "keeps temporary copies in your temp folder."
)


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    conf: float
    cls: int


def filter_detections(dets: list[Detection], conf: float) -> list[Detection]:
    return [d for d in dets if d.conf >= conf]


def count_by_class(dets: list[Detection]) -> dict[str, int]:
    counts = dict.fromkeys(CLASS_NAMES, 0)
    for d in dets:
        counts[CLASS_NAMES[d.cls]] += 1
    return counts


def counts_table(counts: dict[str, int]) -> list[list]:
    rows = [[name, counts[name]] for name in CLASS_NAMES]
    rows.append(["total", sum(counts.values())])
    return rows


def draw_detections(image: Image.Image, dets: list[Detection]) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    width = max(2, round(min(out.size) / 200))
    for d in dets:
        color = COLORS[CLASS_NAMES[d.cls]]
        draw.rectangle(d.box, outline=color, width=width)
        label = f"{CLASS_NAMES[d.cls]} {d.conf:.2f}"
        x, y = d.box[0], max(0, d.box[1] - 12)
        draw.rectangle((x, y, x + 6 * len(label) + 4, y + 12), fill=color)
        draw.text((x + 2, y), label, fill=(255, 255, 255))
    return out


def run_model(model, image: Image.Image, iou: float = 0.7, imgsz: int = 672) -> list[Detection]:
    """Detect once at the lowest allowed confidence; the slider then filters the result."""
    result = model.predict(
        source=image.convert("RGB"),
        conf=MIN_CONF,
        iou=iou,
        imgsz=imgsz,
        device="cpu",
        verbose=False,
    )[0]
    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    return [
        Detection(tuple(float(v) for v in b), float(c), int(k))
        for b, c, k in zip(xyxy, conf, cls, strict=True)
    ]


def load_model(weights: str | Path):
    from ultralytics import YOLO

    return YOLO(str(weights))


def render(image: Image.Image | None, dets: list[Detection] | None, conf: float):
    """What the page shows for the current image, detections and slider value."""
    if image is None or dets is None:
        return None, counts_table(dict.fromkeys(CLASS_NAMES, 0))
    shown = filter_detections(dets, conf)
    return draw_detections(image, shown), counts_table(count_by_class(shown))


def build_app(model):
    import gradio as gr

    def on_image(image, conf):
        if image is None:
            return None, counts_table(dict.fromkeys(CLASS_NAMES, 0)), None
        dets = run_model(model, image)
        annotated, table = render(image, dets, conf)
        return annotated, table, dets

    def on_slider(image, conf, dets):
        annotated, table = render(image, dets, conf)
        return annotated, table

    with gr.Blocks(title="Vehicle detection (car, motorcycle, truck)") as app:
        gr.Markdown(
            "# Vehicle detection: car, motorcycle, truck\n"
            "Upload a road scene, then move the slider to trade missed vehicles against false "
            f"alarms. Default {DEFAULT_CONF} is the value chosen on validation data.\n\n"
            f"**{NOTICE}**"
        )
        with gr.Row():
            inp = gr.Image(type="pil", label="Image")
            out = gr.Image(label="Detections", type="pil")
        conf = gr.Slider(
            MIN_CONF, 0.95, value=DEFAULT_CONF, step=0.01, label="Confidence threshold"
        )
        table = gr.Dataframe(
            headers=["class", "count"],
            datatype=["str", "number"],
            interactive=False,
            label="Counts",
        )
        state = gr.State(None)
        inp.change(on_image, [inp, conf], [out, table, state], api_name="detect")
        conf.change(on_slider, [inp, conf, state], [out, table], api_name="refilter")
    return app


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, default=Path("weights/vehicle_detector_yolov8n.pt"))
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args(argv)
    if not args.weights.exists():
        raise SystemExit(f"Weights not found: {args.weights}")
    build_app(load_model(args.weights)).launch(server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
