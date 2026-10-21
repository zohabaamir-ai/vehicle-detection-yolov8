"""Export a model's detections for one split to JSON.

Needs the [train] extra (ultralytics). Ultralytics is imported only when this runs, so the
rest of the package and its tests never need torch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vehicle_detection.audit import IMAGE_EXTS


def list_images(images_dir: Path) -> list[Path]:
    return sorted(p for p in Path(images_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def predict_split(
    weights: str | Path,
    images_dir: str | Path,
    out_json: str | Path,
    conf: float = 0.001,
    iou: float = 0.7,
    imgsz: int = 672,
    max_det: int = 300,
    device: int | str = 0,
    batch: int = 16,
    allow_test: bool = False,
) -> dict:
    """Run the model on every image in images_dir and save boxes (pixels, xyxy), scores and
    class ids. Uses a very low confidence (0.001) so mAP can be computed from the file; the
    operating threshold is applied later, when scoring.

    The test split is protected: it needs allow_test=True, and an existing test file is never
    overwritten.
    """
    images_dir, out = Path(images_dir), Path(out_json)
    split = images_dir.parent.name
    if split == "test":
        if not allow_test:
            raise SystemExit("Refusing to predict the test split without allow_test=True.")
        if out.exists():
            raise SystemExit(f"{out} already exists. The test split is predicted only once.")
    paths = list_images(images_dir)
    if not paths:
        raise SystemExit(f"No images found in {images_dir}")

    import ultralytics
    from ultralytics import YOLO

    model = YOLO(str(weights))
    records = []
    stream = model.predict(
        source=[str(p) for p in paths],
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        max_det=max_det,
        device=device,
        batch=batch,
        stream=True,
        verbose=False,
    )
    for r in stream:
        height, width = r.orig_shape
        b = r.boxes
        records.append(
            {
                "file": Path(r.path).name,
                "width": int(width),
                "height": int(height),
                "boxes": np.round(b.xyxy.cpu().numpy(), 2).tolist(),
                "scores": np.round(b.conf.cpu().numpy(), 5).tolist(),
                "classes": b.cls.cpu().numpy().astype(int).tolist(),
            }
        )
    meta = {
        "split": split,
        "weights": Path(weights).name,
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "max_det": max_det,
        "ultralytics": ultralytics.__version__,
        "n_images": len(records),
        "n_predictions": sum(len(r["scores"]) for r in records),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "images": records}) + "\n", encoding="utf-8")
    return meta
