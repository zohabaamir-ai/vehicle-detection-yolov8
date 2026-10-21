"""Thin wrapper around Ultralytics training with resume support.

Needs the [train] extra (ultralytics). Ultralytics is imported only when training starts,
so the rest of the package and its tests never need torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vehicle_detection.runs import read_results_csv


@dataclass(frozen=True)
class RunConfig:
    name: str  # run folder name, e.g. "v8n_seed0"
    model: str  # pretrained weights name, e.g. "yolov8n.pt"
    data: str  # path to data.yaml
    project: str  # folder that will hold the run (put it on Google Drive in Colab)
    seed: int = 0
    epochs: int = 100
    imgsz: int = 672
    batch: int = 16
    workers: int = 2
    cache: str = "ram"
    device: int | str = 0


def run_dir(cfg: RunConfig) -> Path:
    return Path(cfg.project) / cfg.name


def train_kwargs(cfg: RunConfig) -> dict:
    """Exactly what is passed to YOLO.train for a fresh run. Everything else is left at the
    Ultralytics 8.4.174 defaults (optimizer 'auto', default augmentation, patience 100).
    """
    return {
        "data": cfg.data,
        "epochs": cfg.epochs,
        "imgsz": cfg.imgsz,
        "batch": cfg.batch,
        "seed": cfg.seed,
        "deterministic": True,
        "workers": cfg.workers,
        "cache": cfg.cache,
        "device": cfg.device,
        "project": cfg.project,
        "name": cfg.name,
        "exist_ok": True,
        "pretrained": True,
        "plots": True,
    }


def run_status(cfg: RunConfig) -> str:
    """'finished' (nothing to do), 'resume' (continue from last.pt) or 'new'."""
    d = run_dir(cfg)
    csv_path, last, best = d / "results.csv", d / "weights" / "last.pt", d / "weights" / "best.pt"
    if csv_path.exists() and best.exists():
        rows = read_results_csv(csv_path)
        if rows and rows[-1]["epoch"] >= cfg.epochs:
            return "finished"
    if last.exists():
        return "resume"
    return "new"


def run_training(cfg: RunConfig) -> Path:
    """Train, resume an interrupted run, or do nothing if the run is already finished."""
    status = run_status(cfg)
    d = run_dir(cfg)
    print(f"Run '{cfg.name}': {status}")
    if status == "finished":
        return d
    from ultralytics import YOLO

    if status == "resume":
        YOLO(str(d / "weights" / "last.pt")).train(resume=True, data=cfg.data)
    else:
        YOLO(cfg.model).train(**train_kwargs(cfg))
    return d
