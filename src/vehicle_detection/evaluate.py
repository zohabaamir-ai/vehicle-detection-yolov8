"""Score saved predictions against the labels of one split. No torch needed.

Usage (validation, also choosing the confidence threshold):
    python -m vehicle_detection.evaluate --preds p_val.json --split-dir D:\\data\\clean\\val ^
        --choose-conf --out results\\val_eval.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from vehicle_detection.audit import IMAGE_EXTS
from vehicle_detection.constants import CLASS_NAMES, ISOLATED_MIN_DISTANCE
from vehicle_detection.labels import parse_label_text
from vehicle_detection.metrics import (
    ImageData,
    bootstrap,
    build_tables,
    choose_threshold,
    confusion_matrix,
    f1_curve,
    metrics_from_tables,
    restrict,
)


def load_predictions(path: Path | str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"meta": data["meta"], "images": {r["file"]: r for r in data["images"]}}


def build_images(split_dir: Path, preds: dict) -> list[ImageData]:
    """Pair every predicted image with its label file. Both sides must cover the same images."""
    on_disk = {p.name for p in (split_dir / "images").iterdir() if p.suffix.lower() in IMAGE_EXTS}
    predicted = set(preds["images"])
    if on_disk != predicted:
        raise SystemExit(
            f"Predictions and images differ: {len(predicted - on_disk)} predicted but not on "
            f"disk, {len(on_disk - predicted)} on disk but not predicted."
        )
    images = []
    for name in sorted(predicted):
        rec = preds["images"][name]
        w, h = rec["width"], rec["height"]
        label = split_dir / "labels" / f"{Path(name).stem}.txt"
        parsed = parse_label_text(label.read_text(encoding="utf-8"))
        gt = np.array(
            [
                [
                    (b.xc - b.w / 2) * w,
                    (b.yc - b.h / 2) * h,
                    (b.xc + b.w / 2) * w,
                    (b.yc + b.h / 2) * h,
                ]
                for b in parsed.boxes
            ]
        ).reshape(-1, 4)
        images.append(
            ImageData(
                name=name,
                gt_boxes=gt,
                gt_cls=np.array([b.cls for b in parsed.boxes], dtype=int),
                pred_boxes=np.array(rec["boxes"], dtype=float).reshape(-1, 4),
                pred_conf=np.array(rec["scores"], dtype=float),
                pred_cls=np.array(rec["classes"], dtype=int),
            )
        )
    return images


def read_manifest(path: Path | str, split: str) -> dict[str, dict]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return {r["file"]: r for r in csv.DictReader(f) if r["split"] == split}


def count_errors(images: list[ImageData], n_classes: int, conf: float) -> dict:
    """How far off the per-image object counts are (what a traffic counter would report)."""
    pred = np.zeros((len(images), n_classes))
    true = np.zeros((len(images), n_classes))
    for i, im in enumerate(images):
        keep = np.asarray(im.pred_conf) >= conf
        for c in np.asarray(im.pred_cls)[keep]:
            pred[i, c] += 1
        for c in np.asarray(im.gt_cls):
            true[i, c] += 1
    per_class = {}
    for c in range(n_classes):
        diff = pred[:, c] - true[:, c]
        per_class[CLASS_NAMES[c]] = {
            "mean_abs_error_per_image": float(np.abs(diff).mean()),
            "mean_error_per_image": float(diff.mean()),
            "true_total": int(true[:, c].sum()),
            "predicted_total": int(pred[:, c].sum()),
        }
    diff_all = pred.sum(axis=1) - true.sum(axis=1)
    return {
        "per_class": per_class,
        "all_vehicles": {
            "mean_abs_error_per_image": float(np.abs(diff_all).mean()),
            "mean_error_per_image": float(diff_all.mean()),
            "true_mean_per_image": float(true.sum(axis=1).mean()),
        },
    }


def _named(values: list[float]) -> dict:
    return {CLASS_NAMES[i]: v for i, v in enumerate(values)}


def _public(m: dict) -> dict:
    """Metrics with per-class lists turned into dicts keyed by class name."""
    out = {k: v for k, v in m.items() if k not in ("ap50", "ap50_95", "precision", "recall", "f1")}
    out["ap50"], out["ap50_95"] = _named(m["ap50"]), _named(m["ap50_95"])
    out["n_gt"] = _named(m["n_gt"])
    if "precision" in m:
        out["precision"], out["recall"], out["f1"] = (
            _named(m[k]) for k in ("precision", "recall", "f1")
        )
    return out


def _rename_ci(ci: dict) -> dict:
    """Bootstrap keys like 'ap50_class1' become 'ap50_motorcycle'."""
    out = {}
    for key, value in ci.items():
        for c, name in enumerate(CLASS_NAMES):
            key = key.replace(f"_class{c}", f"_{name}")
        out[key] = value
    return out


def evaluate(
    preds_path: Path | str,
    split_dir: Path | str,
    manifest_csv: Path | str,
    conf: float | None = None,
    choose_conf: bool = False,
    n_boot: int = 1000,
    seed: int = 0,
    reference: dict | None = None,
) -> dict:
    if (conf is None) == (not choose_conf):
        raise SystemExit("Give either --conf or --choose-conf (not both, not neither).")
    split_dir = Path(split_dir)
    preds = load_predictions(preds_path)
    images = build_images(split_dir, preds)
    n_classes = len(CLASS_NAMES)
    tables = build_tables(images, n_classes)

    manifest = read_manifest(manifest_csv, split_dir.name)
    missing = [im.name for im in images if im.name not in manifest]
    if missing:
        raise SystemExit(f"{len(missing)} images are not in the manifest, e.g. {missing[:3]}")
    group_ids = np.array([int(manifest[im.name]["group"]) for im in images])
    nearest = np.array([float(manifest[im.name]["nearest_train_dist"] or "nan") for im in images])

    curve = f1_curve(tables)
    chosen = choose_threshold(curve) if choose_conf else None
    conf = chosen["conf"] if chosen else float(conf)

    point = metrics_from_tables(tables)
    at_conf = metrics_from_tables(tables, conf_threshold=conf)
    isolated_mask = nearest > ISOLATED_MIN_DISTANCE
    out = {
        "split": split_dir.name,
        "n_images": len(images),
        "predictions_meta": preds["meta"],
        "class_names": list(CLASS_NAMES),
        "conf_threshold": conf,
        "conf_threshold_was_chosen_here": bool(chosen),
        "threshold_choice": chosen,
        "ranking_metrics": {"map50": point["map50"], "map50_95": point["map50_95"]},
        "ranking_metrics_per_class": {
            "ap50": _named(point["ap50"]),
            "ap50_95": _named(point["ap50_95"]),
        },
        "at_threshold": _public(at_conf),
        "confusion_matrix": {
            "rows_are": "predicted class (last = background)",
            "columns_are": "true class (last = background)",
            "labels": [*CLASS_NAMES, "background"],
            "counts": confusion_matrix(images, n_classes, conf).tolist(),
        },
        "counting": count_errors(images, n_classes, conf),
        "bootstrap": {
            "n_resamples": n_boot,
            "by_image": _rename_ci(bootstrap(tables, n_boot, seed, conf)),
            "by_group": _rename_ci(bootstrap(tables, n_boot, seed, conf, group_ids)),
        },
        "curve": curve,
    }
    if isolated_mask.sum() >= 2:
        sub = restrict(tables, isolated_mask)
        out["isolated_subset"] = {
            "definition": f"nearest train image more than {ISOLATED_MIN_DISTANCE} hash bits away",
            "n_images": int(isolated_mask.sum()),
            "metrics": _public(metrics_from_tables(sub, conf_threshold=conf)),
            "bootstrap_by_group": _rename_ci(
                bootstrap(sub, n_boot, seed, conf, group_ids[isolated_mask])
            ),
        }
    if reference:
        out["reference_check"] = {
            k: {"ours": point[k], "reference": v, "difference": point[k] - v}
            for k, v in reference.items()
        }
    return out


def print_summary(res: dict) -> None:
    ci = res["bootstrap"]["by_image"]
    gci = res["bootstrap"]["by_group"]
    print(
        f"split: {res['split']}  images: {res['n_images']}  conf threshold: {res['conf_threshold']}"
    )
    for key in ("map50", "map50_95"):
        v = res["ranking_metrics"][key]
        print(
            f"  {key:9s} {v:.4f}   95% CI by image [{ci[key]['lo']:.3f}, {ci[key]['hi']:.3f}]"
            f"   by group [{gci[key]['lo']:.3f}, {gci[key]['hi']:.3f}]"
        )
    for name in res["class_names"]:
        a = res["ranking_metrics_per_class"]
        t = res["at_threshold"]
        print(
            f"  {name:11s} AP50 {a['ap50'][name]:.3f}  AP50-95 {a['ap50_95'][name]:.3f}  "
            f"P {t['precision'][name]:.3f}  R {t['recall'][name]:.3f}  F1 {t['f1'][name]:.3f}"
        )
    t = res["at_threshold"]
    print(
        f"  macro P {t['macro_precision']:.3f}  R {t['macro_recall']:.3f}  F1 {t['macro_f1']:.3f}"
    )
    c = res["counting"]["all_vehicles"]
    print(
        f"  counting: all vehicles off by {c['mean_abs_error_per_image']:.2f} per image "
        f"(true mean {c['true_mean_per_image']:.2f})"
    )
    if "isolated_subset" in res:
        iso = res["isolated_subset"]
        print(f"  isolated subset ({iso['n_images']} images): ", end="")
        print(f"mAP50 {iso['metrics']['map50']:.3f}, mAP50-95 {iso['metrics']['map50_95']:.3f}")
    for k, v in res.get("reference_check", {}).items():
        print(
            f"  check vs Ultralytics {k}: ours {v['ours']:.4f} vs {v['reference']:.4f} "
            f"(difference {v['difference']:+.4f})"
        )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", required=True, type=Path)
    ap.add_argument("--split-dir", required=True, type=Path, help="folder like clean/val")
    ap.add_argument("--manifest", type=Path, default=Path("results/split_manifest.csv"))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--choose-conf", action="store_true")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reference-map50", type=float, default=None)
    ap.add_argument("--reference-map50-95", type=float, default=None)
    args = ap.parse_args(argv)
    reference = {}
    if args.reference_map50 is not None:
        reference["map50"] = args.reference_map50
    if args.reference_map50_95 is not None:
        reference["map50_95"] = args.reference_map50_95
    res = evaluate(
        args.preds,
        args.split_dir,
        args.manifest,
        args.conf,
        args.choose_conf,
        args.boot,
        args.seed,
        reference,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print_summary(res)
    print(f"Full results written to {args.out}")


if __name__ == "__main__":
    main()
