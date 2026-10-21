"""Detection metrics: mAP, precision/recall, F1 curve, confusion matrix, bootstrap intervals.

No torch needed. Matching and AP follow the conventions Ultralytics uses for validation
(IoU 0.50 to 0.95 in 10 steps, 101-point interpolated AP), so numbers computed here from saved
predictions can be compared with `model.val()` output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10, dtype=np.float32).astype(np.float64)
EPS = 1e-16
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


@dataclass
class ImageData:
    """Ground truth and predictions for one image. Boxes are xyxy in pixels."""

    name: str
    gt_boxes: np.ndarray  # (m, 4)
    gt_cls: np.ndarray  # (m,) ints
    pred_boxes: np.ndarray  # (n, 4)
    pred_conf: np.ndarray  # (n,)
    pred_cls: np.ndarray  # (n,) ints


@dataclass
class Tables:
    """All predictions of a dataset in flat arrays, ready for fast resampling."""

    img_idx: np.ndarray  # (P,) which image each prediction belongs to
    conf: np.ndarray  # (P,)
    cls: np.ndarray  # (P,)
    tp: np.ndarray  # (P, 10) correct at each IoU threshold
    gt_counts: np.ndarray  # (n_images, n_classes) ground-truth boxes per image and class
    names: list[str]  # image names, same order as gt_counts rows


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of xyxy boxes: (n, 4) x (m, 4) -> (n, m)."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)[:, None, :]
    b = np.asarray(b, dtype=float).reshape(-1, 4)[None, :, :]
    w = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    h = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = w * h
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area_a + area_b - inter + 1e-7)


def match_image(
    pred_boxes: np.ndarray, pred_cls: np.ndarray, gt_boxes: np.ndarray, gt_cls: np.ndarray
) -> np.ndarray:
    """Which predictions are correct at each IoU threshold: bool array (n_pred, 10).

    Predictions must be sorted by confidence, highest first. A prediction can only match a
    ground-truth box of its own class. Each prediction keeps its best-IoU box, and each box is
    given to the highest-confidence prediction that chose it.
    """
    n = len(pred_boxes)
    tp = np.zeros((n, len(IOU_THRESHOLDS)), dtype=bool)
    if n == 0 or len(gt_boxes) == 0:
        return tp
    same_class = np.asarray(gt_cls)[:, None] == np.asarray(pred_cls)[None, :]
    iou = box_iou(gt_boxes, pred_boxes) * same_class
    for i, thr in enumerate(IOU_THRESHOLDS):
        matches = np.array(np.nonzero(iou >= thr)).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            tp[matches[:, 1], i] = True
    return tp


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Area under the precision-recall curve (101-point interpolation, as in COCO)."""
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(_trapezoid(np.interp(x, mrec, mpre), x))


def build_tables(images: list[ImageData], n_classes: int) -> Tables:
    idx, conf, cls, tps = [], [], [], []
    gt_counts = np.zeros((len(images), n_classes), dtype=int)
    for i, im in enumerate(images):
        order = np.argsort(-np.asarray(im.pred_conf), kind="stable")
        boxes = np.asarray(im.pred_boxes, dtype=float).reshape(-1, 4)[order]
        c = np.asarray(im.pred_conf, dtype=float)[order]
        k = np.asarray(im.pred_cls, dtype=int)[order]
        gt_boxes = np.asarray(im.gt_boxes, dtype=float).reshape(-1, 4)
        tps.append(match_image(boxes, k, gt_boxes, np.asarray(im.gt_cls)))
        idx.append(np.full(len(c), i))
        conf.append(c)
        cls.append(k)
        for g in np.asarray(im.gt_cls, dtype=int):
            gt_counts[i, g] += 1
    cat = np.concatenate
    return Tables(
        img_idx=cat(idx) if idx else np.zeros(0, int),
        conf=cat(conf) if conf else np.zeros(0),
        cls=cat(cls) if cls else np.zeros(0, int),
        tp=cat(tps) if tps else np.zeros((0, len(IOU_THRESHOLDS)), bool),
        gt_counts=gt_counts,
        names=[im.name for im in images],
    )


def restrict(t: Tables, image_mask: np.ndarray) -> Tables:
    """Tables for a subset of images (for example only the 'isolated' test images)."""
    image_mask = np.asarray(image_mask, dtype=bool)
    new_index = np.cumsum(image_mask) - 1
    keep = image_mask[t.img_idx]
    return Tables(
        img_idx=new_index[t.img_idx[keep]],
        conf=t.conf[keep],
        cls=t.cls[keep],
        tp=t.tp[keep],
        gt_counts=t.gt_counts[image_mask],
        names=[n for n, m in zip(t.names, image_mask, strict=True) if m],
    )


def _class_ap(conf: np.ndarray, tp: np.ndarray, n_gt: int) -> np.ndarray:
    """AP at the 10 IoU thresholds for one class. conf must be sorted high to low."""
    if n_gt == 0 or len(conf) == 0:
        return np.zeros(tp.shape[1])
    tpc = np.cumsum(tp, axis=0)
    fpc = np.cumsum(~tp, axis=0)
    recall = tpc / (n_gt + EPS)
    precision = tpc / (tpc + fpc)
    return np.array([compute_ap(recall[:, j], precision[:, j]) for j in range(tp.shape[1])])


def metrics_from_tables(
    t: Tables, weights: np.ndarray | None = None, conf_threshold: float | None = None
) -> dict:
    """mAP and per-class AP; with conf_threshold also precision, recall and F1 at IoU 0.5.

    weights[i] says how many times image i is counted (used by the bootstrap).
    """
    n_classes = t.gt_counts.shape[1]
    w = np.ones(len(t.gt_counts), dtype=int) if weights is None else np.asarray(weights, dtype=int)
    rep = np.repeat(np.arange(len(t.conf)), w[t.img_idx])
    conf, cls, tp = t.conf[rep], t.cls[rep], t.tp[rep]
    order = np.argsort(-conf, kind="stable")
    conf, cls, tp = conf[order], cls[order], tp[order]
    n_gt = (t.gt_counts * w[:, None]).sum(axis=0)

    ap = np.zeros((n_classes, tp.shape[1]))
    for c in range(n_classes):
        m = cls == c
        ap[c] = _class_ap(conf[m], tp[m], int(n_gt[c]))
    present = n_gt > 0
    out = {
        "n_gt": n_gt.tolist(),
        "map50": float(ap[present, 0].mean()) if present.any() else 0.0,
        "map50_95": float(ap[present].mean()) if present.any() else 0.0,
        "ap50": ap[:, 0].tolist(),
        "ap50_95": ap.mean(axis=1).tolist(),
    }
    if conf_threshold is not None:
        keep = conf >= conf_threshold
        prec, rec, f1 = np.zeros(n_classes), np.zeros(n_classes), np.zeros(n_classes)
        for c in range(n_classes):
            m = keep & (cls == c)
            hits = int(tp[m, 0].sum())
            n_pred = int(m.sum())
            prec[c] = hits / n_pred if n_pred else 0.0
            rec[c] = hits / n_gt[c] if n_gt[c] else 0.0
            f1[c] = 2 * prec[c] * rec[c] / (prec[c] + rec[c] + EPS)
        out["conf_threshold"] = conf_threshold
        out["precision"], out["recall"], out["f1"] = prec.tolist(), rec.tolist(), f1.tolist()
        out["macro_precision"] = float(prec[present].mean()) if present.any() else 0.0
        out["macro_recall"] = float(rec[present].mean()) if present.any() else 0.0
        out["macro_f1"] = float(f1[present].mean()) if present.any() else 0.0
    return out


def f1_curve(t: Tables, grid: np.ndarray | None = None) -> dict:
    """Precision, recall and F1 (IoU 0.5) for each class at each confidence threshold."""
    grid = np.linspace(0.01, 0.99, 99) if grid is None else np.asarray(grid)
    rows = [metrics_from_tables(t, conf_threshold=float(g)) for g in grid]
    return {
        "conf": grid.tolist(),
        "precision": [r["precision"] for r in rows],
        "recall": [r["recall"] for r in rows],
        "f1": [r["f1"] for r in rows],
        "macro_precision": [r["macro_precision"] for r in rows],
        "macro_recall": [r["macro_recall"] for r in rows],
        "macro_f1": [r["macro_f1"] for r in rows],
    }


def choose_threshold(curve: dict, smooth_window: int = 5) -> dict:
    """Confidence threshold with the best macro F1, after a small moving-average smoothing.

    If several thresholds tie for the best score, the one in the middle of the tied region is
    used, which is less sensitive to small changes in the data than the edge of a plateau.
    """
    y = np.asarray(curve["macro_f1"], dtype=float)
    half = smooth_window // 2
    padded = np.concatenate((np.full(half, y[0]), y, np.full(half, y[-1])))
    smooth = np.convolve(padded, np.ones(smooth_window) / smooth_window, mode="valid")
    best = np.flatnonzero(smooth >= smooth.max() - 1e-9)
    i = int(best[len(best) // 2])
    return {
        "conf": round(float(curve["conf"][i]), 4),
        "macro_f1_smoothed": float(smooth[i]),
        "macro_f1_raw": float(y[i]),
        "macro_precision": float(curve["macro_precision"][i]),
        "macro_recall": float(curve["macro_recall"][i]),
    }


def confusion_matrix(
    images: list[ImageData], n_classes: int, conf_threshold: float, iou_threshold: float = 0.5
) -> np.ndarray:
    """Counts with rows = predicted class and columns = true class; the last index is background.

    Boxes are paired one-to-one by IoU regardless of class, so wrong-class detections show up
    as off-diagonal entries. Unpaired predictions are false alarms (true = background) and
    unpaired ground-truth boxes are misses (predicted = background).
    """
    cm = np.zeros((n_classes + 1, n_classes + 1), dtype=int)
    bg = n_classes
    for im in images:
        keep = np.asarray(im.pred_conf) >= conf_threshold
        pb = np.asarray(im.pred_boxes, dtype=float).reshape(-1, 4)[keep]
        pc = np.asarray(im.pred_cls, dtype=int)[keep]
        gb = np.asarray(im.gt_boxes, dtype=float).reshape(-1, 4)
        gc = np.asarray(im.gt_cls, dtype=int)
        used_p, used_g = set(), set()
        if len(pb) and len(gb):
            iou = box_iou(gb, pb)
            pairs = np.argwhere(iou >= iou_threshold)
            pairs = pairs[np.argsort(-iou[pairs[:, 0], pairs[:, 1]], kind="stable")]
            for g, p in pairs:
                if g in used_g or p in used_p:
                    continue
                used_g.add(g)
                used_p.add(p)
                cm[pc[p], gc[g]] += 1
        for p in range(len(pb)):
            if p not in used_p:
                cm[pc[p], bg] += 1
        for g in range(len(gb)):
            if g not in used_g:
                cm[bg, gc[g]] += 1
    return cm


def _summary(values: np.ndarray, alpha: float) -> dict:
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(values.mean()), "lo": float(lo), "hi": float(hi)}


def bootstrap(
    t: Tables,
    n_boot: int = 1000,
    seed: int = 0,
    conf_threshold: float | None = None,
    group_ids: np.ndarray | None = None,
    alpha: float = 0.05,
) -> dict:
    """Percentile bootstrap intervals. Resamples images, or whole groups when group_ids is given.

    Returns, for each metric, the mean over resamples and the (1 - alpha) interval.
    """
    rng = np.random.default_rng(seed)
    n_img = len(t.gt_counts)
    if group_ids is None:
        member = np.arange(n_img)
    else:
        _, member = np.unique(np.asarray(group_ids), return_inverse=True)
    n_units = int(member.max()) + 1
    collected: dict[str, list] = {}
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, n_units, size=n_units), minlength=n_units)
        weights = counts[member]
        m = metrics_from_tables(t, weights, conf_threshold)
        flat = {"map50": m["map50"], "map50_95": m["map50_95"]}
        for c, (a, b) in enumerate(zip(m["ap50"], m["ap50_95"], strict=True)):
            flat[f"ap50_class{c}"], flat[f"ap50_95_class{c}"] = a, b
        if conf_threshold is not None:
            for key in ("macro_precision", "macro_recall", "macro_f1"):
                flat[key] = m[key]
            for c in range(len(m["precision"])):
                flat[f"precision_class{c}"] = m["precision"][c]
                flat[f"recall_class{c}"] = m["recall"][c]
        for k, v in flat.items():
            collected.setdefault(k, []).append(v)
    return {k: _summary(np.array(v), alpha) for k, v in collected.items()}
