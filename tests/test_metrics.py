import numpy as np
import pytest

from vehicle_detection.metrics import (
    ImageData,
    bootstrap,
    box_iou,
    build_tables,
    choose_threshold,
    compute_ap,
    confusion_matrix,
    f1_curve,
    match_image,
    metrics_from_tables,
    restrict,
)

BOX = [0, 0, 100, 100]


def image(gt, preds, name="a"):
    """gt: list of (cls, box). preds: list of (cls, conf, box)."""
    return ImageData(
        name=name,
        gt_boxes=np.array([b for _, b in gt], dtype=float).reshape(-1, 4),
        gt_cls=np.array([c for c, _ in gt], dtype=int),
        pred_boxes=np.array([b for _, _, b in preds], dtype=float).reshape(-1, 4),
        pred_conf=np.array([s for _, s, _ in preds], dtype=float),
        pred_cls=np.array([c for c, _, _ in preds], dtype=int),
    )


def test_box_iou_values():
    iou = box_iou(
        np.array([[0, 0, 10, 10]]), np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    )
    assert iou[0] == pytest.approx([1.0, 1 / 3, 0.0], abs=1e-6)


def test_match_image_marks_the_thresholds_a_box_passes():
    # shifted by 20 px: IoU = 8000 / 12000 = 0.667, so it passes 0.50, 0.55, 0.60 and 0.65
    tp = match_image(np.array([[20, 0, 120, 100]]), np.array([0]), np.array([BOX]), np.array([0]))
    assert tp[0].tolist() == [True] * 4 + [False] * 6


def test_wrong_class_never_matches():
    tp = match_image(np.array([BOX]), np.array([1]), np.array([BOX]), np.array([0]))
    assert not tp.any()


def test_duplicate_predictions_only_the_first_is_correct():
    boxes = np.array([BOX, BOX])
    tp = match_image(boxes, np.array([0, 0]), np.array([BOX]), np.array([0]))
    assert tp[:, 0].tolist() == [True, False]


def test_a_prediction_goes_to_its_best_overlapping_box():
    gt = np.array([[0, 0, 100, 100], [50, 0, 150, 100]])
    pred = np.array([[48, 0, 148, 100]])  # overlaps both, much better with the second
    tp = match_image(pred, np.array([0]), gt, np.array([0, 0]))
    assert tp[0, 0]
    both = match_image(
        np.array([[48, 0, 148, 100], [0, 0, 100, 100]]), np.array([0, 0]), gt, np.array([0, 0])
    )
    assert both[:, 0].tolist() == [True, True]  # each box gets its own prediction


def test_compute_ap_known_values():
    assert compute_ap(np.array([0.0, 1.0]), np.array([0.0, 0.5])) == pytest.approx(0.4975)
    assert compute_ap(np.array([1.0]), np.array([1.0])) == pytest.approx(0.995)  # 101-point rule


def wrong_first():
    """One object; a confident wrong box ranks above the correct low-confidence one."""
    return image([(0, BOX)], [(0, 0.9, [200, 200, 300, 300]), (0, 0.5, BOX)])


def test_map_and_precision_recall_on_a_hand_computed_case():
    t = build_tables([wrong_first()], 1)
    m = metrics_from_tables(t, conf_threshold=0.3)
    assert m["map50"] == pytest.approx(0.4975)
    assert m["precision"] == [0.5] and m["recall"] == [1.0]
    high = metrics_from_tables(t, conf_threshold=0.6)  # only the wrong box survives
    assert high["precision"] == [0.0] and high["recall"] == [0.0]


def test_perfect_detections_score_near_one():
    ims = [
        image(
            [(0, BOX), (1, [200, 200, 260, 260])], [(0, 0.9, BOX), (1, 0.8, [200, 200, 260, 260])]
        )
    ]
    m = metrics_from_tables(build_tables(ims, 2), conf_threshold=0.5)
    assert m["map50"] > 0.99 and m["map50_95"] > 0.99
    assert m["macro_f1"] == pytest.approx(1.0)


def test_weights_are_the_same_as_repeating_an_image():
    a, b = wrong_first(), image([(0, BOX)], [(0, 0.8, BOX)], "b")
    weighted = metrics_from_tables(build_tables([a, b], 1), weights=np.array([2, 1]))
    repeated = metrics_from_tables(build_tables([a, a, b], 1))
    assert weighted["map50"] == pytest.approx(repeated["map50"])
    assert weighted["n_gt"] == repeated["n_gt"]


def test_restrict_equals_building_from_the_subset():
    a, b, c = wrong_first(), image([(0, BOX)], [(0, 0.8, BOX)], "b"), image([(0, BOX)], [], "c")
    t = build_tables([a, b, c], 1)
    sub = restrict(t, np.array([True, False, True]))
    expected = build_tables([a, c], 1)
    assert sub.names == ["a", "c"]
    assert metrics_from_tables(sub)["map50"] == pytest.approx(
        metrics_from_tables(expected)["map50"]
    )
    assert sub.gt_counts.tolist() == expected.gt_counts.tolist()


def test_threshold_choice_lands_between_the_false_and_true_detections():
    gt = [(0, [i * 20, 0, i * 20 + 10, 10]) for i in range(10)]
    hits = [(0, 0.8, b) for _, b in gt]
    junk = [(0, 0.3, [i * 20, 500, i * 20 + 10, 510]) for i in range(10)]
    t = build_tables([image(gt, hits + junk)], 1)
    curve = f1_curve(t)
    chosen = choose_threshold(curve)
    assert 0.35 <= chosen["conf"] <= 0.75
    assert chosen["macro_f1_raw"] == pytest.approx(1.0)
    assert len(curve["conf"]) == 99


def test_confusion_matrix_counts_confusions_misses_and_false_alarms():
    im = image(
        [(0, [0, 0, 50, 50]), (2, [100, 100, 150, 150])],
        [
            (1, 0.9, [0, 0, 50, 50]),  # right place, wrong class
            (0, 0.8, [300, 300, 350, 350]),  # nothing there
            (2, 0.1, [100, 100, 150, 150]),  # below the threshold, ignored
        ],
    )
    cm = confusion_matrix([im], 3, conf_threshold=0.25)
    assert cm[1, 0] == 1  # predicted class 1, true class 0
    assert cm[0, 3] == 1  # false alarm (true = background)
    assert cm[3, 2] == 1  # missed class-2 object
    assert cm.sum() == 3


def scene(seed=0, n_img=40, nc=3):
    rng = np.random.default_rng(seed)
    ims = []
    for i in range(n_img):
        gtb, gtc = [], []
        for _ in range(rng.integers(0, 7)):
            x, y = rng.uniform(0, 500, 2)
            w, h = rng.uniform(20, 140, 2)
            gtb.append([x, y, x + w, y + h])
            gtc.append(rng.integers(0, nc))
        gtb, gtc = np.array(gtb).reshape(-1, 4), np.array(gtc, int)
        pb, pc, pf = [], [], []
        for b, c in zip(gtb, gtc, strict=True):
            if rng.random() < 0.85:
                pb.append(b + rng.normal(0, 6, 4))
                pc.append(c if rng.random() < 0.9 else rng.integers(0, nc))
                pf.append(rng.uniform(0.2, 1))
            if rng.random() < 0.2:
                pb.append(b + rng.normal(0, 6, 4))
                pc.append(c)
                pf.append(rng.uniform(0.001, 0.5))
        for _ in range(rng.integers(0, 5)):
            x, y = rng.uniform(0, 500, 2)
            w, h = rng.uniform(20, 140, 2)
            pb.append([x, y, x + w, y + h])
            pc.append(rng.integers(0, nc))
            pf.append(rng.uniform(0.001, 0.4))
        ims.append(
            ImageData(
                f"i{i}", gtb, gtc, np.array(pb).reshape(-1, 4), np.array(pf), np.array(pc, int)
            )
        )
    return ims


def test_matches_ultralytics_ap_on_a_fixed_random_scene():
    # Golden values were produced by Ultralytics 8.4.174's own ap_per_class on this exact scene.
    m = metrics_from_tables(build_tables(scene(), 3))
    assert m["map50"] == pytest.approx(0.6995703660148905, abs=1e-9)
    assert m["map50_95"] == pytest.approx(0.3330286989557534, abs=1e-9)


def test_bootstrap_is_deterministic_and_brackets_the_estimate():
    t = build_tables(scene(), 3)
    a = bootstrap(t, n_boot=60, seed=1, conf_threshold=0.25)
    assert a == bootstrap(t, n_boot=60, seed=1, conf_threshold=0.25)
    assert a != bootstrap(t, n_boot=60, seed=2, conf_threshold=0.25)
    point = metrics_from_tables(t)["map50"]
    assert a["map50"]["lo"] < point < a["map50"]["hi"]
    assert {"macro_f1", "recall_class2", "ap50_95_class0"} <= set(a)


def test_bootstrapping_one_single_group_cannot_vary():
    t = build_tables(scene(), 3)
    point = metrics_from_tables(t)["map50"]
    same = bootstrap(t, n_boot=20, group_ids=np.zeros(len(t.gt_counts), dtype=int))
    assert same["map50"]["lo"] == same["map50"]["hi"] == pytest.approx(point)
    by_image = bootstrap(t, n_boot=60, seed=0)
    by_pair = bootstrap(t, n_boot=60, seed=0, group_ids=np.arange(len(t.gt_counts)) // 2)
    assert (by_pair["map50"]["hi"] - by_pair["map50"]["lo"]) > 0
    assert (by_image["map50"]["hi"] - by_image["map50"]["lo"]) > 0
