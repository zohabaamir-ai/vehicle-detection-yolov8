import re
from pathlib import Path

import pytest

from vehicle_detection import report
from vehicle_detection.constants import CLASS_NAMES

ROOT = Path(__file__).resolve().parents[1]
CI = {"mean": 0.5, "lo": 0.4, "hi": 0.6}


def split_result(map50, map50_95, n_images, with_final=False):
    names = list(CLASS_NAMES)
    boot = {
        k: dict(CI) for k in ("map50", "map50_95", "macro_precision", "macro_recall", "macro_f1")
    }
    boot.update({f"ap50_{n}": dict(CI) for n in names})
    res = {
        "n_images": n_images,
        "conf_threshold": 0.29,
        "ranking_metrics": {"map50": map50, "map50_95": map50_95},
        "ranking_metrics_per_class": {
            "ap50": dict.fromkeys(names, map50),
            "ap50_95": dict.fromkeys(names, map50_95),
        },
        "at_threshold": {
            "precision": dict.fromkeys(names, 0.7),
            "recall": dict.fromkeys(names, 0.5),
            "f1": dict.fromkeys(names, 0.58),
            "n_gt": dict.fromkeys(names, 10),
            "macro_precision": 0.7,
            "macro_recall": 0.5,
            "macro_f1": 0.58,
        },
        "bootstrap": {"n_resamples": 1000, "by_image": boot, "by_group": boot},
        "counting": {
            "all_vehicles": {"mean_abs_error_per_image": 1.5, "true_mean_per_image": 5.0},
            "per_class": {
                n: {"true_total": 10, "predicted_total": 8, "mean_error_per_image": -0.2}
                for n in names
            },
        },
        "isolated_subset": {
            "n_images": 3,
            "metrics": {"map50": map50 - 0.01, "map50_95": map50_95 - 0.01},
            "bootstrap_by_group": {"map50": dict(CI), "map50_95": dict(CI)},
        },
        "confusion_matrix": {
            "labels": [*names, "background"],
            "counts": [[1, 0, 0, 2], [0, 1, 0, 2], [0, 0, 1, 2], [3, 3, 3, 0]],
        },
        "curve": {
            "conf": [0.1, 0.5, 0.9],
            "f1": [[0.3, 0.3, 0.3]] * 3,
            "macro_f1": [0.3, 0.6, 0.2],
            "macro_precision": [0.4, 0.7, 0.9],
            "macro_recall": [0.8, 0.5, 0.1],
        },
    }
    if with_final:
        res["final_model"] = {"weights_file": "v8n_seed0_best.pt", "weights_sha256": "ab" * 32}
    return res


def error_result(rate):
    sizes = ("tiny", "small", "medium", "large")
    return {
        "detection_rate_by_object_size": {
            s: {"n_objects": 10, "detected_correctly": rate} for s in sizes
        }
    }


@pytest.fixture
def res():
    run = {
        "best_epoch": 53,
        "best": {"map50": 0.796, "map50_95": 0.476},
        "last_window": {"mean": {"map50_95": 0.461}},
        "run": {"model": "yolov8n.pt", "imgsz": 672, "seed": 0},
    }
    return {
        "test": split_result(0.598, 0.375, 126, with_final=True),
        "val": split_result(0.781, 0.463, 135),
        "split": {
            "splits": {
                n: {
                    "n_images": 10,
                    "n_groups": 5,
                    "n_boxes": 30,
                    "boxes_per_class": dict.fromkeys(CLASS_NAMES, 10),
                }
                for n in ("train", "val", "test")
            },
            "isolated_images": {"val": 7, "test": 4},
        },
        "audit": {
            "splits": {
                n: {"tiny_box_fraction": frac, "boxes_per_image_mean": 5.0}
                for n, frac in (("train", 0.24), ("val", 0.22), ("test", 0.47))
            }
        },
        "err_val": error_result(0.7),
        "err_test": error_result(0.5),
        "runs": {"v8n_seed0": run},
    }


def test_the_headline_and_tables_contain_the_numbers_from_the_results(res):
    head = report.headline(res)
    assert "0.598" in head and "0.375" in head and "126 images" in head and "30 objects" in head
    block = report.results_block(res)
    for needle in ("0.781", "0.463", "0.400 to 0.600", "v8n_seed0", "yolov8n.pt", "1000 bootstrap"):
        assert needle in block, needle
    for name in CLASS_NAMES:
        assert f"| {name} |" in block
    assert "ab" * 32 in block and "50%" in block and "70%" in block
    assert "predicted background" in block and "true background" in block
    assert "| 47% |" in block and "tiny boxes (share)" in block


def test_replace_block_only_touches_the_text_between_the_markers():
    markers = ("<!-- A -->", "<!-- B -->")
    text = "before\n<!-- A -->\nold\n<!-- B -->\nafter"
    assert (
        report.replace_block(text, markers, "new") == "before\n<!-- A -->\nnew\n<!-- B -->\nafter"
    )
    with pytest.raises(SystemExit):
        report.replace_block("no markers here", markers, "new")
    with pytest.raises(SystemExit):
        report.replace_block("<!-- B -->\n<!-- A -->", markers, "new")


def test_rendering_the_readme_is_stable_and_follows_the_files_present(res, tmp_path):
    template = "".join(
        f"<!-- {name}:START -->\nx\n<!-- {name}:END -->\n"
        for name in ("HEADLINE", "DEMO", "RESULTS")
    )
    once = report.render_readme(template, res, tmp_path)
    assert "not added yet" in once and report.render_readme(once, res, tmp_path) == once
    (tmp_path / "docs" / "images").mkdir(parents=True)
    (tmp_path / "docs" / "images" / "demo.png").write_bytes(b"png")
    (tmp_path / "samples").mkdir()
    (tmp_path / "samples" / "road.jpg").write_bytes(b"jpg")
    again = report.render_readme(once, res, tmp_path)
    assert "docs/images/demo.png" in again and "`samples/road.jpg`" in again
    assert "not added yet" not in again


def test_figures_are_drawn(res, tmp_path):
    pytest.importorskip("matplotlib")
    files = report.figures(res, tmp_path)
    assert sorted(files) == [
        "confusion_matrix_test.png",
        "f1_vs_confidence_val.png",
        "precision_recall_val.png",
    ]
    assert all((tmp_path / f).stat().st_size > 1000 for f in files)


def test_the_committed_readme_matches_the_committed_results():
    if not (ROOT / "results" / "test_result.json").exists():
        pytest.skip("results/ not present")
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    res = report.load_results(ROOT / "results")
    assert report.render_readme(text, res, ROOT) == text, report.OUT_OF_DATE


def test_every_local_link_and_image_in_the_readme_exists():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    links = re.findall(r"!?\[[^\]]*\]\(([^)\s]+)\)", text)
    local = [x.split("#")[0] for x in links if not x.startswith(("http", "#", "mailto"))]
    missing = [x for x in local if x and not (ROOT / x).exists()]
    assert not missing, f"README links to files that do not exist: {missing}"
