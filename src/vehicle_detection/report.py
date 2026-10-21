"""Build the README's result tables and figures from the saved result files.

Nothing is typed by hand: every number comes from results/*.json. Tables need only the
standard library; figures need the [report] extra (matplotlib).

    python -m vehicle_detection.report --write     # update the tables in README.md
    python -m vehicle_detection.report --check     # fail if README.md is out of date
    python -m vehicle_detection.report --figures   # draw the figures into docs/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vehicle_detection.constants import CLASS_NAMES

HEADLINE = ("<!-- HEADLINE:START -->", "<!-- HEADLINE:END -->")
RESULTS = ("<!-- RESULTS:START -->", "<!-- RESULTS:END -->")
DEMO = ("<!-- DEMO:START -->", "<!-- DEMO:END -->")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
OUT_OF_DATE = (
    "README.md is out of date with results/. Run: python -m vehicle_detection.report --write"
)


def load_results(results_dir: Path | str) -> dict:
    r = Path(results_dir)

    def read(name):
        return json.loads((r / name).read_text(encoding="utf-8"))

    return {
        "test": read("test_result.json"),
        "val": read("val_eval_v8n_seed0.json"),
        "split": read("split_summary.json"),
        "audit": read("clean_audit.json"),
        "err_val": read("error_analysis_val.json"),
        "err_test": read("error_analysis_test.json"),
        "runs": {
            p.parent.name: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((r / "runs").glob("*/run_summary.json"))
        },
    }


def f(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def ci(boot: dict, key: str) -> str:
    return f"{f(boot[key]['lo'])} to {f(boot[key]['hi'])}"


def md_table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def n_objects(result: dict) -> int:
    return int(sum(result["at_threshold"]["n_gt"].values()))


def headline(res: dict) -> str:
    t = res["test"]
    by_img, by_grp = t["bootstrap"]["by_image"], t["bootstrap"]["by_group"]
    rank, a = t["ranking_metrics"], t["at_threshold"]
    return (
        f"**Held-out test split ({t['n_images']} images, {n_objects(t)} objects), scored once:** "
        f"mAP@0.5 **{f(rank['map50'])}** "
        f"(95% CI {ci(by_img, 'map50')} by image, {ci(by_grp, 'map50')} by group) and "
        f"mAP@0.5:0.95 **{f(rank['map50_95'])}** "
        f"(95% CI {ci(by_img, 'map50_95')} by image, {ci(by_grp, 'map50_95')} by group). "
        f"At the confidence threshold {t['conf_threshold']}, averaged over the three classes, "
        f"precision is {f(a['macro_precision'], 2)} and recall is {f(a['macro_recall'], 2)}."
    )


def main_table(res: dict) -> str:
    t, v = res["test"], res["val"]
    ti, tg = t["bootstrap"]["by_image"], t["bootstrap"]["by_group"]
    rows = []
    for label, key in (("mAP@0.5", "map50"), ("mAP@0.5:0.95", "map50_95")):
        rank_v, rank_t = v["ranking_metrics"][key], t["ranking_metrics"][key]
        rows.append([label, f(rank_v), f(rank_t), ci(ti, key), ci(tg, key)])
    for label, key in (
        ("precision (macro)", "macro_precision"),
        ("recall (macro)", "macro_recall"),
        ("F1 (macro)", "macro_f1"),
    ):
        at_v, at_t = v["at_threshold"][key], t["at_threshold"][key]
        rows.append([label, f(at_v), f(at_t), ci(ti, key), ci(tg, key)])
    cv, ct = v["counting"]["all_vehicles"], t["counting"]["all_vehicles"]

    def count_cell(c):
        return f"{f(c['mean_abs_error_per_image'], 2)} (true mean {f(c['true_mean_per_image'], 1)})"

    rows.append(["count error (vehicles per image)", count_cell(cv), count_cell(ct), "n/a", "n/a"])
    header = [
        "metric",
        f"validation ({v['n_images']} images)",
        f"**test** ({t['n_images']} images)",
        "test 95% CI, resampling images",
        "test 95% CI, resampling groups",
    ]
    return md_table(header, rows)


def per_class_table(res: dict) -> str:
    t, v = res["test"], res["val"]
    ti = t["bootstrap"]["by_image"]
    a, rank = t["at_threshold"], t["ranking_metrics_per_class"]
    rows = []
    for name in CLASS_NAMES:
        rows.append(
            [
                name,
                a["n_gt"][name],
                f(rank["ap50"][name]),
                ci(ti, f"ap50_{name}"),
                f(rank["ap50_95"][name]),
                f(a["precision"][name], 2),
                f(a["recall"][name], 2),
                f(v["ranking_metrics_per_class"]["ap50"][name]),
            ]
        )
    header = [
        "class",
        "test objects",
        "test AP@0.5",
        "95% CI (images)",
        "test AP@0.5:0.95",
        "precision",
        "recall",
        "validation AP@0.5",
    ]
    return md_table(header, rows)


def isolated_table(res: dict) -> str:
    rows = []
    for label, r in (("validation", res["val"]), ("test", res["test"])):
        iso = r["isolated_subset"]
        g = iso["bootstrap_by_group"]
        all_cell = f"{f(r['ranking_metrics']['map50'])} / {f(r['ranking_metrics']['map50_95'])}"
        iso_cell = f"{f(iso['metrics']['map50'])} / {f(iso['metrics']['map50_95'])}"
        rows.append(
            [
                label,
                f"{iso['n_images']} of {r['n_images']}",
                all_cell,
                iso_cell,
                f"{ci(g, 'map50')} / {ci(g, 'map50_95')}",
            ]
        )
    header = [
        "split",
        "isolated images",
        "all images (mAP@0.5 / 0.5:0.95)",
        "isolated only",
        "isolated only, 95% CI by group",
    ]
    return md_table(header, rows)


def runs_table(res: dict) -> str:
    rows = []
    for name, s in res["runs"].items():
        run = s["run"]
        rows.append(
            [
                f"`{name}`",
                run["model"],
                run["imgsz"],
                run["seed"],
                s["best_epoch"],
                f(s["best"]["map50"]),
                f(s["best"]["map50_95"]),
                f(s["last_window"]["mean"]["map50_95"]),
            ]
        )
    header = [
        "run",
        "model",
        "image size",
        "seed",
        "best epoch",
        "val mAP@0.5",
        "val mAP@0.5:0.95",
        "mean of last 10 epochs (mAP@0.5:0.95)",
    ]
    return md_table(header, rows)


def dataset_table(res: dict) -> str:
    s, audit = res["split"], res["audit"]["splits"]
    rows = []
    for name, sp in s["splits"].items():
        b = sp["boxes_per_class"]
        a = audit[name]
        rows.append(
            [name, sp["n_images"], sp["n_groups"], sp["n_boxes"]]
            + [b[c] for c in CLASS_NAMES]
            + [
                f"{a['tiny_box_fraction']:.0%}",
                f(a["boxes_per_image_mean"], 1),
                s["isolated_images"].get(name, "-"),
            ]
        )
    header = [
        "split",
        "images",
        "groups",
        "boxes",
        *CLASS_NAMES,
        "tiny boxes (share)",
        "boxes per image",
        "isolated images",
    ]
    return md_table(header, rows)


def size_table(res: dict) -> str:
    rows = []
    for size, v in res["err_val"]["detection_rate_by_object_size"].items():
        t = res["err_test"]["detection_rate_by_object_size"][size]
        rows.append(
            [
                size,
                v["n_objects"],
                f"{v['detected_correctly']:.0%}",
                t["n_objects"],
                f"{t['detected_correctly']:.0%}",
            ]
        )
    header = [
        "object size (share of image area)",
        "validation objects",
        "found correctly",
        "test objects",
        "found correctly",
    ]
    return md_table(header, rows)


def counting_table(res: dict) -> str:
    c = res["test"]["counting"]["per_class"]
    rows = [
        [
            name,
            c[name]["true_total"],
            c[name]["predicted_total"],
            f(c[name]["mean_error_per_image"], 2),
        ]
        for name in CLASS_NAMES
    ]
    header = ["class", "true objects (test)", "predicted at threshold", "mean error per image"]
    return md_table(header, rows)


def confusion_table(res: dict) -> str:
    cm = res["test"]["confusion_matrix"]
    labels = cm["labels"]
    rows = [[f"predicted {labels[i]}", *map(str, row)] for i, row in enumerate(cm["counts"])]
    return md_table([""] + [f"true {x}" for x in labels], rows)


def environment_sentence(run: dict) -> str:
    v = {key: run.get(key, "unknown") for key in ("ultralytics", "torch", "python", "gpu")}
    return (
        f"Training environment of the final run: Ultralytics {v['ultralytics']}, "
        f"PyTorch {v['torch']}, Python {v['python']}, GPU {v['gpu']}."
    )


def results_block(res: dict) -> str:
    t = res["test"]
    fm = t["final_model"]
    run = fm["weights_file"].split("_best")[0]
    parts = [
        "### Main results",
        main_table(res),
        f"The confidence threshold ({t['conf_threshold']}) was chosen on validation data "
        "(best macro F1) and not changed afterwards. Intervals come from "
        f"{t['bootstrap']['n_resamples']} bootstrap resamples; resampling groups of "
        "near-identical images is the more honest of the two.",
        "### Per class (test split)",
        per_class_table(res),
        "### Images with no near-duplicate in the training set",
        isolated_table(res),
        "### What is found, by object size",
        size_table(res),
        "### Counting (test split)",
        counting_table(res),
        f"### Confusion matrix (test split, threshold {t['conf_threshold']})",
        confusion_table(res),
        "### Data splits",
        dataset_table(res),
        "### All training runs (validation split)",
        runs_table(res),
        f"The final model is the `best.pt` of run `{run}` (file `{fm['weights_file']}`, "
        f"sha256 `{fm['weights_sha256']}`).",
        environment_sentence(res["runs"].get(run, {}).get("run", {})),
    ]
    return "\n\n".join(parts)


def demo_block(root: Path) -> str:
    """The demo screenshot and sample list, shown only for files that exist in the repo."""
    root = Path(root)
    lines = []
    if (root / "docs" / "images" / "demo.png").exists():
        lines.append("![Screenshot of the demo app](docs/images/demo.png)")
    else:
        lines.append("_Demo screenshot: not added yet._")
    samples_dir = root / "samples"
    names = (
        sorted(p.name for p in samples_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if samples_dir.is_dir()
        else []
    )
    if names:
        listed = ", ".join(f"`samples/{n}`" for n in names)
        lines.append(
            f"Sample images you can try: {listed}. Sources and licences: `samples/ATTRIBUTION.md`."
        )
    return "\n\n".join(lines)


def replace_block(text: str, markers: tuple[str, str], new: str) -> str:
    start, end = markers
    if start not in text or end not in text or text.index(start) > text.index(end):
        raise SystemExit(f"README is missing the markers {start} ... {end}")
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{start}\n{new}\n{end}{tail}"


def render_readme(text: str, res: dict, root: Path | str = ".") -> str:
    text = replace_block(text, HEADLINE, headline(res))
    text = replace_block(text, DEMO, demo_block(Path(root)))
    return replace_block(text, RESULTS, results_block(res))


def figures(res: dict, out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    val, test = res["val"], res["test"]
    curve, chosen = val["curve"], val["conf_threshold"]
    conf = np.array(curve["conf"])
    files = []

    fig, ax = plt.subplots(figsize=(6, 4))
    per_class = np.array(curve["f1"])
    for i, name in enumerate(CLASS_NAMES):
        ax.plot(conf, per_class[:, i], label=name, alpha=0.7)
    ax.plot(conf, curve["macro_f1"], "k", lw=2, label="macro average")
    ax.axvline(chosen, color="red", ls="--", label=f"chosen threshold {chosen}")
    ax.set(xlabel="confidence threshold", ylabel="F1 (IoU 0.5)")
    ax.set_title("Validation: F1 vs confidence")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "f1_vs_confidence_val.png", dpi=120)
    plt.close(fig)
    files.append("f1_vs_confidence_val.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    recall, precision = curve["macro_recall"], curve["macro_precision"]
    ax.plot(recall, precision, "k-")
    i = int(np.argmin(np.abs(conf - chosen)))
    ax.plot(recall[i], precision[i], "ro", label=f"threshold {chosen}")
    ax.set(xlabel="recall (macro)", ylabel="precision (macro)")
    ax.set_title("Validation: precision vs recall")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "precision_recall_val.png", dpi=120)
    plt.close(fig)
    files.append("precision_recall_val.png")

    cm = np.array(test["confusion_matrix"]["counts"])
    labels = test["confusion_matrix"]["labels"]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(cm, cmap="Blues")
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            color = "white" if cm[r, c] > cm.max() / 2 else "black"
            ax.text(c, r, cm[r, c], ha="center", va="center", color=color)
    ax.set_xticks(range(len(labels)), labels, rotation=30)
    ax.set_yticks(range(len(labels)), labels)
    ax.set(xlabel="true", ylabel="predicted")
    ax.set_title("Test: confusion matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix_test.png", dpi=120)
    plt.close(fig)
    files.append("confusion_matrix_test.png")
    return files


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--readme", type=Path, default=Path("README.md"))
    ap.add_argument("--figures-dir", type=Path, default=Path("docs/figures"))
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--figures", action="store_true")
    args = ap.parse_args(argv)
    res = load_results(args.results)
    if args.figures:
        print("figures:", figures(res, args.figures_dir))
        return
    text = args.readme.read_text(encoding="utf-8")
    new = render_readme(text, res, args.readme.resolve().parent)
    if args.check:
        if new != text:
            raise SystemExit(OUT_OF_DATE)
        print("README.md matches results/")
        return
    args.readme.write_text(new, encoding="utf-8")
    t = res["test"]
    rank, a = t["ranking_metrics"], t["at_threshold"]
    print("README.md updated. Key numbers from results/test_result.json, to compare by eye:")
    print(f"  test mAP@0.5 {rank['map50']:.4f}   mAP@0.5:0.95 {rank['map50_95']:.4f}")
    print(f"  threshold {t['conf_threshold']}   macro precision {a['macro_precision']:.4f}", end="")
    print(f"   macro recall {a['macro_recall']:.4f}")
    print(f"  test images {t['n_images']}   objects {n_objects(t)}")


if __name__ == "__main__":
    main()
