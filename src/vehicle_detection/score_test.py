"""Score the held-out test split ONCE, for the final model only. No torch needed.

The confidence threshold must be the one chosen on validation (read from the validation result
file). The result is written to a new file; an existing result is never overwritten.

Real run:
    python -m vehicle_detection.score_test --preds p_test.json --split-dir D:\\data\\clean\\test ^
        --weights best.pt --conf 0.29

Rehearsal on validation (same pipeline, can be repeated, never touches the test split):
    python -m vehicle_detection.score_test --preds p_val.json --split-dir D:\\data\\clean\\val ^
        --weights best.pt --conf 0.29 --rehearsal --out data\\audit\\rehearsal_val.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from vehicle_detection import __version__
from vehicle_detection.evaluate import evaluate, print_summary

FINAL_RESULT = Path("results/test_result.json")
VAL_RESULT = Path("results/val_eval_v8n_seed0.json")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_threshold_from_validation(val_result: Path, conf: float) -> dict:
    """The threshold must equal the one chosen on the validation split."""
    val = json.loads(Path(val_result).read_text(encoding="utf-8"))
    if val.get("split") != "val" or not val.get("conf_threshold_was_chosen_here"):
        raise SystemExit(f"{val_result} is not a validation result that chose a threshold.")
    if abs(val["conf_threshold"] - conf) > 1e-9:
        raise SystemExit(
            f"--conf {conf} is not the threshold chosen on validation ({val['conf_threshold']}). "
            "The test split must not be used to pick a threshold."
        )
    return val


def score_final(
    preds: Path,
    split_dir: Path,
    manifest: Path,
    weights: Path,
    conf: float,
    val_result: Path = VAL_RESULT,
    out: Path = FINAL_RESULT,
    rehearsal: bool = False,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    split_dir, out = Path(split_dir), Path(out)
    expected = "val" if rehearsal else "test"
    if split_dir.name != expected:
        raise SystemExit(
            f"This mode scores the '{expected}' split but got '{split_dir.name}'. "
            + ("Use --rehearsal only with the validation split." if rehearsal else "")
        )
    if rehearsal and out.name == FINAL_RESULT.name:
        raise SystemExit(f"A rehearsal must not write {FINAL_RESULT.name}.")
    if not rehearsal and out.exists():
        raise SystemExit(f"{out} already exists. The test split is scored only once.")
    val = check_threshold_from_validation(val_result, conf)

    res = evaluate(preds, split_dir, manifest, conf=conf, n_boot=n_boot, seed=seed)
    if res["predictions_meta"].get("split") != expected:
        raise SystemExit("The predictions file was not made from this split.")
    res["final_model"] = {
        "weights_file": Path(weights).name,
        "weights_sha256": sha256_of(weights),
        "ultralytics": res["predictions_meta"].get("ultralytics"),
        "vehicle_detection": __version__,
    }
    res["protocol"] = {
        "rehearsal": rehearsal,
        "scored_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "threshold_source": (
            f"{Path(val_result).name} (chosen on validation, {val['n_images']} images)"
        ),
        "note": "Test split scored once for the final model; nothing was chosen from this result.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out.open("w" if rehearsal else "x", encoding="utf-8") as f:
            f.write(json.dumps(res, indent=1) + "\n")
    except FileExistsError:
        raise SystemExit(f"{out} already exists. The test split is scored only once.") from None
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", required=True, type=Path)
    ap.add_argument("--split-dir", required=True, type=Path)
    ap.add_argument("--weights", required=True, type=Path, help="the final model's weights file")
    ap.add_argument("--conf", required=True, type=float, help="threshold chosen on validation")
    ap.add_argument("--manifest", type=Path, default=Path("results/split_manifest.csv"))
    ap.add_argument("--val-result", type=Path, default=VAL_RESULT)
    ap.add_argument("--out", type=Path, default=FINAL_RESULT)
    ap.add_argument("--rehearsal", action="store_true")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args(argv)
    res = score_final(
        args.preds,
        args.split_dir,
        args.manifest,
        args.weights,
        args.conf,
        args.val_result,
        args.out,
        args.rehearsal,
        args.boot,
    )
    print("REHEARSAL on validation" if args.rehearsal else "FINAL TEST RESULT (scored once)")
    print_summary(res)
    print(f"weights sha256: {res['final_model']['weights_sha256']}")
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
