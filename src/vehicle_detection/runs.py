"""Read and compare training runs (results.csv written by Ultralytics). No torch needed."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

COLUMNS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}


def read_results_csv(path: Path | str) -> list[dict[str, float]]:
    """Rows of results.csv as dicts of floats (column names are stripped of spaces)."""
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [{k.strip(): float(v) for k, v in row.items() if k} for row in reader]


def _pick(row: dict[str, float]) -> dict[str, float]:
    return {name: round(row[col], 5) for name, col in COLUMNS.items()}


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    return {name: round(sum(r[col] for r in rows) / len(rows), 5) for name, col in COLUMNS.items()}


def summarize_run(rows: list[dict[str, float]], window: int = 10) -> dict:
    """Best epoch (by validation mAP@0.5:0.95), final epoch, and how settled the curve is."""
    if not rows:
        raise ValueError("results.csv has no rows")
    key = COLUMNS["map50_95"]
    best = max(rows, key=lambda r: (r[key], -r["epoch"]))  # ties: earliest epoch
    tail = rows[-window:]
    out = {
        "n_epochs": len(rows),
        "best_epoch": int(best["epoch"]),
        "best": _pick(best),
        "final_epoch": int(rows[-1]["epoch"]),
        "final": _pick(rows[-1]),
        "last_window": {
            "epochs": len(tail),
            "mean": _mean(tail),
            "map50_min": round(min(r[COLUMNS["map50"]] for r in tail), 5),
            "map50_max": round(max(r[COLUMNS["map50"]] for r in tail), 5),
            "map50_95_min": round(min(r[key] for r in tail), 5),
            "map50_95_max": round(max(r[key] for r in tail), 5),
        },
    }
    prev = rows[-2 * window : -window]
    if len(prev) == window:
        gain = _mean(tail)["map50_95"] - _mean(prev)["map50_95"]
        out["map50_95_gain_last_window_vs_previous"] = round(gain, 5)
    else:
        out["map50_95_gain_last_window_vs_previous"] = None
    return out


def adopt_option(baseline_scores: list[float], candidate_score: float) -> dict:
    """The agreed rule: adopt a candidate only if it beats the baseline mean by more than the
    seed-to-seed spread (max minus min) of the baseline runs. Scores are validation mAP@0.5:0.95.
    """
    if len(baseline_scores) < 2:
        raise ValueError("need at least two baseline runs to measure the seed spread")
    mean = sum(baseline_scores) / len(baseline_scores)
    spread = max(baseline_scores) - min(baseline_scores)
    margin = candidate_score - mean
    return {
        "baseline_mean": round(mean, 5),
        "baseline_spread": round(spread, 5),
        "candidate": round(candidate_score, 5),
        "margin": round(margin, 5),
        "adopt": margin > spread,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Summarise one or more results.csv files.")
    ap.add_argument("csv", nargs="+", type=Path)
    args = ap.parse_args(argv)
    for path in args.csv:
        print(path)
        print(json.dumps(summarize_run(read_results_csv(path)), indent=2))


if __name__ == "__main__":
    main()
