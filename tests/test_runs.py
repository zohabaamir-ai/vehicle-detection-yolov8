import pytest

from vehicle_detection.runs import adopt_option, main, read_results_csv, summarize_run

HEADER = (
    "epoch,time,train/box_loss,metrics/precision(B),metrics/recall(B),"
    "metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,lr/pg0"
)


def write_csv(path, map50_95, map50=None, pad=False):
    map50 = map50 or [m + 0.3 for m in map50_95]
    header = HEADER.replace(",", ",  ") if pad else HEADER  # old versions padded the names
    lines = [header]
    for i, (a, b) in enumerate(zip(map50_95, map50, strict=True), start=1):
        lines.append(f"{i},{i * 10.0},1.0,0.5,0.4,{b},{a},1.1,0.01")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_read_results_csv_strips_padded_names(tmp_path):
    rows = read_results_csv(write_csv(tmp_path / "r.csv", [0.1, 0.2], pad=True))
    assert rows[1]["metrics/mAP50-95(B)"] == 0.2 and rows[1]["epoch"] == 2.0


def test_summary_finds_best_epoch_and_final_epoch(tmp_path):
    series = [0.1, 0.2, 0.5, 0.4, 0.45]
    rows = read_results_csv(write_csv(tmp_path / "r.csv", series))
    s = summarize_run(rows, window=2)
    assert s["n_epochs"] == 5 and s["best_epoch"] == 3 and s["best"]["map50_95"] == 0.5
    assert s["final_epoch"] == 5 and s["final"]["map50_95"] == 0.45
    assert s["last_window"]["mean"]["map50_95"] == pytest.approx(0.425)
    assert s["last_window"]["map50_95_min"] == 0.4 and s["last_window"]["map50_95_max"] == 0.45
    # mean of last 2 (0.425) minus mean of the 2 before (0.35)
    assert s["map50_95_gain_last_window_vs_previous"] == pytest.approx(0.075)


def test_best_epoch_ties_go_to_the_earliest(tmp_path):
    rows = read_results_csv(write_csv(tmp_path / "r.csv", [0.3, 0.5, 0.5, 0.2]))
    assert summarize_run(rows, window=2)["best_epoch"] == 2


def test_gain_is_none_when_the_run_is_too_short(tmp_path):
    rows = read_results_csv(write_csv(tmp_path / "r.csv", [0.1, 0.2, 0.3]))
    assert summarize_run(rows, window=10)["map50_95_gain_last_window_vs_previous"] is None
    with pytest.raises(ValueError):
        summarize_run([])


def test_adopt_option_needs_a_margin_bigger_than_the_seed_spread():
    # baseline seeds 0.500 and 0.520 -> mean 0.510, spread 0.020
    assert adopt_option([0.50, 0.52], 0.540)["adopt"] is True  # +0.030 > 0.020
    assert adopt_option([0.50, 0.52], 0.525)["adopt"] is False  # +0.015 < 0.020
    assert adopt_option([0.50, 0.52], 0.530)["adopt"] is False  # +0.020 is not more than 0.020
    out = adopt_option([0.50, 0.52], 0.54)
    assert out["baseline_mean"] == 0.51 and out["baseline_spread"] == 0.02
    with pytest.raises(ValueError):
        adopt_option([0.5], 0.6)


def test_cli_prints_summary(tmp_path, capsys):
    p = write_csv(tmp_path / "r.csv", [0.1, 0.2, 0.3])
    main([str(p)])
    assert '"best_epoch": 3' in capsys.readouterr().out
