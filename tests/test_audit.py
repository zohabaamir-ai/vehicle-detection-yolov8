import json

from vehicle_detection.audit import audit_dataset, main, roboflow_base_name

H1 = "a" * 32
H2 = "b" * 32


def test_roboflow_base_name():
    assert roboflow_base_name(f"road_jpg.rf.{H1}.jpg") == "road_jpg"
    assert roboflow_base_name(f"road_jpg.rf.{H2}.txt") == "road_jpg"
    assert roboflow_base_name("plain_name.jpg") == "plain_name"
    assert roboflow_base_name("a.b.c.jpg") == "a.b.c"


def test_audit_finds_planted_problems(dataset_root):
    summary, _ = audit_dataset(dataset_root)
    tr, te = summary["splits"]["train"], summary["splits"]["test"]
    assert tr["n_images"] == 6
    assert tr["n_label_files"] == 5
    assert tr["n_images_without_label"] == 1
    assert tr["n_empty_label_files"] == 1
    assert tr["n_label_files_with_polygons"] == 1
    # the 0.3 x 0.1 box at x=0.9 sticks out of the image
    assert tr["n_label_files_with_problems"] == 1
    assert tr["boxes_per_class"] == {"0": 4, "1": 1, "2": 1}
    # tiny = area < 0.0025: the 0.01 x 0.01 box, and the 0.1 x 0.1 box (area 0.01) is not
    assert tr["n_tiny_boxes"] == 1
    assert tr["n_source_names_with_copies"] == 1
    assert te["n_images"] == 2


def test_audit_finds_train_test_overlap(dataset_root):
    summary, _ = audit_dataset(dataset_root)
    (overlap,) = summary["overlap"].values()
    assert overlap["n_shared_source_names"] == 1
    assert overlap["examples"] == ["road_jpg"]


def test_cli_writes_files_and_hides_absolute_path(dataset_root, tmp_path, capsys):
    out, det = tmp_path / "s.json", tmp_path / "d.json"
    main(["--root", str(dataset_root), "--out", str(out), "--details", str(det)])
    data = json.loads(out.read_text())
    assert data["dataset_folder_name"] == "ObjectsDataset"
    assert str(tmp_path) not in out.read_text()
    assert det.exists()
    assert "overlap" in capsys.readouterr().out


def test_audit_counts_mixed_polygon_and_box_files(tmp_path, write_sample):
    split = tmp_path / "train"
    write_sample(split, f"a_jpg.rf.{H1}", "0 0.5 0.5 0.2 0.2\n2 0.2 0.2 0.6 0.2 0.6 0.8 0.2 0.8\n")
    write_sample(split, f"b_jpg.rf.{H1}", "0 0.5 0.5 0.2 0.2\n")
    summary, details = audit_dataset(tmp_path)
    s = summary["splits"]["train"]
    assert s["n_label_files_mixed_polygon_and_box"] == 1
    assert s["n_plain_box_lines_in_mixed_files"] == 1
    assert details["splits"]["train"]["mixed_polygon_box_files"] == [f"a_jpg.rf.{H1}.txt"]
