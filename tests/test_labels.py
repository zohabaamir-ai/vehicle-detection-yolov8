from vehicle_detection.labels import Box, parse_label_text, polygon_to_box


def test_parses_normal_boxes():
    p = parse_label_text("0 0.5 0.5 0.2 0.2\n2 0.3 0.3 0.1 0.1\n")
    assert [b.cls for b in p.boxes] == [0, 2]
    assert p.problems == [] and p.n_polygons == 0


def test_blank_file_is_empty_not_a_problem():
    p = parse_label_text("\n\n")
    assert p.boxes == [] and p.problems == []


def test_polygon_becomes_its_bounding_box():
    p = parse_label_text("1 0.2 0.2 0.6 0.2 0.6 0.8 0.2 0.8\n")
    assert p.n_polygons == 1
    (b,) = p.boxes
    assert b.cls == 1
    assert (round(b.xc, 3), round(b.yc, 3), round(b.w, 3), round(b.h, 3)) == (0.4, 0.5, 0.4, 0.6)


def test_polygon_to_box_direct():
    b = polygon_to_box(0, [0.1, 0.1, 0.3, 0.1, 0.2, 0.5])
    assert round(b.w, 3) == 0.2
    assert round(b.h, 3) == 0.4


def test_problems_are_reported_and_skipped():
    text = "0 0.5 0.5\nx 0.5 0.5 0.1 0.1\n1.5 0.5 0.5 0.1 0.1\n0 0.5 0.5 0 0.1\n0 nan 0.5 0.1 0.1\n"
    p = parse_label_text(text)
    assert p.boxes == []
    assert len(p.problems) == 5


def test_box_outside_image_is_flagged_but_kept():
    p = parse_label_text("0 0.95 0.5 0.2 0.2\n")
    assert len(p.boxes) == 1
    assert any("outside" in m for m in p.problems)


def test_area():
    assert abs(Box(0, 0.5, 0.5, 0.2, 0.5).area - 0.1) < 1e-12


def test_mixed_polygon_and_box_file_is_detected():
    p = parse_label_text("0 0.5 0.5 0.2 0.2\n2 0.2 0.2 0.6 0.2 0.6 0.8 0.2 0.8\n")
    assert p.is_mixed
    assert (p.n_plain_boxes, p.n_polygons) == (1, 1)
    assert not parse_label_text("0 0.5 0.5 0.2 0.2\n").is_mixed
    assert not parse_label_text("2 0.2 0.2 0.6 0.2 0.6 0.8 0.2 0.8\n").is_mixed
