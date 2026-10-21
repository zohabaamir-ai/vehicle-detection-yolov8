from vehicle_detection.crop_sheet import box_to_pixels, collect_boxes, make_sheet
from vehicle_detection.labels import Box


def test_box_to_pixels_is_clipped_to_image():
    left, top, right, bottom = box_to_pixels(Box(0, 0.95, 0.5, 0.2, 0.2), 100, 100)
    assert 0 <= left < right <= 100
    assert 0 <= top < bottom <= 100


def test_collect_and_make_sheet(dataset_root):
    items = collect_boxes(dataset_root, 0)
    assert len(items) == 5
    sheet = make_sheet(items, n=4, seed=0, cell=64)
    assert sheet.size == (6 * 64, 64)
