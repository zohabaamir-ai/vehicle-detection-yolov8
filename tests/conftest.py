from pathlib import Path

import pytest
from PIL import Image

H1 = "a" * 32
H2 = "b" * 32
H3 = "c" * 32


def _write(split: Path, name: str, label_text: str | None, size=(64, 48)):
    (split / "images").mkdir(parents=True, exist_ok=True)
    (split / "labels").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "gray").save(split / "images" / f"{name}.jpg")
    if label_text is not None:
        (split / "labels" / f"{name}.txt").write_text(label_text)


def build_dataset(root: Path) -> Path:
    """A tiny fake dataset with planted problems (see tests/test_audit.py)."""
    train, test = root / "train", root / "test"
    _write(train, f"road_jpg.rf.{H1}", "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")
    _write(train, f"road_jpg.rf.{H2}", "0 0.5 0.5 0.2 0.2\n")  # augmented copy of same source
    _write(train, f"poly_jpg.rf.{H1}", "2 0.2 0.2 0.6 0.2 0.6 0.8 0.2 0.8\n")
    _write(train, f"empty_jpg.rf.{H1}", "")
    _write(train, f"nolabel_jpg.rf.{H1}", None)
    _write(train, f"tiny_jpg.rf.{H1}", "0 0.5 0.5 0.01 0.01\n0 0.9 0.5 0.3 0.1\n")
    _write(test, f"road_jpg.rf.{H3}", "0 0.5 0.5 0.2 0.2\n")  # same source as a train image
    _write(test, f"other_jpg.rf.{H1}", "2 0.5 0.5 0.4 0.4\n", size=(32, 32))
    return root


@pytest.fixture
def write_sample():
    """Fixture giving the helper that writes one image + optional label file."""
    return _write


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    return build_dataset(tmp_path / "ObjectsDataset")
