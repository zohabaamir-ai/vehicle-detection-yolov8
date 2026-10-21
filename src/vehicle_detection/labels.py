"""Parse and validate YOLO-format label files. No torch needed."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

EPS = 1e-6


@dataclass(frozen=True)
class Box:
    """One object: class id plus a normalised centre/size box (all in 0..1)."""

    cls: int
    xc: float
    yc: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return self.w * self.h


@dataclass
class ParsedLabel:
    boxes: list[Box] = field(default_factory=list)
    n_polygons: int = 0
    n_plain_boxes: int = 0
    problems: list[str] = field(default_factory=list)

    @property
    def is_mixed(self) -> bool:
        """True if the file has both polygon lines and plain box lines."""
        return self.n_polygons > 0 and self.n_plain_boxes > 0


def polygon_to_box(cls_id: int, coords: list[float]) -> Box:
    """Turn polygon points (x1 y1 x2 y2 ...) into their bounding box."""
    xs, ys = coords[0::2], coords[1::2]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return Box(cls_id, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)


def parse_label_text(text: str) -> ParsedLabel:
    """Parse the text of one label file.

    Lines with 5 numbers are boxes. Lines with an even number (>= 6) of numbers after
    the class id are polygons; they are converted to their bounding box and counted.
    Anything else is recorded as a problem and skipped.
    """
    out = ParsedLabel()
    for lineno, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            out.problems.append(f"line {lineno}: non-numeric value")
            continue
        if not all(math.isfinite(n) for n in nums):
            out.problems.append(f"line {lineno}: nan or inf value")
            continue
        cls_f, values = nums[0], nums[1:]
        if cls_f < 0 or cls_f != int(cls_f):
            out.problems.append(f"line {lineno}: bad class id {parts[0]}")
            continue
        cls_id = int(cls_f)
        if len(values) == 4:
            out.n_plain_boxes += 1
            box = Box(cls_id, *values)
        elif len(values) >= 6 and len(values) % 2 == 0:
            out.n_polygons += 1
            box = polygon_to_box(cls_id, values)
        else:
            out.problems.append(f"line {lineno}: {len(values)} numbers after class id")
            continue
        if box.w <= EPS or box.h <= EPS:
            out.problems.append(f"line {lineno}: zero-size box")
            continue
        if (
            box.xc - box.w / 2 < -EPS
            or box.xc + box.w / 2 > 1 + EPS
            or box.yc - box.h / 2 < -EPS
            or box.yc + box.h / 2 > 1 + EPS
        ):
            out.problems.append(f"line {lineno}: box outside the image")
        out.boxes.append(box)
    return out
