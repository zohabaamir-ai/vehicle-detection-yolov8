"""Project-wide settings. Change them here, nowhere else."""

# Lower-case COCO names on purpose: Ultralytics copies the pretrained detection-head rows for
# classes whose names match COCO's (case-insensitive), so all three classes get a warm start.
CLASS_NAMES = ("car", "motorcycle", "truck")
SPLIT_NAMES = ("train", "val", "test")
SPLIT_RATIOS = (0.70, 0.15, 0.15)
SEED = 42  # seed for training runs and for the stand-alone split helper
SPLIT_SEED_CANDIDATES = 50  # split seeds tried; the best-balanced one is kept
DEDUP_THRESHOLD = 12  # max hash distance (of 64) for two images to count as the same scene
ISOLATED_MIN_DISTANCE = 16  # a val/test image is "isolated" if no train image is within this
