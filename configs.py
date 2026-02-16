"""
Global configuration settings for RT-DETRv4 training on chicken detection dataset.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.resolve()

# Dataset configuration
DATASET_CONFIG = {
    # Dataset paths (relative to project root)
    "train_ann_file": "chicken-detection-labelme-format/coco-format/train/_annotations.coco.json",
    "val_ann_file": "chicken-detection-labelme-format/coco-format/valid/_annotations.coco.json",
    
    # Image folders (relative to project root)
    "train_img_folder": "chicken-detection-labelme-format/coco-format/train",
    "val_img_folder": "chicken-detection-labelme-format/coco-format/valid",
    
    # Dataset settings
    "num_classes": 2,  # chicken, not-chicken
    "remap_mscoco_category": False,
    
    # Class names
    "class_names": ["chicken", "not-chicken"],
}

# RT-DETRv4 paths
RTDETRV4_PATH = PROJECT_ROOT / "RT-DETRv4"
RTDETRV4_TRAIN_SCRIPT = RTDETRV4_PATH / "train.py"

# Config file paths (relative to chicken-detection-labelme-format directory)
CONFIG_PATHS = {
    "s": "chicken-detection-labelme-format/configs/rtv4/rtv4_hgnetv2_s_chicken.yml",
    "m": "chicken-detection-labelme-format/configs/rtv4/rtv4_hgnetv2_m_chicken.yml",
    "l": "chicken-detection-labelme-format/configs/rtv4/rtv4_hgnetv2_l_chicken.yml",
    "x": "chicken-detection-labelme-format/configs/rtv4/rtv4_hgnetv2_x_chicken.yml",
}

# Default training settings
DEFAULT_TRAINING_CONFIG = {
    "use_amp": False,  # Set to True to enable Automatic Mixed Precision
    "seed": 0,
    "num_workers": 4,
}

# Output directories (relative to project root)
OUTPUT_DIRS = {
    "s": "models/rt-detr-v4-s",
    "m": "models/rt-detr-v4-m",
    "l": "models/rt-detr-v4-l",
    "x": "models/rt-detr-v4-x",
}
