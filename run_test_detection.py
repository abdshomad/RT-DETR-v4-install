#!/usr/bin/env python3
"""
Run RT-DETRv4-S detection on chicken-detection-labelme-format/coco-format/test images
using the latest checkpoint and draw results with supervision. Saves to test-result.
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image
import supervision as sv

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent
# RT-DETRv4 must be on path for engine imports
RTDETRV4_ROOT = PROJECT_ROOT / "RT-DETRv4"
sys.path.insert(0, str(RTDETRV4_ROOT))

from engine.core import YAMLConfig

# Paths
CONFIG_PATH = PROJECT_ROOT / "chicken-detection-labelme-format/configs/rtv4/rtv4_hgnetv2_s_chicken.yml"
CHECKPOINT_DIR = RTDETRV4_ROOT / "models/rt-detr-v4-s"
TEST_IMAGE_DIR = PROJECT_ROOT / "chicken-detection-labelme-format/coco-format/test"
OUTPUT_DIR = PROJECT_ROOT / "chicken-detection-labelme-format/coco-format/test-result"

CLASS_NAMES = ["chicken", "not-chicken"]
CONFIDENCE_THRESHOLD = 0.25
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def get_latest_checkpoint(checkpoint_dir: Path) -> Path:
    """Prefer best_stg1.pth, else last.pth, else newest .pth."""
    best = checkpoint_dir / "best_stg1.pth"
    if best.exists():
        return best
    last = checkpoint_dir / "last.pth"
    if last.exists():
        return last
    pths = list(checkpoint_dir.glob("*.pth"))
    if not pths:
        raise FileNotFoundError(f"No .pth checkpoint found in {checkpoint_dir}")
    return max(pths, key=lambda p: p.stat().st_mtime)


def load_model(config_path: Path, checkpoint_path: Path, device: str = "cuda:0"):
    """Load RT-DETRv4 model from config and checkpoint."""
    cfg = YAMLConfig(str(config_path), resume=str(checkpoint_path))

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    else:
        state = checkpoint["model"]

    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(device)
    model.eval()
    return model


def run_inference(model, image_path: Path, device: str):
    """Run detection on one image. Returns (labels, boxes_xyxy, scores) on CPU."""
    im_pil = Image.open(image_path).convert("RGB")
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]], dtype=torch.float32).to(device)

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        labels, boxes, scores = model(im_data, orig_size)

    # Batch size 1
    labels = labels[0].cpu().numpy()
    boxes = boxes[0].cpu().numpy()
    scores = scores[0].cpu().numpy()
    return im_pil, labels, boxes, scores


def to_supervision_detections(labels, boxes, scores, conf_thr: float):
    """Convert model output to supervision Detections (filter by confidence)."""
    mask = scores >= conf_thr
    if not np.any(mask):
        return sv.Detections.empty()

    xyxy = boxes[mask].astype(np.float32)
    class_id = labels[mask].astype(int)
    confidence = scores[mask].astype(np.float32)
    return sv.Detections(xyxy=xyxy, class_id=class_id, confidence=confidence)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run RT-DETRv4-S on test images, draw with supervision")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (cuda:0 or cpu)")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path (default: latest in model dir)")
    args = parser.parse_args()

    if not CONFIG_PATH.exists():
        print(f"Config not found: {CONFIG_PATH}")
        sys.exit(1)
    if not CHECKPOINT_DIR.exists():
        print(f"Checkpoint dir not found: {CHECKPOINT_DIR}")
        sys.exit(1)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else get_latest_checkpoint(CHECKPOINT_DIR)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    print(f"Using checkpoint: {checkpoint_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    model = load_model(CONFIG_PATH, checkpoint_path, device)

    image_paths = [
        p for p in TEST_IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    image_paths.sort()
    print(f"Found {len(image_paths)} images in {TEST_IMAGE_DIR}")

    # Chicken = green, not-chicken = red (class_id 0 and 1)
    color_palette = sv.ColorPalette([
        sv.Color.from_hex("#00FF00"),  # chicken
        sv.Color.from_hex("#0000FF"),   # not-chicken
    ])
    box_annotator = sv.BoxAnnotator(thickness=2, color=color_palette)
    label_annotator = sv.LabelAnnotator(
        text_position=sv.Position.TOP_LEFT, text_scale=0.5, color=color_palette
    )

    for i, image_path in enumerate(image_paths):
        im_pil, labels, boxes, scores = run_inference(model, image_path, device)
        detections = to_supervision_detections(labels, boxes, scores, args.conf)
        frame = np.array(im_pil)

        if detections.xyxy is not None and len(detections.xyxy) > 0:
            labels_sv = [
                f"{CLASS_NAMES[c]} {s:.2f}"
                for c, s in zip(detections.class_id, detections.confidence)
            ]
            frame = box_annotator.annotate(scene=frame, detections=detections)
            # frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels_sv)

        out_path = OUTPUT_DIR / image_path.name
        Image.fromarray(frame).save(out_path)
        if (i + 1) % 50 == 0 or (i + 1) == len(image_paths):
            print(f"  Saved {i + 1}/{len(image_paths)} -> {out_path.name}")

    print(f"Done. Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
