#!/usr/bin/env python3
"""
Train RT-DETRv4-S (Small) model on chicken detection dataset.

Usage:
    python train-rt-detr-v4-s.py [--use-amp] [--seed SEED] [--resume PATH] [--tune PATH] [--test-only]
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

# Import global configs
import configs


def main():
    """Main training function for RT-DETRv4-S."""
    parser = argparse.ArgumentParser(
        description="Train RT-DETRv4-S model on chicken detection dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--use-amp",
        action="store_true",
        default=configs.DEFAULT_TRAINING_CONFIG["use_amp"],
        help="Use automatic mixed precision training"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=configs.DEFAULT_TRAINING_CONFIG["seed"],
        help="Random seed"
    )
    
    parser.add_argument(
        "-r", "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint path"
    )
    
    parser.add_argument(
        "-t", "--tune",
        type=str,
        default=None,
        help="Fine-tune from checkpoint path"
    )
    
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only run testing/evaluation"
    )
    
    parser.add_argument(
        "--rtdetrv4-path",
        type=str,
        default=None,
        help="Path to RT-DETRv4 repository (if not in standard location)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for models, logs, and checkpoints (overrides config file)"
    )
    
    args = parser.parse_args()
    
    # Determine RT-DETRv4 path
    if args.rtdetrv4_path:
        rtdetrv4_root = Path(args.rtdetrv4_path).resolve()
        train_script = rtdetrv4_root / "train.py"
    else:
        rtdetrv4_root = configs.RTDETRV4_PATH
        train_script = configs.RTDETRV4_TRAIN_SCRIPT
    
    if not train_script.exists():
        print(f"Error: RT-DETRv4 train.py not found at {train_script}")
        print("\nPlease ensure RT-DETRv4 submodule is initialized:")
        print("   git submodule update --init --recursive")
        sys.exit(1)
    
    # Get config file path (use absolute path)
    config_path = (configs.PROJECT_ROOT / configs.CONFIG_PATHS["s"]).resolve()
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Build command (use absolute path for config)
    cmd = [sys.executable, str(train_script), "-c", str(config_path)]
    
    if args.use_amp:
        cmd.append("--use-amp")
    
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    
    if args.resume:
        cmd.extend(["-r", args.resume])
    
    if args.tune:
        cmd.extend(["-t", args.tune])
    
    if args.test_only:
        cmd.append("--test-only")
    
    if args.output_dir:
        cmd.extend(["--update", f"output_dir={args.output_dir}"])
    
    # Print configuration
    print("\n" + "=" * 60)
    print("RT-DETRv4-S Training Configuration")
    print("=" * 60)
    print(f"Config file: {config_path}")
    print(f"Train annotations: {configs.DATASET_CONFIG['train_ann_file']}")
    print(f"Val annotations: {configs.DATASET_CONFIG['val_ann_file']}")
    print(f"Number of classes: {configs.DATASET_CONFIG['num_classes']}")
    print(f"Classes: {', '.join(configs.DATASET_CONFIG['class_names'])}")
    print(f"RT-DETRv4 train.py: {train_script}")
    if args.resume:
        print(f"Resume from: {args.resume}")
    if args.tune:
        print(f"Fine-tune from: {args.tune}")
    if args.test_only:
        print("Mode: Testing/Evaluation only")
    else:
        print("Mode: Training")
    if args.use_amp:
        print("AMP: Enabled")
    print(f"Seed: {args.seed}")
    print("=" * 60 + "\n")
    
    # Change to RT-DETRv4 directory and run
    original_cwd = os.getcwd()
    try:
        os.chdir(rtdetrv4_root)
        print(f"Running: {' '.join(cmd)}\n")
        result = subprocess.run(cmd, check=False)
        sys.exit(result.returncode)
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
