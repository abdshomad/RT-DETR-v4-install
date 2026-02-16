#!/usr/bin/env python3
"""
Train all RT-DETRv4 variants (S, M, L, X) on chicken detection dataset.

Runs training sequentially for each variant: Small -> Medium -> Large -> X-Large.

Usage:
    python train-rt-detr-v4-all-variants.py [--use-amp] [--seed SEED] [--test-only]
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

import torch

# Import global configs
import configs

VARIANTS = ["s", "m", "l", "x"]
VARIANT_NAMES = {"s": "Small", "m": "Medium", "l": "Large", "x": "X-Large"}


def main():
    """Main training function - runs all RT-DETRv4 variants sequentially."""
    parser = argparse.ArgumentParser(
        description="Train all RT-DETRv4 variants (S, M, L, X) on chicken detection dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--use-amp",
        action="store_true",
        default=configs.DEFAULT_TRAINING_CONFIG["use_amp"],
        help="Use automatic mixed precision training",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=configs.DEFAULT_TRAINING_CONFIG["seed"],
        help="Random seed",
    )

    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only run testing/evaluation",
    )

    parser.add_argument(
        "--rtdetrv4-path",
        type=str,
        default=None,
        help="Path to RT-DETRv4 repository (if not in standard location)",
    )

    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        choices=VARIANTS,
        default=VARIANTS,
        help=f"Variants to train (default: all {VARIANTS})",
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

    # Detect available GPUs
    num_gpus = torch.cuda.device_count()
    use_distributed = num_gpus >= 1
    master_port = 29500

    # Print overall configuration
    print("\n" + "=" * 60)
    print("RT-DETRv4 All Variants Training")
    print("=" * 60)
    print(f"Variants: {[VARIANT_NAMES[v] for v in args.variants]}")
    print(f"RT-DETRv4 train.py: {train_script}")
    print(f"GPUs: {num_gpus} available, using {'torchrun (distributed)' if use_distributed else 'single process (CPU)'}")
    print(f"AMP: {'Enabled' if args.use_amp else 'Disabled'}")
    print(f"Seed: {args.seed}")
    print(f"Mode: {'Testing only' if args.test_only else 'Training'}")
    print("=" * 60 + "\n")

    original_cwd = os.getcwd()
    results = []

    try:
        os.chdir(rtdetrv4_root)

        for variant in args.variants:
            config_path = (
                configs.PROJECT_ROOT / configs.CONFIG_PATHS[variant]
            ).resolve()
            if not config_path.exists():
                print(f"Error: Config file not found for {VARIANT_NAMES[variant]}: {config_path}")
                results.append((variant, 1))
                continue

            if use_distributed:
                cmd = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    f"--nproc_per_node={num_gpus}",
                    f"--master_port={master_port}",
                    str(train_script),
                    "-c",
                    str(config_path),
                ]
            else:
                cmd = [sys.executable, str(train_script), "-c", str(config_path)]

            if args.use_amp:
                cmd.append("--use-amp")
            if args.seed is not None:
                cmd.extend(["--seed", str(args.seed)])
            if args.test_only:
                cmd.append("--test-only")

            print("\n" + "-" * 60)
            print(f"Training RT-DETRv4-{VARIANT_NAMES[variant]} ({variant})")
            print("-" * 60)
            print(f"Config: {config_path}")
            print(f"Running: {' '.join(cmd)}\n")

            result = subprocess.run(cmd)
            results.append((variant, result.returncode))

            if result.returncode != 0:
                print(
                    f"\nWarning: RT-DETRv4-{VARIANT_NAMES[variant]} exited with code {result.returncode}"
                )
                # Continue with next variant; user may want to see all results

        os.chdir(original_cwd)

        # Summary
        print("\n" + "=" * 60)
        print("Training Summary")
        print("=" * 60)
        for variant, code in results:
            status = "OK" if code == 0 else f"FAILED ({code})"
            print(f"  RT-DETRv4-{VARIANT_NAMES[variant]}: {status}")
        print("=" * 60 + "\n")

        # Exit with first non-zero code, or 0 if all succeeded
        exit_code = next((c for _, c in results if c != 0), 0)
        sys.exit(exit_code)

    except KeyboardInterrupt:
        os.chdir(original_cwd)
        print("\n\nTraining interrupted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
