# RT-DETR-v4-install

Training setup for RT-DETRv4 on chicken detection dataset using COCO format annotations.

## 📋 Overview

This repository provides a streamlined setup for training RT-DETRv4 models on a custom chicken detection dataset. The dataset contains 2 classes: `chicken` and `not-chicken`, formatted in COCO annotation format.

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- CUDA-capable GPU(s)
- Git submodules initialized

### Setup

1. **Initialize git submodules:**
   ```bash
   git submodule update --init --recursive
   ```

2. **Create virtual environment and install dependencies:**
   ```bash
   uv venv
   uv sync
   ```

3. **Activate virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

### Dataset Structure

The dataset is located in `chicken-detection-labelme-format/coco-format/`:

```
coco-format/
├── train/
│   ├── _annotations.coco.json
│   └── [training images]
├── valid/
│   ├── _annotations.coco.json
│   └── [validation images]
└── test/
    ├── _annotations.coco.json
    └── [test images]
```

**Dataset Configuration:**
- **Training annotations:** `chicken-detection-labelme-format/coco-format/train/_annotations.coco.json`
- **Validation annotations:** `chicken-detection-labelme-format/coco-format/valid/_annotations.coco.json`
- **Number of classes:** 2 (chicken, not-chicken)
- **Category remapping:** Disabled (`remap_mscoco_category: False`)

## 🎯 Training

### Train All Variants (Recommended)

Train all four model variants (Small → Medium → Large → X-Large) sequentially. **Automatically uses all available GPUs** via distributed training:

```bash
# Using the shell script (via uv venv)
./train-rt-detr-v4-all-variants.sh --use-amp

# Or directly with uv
uv run train-rt-detr-v4-all-variants.py --use-amp
```

Options:
- `--variants s m l x` — Train specific variants only (default: all)
- `--use-amp` — Enable mixed precision
- `--test-only` — Evaluation only

### Single Variant Training

Train any of the four model variants (Small, Medium, Large, X-Large):

```bash
# Small variant
uv run train-rt-detr-v4-s.py --use-amp

# Medium variant
uv run train-rt-detr-v4-m.py --use-amp

# Large variant
uv run train-rt-detr-v4-l.py --use-amp

# X-Large variant
uv run train-rt-detr-v4-x.py --use-amp
```

### Multi-GPU Training

The **train-rt-detr-v4-all-variants.py** script automatically detects and uses all available GPUs via `torchrun`. For single-variant multi-GPU training, use `torchrun` directly:

```bash
# Example: Train Small variant on 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun \
    --master_port=7777 \
    --nproc_per_node=4 \
    train-rt-detr-v4-s.py \
    --use-amp \
    --seed=0
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--use-amp` | Enable Automatic Mixed Precision training | False |
| `--seed SEED` | Random seed for reproducibility | 0 |
| `-r, --resume PATH` | Resume training from checkpoint (single-variant scripts) | None |
| `-t, --tune PATH` | Fine-tune from checkpoint (single-variant scripts) | None |
| `--test-only` | Only run evaluation/testing | False |
| `--variants s m l x` | Variants to train (all-variants script only) | All |
| `--rtdetrv4-path PATH` | Path to RT-DETRv4 repository | `RT-DETRv4` |
| `--output-dir DIR` | Output directory (single-variant scripts) | From config file |

### Examples

#### Train All Variants with AMP (uses all GPUs)
```bash
./train-rt-detr-v4-all-variants.sh --use-amp
```

#### Train Specific Variants Only
```bash
uv run train-rt-detr-v4-all-variants.py --variants s m --use-amp
```

#### Train Single Variant with AMP
```bash
uv run train-rt-detr-v4-s.py --use-amp
```

#### Resume Training
```bash
uv run train-rt-detr-v4-m.py --resume models/rt-detr-v4-m/checkpoint.pth
```

#### Fine-tune from Pre-trained Model
```bash
uv run train-rt-detr-v4-l.py --tune path/to/pretrained_model.pth --use-amp
```

#### Evaluation Only
```bash
uv run train-rt-detr-v4-x.py --test-only --resume models/rt-detr-v4-x/checkpoint.pth
```

## ⚙️ Configuration

### Global Configuration (`configs.py`)

The `configs.py` file contains global settings used by all training scripts:

- **Dataset paths:** Training and validation annotation files
- **Number of classes:** 2 (chicken, not-chicken)
- **RT-DETRv4 paths:** Paths to RT-DETRv4 repository and training script
- **Config file paths:** Paths to model-specific YAML configs
- **Default training settings:** AMP, seed, num_workers, etc.

### Model-Specific Configs

Each model variant has its own YAML configuration file:

- **Small (S):** `chicken-detection-labelme-format/configs/rtv4/rtv4_hgnetv2_s_chicken.yml`
- **Medium (M):** `chicken-detection-labelme-format/configs/rtv4/rtv4_hgnetv2_m_chicken.yml`
- **Large (L):** `chicken-detection-labelme-format/configs/rtv4/rtv4_hgnetv2_l_chicken.yml`
- **X-Large (X):** `chicken-detection-labelme-format/configs/rtv4/rtv4_hgnetv2_x_chicken.yml`

These configs include:
- Model architecture settings
- Training epochs (S: 132, M: 102, L/X: 58)
- Learning rate schedules
- Data augmentation policies
- DINOv3 teacher model configuration

### Dataset Config

The dataset configuration is in:
`chicken-detection-labelme-format/configs/dataset/chicken_detection.yml`

This file specifies:
- Dataset paths (images and annotations)
- Number of classes
- DataLoader settings
- Transform configurations

## 📁 Project Structure

```
RT-DETR-v4-install/
├── configs.py                          # Global configuration
├── train-rt-detr-v4-all-variants.py   # Train all variants (auto multi-GPU)
├── train-rt-detr-v4-all-variants.sh   # One-liner to run via uv venv
├── train-rt-detr-v4-s.py              # Small variant training script
├── train-rt-detr-v4-m.py              # Medium variant training script
├── train-rt-detr-v4-l.py              # Large variant training script
├── train-rt-detr-v4-x.py              # X-Large variant training script
├── RT-DETRv4/                         # RT-DETRv4 submodule
├── chicken-detection-labelme-format/  # Dataset submodule
│   ├── coco-format/                   # COCO format dataset
│   │   ├── train/
│   │   ├── valid/
│   │   └── test/
│   └── configs/                       # Model configs
│       ├── dataset/
│       └── rtv4/
└── pyproject.toml                      # Python dependencies
```

## 🔧 Dependencies

Dependencies are managed via `uv` and `pyproject.toml`. Install with:

```bash
uv sync
```

## 📝 Notes

### DINOv3 Teacher Model

RT-DETRv4 uses DINOv3 as a teacher model for knowledge distillation. Ensure the DINOv3 repository and weights are configured in the model config files:

```yaml
teacher_model:
  type: "DINOv3TeacherModel"
  dinov3_repo_path: dinov3/
  dinov3_weights_path: pretrain/dinov3_vitb16_pretrain_lvd1689m.pth
```

### Output Directories

Trained models, logs, and checkpoints are saved to:
- **Small:** `models/rt-detr-v4-s/`
- **Medium:** `models/rt-detr-v4-m/`
- **Large:** `models/rt-detr-v4-l/`
- **X-Large:** `models/rt-detr-v4-x/`

### Training Epochs

Default training epochs per variant:
- **Small (S):** 132 epochs
- **Medium (M):** 102 epochs
- **Large (L):** 58 epochs
- **X-Large (X):** 58 epochs

## 🤝 Contributing

This repository uses git submodules. When updating submodules:

```bash
git submodule update --remote
```

**⚠️ Important:** Do not modify files within submodule directories directly. See `AGENTS.md` for more information.

## 📚 References

- [RT-DETRv4 Paper](https://arxiv.org/abs/2510.25257)
- [RT-DETRv4 Repository](https://github.com/RT-DETRs/RT-DETRv4)
- [DINOv3 Repository](https://github.com/facebookresearch/dinov3)

## 📄 License

See the RT-DETRv4 repository for license information.
