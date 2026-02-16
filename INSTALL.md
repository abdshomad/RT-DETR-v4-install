# Installation Guide

Complete step-by-step installation guide for RT-DETRv4 training setup on chicken detection dataset.

## 📋 Prerequisites

Before starting, ensure you have:

- **Python 3.11+** installed
- **CUDA-capable GPU(s)** with NVIDIA drivers installed
- **Git** installed
- **`uv`** package manager installed ([Installation guide](https://github.com/astral-sh/uv))
- **nvidia-smi** available (for GPU monitoring)

### Verify Prerequisites

```bash
# Check Python version
python3 --version  # Should be 3.11 or higher

# Check CUDA availability
nvidia-smi  # Should show GPU information

# Check uv installation
uv --version
```

## 🚀 Installation Steps

### Step 1: Clone Repository and Initialize Submodules

```bash
# If cloning the repository for the first time
git clone <repository-url>
cd RT-DETR-v4-install

# Initialize git submodules (RT-DETRv4 and dataset)
git submodule update --init --recursive
```

**Important:** The repository uses git submodules for:
- `RT-DETRv4/` - The RT-DETRv4 framework
- `chicken-detection-labelme-format/` - The dataset and configs

### Step 2: Set Up Python Virtual Environment

```bash
# Create virtual environment using uv
uv venv

# Sync dependencies from pyproject.toml
uv sync
```

This will:
- Create a virtual environment in `.venv/`
- Install all required dependencies (PyTorch, torchvision, etc.)

### Step 3: Activate Virtual Environment (Optional)

While `uv run` can execute commands directly, you can also activate the environment:

```bash
source .venv/bin/activate
# On Windows: .venv\Scripts\activate
```

### Step 4: Set Up DINOv3 Teacher Model

RT-DETRv4 uses DINOv3 as a teacher model for knowledge distillation. This requires two components:

#### 4.1 Clone DINOv3 Repository

The DINOv3 repository should be cloned into the RT-DETRv4 directory:

```bash
cd RT-DETRv4
git clone https://github.com/facebookresearch/dinov3.git dinov3
cd ..
```

**Verify:** Check that `RT-DETRv4/dinov3/hubconf.py` exists.

#### 4.2 Download DINOv3 Pretrained Weights

Download the DINOv3 ViT-B/16 pretrained weights:

```bash
# Create pretrain directory if it doesn't exist
mkdir -p RT-DETRv4/pretrain

# Download the weights file
# Note: You may need to obtain a valid download URL from Meta's DINOv3 website
# The file should be saved as: RT-DETRv4/pretrain/dinov3_vitb16_pretrain_lvd1689m.pth
```

**Download URL:** The download URL is available in `secrets/DINOv3-download.md`. The file name should be:
- `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`

**File Size:** Approximately ~1.1 GB

**Verify:** Check that `RT-DETRv4/pretrain/dinov3_vitb16_pretrain_lvd1689m.pth` exists.

### Step 5: Verify Configuration

Verify that all configuration files are in place:

```bash
# Check config files exist
ls chicken-detection-labelme-format/configs/rtv4/rtv4_hgnetv2_*_chicken.yml

# Check dataset files exist
ls chicken-detection-labelme-format/coco-format/train/_annotations.coco.json
ls chicken-detection-labelme-format/coco-format/valid/_annotations.coco.json

# Check training scripts exist
ls train-rt-detr-v4-*.py
```

### Step 6: Verify Dataset Structure

Ensure your dataset follows this structure:

```
chicken-detection-labelme-format/
└── coco-format/
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

## 🔧 Configuration

### Global Configuration (`configs.py`)

The `configs.py` file contains global settings:
- Dataset paths (training and validation annotations)
- Number of classes: 2 (chicken, not-chicken)
- RT-DETRv4 paths
- Default training settings

### Model-Specific Configs

Each model variant has its own YAML configuration:
- **Small (S):** `chicken-detection-labelme-format/configs/rtv4/rtv4_hgnetv2_s_chicken.yml`
- **Medium (M):** `chicken-detection-labelme-format/configs/rtv4/rtv4_hgnetv2_m_chicken.yml`
- **Large (L):** `chicken-detection-labelme-format/configs/rtv4/rtv4_hgnetv2_l_chicken.yml`
- **X-Large (X):** `chicken-detection-labelme-format/configs/rtv4/rtv4_hgnetv2_x_chicken.yml`

### Batch Size Configuration

Batch sizes have been optimized to avoid OOM errors:

| Model | Train Batch Size | Val Batch Size | Notes |
|-------|------------------|----------------|-------|
| Small (S) | 16 | 16 | Reduced from 32/64 |
| Medium (M) | 12 | 12 | Reduced from 32/64 |
| Large (L) | 8 | 8 | Reduced from 32/64 |
| X-Large (X) | 8 | 8 | Reduced from 32/64 |

To adjust batch sizes, edit the `total_batch_size` in the respective config files.

## 🧪 Testing Installation

### Quick Test

Run a quick test to verify everything is set up correctly:

```bash
# Test import of key modules
uv run python -c "import torch; import torchvision; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

### Test Training Script

Verify the training script can load configurations:

```bash
# This should print configuration without errors
uv run python train-rt-detr-v4-s.py --test-only --resume /nonexistent 2>&1 | head -20
```

## 🎯 First Training Run

### Free GPU Memory (Recommended)

Before training, free up GPU memory:

```bash
# List processes using GPUs
uv run python free_gpu.py

# Kill all GPU processes (if needed)
uv run python free_gpu.py --kill
```

### Start Training

```bash
# Train Small variant with AMP
uv run train-rt-detr-v4-s.py --use-amp

# Or activate venv and run directly
source .venv/bin/activate
python train-rt-detr-v4-s.py --use-amp
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Git Submodules Not Initialized

**Error:** `FileNotFoundError: RT-DETRv4/train.py not found`

**Solution:**
```bash
git submodule update --init --recursive
```

#### 2. DINOv3 Repository Missing

**Error:** `FileNotFoundError: RT-DETRv4/dinov3/hubconf.py`

**Solution:**
```bash
cd RT-DETRv4
git clone https://github.com/facebookresearch/dinov3.git dinov3
cd ..
```

#### 3. DINOv3 Weights Missing

**Error:** `FileNotFoundError: pretrain/dinov3_vitb16_pretrain_lvd1689m.pth`

**Solution:**
- Download weights from Meta's DINOv3 website
- Save to `RT-DETRv4/pretrain/dinov3_vitb16_pretrain_lvd1689m.pth`
- See `secrets/DINOv3-download.md` for download URL

#### 4. CUDA Out of Memory (OOM)

**Error:** `torch.OutOfMemoryError: CUDA out of memory`

**Solutions:**
1. **Free GPU memory:**
   ```bash
   uv run python free_gpu.py --kill
   ```

2. **Reduce batch size** in config files:
   - Edit `total_batch_size` in model config YAML files
   - Start with smaller values (e.g., 4 or 8)

3. **Use smaller model variant:**
   - Try Small (S) instead of Medium/Large/X-Large

4. **Enable gradient checkpointing** (if supported)

#### 5. Missing Python Dependencies

**Error:** `ModuleNotFoundError: No module named 'X'`

**Solution:**
```bash
# Sync dependencies
uv sync

# Or add missing package
uv add <package-name>
```

#### 6. YAML Configuration Errors

**Error:** `yaml.parser.ParserError: while parsing a block mapping`

**Solution:**
- Check YAML indentation (must use 2 spaces, not tabs)
- Verify all nested blocks are properly indented
- Check for missing colons or incorrect list syntax

#### 7. Path Resolution Errors

**Error:** `FileNotFoundError: [Errno 2] No such file or directory`

**Solution:**
- Verify dataset paths in `chicken-detection-labelme-format/configs/dataset/chicken_detection.yml`
- Ensure paths are relative to `RT-DETRv4/` directory (where train.py runs)
- Check that annotation files exist at specified paths

#### 8. Distributed Training Errors

**Error:** `ValueError: Default process group has not been initialized`

**Solution:**
- This is normal for single-GPU training
- The code has been fixed to handle non-distributed mode
- If using multi-GPU, ensure `torchrun` is used correctly

### Getting Help

If you encounter issues not covered here:

1. Check the error message carefully
2. Verify all prerequisites are met
3. Ensure all submodules and dependencies are installed
4. Check GPU memory availability with `nvidia-smi`
5. Review the `README.md` for additional information

## 📦 Dependency Management

### Adding Dependencies

```bash
# Add a new package
uv add <package-name>

# Add a development dependency
uv add --dev <package-name>
```

### Updating Dependencies

```bash
# Sync with pyproject.toml
uv sync

# Update a specific package
uv add <package-name>@latest
```

### Current Dependencies

See `pyproject.toml` for the complete list. Key dependencies include:
- `torch` - PyTorch deep learning framework
- `torchvision` - Computer vision utilities
- `faster-coco-eval` - COCO evaluation metrics
- `PyYAML` - YAML configuration parsing
- `tensorboard` - Training visualization
- `transformers` - Hugging Face transformers (for DINOv3)
- `torchmetrics` - PyTorch metrics
- `termcolor` - Colored terminal output

## 🔄 Updating the Installation

### Update Git Submodules

```bash
# Update submodules to latest commits
git submodule update --remote

# Or update to specific commits
cd RT-DETRv4
git pull origin main
cd ..
```

**⚠️ Important:** Do not modify files within submodule directories directly. See `AGENTS.md` for more information.

### Update Python Dependencies

```bash
# Sync dependencies after pyproject.toml changes
uv sync
```

## ✅ Installation Checklist

Use this checklist to verify your installation:

- [ ] Python 3.11+ installed
- [ ] CUDA and GPU drivers installed (`nvidia-smi` works)
- [ ] `uv` package manager installed
- [ ] Repository cloned
- [ ] Git submodules initialized (`RT-DETRv4/` and `chicken-detection-labelme-format/`)
- [ ] Python virtual environment created (`uv venv`)
- [ ] Dependencies installed (`uv sync`)
- [ ] DINOv3 repository cloned (`RT-DETRv4/dinov3/`)
- [ ] DINOv3 weights downloaded (`RT-DETRv4/pretrain/dinov3_vitb16_pretrain_lvd1689m.pth`)
- [ ] Dataset files present (train/valid annotations)
- [ ] Configuration files present (model configs)
- [ ] Training scripts present (`train-rt-detr-v4-*.py`)
- [ ] GPU memory available (checked with `nvidia-smi`)

## 📚 Next Steps

After installation:

1. **Review Configuration:** Check `configs.py` and model-specific YAML files
2. **Free GPU Memory:** Run `python free_gpu.py --kill` if needed
3. **Start Training:** Run `uv run train-rt-detr-v4-s.py --use-amp`
4. **Monitor Training:** Use TensorBoard or check log files
5. **Evaluate Models:** Use `--test-only` flag for evaluation

For more information, see:
- `README.md` - Project overview and usage
- `AGENTS.md` - Development guidelines
- Model config files - Training hyperparameters

## 📄 License

See the RT-DETRv4 repository for license information.
