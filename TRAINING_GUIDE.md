# Training Guide

This guide explains how to train the baseline and full models for comparison.

## Two Models for Comparison

### 1. Baseline Model
**File:** `train_baseline.py`

Standard HiDDeN model WITHOUT novel contributions:
- ❌ No NECST channel coding
- ❌ No FFT consistency loss
- ❌ No distortion pool (only AttackNetwork)

### 2. Full Model
**File:** `train_full.py`

Enhanced model WITH all novel contributions:
- ✅ NECST channel coding
- ✅ FFT consistency loss
- ✅ HybridDistorter with distortion pool

## Local Training

### Train Baseline Model
```bash
cd CV_PROJ
python train_baseline.py
```

Checkpoints saved to: `runs/baseline_model_epoch_*.pth`

### Train Full Model
```bash
cd CV_PROJ
python train_full.py
```

Checkpoints saved to: `runs/full_model_epoch_*.pth`

## Google Colab Training

### Option 1: Modify the Colab Notebook

In `train_colab.ipynb`, change the configuration import:

**For Baseline:**
```python
from configs import get_baseline_config, get_training_options

hidden_config = get_baseline_config()
training_options = get_training_options('baseline_model')
```

**For Full Model:**
```python
from configs import get_full_config, get_training_options

hidden_config = get_full_config()
training_options = get_training_options('full_model')
```

### Option 2: Run Training Scripts Directly

In Colab, after cloning the repo:

**Baseline:**
```python
!python train_baseline.py
```

**Full:**
```python
!python train_full.py
```

## Configurations Overview

All configurations are in `configs.py`:

```python
from configs import (
    get_baseline_config,    # Baseline model
    get_full_config,        # Full model with all features
    get_necst_only_config,  # Only NECST enabled
    get_fft_only_config,    # Only FFT loss enabled
    get_distortion_only_config,  # Only distortion pool enabled
    get_training_options    # Training hyperparameters
)
```

## Comparing Results

After training both models, compare:

### Metrics to Compare:
1. **Bitwise Error** - Lower is better (recovery accuracy)
2. **Encoder MSE** - Lower is better (imperceptibility)
3. **Total Loss** - Lower is better (overall performance)
4. **FFT Loss** - Only for full model (frequency preservation)

### Expected Results:

**Baseline Model:**
- Good baseline performance
- May struggle with severe distortions
- No frequency domain preservation

**Full Model:**
- Better robustness to distortions (distortion pool)
- Better error correction (NECST)
- Better frequency preservation (FFT loss)
- Should show improved metrics overall

## Quick Test Run

For quick testing with smaller dataset:

```bash
# Download only validation set
python setup_coco_data.py --quick

# Modify configs.py training options:
# batch_size=8, number_of_epochs=5
```

## Checkpoint Format

Checkpoints contain:
```python
{
    'epoch': int,
    'encoder_decoder_state_dict': state_dict,
    'discriminator_state_dict': state_dict,
    'optimizer_enc_dec_state_dict': state_dict,
    'optimizer_discrim_state_dict': state_dict
}
```

## Loading Checkpoints

```python
checkpoint = torch.load('runs/baseline_model_epoch_50.pth')
hidden_net.encoder_decoder.load_state_dict(checkpoint['encoder_decoder_state_dict'])
hidden_net.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
```

## Training Time Estimates

On Google Colab T4 GPU:

**Baseline Model:**
- NECST pretraining: 0 min (disabled)
- Per epoch: ~25-35 minutes
- 50 epochs: ~20-30 hours

**Full Model:**
- NECST pretraining: ~15-20 minutes
- Per epoch: ~30-45 minutes
- 50 epochs: ~25-40 hours

## Troubleshooting

**Out of Memory:**
- Reduce batch size in `configs.py`
- Reduce image size (H, W parameters)

**Import Errors:**
- Make sure you're in CV_PROJ directory
- Check all files are present after git clone

**NECST Pretraining Takes Too Long:**
- Reduce `necst_iter` in configs (e.g., 5000 instead of 10000)
