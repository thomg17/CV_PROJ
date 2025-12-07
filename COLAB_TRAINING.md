# Training on Google Colab with MS COCO

This guide explains how to train the distortion-robust watermarking model on Google Colab using the MS COCO dataset.

## Quick Start

### Option 1: Using the Jupyter Notebook (Recommended)

1. **Upload the notebook to Google Colab:**
   - Go to [Google Colab](https://colab.research.google.com/)
   - Click `File` → `Upload notebook`
   - Upload `train_colab.ipynb`

2. **Enable GPU:**
   - Click `Runtime` → `Change runtime type`
   - Select `GPU` as Hardware accelerator (T4 or better)
   - Click `Save`

3. **Run all cells:**
   - Click `Runtime` → `Run all`
   - The notebook will:
     - Clone the repository
     - Install dependencies
     - Download MS COCO dataset (~19GB)
     - Pretrain NECST
     - Train the model
     - Save checkpoints to Google Drive

### Option 2: Manual Setup

1. **Open a new Colab notebook and run:**

```python
# Clone repository
!git clone https://github.com/thomg17/CV_PROJ.git
%cd CV_PROJ
!git checkout Ben

# Install dependencies
!pip install torch torchvision numpy Pillow opencv-python pycocotools tqdm

# Download COCO dataset
!python setup_coco_data.py

# Run training
!python train.py
```

## Dataset Information

### MS COCO 2017
- **Train set:** ~118,000 images (18GB)
- **Validation set:** ~5,000 images (1GB)
- **Total download:** ~19GB

### Quick Testing
If you want to test the training pipeline quickly without downloading the full dataset:

```python
!python setup_coco_data.py --quick
```

This downloads only the validation set (~1GB) for quick testing.

## Training Configuration

The default configuration in `train_colab.ipynb`:

```python
- Image size: 128x128
- Batch size: 16 (adjust based on GPU memory)
- Epochs: 50
- Message length: 30 bits
- NECST redundant length: 60 bits (2x)

Novel features enabled:
- NECST channel coding: ✓
- FFT consistency loss: ✓
- Distortion pool (50% probability): ✓
```

### Adjusting for GPU Memory

If you run out of GPU memory:

**For T4 GPU (16GB):**
- Batch size: 12-16
- Image size: 128x128

**For smaller GPUs:**
```python
training_options = TrainingOptions(
    batch_size=8,  # Reduce batch size
    ...
)
```

## Monitoring Training

The notebook includes live plotting that updates every 5 epochs showing:
- Training vs Validation Loss
- Training vs Validation Bitwise Error

## Saving Checkpoints

Checkpoints are automatically saved:
- Every 5 epochs
- When achieving best validation loss

The final cell copies all checkpoints to Google Drive at:
```
/content/drive/MyDrive/watermark_checkpoints/
```

## Estimated Training Time

On Google Colab with T4 GPU:
- NECST pretraining: ~15-20 minutes
- Per epoch: ~30-45 minutes
- Full 50 epochs: ~25-40 hours

**Tip:** Colab may disconnect after 12 hours. To work around this:
1. Train in smaller chunks (e.g., 10 epochs at a time)
2. Use `start_epoch` parameter to resume
3. Consider Colab Pro for longer sessions

## Resuming Training

To resume from a checkpoint:

```python
# Load checkpoint
checkpoint = torch.load('runs/coco_watermark_epoch_10.pth')
hidden_net.encoder_decoder.load_state_dict(checkpoint['encoder_decoder_state_dict'])
hidden_net.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
hidden_net.optimizer_enc_dec.load_state_dict(checkpoint['optimizer_enc_dec_state_dict'])
hidden_net.optimizer_discrim.load_state_dict(checkpoint['optimizer_discrim_state_dict'])

# Update start epoch
training_options.start_epoch = checkpoint['epoch']
```

## Troubleshooting

### Out of Memory Error
- Reduce batch size
- Reduce image size to 64x64 or 96x96
- Use fewer workers: `num_workers=0`

### Dataset Download Fails
- Retry the download cell
- Check internet connection
- The script automatically resumes if interrupted

### Training is Slow
- Ensure GPU is enabled (`Runtime` → `Change runtime type`)
- Reduce `num_workers` if data loading is bottleneck
- Consider using smaller validation set

## Expected Results

After training, you should see:
- Bitwise error: <0.05 (95%+ accuracy)
- Encoder MSE: <0.01 (imperceptible watermarks)
- Adversarial BCE: ~0.7 (discriminator fooled)

The model will be robust to:
- JPEG compression (quality 50-100)
- Downsampling (0.5x-2x)
- Color jitter
- Random flipping
- Gaussian noise
