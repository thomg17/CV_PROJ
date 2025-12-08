"""
Training script for BASELINE model
(HiDDeN without novel contributions: no NECST, no FFT loss, no distortion pool)
"""

import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image
from model.hidden import Hidden
from configs import get_baseline_config, get_training_options


class ImageDataset(Dataset):
    """Custom dataset for loading images from a flat directory (e.g., COCO)"""
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.image_files = [
            f for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, 0  # Return dummy label (0) since we don't need classes


def create_data_loaders(train_folder, validation_folder, batch_size, image_size):
    """Create data loaders for training and validation"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    train_dataset = ImageDataset(train_folder, transform=transform)
    validation_dataset = ImageDataset(validation_folder, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, validation_loader


def generate_random_messages(batch_size, message_length, device):
    """Generate random binary messages for watermarking"""
    return torch.Tensor(np.random.choice([0, 1], (batch_size, message_length))).to(device)


def train_epoch(hidden_net, train_loader, epoch, device, message_length):
    """Train for one epoch"""
    hidden_net.encoder_decoder.train()
    hidden_net.discriminator.train()

    epoch_losses = {
        'loss           ': [],
        'encoder_mse    ': [],
        'dec_mse        ': [],
        'fft_loss       ': [],
        'bitwise-error  ': [],
        'adversarial_bce': [],
        'discr_cover_bce': [],
        'discr_encod_bce': []
    }

    batch_count = len(train_loader)

    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.to(device)
        batch_size = images.shape[0]
        messages = generate_random_messages(batch_size, message_length, device)

        losses, _ = hidden_net.train_on_batch([images, messages])

        for key in epoch_losses.keys():
            epoch_losses[key].append(losses[key])

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == batch_count:
            print(f'Epoch {epoch} [{batch_idx + 1}/{batch_count}] | '
                  f'Loss: {losses["loss           "]:.4f} | '
                  f'Enc MSE: {losses["encoder_mse    "]:.4f} | '
                  f'Dec MSE: {losses["dec_mse        "]:.4f} | '
                  f'BitErr: {losses["bitwise-error  "]:.4f}')

    avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
    return avg_losses


def validate(hidden_net, validation_loader, device, message_length):
    """Run validation"""
    hidden_net.encoder_decoder.eval()
    hidden_net.discriminator.eval()

    validation_losses = {
        'loss           ': [],
        'encoder_mse    ': [],
        'dec_mse        ': [],
        'fft_loss       ': [],
        'bitwise-error  ': [],
        'adversarial_bce': [],
        'discr_cover_bce': [],
        'discr_encod_bce': []
    }

    with torch.no_grad():
        for images, _ in validation_loader:
            images = images.to(device)
            batch_size = images.shape[0]
            messages = generate_random_messages(batch_size, message_length, device)

            losses, _ = hidden_net.validate_on_batch([images, messages])

            for key in validation_losses.keys():
                validation_losses[key].append(losses[key])

    avg_losses = {key: np.mean(values) for key, values in validation_losses.items()}
    return avg_losses


def save_checkpoint(hidden_net, epoch, checkpoint_dir, experiment_name):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'{experiment_name}_epoch_{epoch}.pth')

    torch.save({
        'epoch': epoch,
        'encoder_decoder_state_dict': hidden_net.encoder_decoder.state_dict(),
        'discriminator_state_dict': hidden_net.discriminator.state_dict(),
        'optimizer_enc_dec_state_dict': hidden_net.optimizer_enc_dec.state_dict(),
        'optimizer_discrim_state_dict': hidden_net.optimizer_discrim.state_dict(),
    }, checkpoint_path)

    print(f'Checkpoint saved: {checkpoint_path}')


def main():
    """Main training function for BASELINE model"""
    print('\n' + '='*60)
    print('TRAINING BASELINE MODEL (No Novel Contributions)')
    print('='*60)

    # Get configurations
    hidden_config = get_baseline_config()
    training_options = get_training_options('baseline_model')

    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'\nUsing device: {device}')

    # Create model
    print('\n=== Initializing Baseline Model ===')
    print('Features:')
    print(f'  - NECST: {hidden_config.use_necst}')
    print(f'  - FFT Loss: {hidden_config.use_fft_loss}')
    print(f'  - Distortion Pool: {hidden_config.use_distortion_pool}')

    hidden_net = Hidden(hidden_config, device)

    # Create data loaders
    print('\n=== Loading Datasets ===')
    train_loader, validation_loader = create_data_loaders(
        training_options.train_folder,
        training_options.validation_folder,
        training_options.batch_size,
        hidden_config.H
    )
    print(f'Training batches: {len(train_loader)}')
    print(f'Validation batches: {len(validation_loader)}')

    # Training loop
    print('\n=== Starting Training ===')
    best_val_loss = float('inf')

    for epoch in range(training_options.start_epoch, training_options.number_of_epochs):
        epoch_start_time = time.time()

        print(f'\n--- Epoch {epoch + 1}/{training_options.number_of_epochs} ---')

        # Train
        train_losses = train_epoch(
            hidden_net,
            train_loader,
            epoch + 1,
            device,
            hidden_config.message_length
        )

        # Validate
        print('Running validation...')
        val_losses = validate(
            hidden_net,
            validation_loader,
            device,
            hidden_config.message_length
        )

        epoch_time = time.time() - epoch_start_time

        # Print epoch summary
        print(f'\n=== Epoch {epoch + 1} Summary (Time: {epoch_time:.1f}s) ===')
        print(f'Train Loss: {train_losses["loss           "]:.4f} | '
              f'Val Loss: {val_losses["loss           "]:.4f}')
        print(f'Train BitErr: {train_losses["bitwise-error  "]:.4f} | '
              f'Val BitErr: {val_losses["bitwise-error  "]:.4f}')
        print(f'Train Enc MSE: {train_losses["encoder_mse    "]:.4f} | '
              f'Val Enc MSE: {val_losses["encoder_mse    "]:.4f}')

        # Save checkpoint
        if (epoch + 1) % 10 == 0 or val_losses["loss           "] < best_val_loss:
            save_checkpoint(
                hidden_net,
                epoch + 1,
                training_options.runs_folder,
                training_options.experiment_name
            )

            if val_losses["loss           "] < best_val_loss:
                best_val_loss = val_losses["loss           "]
                print(f'New best validation loss: {best_val_loss:.4f}')

    print('\n=== Training Complete ===')
    print(f'Best validation loss: {best_val_loss:.4f}')


if __name__ == '__main__':
    main()
