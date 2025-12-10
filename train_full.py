"""
Training script for FULL model
(HiDDeN WITH novel contributions: NECST, FFT loss, and distortion pool)
"""

import os
import time
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from model.hidden import Hidden
from configs import get_full_config, get_training_options


def create_data_loaders(train_folder, validation_folder, batch_size, image_size):
    """Create data loaders for training and validation"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(train_folder, transform=transform)
    validation_dataset = datasets.ImageFolder(validation_folder, transform=transform)

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
                  f'FFT: {losses["fft_loss       "]:.4f} | '
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


def save_training_stats(stats, checkpoint_dir, experiment_name):
    """Save training statistics to JSON file"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    stats_path = os.path.join(checkpoint_dir, f'{experiment_name}_training_stats.json')

    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f'Training stats saved: {stats_path}')


def plot_training_graphs(stats, checkpoint_dir, experiment_name):
    """Generate and save training progress graphs"""
    os.makedirs(checkpoint_dir, exist_ok=True)

    epochs = stats['epochs']

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Loss
    ax1.plot(epochs, stats['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, stats['val_loss'], 'orange', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Progress - Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Bitwise Error
    ax2.plot(epochs, stats['train_biterr'], 'b-', label='Train BitErr', linewidth=2)
    ax2.plot(epochs, stats['val_biterr'], 'orange', label='Val BitErr', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Bitwise Error', fontsize=12)
    ax2.set_title('Training Progress - Bitwise Error', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    graph_path = os.path.join(checkpoint_dir, f'{experiment_name}_training_progress.png')
    plt.savefig(graph_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'Training graph saved: {graph_path}')

    # If FFT loss is being tracked, create additional graph
    if 'train_fft' in stats and len(stats['train_fft']) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.plot(epochs, stats['train_fft'], 'b-', label='Train FFT Loss', linewidth=2)
        ax.plot(epochs, stats['val_fft'], 'orange', label='Val FFT Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('FFT Loss', fontsize=12)
        ax.set_title('Training Progress - FFT Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        fft_graph_path = os.path.join(checkpoint_dir, f'{experiment_name}_fft_loss.png')
        plt.savefig(fft_graph_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f'FFT loss graph saved: {fft_graph_path}')


def main():
    """Main training function for FULL model"""
    print('\n' + '='*60)
    print('TRAINING FULL MODEL (With All Novel Contributions)')
    print('='*60)

    # Get configurations
    hidden_config = get_full_config()
    training_options = get_training_options('full_model')

    # Override runs_folder to use Google Drive if available
    gdrive_path = '/content/drive/MyDrive/CV_Project_Checkpoints'
    if os.path.exists('/content/drive/MyDrive'):
        training_options.runs_folder = gdrive_path
        os.makedirs(gdrive_path, exist_ok=True)
        print(f'\nGoogle Drive detected! Checkpoints will be saved to: {gdrive_path}')
    else:
        print(f'\nGoogle Drive not mounted. Using local path: {training_options.runs_folder}')

    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create model
    print('\n=== Initializing Full Model ===')
    print('Features:')
    print(f'  - NECST: {hidden_config.use_necst}')
    print(f'  - FFT Loss: {hidden_config.use_fft_loss} (weight: {hidden_config.fft_loss_weight})')
    print(f'  - Distortion Pool: {hidden_config.use_distortion_pool} (prob: {hidden_config.distortion_prob})')

    hidden_net = Hidden(hidden_config, device)

    # Pretrain NECST
    if hidden_config.use_necst:
        print('\n=== Pretraining NECST Channel Encoder/Decoder ===')
        hidden_net.necst.pretrain()
        print('NECST pretraining complete')

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

    # Initialize statistics tracking
    training_stats = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'train_biterr': [],
        'val_biterr': [],
        'train_enc_mse': [],
        'val_enc_mse': [],
        'train_fft': [],
        'val_fft': [],
        'epoch_times': []
    }

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

        # Update statistics
        training_stats['epochs'].append(epoch + 1)
        training_stats['train_loss'].append(float(train_losses["loss           "]))
        training_stats['val_loss'].append(float(val_losses["loss           "]))
        training_stats['train_biterr'].append(float(train_losses["bitwise-error  "]))
        training_stats['val_biterr'].append(float(val_losses["bitwise-error  "]))
        training_stats['train_enc_mse'].append(float(train_losses["encoder_mse    "]))
        training_stats['val_enc_mse'].append(float(val_losses["encoder_mse    "]))
        training_stats['train_fft'].append(float(train_losses["fft_loss       "]))
        training_stats['val_fft'].append(float(val_losses["fft_loss       "]))
        training_stats['epoch_times'].append(float(epoch_time))

        # Print epoch summary
        print(f'\n=== Epoch {epoch + 1} Summary (Time: {epoch_time:.1f}s) ===')
        print(f'Train Loss: {train_losses["loss           "]:.4f} | '
              f'Val Loss: {val_losses["loss           "]:.4f}')
        print(f'Train BitErr: {train_losses["bitwise-error  "]:.4f} | '
              f'Val BitErr: {val_losses["bitwise-error  "]:.4f}')
        print(f'Train Enc MSE: {train_losses["encoder_mse    "]:.4f} | '
              f'Val Enc MSE: {val_losses["encoder_mse    "]:.4f}')
        print(f'Train FFT: {train_losses["fft_loss       "]:.4f} | '
              f'Val FFT: {val_losses["fft_loss       "]:.4f}')

        # Save checkpoint and generate graphs
        if (epoch + 1) % 10 == 0 or val_losses["loss           "] < best_val_loss:
            save_checkpoint(
                hidden_net,
                epoch + 1,
                training_options.runs_folder,
                training_options.experiment_name
            )

            # Save statistics and generate graphs
            save_training_stats(training_stats, training_options.runs_folder, training_options.experiment_name)
            plot_training_graphs(training_stats, training_options.runs_folder, training_options.experiment_name)

            if val_losses["loss           "] < best_val_loss:
                best_val_loss = val_losses["loss           "]
                print(f'New best validation loss: {best_val_loss:.4f}')

    print('\n=== Training Complete ===')
    print(f'Best validation loss: {best_val_loss:.4f}')

    # Final save of statistics and graphs
    save_training_stats(training_stats, training_options.runs_folder, training_options.experiment_name)
    plot_training_graphs(training_stats, training_options.runs_folder, training_options.experiment_name)
    print('\nFinal statistics and graphs saved!')


if __name__ == '__main__':
    main()
