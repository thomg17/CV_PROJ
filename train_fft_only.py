"""
Training script for FFT-only ablation study
Tests the impact of FFT consistency loss without NECST or distortion pool
"""
import os
import torch
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from configs import get_fft_only_config, get_training_options
from model.hidden import Hidden
from train_baseline import train_epoch, validate, save_checkpoint, create_data_loaders


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
    graph_path = os.path.join(checkpoint_dir, f'{experiment_name}_training_progress.png')
    plt.savefig(graph_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Training graph saved: {graph_path}')

    # FFT loss graph
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Get FFT-only configuration
    hidden_config = get_fft_only_config()
    training_options = get_training_options('fft_only')

    # Override runs_folder to use Google Drive if available
    gdrive_path = '/content/drive/MyDrive/CV_Project_Checkpoints'
    if os.path.exists('/content/drive/MyDrive'):
        training_options.runs_folder = gdrive_path
        os.makedirs(gdrive_path, exist_ok=True)
        print(f'\nGoogle Drive detected! Checkpoints will be saved to: {gdrive_path}')
    else:
        print(f'\nGoogle Drive not mounted. Using local path: {training_options.runs_folder}')

    print('\n=== FFT-Only Ablation Study Configuration ===')
    print(f'Message Length: {hidden_config.message_length}')
    print(f'Image Size: {hidden_config.H}x{hidden_config.W}')
    print(f'\nNovel Features:')
    print(f'  - NECST: {hidden_config.use_necst}')
    print(f'  - FFT Loss: {hidden_config.use_fft_loss} (weight: {hidden_config.fft_loss_weight})')
    print(f'  - Distortion Pool: {hidden_config.use_distortion_pool}')
    print(f'\nTraining:')
    print(f'  - Batch Size: {training_options.batch_size}')
    print(f'  - Epochs: {training_options.number_of_epochs}')
    print(f'  - Experiment: {training_options.experiment_name}')

    # Initialize model
    print('\n=== Initializing Model ===')
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
        train_losses, _ = train_epoch(  # Unpack tuple (losses, distortion_stats)
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
        print(f'\nEpoch {epoch + 1} Summary ({epoch_time:.1f}s):')
        print(f'Train Loss: {train_losses["loss           "]:.4f} | Val Loss: {val_losses["loss           "]:.4f}')
        print(f'Train BitErr: {train_losses["bitwise-error  "]:.4f} | Val BitErr: {val_losses["bitwise-error  "]:.4f}')
        if hidden_config.use_fft_loss:
            print(f'Train FFT: {train_losses["fft_loss       "]:.4f} | Val FFT: {val_losses["fft_loss       "]:.4f}')

        # Save checkpoint and generate graphs
        if val_losses["loss           "] < best_val_loss:
            best_val_loss = val_losses["loss           "]
            print(f'New best validation loss: {best_val_loss:.4f}')

        if (epoch + 1) % 10 == 0 or val_losses["loss           "] == best_val_loss:
            save_checkpoint(
                hidden_net,
                epoch + 1,
                training_options.runs_folder,
                training_options.experiment_name
            )
            save_training_stats(training_stats, training_options.runs_folder, training_options.experiment_name)
            plot_training_graphs(training_stats, training_options.runs_folder, training_options.experiment_name)

    print('\n=== Training Complete ===')
    print(f'Best validation loss: {best_val_loss:.4f}')

    # Final save of statistics and graphs
    save_training_stats(training_stats, training_options.runs_folder, training_options.experiment_name)
    plot_training_graphs(training_stats, training_options.runs_folder, training_options.experiment_name)
    print('\nFinal statistics and graphs saved!')

if __name__ == '__main__':
    main()
