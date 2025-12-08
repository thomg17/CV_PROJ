"""
Training script for FFT-only ablation study
Tests the impact of FFT consistency loss without NECST or distortion pool
"""
import torch
import time
from configs import get_fft_only_config, get_training_options
from model.hidden import Hidden
from train_baseline import train_epoch, validate, save_checkpoint, create_data_loaders

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Get FFT-only configuration
    hidden_config = get_fft_only_config()
    training_options = get_training_options('fft_only')

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
        print(f'\nEpoch {epoch + 1} Summary ({epoch_time:.1f}s):')
        print(f'Train Loss: {train_losses["loss           "]:.4f} | Val Loss: {val_losses["loss           "]:.4f}')
        print(f'Train BitErr: {train_losses["bitwise-error  "]:.4f} | Val BitErr: {val_losses["bitwise-error  "]:.4f}')
        if hidden_config.use_fft_loss:
            print(f'Train FFT: {train_losses["fft_loss       "]:.4f} | Val FFT: {val_losses["fft_loss       "]:.4f}')

        # Save checkpoint
        is_best = val_losses["loss           "] < best_val_loss
        if is_best:
            best_val_loss = val_losses["loss           "]
            print(f'New best validation loss: {best_val_loss:.4f}')

        if (epoch + 1) % 10 == 0 or is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': hidden_net.encoder_decoder.state_dict(),
                'best_val_loss': best_val_loss,
                'optimizer': hidden_net.optimizer_enc_dec.state_dict(),
            }, is_best, training_options.runs_folder, training_options.experiment_name)

    print('\n=== Training Complete ===')
    print(f'Best validation loss: {best_val_loss:.4f}')

if __name__ == '__main__':
    main()
