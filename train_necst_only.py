"""
Training script for NECST-only ablation study
Tests the impact of NECST error correction without FFT loss or distortion pool
"""
import torch
from configs import get_necst_only_config, get_training_options
from model.hidden import Hidden
from train_baseline import train_epoch, validate, save_checkpoint

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Get NECST-only configuration
    hidden_config = get_necst_only_config()
    training_options = get_training_options('necst_only')

    print('\n=== NECST-Only Ablation Study Configuration ===')
    print(f'Message Length: {hidden_config.message_length}')
    print(f'Image Size: {hidden_config.H}x{hidden_config.W}')
    print(f'\nNovel Features:')
    print(f'  - NECST: {hidden_config.use_necst} (redundant length: {hidden_config.redundant_length})')
    print(f'  - FFT Loss: {hidden_config.use_fft_loss}')
    print(f'  - Distortion Pool: {hidden_config.use_distortion_pool}')
    print(f'\nTraining:')
    print(f'  - Batch Size: {training_options.batch_size}')
    print(f'  - Epochs: {training_options.number_of_epochs}')
    print(f'  - Experiment: {training_options.experiment_name}')

    # Initialize model
    print('\n=== Initializing Model ===')
    hidden_net = Hidden(hidden_config, device, None, None)

    # Pretrain NECST
    if hidden_config.use_necst:
        print('\n=== Pretraining NECST ===')
        hidden_net.necst.pretrain(iterations=hidden_config.necst_iter, batch_size=training_options.batch_size)
        print('NECST pretrained successfully!')

    # Training loop
    print('\n=== Starting Training ===')
    best_val_loss = float('inf')

    for epoch in range(training_options.start_epoch, training_options.number_of_epochs):
        print(f'\nEpoch {epoch + 1}/{training_options.number_of_epochs}')

        # Train
        train_losses = train_epoch(hidden_net, training_options, epoch, device)

        # Validate
        val_losses = validate(hidden_net, training_options, epoch, device)

        # Print epoch summary
        print(f'\nEpoch {epoch + 1} Summary:')
        print(f'Train Loss: {train_losses["loss           "]:.4f} | Val Loss: {val_losses["loss           "]:.4f}')
        print(f'Train BitErr: {train_losses["bitwise-error  "]:.4f} | Val BitErr: {val_losses["bitwise-error  "]:.4f}')

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
