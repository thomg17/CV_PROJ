"""
Resume training from a checkpoint
Usage: python resume_training.py <experiment_name> <checkpoint_path> <resume_epoch>
Example: python resume_training.py fft_only runs/fft_only_epoch_1.pth 1
"""
import sys
import torch
import time
from configs import get_fft_only_config, get_necst_only_config, get_distortion_only_config, get_training_options
from model.hidden import Hidden
from train_baseline import train_epoch, validate, save_checkpoint, create_data_loaders


def main():
    if len(sys.argv) < 4:
        print("Usage: python resume_training.py <experiment_name> <checkpoint_path> <resume_epoch>")
        print("Example: python resume_training.py fft_only runs/fft_only_epoch_1.pth 1")
        sys.exit(1)

    experiment_name = sys.argv[1]
    checkpoint_path = sys.argv[2]
    resume_epoch = int(sys.argv[3])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Resuming {experiment_name} from epoch {resume_epoch}')

    # Get configuration based on experiment name
    if experiment_name == 'fft_only':
        hidden_config = get_fft_only_config()
    elif experiment_name == 'necst_only':
        hidden_config = get_necst_only_config()
    elif experiment_name == 'distortion_only':
        hidden_config = get_distortion_only_config()
    else:
        print(f"Unknown experiment: {experiment_name}")
        sys.exit(1)

    training_options = get_training_options(experiment_name)

    # Initialize model
    print('\n=== Initializing Model ===')
    hidden_net = Hidden(hidden_config, device)

    # Load checkpoint
    print(f'\n=== Loading Checkpoint: {checkpoint_path} ===')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hidden_net.encoder_decoder.load_state_dict(checkpoint['encoder_decoder_state_dict'])
    hidden_net.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    hidden_net.optimizer_enc_dec.load_state_dict(checkpoint['optimizer_enc_dec_state_dict'])
    hidden_net.optimizer_discrim.load_state_dict(checkpoint['optimizer_discrim_state_dict'])
    print(f'Checkpoint loaded! Resuming from epoch {resume_epoch}')

    # Pretrain NECST if needed (only for necst_only and not already pretrained)
    if experiment_name == 'necst_only' and resume_epoch == 0:
        print('\n=== Pretraining NECST ===')
        hidden_net.necst.pretrain()
        print('NECST pretrained successfully!')

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

    # Training loop - start from resume_epoch + 1
    print(f'\n=== Resuming Training from Epoch {resume_epoch + 1} ===')
    best_val_loss = float('inf')

    for epoch in range(resume_epoch, training_options.number_of_epochs):
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

    print('\n=== Training Complete ===')
    print(f'Best validation loss: {best_val_loss:.4f}')


if __name__ == '__main__':
    main()
