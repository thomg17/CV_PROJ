"""
Configuration presets for different model experiments
"""

from options import HiDDenConfiguration, TrainingOptions


def get_baseline_config():
    """
    Baseline HiDDeN model WITHOUT novel contributions
    - No NECST channel coding
    - No FFT consistency loss
    - No distortion pool (just AttackNetwork)
    """
    return HiDDenConfiguration(
        H=128,
        W=128,
        message_length=30,
        encoder_blocks=4,
        encoder_channels=64,
        decoder_blocks=7,
        decoder_channels=64,
        use_discriminator=True,
        use_vgg=False,
        discriminator_blocks=3,
        discriminator_channels=64,
        decoder_loss=1.0,
        encoder_loss=0.7,
        adversarial_loss=0.001,
        enable_fp16=False,
        # Novel features DISABLED
        use_necst=False,
        redundant_length=None,
        necst_iter=0,
        use_fft_loss=False,
        fft_loss_weight=0.0,
        use_distortion_pool=False,
        distortion_prob=0.0
    )


def get_full_config():
    """
    Full model WITH all novel contributions
    - NECST channel coding
    - FFT consistency loss
    - HybridDistorter with distortion pool
    """
    return HiDDenConfiguration(
        H=128,
        W=128,
        message_length=30,
        encoder_blocks=4,
        encoder_channels=64,
        decoder_blocks=7,
        decoder_channels=64,
        use_discriminator=True,
        use_vgg=False,
        discriminator_blocks=3,
        discriminator_channels=64,
        decoder_loss=1.0,
        encoder_loss=0.7,
        adversarial_loss=0.001,
        enable_fp16=False,
        # Novel features ENABLED
        use_necst=True,
        redundant_length=60,  # 2x message length
        necst_iter=10000,
        use_fft_loss=True,
        fft_loss_weight=0.1,
        use_distortion_pool=True,
        distortion_prob=0.2  # Reduced from 0.5 - too aggressive for early training
    )


def get_training_options(experiment_name):
    """
    Training options for experiments

    Args:
        experiment_name: Name for the experiment (e.g., 'baseline', 'full')
    """
    return TrainingOptions(
        batch_size=16,
        number_of_epochs=30,  # Reduced from 50 - diminishing returns after 30
        train_folder='data/train',
        validation_folder='data/validation',
        runs_folder='runs',
        start_epoch=0,
        experiment_name=experiment_name
    )


# Optional: Individual feature ablation studies
def get_necst_only_config():
    """Only NECST enabled"""
    config = get_baseline_config()
    config.use_necst = True
    config.redundant_length = 60
    config.necst_iter = 10000
    return config


def get_fft_only_config():
    """Only FFT loss enabled"""
    config = get_baseline_config()
    config.use_fft_loss = True
    config.fft_loss_weight = 0.1
    return config


def get_distortion_only_config():
    """Only distortion pool enabled"""
    config = get_baseline_config()
    config.use_distortion_pool = True
    config.distortion_prob = 0.1  # Low probability for easier convergence
    return config
