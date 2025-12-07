"""
Distortion pool for applying random non-differentiable distortions during training.
This implements the novel contribution from the project proposal.
"""
import torch
import numpy as np
import random
from PIL import Image
from training.distortions import sample, compression, quantize, color_change, flipper


class DistortionPool:
    """
    Pool of distortion methods that can be randomly selected during training.
    These are non-differentiable, real-world distortions.
    """
    def __init__(self, device='cpu'):
        self.device = device
        self.distortions = [
            'downsample_upsample',
            'compression',
            'quantization',
            'color_change',
            'flipper',
            'identity'  # Sometimes no distortion
        ]

    def apply_random_distortion(self, images_tensor):
        """
        Apply a random distortion from the pool to a batch of images.

        Args:
            images_tensor: Torch tensor (B, C, H, W) in range [0, 1]

        Returns:
            Distorted images tensor (B, C, H, W)
        """
        batch_size = images_tensor.shape[0]
        target_h, target_w = images_tensor.shape[2], images_tensor.shape[3]
        distorted_batch = []

        for i in range(batch_size):
            # Select random distortion
            distortion_type = random.choice(self.distortions)

            # Convert tensor to numpy/PIL for processing
            img_tensor = images_tensor[i]  # (C, H, W)
            img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            # Apply distortion
            try:
                if distortion_type == 'downsample_upsample':
                    img_distorted = sample(rounds=1, img=img_np)
                elif distortion_type == 'compression':
                    img_pil = Image.fromarray(img_np, mode='RGB')
                    img_distorted = compression(rounds=1, img=img_pil)
                    img_distorted = np.array(img_distorted)
                elif distortion_type == 'quantization':
                    img_pil = Image.fromarray(img_np, mode='RGB')
                    img_distorted = quantize(rounds=1, img=img_pil)
                elif distortion_type == 'color_change':
                    img_pil = Image.fromarray(img_np, mode='RGB')
                    img_distorted = color_change(img=img_pil)
                    img_distorted = np.array(img_distorted)
                elif distortion_type == 'flipper':
                    img_pil = Image.fromarray(img_np, mode='RGB')
                    img_distorted = flipper(img=img_pil)
                    img_distorted = np.array(img_distorted)
                else:  # identity
                    img_distorted = img_np
            except Exception as e:
                # If distortion fails, use original image
                print(f"Warning: Distortion {distortion_type} failed: {e}")
                img_distorted = img_np

            # Ensure correct format - convert PIL to numpy
            if isinstance(img_distorted, Image.Image):
                img_distorted = np.array(img_distorted)

            # Handle grayscale conversion
            if img_distorted.ndim == 2:
                img_distorted = np.stack([img_distorted] * 3, axis=-1)

            # Ensure uint8 type
            if img_distorted.dtype != np.uint8:
                img_distorted = img_distorted.astype(np.uint8)

            # ALWAYS resize to ensure consistent dimensions (some distortions may change size slightly)
            # Convert to PIL for resizing
            img_pil_resized = Image.fromarray(img_distorted, mode='RGB')
            img_pil_resized = img_pil_resized.resize((target_w, target_h), Image.BILINEAR)
            img_distorted = np.array(img_pil_resized)

            # Convert back to tensor
            img_distorted = torch.from_numpy(img_distorted).float() / 255.0
            img_distorted = img_distorted.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

            # FINAL SAFETY CHECK: Verify shape and force replace if wrong
            if img_distorted.shape != (3, target_h, target_w):
                print(f"CRITICAL ERROR [{i}]: Distortion {distortion_type} STILL has wrong shape {img_distorted.shape} after resize! Using original tensor.")
                img_distorted = img_tensor

            distorted_batch.append(img_distorted)

        # Stack back into batch
        distorted_tensor = torch.stack(distorted_batch).to(self.device)
        return distorted_tensor


class HybridDistorter:
    """
    Combines the adversarial attack network with the distortion pool.
    Randomly chooses between them during training.

    This implements the key novelty: at training time, the model can select
    between the adversarial generator or a hand-coded distortion method.
    """
    def __init__(self, attack_network, distortion_pool, distortion_prob=0.5):
        """
        Args:
            attack_network: The differentiable AttackNetwork
            distortion_pool: The DistortionPool with non-differentiable distortions
            distortion_prob: Probability of using distortion pool vs attack network
        """
        self.attack_network = attack_network
        self.distortion_pool = distortion_pool
        self.distortion_prob = distortion_prob

    def __call__(self, encoded_images):
        """
        Apply either attack network or random distortion.

        Args:
            encoded_images: Encoded images tensor (B, C, H, W)

        Returns:
            Noised images tensor (B, C, H, W)
        """
        if random.random() < self.distortion_prob:
            # Use distortion pool (non-differentiable)
            with torch.no_grad():
                noised_images = self.distortion_pool.apply_random_distortion(encoded_images)
        else:
            # Use attack network (differentiable)
            noised_images = self.attack_network(encoded_images)

        return noised_images
