# CMSC 672/472 Computer Vision: Distortion Robust Image Watermarking
Authors:
- `Jossie Batisso`
- `Bemnet Demere`
- `Benjamin Fradkin`
- `Thomas Getachew`
- `Joe Kim`

## Description

This project implements Distortion-Agnostic Deep Watermarking (DADW), a robust image watermarking system that embeds 30-bit hidden messages into images while maintaining perceptual quality and resisting various distortions. Building on the HiDDeN baseline architecture, we introduce three novel contributions: NECST channel coding for error correction (expanding messages to 60 bits with soft Turbo-like decoding), FFT consistency loss to preserve frequency-domain characteristics, and a hybrid distortion approach combining a learned attack network with hand-crafted distortions (JPEG compression, downsampling, quantization, color changes, and flipping). The system is trained on MS COCO using an encoder-decoder architecture with adversarial discriminator, achieving strong baseline performance (99.96% bitwise accuracy) while exploring the effectiveness of each novel component through ablation studies.

## Sources
The base hidden architecture is sourced from ando khachatryan [repo](https://github.com/ando-khachatryan/HiDDeN).
