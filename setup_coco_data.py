"""
Setup script for downloading and preparing MS COCO dataset for training
Run this script in Google Colab or locally to prepare the dataset
"""

import os
import urllib.request
import zipfile
from tqdm import tqdm


def download_with_progress(url, filename):
    """Download file with progress bar"""
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
        urllib.request.urlretrieve(url, filename=filename, reporthook=t.update_to)


def setup_coco_dataset(use_full_dataset=True):
    """
    Download and setup MS COCO dataset

    Args:
        use_full_dataset: If True, downloads full train2017 (~18GB) and val2017 (~1GB)
                         If False, downloads only val2017 for quick testing
    """
    os.makedirs('data', exist_ok=True)

    if use_full_dataset:
        # Download train2017 (18GB, ~118k images)
        if not os.path.exists('data/train2017'):
            print('Downloading COCO train2017 (18GB)...')
            download_with_progress(
                'http://images.cocodataset.org/zips/train2017.zip',
                'data/train2017.zip'
            )
            print('Extracting train2017...')
            with zipfile.ZipFile('data/train2017.zip', 'r') as zip_ref:
                for member in tqdm(zip_ref.namelist(), desc='Extracting'):
                    zip_ref.extract(member, 'data/')
            os.remove('data/train2017.zip')
            print('Train set ready!')
        else:
            print('Train2017 already exists, skipping download.')

    # Download val2017 (1GB, ~5k images)
    if not os.path.exists('data/val2017'):
        print('Downloading COCO val2017 (1GB)...')
        download_with_progress(
            'http://images.cocodataset.org/zips/val2017.zip',
            'data/val2017.zip'
        )
        print('Extracting val2017...')
        with zipfile.ZipFile('data/val2017.zip', 'r') as zip_ref:
            for member in tqdm(zip_ref.namelist(), desc='Extracting'):
                zip_ref.extract(member, 'data/')
        os.remove('data/val2017.zip')
        print('Validation set ready!')
    else:
        print('Val2017 already exists, skipping download.')

    # Create directory structure for ImageFolder
    print('\nCreating directory structure...')
    os.makedirs('data/train/images', exist_ok=True)
    os.makedirs('data/validation/images', exist_ok=True)

    # Move/symlink files for ImageFolder format
    import shutil

    # For train
    if use_full_dataset and os.path.exists('data/train2017'):
        train_src = 'data/train2017'
        train_dst = 'data/train/images'
        train_images = os.listdir(train_src)

        if len(os.listdir(train_dst)) < len(train_images):
            print(f'Setting up training images ({len(train_images)} images)...')
            for img in tqdm(train_images, desc='Copying train images'):
                src_path = os.path.join(train_src, img)
                dst_path = os.path.join(train_dst, img)
                if not os.path.exists(dst_path):
                    shutil.copy(src_path, dst_path)

    # For validation
    if os.path.exists('data/val2017'):
        val_src = 'data/val2017'
        val_dst = 'data/validation/images'
        val_images = os.listdir(val_src)

        if len(os.listdir(val_dst)) < len(val_images):
            print(f'Setting up validation images ({len(val_images)} images)...')
            for img in tqdm(val_images, desc='Copying val images'):
                src_path = os.path.join(val_src, img)
                dst_path = os.path.join(val_dst, img)
                if not os.path.exists(dst_path):
                    shutil.copy(src_path, dst_path)

    # Print summary
    train_count = len(os.listdir('data/train/images')) if os.path.exists('data/train/images') else 0
    val_count = len(os.listdir('data/validation/images')) if os.path.exists('data/validation/images') else 0

    print('\n=== Dataset Setup Complete ===')
    print(f'Training images: {train_count}')
    print(f'Validation images: {val_count}')
    print(f'Total size: {train_count + val_count} images')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Download and setup MS COCO dataset')
    parser.add_argument('--quick', action='store_true',
                        help='Quick setup with validation set only (for testing)')

    args = parser.parse_args()

    setup_coco_dataset(use_full_dataset=not args.quick)
