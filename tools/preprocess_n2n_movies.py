#!/usr/bin/env python3
"""
Preprocess raw cryo-EM movie stacks for Noise2Noise training.
Creates odd/even frame averages as input/target pairs.
Outputs streaming binary format for Fortran training.
"""

import numpy as np
import mrcfile
import os
import struct
from pathlib import Path
from tqdm import tqdm
import argparse

def extract_patches_from_movie(movie_path, patch_size=64, stride=128, max_patches_per_movie=100):
    """
    Extract patches from a movie stack using odd/even frame splitting.
    
    Returns:
        List of (input_patch, target_patch) tuples
    """
    patches = []
    
    with mrcfile.open(movie_path, permissive=True) as mrc:
        data = mrc.data.astype(np.float32)
        
        n_frames, h, w = data.shape
        
        # Split into odd/even frames
        odd_frames = data[0::2]   # frames 0, 2, 4, ... (indices)
        even_frames = data[1::2]  # frames 1, 3, 5, ...
        
        # Average each set
        odd_avg = np.mean(odd_frames, axis=0)
        even_avg = np.mean(even_frames, axis=0)
        
        # Normalize each image independently to [0, 1]
        def normalize(img):
            img_min, img_max = img.min(), img.max()
            if img_max - img_min > 1e-6:
                return (img - img_min) / (img_max - img_min)
            return img - img_min
        
        odd_norm = normalize(odd_avg)
        even_norm = normalize(even_avg)
        
        # Extract patches
        patch_count = 0
        positions = []
        
        # Create list of all valid positions
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                positions.append((y, x))
        
        # Randomly sample if too many positions
        if len(positions) > max_patches_per_movie:
            np.random.shuffle(positions)
            positions = positions[:max_patches_per_movie]
        
        for y, x in positions:
            input_patch = odd_norm[y:y+patch_size, x:x+patch_size]
            target_patch = even_norm[y:y+patch_size, x:x+patch_size]
            
            # Skip patches with very low variance (likely empty/ice)
            if input_patch.std() < 0.01:
                continue
                
            patches.append((input_patch.copy(), target_patch.copy()))
    
    return patches


def write_binary_dataset(patches, output_dir, patch_size, split_ratio=0.9):
    """
    Write patches to binary format for Fortran streaming loader.
    
    Format per sample: patch_size * patch_size * 4 bytes (float32) for input,
                       patch_size * patch_size * 4 bytes (float32) for target
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Shuffle patches
    np.random.shuffle(patches)
    
    # Split into train/val
    n_train = int(len(patches) * split_ratio)
    train_patches = patches[:n_train]
    val_patches = patches[n_train:]
    
    # Write training data
    train_file = output_dir / 'train_n2n.bin'
    with open(train_file, 'wb') as f:
        # Header: n_samples, patch_size, patch_size, channels (1)
        f.write(struct.pack('iiii', len(train_patches), patch_size, patch_size, 1))
        
        for input_patch, target_patch in tqdm(train_patches, desc='Writing train'):
            f.write(input_patch.astype(np.float32).tobytes())
            f.write(target_patch.astype(np.float32).tobytes())
    
    # Write validation data
    val_file = output_dir / 'val_n2n.bin'
    with open(val_file, 'wb') as f:
        f.write(struct.pack('iiii', len(val_patches), patch_size, patch_size, 1))
        
        for input_patch, target_patch in tqdm(val_patches, desc='Writing val'):
            f.write(input_patch.astype(np.float32).tobytes())
            f.write(target_patch.astype(np.float32).tobytes())
    
    print(f"\nDataset created:")
    print(f"  Training samples: {len(train_patches)}")
    print(f"  Validation samples: {len(val_patches)}")
    print(f"  Patch size: {patch_size}x{patch_size}")
    print(f"  Train file: {train_file} ({train_file.stat().st_size / 1e6:.1f} MB)")
    print(f"  Val file: {val_file} ({val_file.stat().st_size / 1e6:.1f} MB)")
    
    return len(train_patches), len(val_patches)


def main():
    parser = argparse.ArgumentParser(description='Preprocess cryo-EM movies for N2N training')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with .mrc movie files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for binary files')
    parser.add_argument('--patch_size', type=int, default=64, help='Patch size (default: 64)')
    parser.add_argument('--stride', type=int, default=128, help='Stride between patches (default: 128)')
    parser.add_argument('--max_patches', type=int, default=100, help='Max patches per movie (default: 100)')
    parser.add_argument('--max_movies', type=int, default=None, help='Max movies to process (default: all)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    
    # Find all movie files
    input_dir = Path(args.input_dir)
    movie_files = sorted(list(input_dir.glob('*.mrc')) + list(input_dir.glob('*.mrcs')))
    
    if args.max_movies:
        movie_files = movie_files[:args.max_movies]
    
    print(f"Found {len(movie_files)} movie files")
    print(f"Processing with patch_size={args.patch_size}, stride={args.stride}")
    print(f"Max patches per movie: {args.max_patches}")
    
    # Extract patches from all movies
    all_patches = []
    
    for movie_path in tqdm(movie_files, desc='Processing movies'):
        try:
            patches = extract_patches_from_movie(
                movie_path, 
                patch_size=args.patch_size,
                stride=args.stride,
                max_patches_per_movie=args.max_patches
            )
            all_patches.extend(patches)
        except Exception as e:
            print(f"\nError processing {movie_path}: {e}")
            continue
    
    print(f"\nTotal patches extracted: {len(all_patches)}")
    
    # Write to binary format
    write_binary_dataset(all_patches, args.output_dir, args.patch_size)


if __name__ == '__main__':
    main()
