"""
Diagnostic script to analyze mask connected components.

This tool helps troubleshoot skeletonization issues by analyzing the
connected components in a binary mask. It reports:
  - Total number of components
  - Size of each component (in voxels)
  - Which components will be kept vs filtered by dust_threshold

Use this when:
  - Skeleton output is missing expected structures
  - Uncertain about appropriate dust_threshold value
  - Mask preprocessing needs validation

Usage:
  1. Update the filepath in __main__ section
  2. Run: python check_mask_components.py
  3. Review recommendations for dust_threshold adjustment
"""

import numpy as np
from PIL import Image
from scipy import ndimage


def load_tiff_stack(filepath):
    """Load TIFF stack into 3D numpy array."""
    img = Image.open(filepath)
    n_frames = getattr(img, 'n_frames', 1)

    frames = []
    for i in range(n_frames):
        img.seek(i)
        frames.append(np.array(img))

    return np.array(frames)


def analyze_components(labels):
    """Analyze connected components in the mask."""
    # Convert to binary
    binary = (labels > 0).astype(np.uint8)

    # Find connected components
    labeled, num_components = ndimage.label(binary)

    print(f"Total connected components: {num_components}")
    print(f"Total non-zero voxels: {np.count_nonzero(binary)}")
    print()

    # Analyze each component
    component_sizes = []
    for i in range(1, num_components + 1):
        size = np.sum(labeled == i)
        component_sizes.append((i, size))

    # Sort by size (largest first)
    component_sizes.sort(key=lambda x: x[1], reverse=True)

    print("Connected components (sorted by size):")
    print("-" * 60)
    print(f"{'ID':<6} {'Voxels':<12} {'Status (dust_threshold=500)'}")
    print("-" * 60)

    total_kept = 0
    total_filtered = 0

    for idx, (comp_id, size) in enumerate(component_sizes):
        if size >= 500:
            status = "KEPT"
            total_kept += 1
        else:
            status = "FILTERED OUT"
            total_filtered += 1

        print(f"{comp_id:<6} {size:<12} {status}")

        if idx >= 20:  # Show first 20
            print(f"... and {num_components - 21} more components")
            break

    print("-" * 60)
    print(f"Components KEPT (>= 500 voxels): {total_kept}")
    print(f"Components FILTERED (< 500 voxels): {total_filtered}")
    print()

    # Recommendations
    print("Recommendations:")
    print("=" * 60)

    if total_filtered > 0:
        largest_filtered = max([size for _, size in component_sizes if size < 500])
        print(f"Largest filtered component: {largest_filtered} voxels")
        print()
        print(f"Solution 1: Reduce dust_threshold to keep more components")
        print(f"  Suggested: dust_threshold = {largest_filtered // 2}")
        print(f"  Or: dust_threshold = 10 (keep almost all)")
        print()
        print(f"Solution 2: Fix mask connectivity in Fiji/ImageJ")
        print(f"  Use: Process -> Binary -> Close")
        print(f"  This will connect broken segments")
    else:
        print("All components are being kept (>= 500 voxels)")
        print("The issue might be with teasar_params (scale/const)")


if __name__ == '__main__':
    filepath = '/home/aero/charliechang/projects/skeleton/DATA/mask_fiji_clean/skeleton_roi_8bit_z60-630_mask_otsu_stacktrd_clean3d.tif'

    print(f"Loading mask: {filepath}")
    labels = load_tiff_stack(filepath)
    print(f"Shape: {labels.shape}")
    print()

    analyze_components(labels)
