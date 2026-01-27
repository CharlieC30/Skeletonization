"""Step 02: Otsu thresholding.

Applies stack histogram mode (single threshold for entire 3D volume).
"""
import os
import logging
from pathlib import Path

import numpy as np
import tifffile
from natsort import natsorted
from skimage.filters import threshold_otsu


STEP_NAME = "02_otsu"
STEP_DESCRIPTION = "Otsu thresholding"
REQUIRES = "01_format"


def compute_stack_otsu_threshold(image: np.ndarray, logger: logging.Logger) -> float:
    """Compute single Otsu threshold from entire 3D stack.

    Args:
        image: 3D numpy array with shape (Z, Y, X).
        logger: Logger instance.

    Returns:
        Otsu threshold value.
    """
    if image.ndim != 3:
        raise ValueError(
            f"Expected 3D image, got {image.ndim}D. "
            "Run step01_format first."
        )

    # Determine histogram range based on dtype
    if image.dtype == np.uint8:
        hist_range = (0, 256)
        bins = 256
    else:
        logger.warning(
            f"Input dtype is {image.dtype}, expected uint8. "
            "Consider running step01_format first."
        )
        if np.issubdtype(image.dtype, np.integer):
            info = np.iinfo(image.dtype)
            hist_range = (info.min, info.max + 1)
            bins = min(256, info.max - info.min + 1)
        else:
            hist_range = (float(image.min()), float(image.max()))
            bins = 256

    # Compute histogram to avoid high memory usage
    counts, _ = np.histogram(image, bins=bins, range=hist_range)
    threshold = threshold_otsu(hist=counts)
    return float(threshold)


def apply_threshold(image: np.ndarray, threshold: float) -> np.ndarray:
    """Apply threshold to create binary mask (0/255).

    Args:
        image: Input numpy array.
        threshold: Threshold value.

    Returns:
        Binary uint8 array with values 0 or 255.
    """
    return (image >= threshold).astype(np.uint8) * 255


def run(input_path: str, output_dir: str, config: dict, logger: logging.Logger) -> str:
    """Run Step 02: Otsu thresholding.

    Args:
        input_path: Path to input (previous step output directory).
        output_dir: Base output directory.
        config: Configuration dictionary.
        logger: Logger instance.

    Returns:
        Output directory path for this step.
    """
    # Input is previous step's output directory
    input_dir = Path(input_path)
    if not input_dir.exists():
        # Try finding 01_format in output_dir
        input_dir = Path(output_dir) / "01_format"

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_path = Path(output_dir) / STEP_NAME
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_path}")

    # Find all TIF files
    tif_files = natsorted([
        f for f in input_dir.iterdir()
        if f.suffix.lower() in ('.tif', '.tiff') and '_otsu' not in f.name
    ])

    if not tif_files:
        raise ValueError(f"No TIF files found in: {input_dir}")

    logger.info(f"Found {len(tif_files)} TIF files")

    for idx, tif_file in enumerate(tif_files, start=1):
        logger.info(f"[{idx}/{len(tif_files)}] Processing: {tif_file.name}")

        image = tifffile.imread(str(tif_file))
        logger.debug(f"  Shape: {image.shape}, dtype: {image.dtype}")

        # Compute Otsu threshold
        threshold = compute_stack_otsu_threshold(image, logger)
        logger.debug(f"  Otsu threshold: {threshold:.2f}")

        # Apply threshold
        binary = apply_threshold(image, threshold)

        # Save with _otsu suffix
        out_file = output_path / f"{tif_file.stem}_otsu.tif"
        tifffile.imwrite(str(out_file), binary, imagej=True, metadata={'axes': 'ZYX'})
        logger.debug(f"  Saved to: {out_file}")

    return str(output_path)
