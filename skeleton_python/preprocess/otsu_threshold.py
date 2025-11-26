"""
Otsu thresholding for TIF stacks.
Applies stack histogram mode (single threshold for entire 3D volume).
"""

import os
import sys
import logging
from pathlib import Path

import numpy as np
import tifffile
from skimage.filters import threshold_otsu

from pipeline.utils import ensure_3d, auto_detect_subdir, setup_logging

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.resolve()


def compute_stack_otsu_threshold(image: np.ndarray) -> float:
    """Compute single Otsu threshold from entire 3D stack.

    Args:
        image: 2D or 3D numpy array.

    Returns:
        Otsu threshold value.

    Raises:
        ValueError: If image is not 2D or 3D.
    """
    if image.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D array, got {image.ndim}D")

    threshold = threshold_otsu(image.ravel())
    return float(threshold)


def apply_threshold(image: np.ndarray, threshold: float) -> np.ndarray:
    """Apply threshold to create binary mask (0/255).

    Args:
        image: Input numpy array.
        threshold: Threshold value.

    Returns:
        Binary uint8 array with values 0 or 255.
    """
    binary = (image >= threshold).astype(np.uint8) * 255
    return binary


def process_single_file(input_path: str, output_dir: str, progress: str = "") -> str:
    """Process single TIF file: compute Otsu threshold and binarize.

    Args:
        input_path: Path to input TIF file.
        output_dir: Output directory path.
        progress: Optional progress string (e.g., "1/10").

    Returns:
        Path to output file.
    """
    filename = os.path.basename(input_path)
    progress_prefix = f"[{progress}] " if progress else ""
    logger.info(f"{progress_prefix}Processing: {filename}")

    # Load image (assuming already processed by check_tif_format.py)
    image = tifffile.imread(input_path)
    image = ensure_3d(image)

    logger.debug(f"  Shape: {image.shape}, dtype: {image.dtype}")

    # Compute Otsu threshold on entire stack
    threshold = compute_stack_otsu_threshold(image)
    logger.debug(f"  Otsu threshold: {threshold:.2f}")

    # Apply threshold
    binary = apply_threshold(image, threshold)

    # Generate output path with _otsu suffix
    input_name = Path(input_path).stem
    output_path = os.path.join(output_dir, f"{input_name}_otsu.tif")

    # Save binary mask
    tifffile.imwrite(output_path, binary, imagej=True, metadata={'axes': 'ZYX'})
    logger.debug(f"  Saved to: {output_path}")

    return output_path


def process_directory(input_dir: str, output_dir: str = None) -> None:
    """Process all TIF files in directory with Otsu thresholding.

    Args:
        input_dir: Input directory containing TIF files.
        output_dir: Output directory (default: auto-detect as 02_otsu).

    Raises:
        FileNotFoundError: If input directory does not exist.
        ValueError: If no TIF files found.
    """
    input_dir_obj = Path(input_dir)
    if not input_dir_obj.is_absolute():
        input_dir_obj = BASE_DIR / input_dir
    input_dir = str(input_dir_obj.resolve())

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if not os.path.isdir(input_dir):
        raise ValueError(f"Input path is not a directory: {input_dir}")

    # Auto-detect 01_format subdirectory if current dir has no TIF files
    detected_dir = auto_detect_subdir(input_dir, '01_format')
    if detected_dir != input_dir:
        input_dir = detected_dir
        logger.info(f"Auto-detected input directory: {input_dir}")

    # Auto-detect output directory based on input path structure
    if output_dir is None:
        input_path = Path(input_dir)
        if input_path.name == '01_format' and input_path.parent.parent.name == 'preprocess_output':
            # Input is preprocess_output/YYYYMMDD_HHMMSS/01_format/ -> output to 02_otsu/
            output_dir = str(input_path.parent / '02_otsu')
        else:
            # Create 02_otsu/ in parent directory
            output_dir = str(input_path.parent / '02_otsu')

    # Find all TIF files
    tif_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(('.tif', '.tiff')) and '_otsu' not in f
    ]

    if not tif_files:
        raise ValueError(f"No TIF files found in directory: {input_dir}")

    logger.info(f"Found {len(tif_files)} TIF files in directory")
    logger.info(f"Output directory: {output_dir}")

    # Create output directory before processing
    os.makedirs(output_dir, exist_ok=True)

    # Process each file
    for idx, tif_file in enumerate(tif_files, start=1):
        try:
            process_single_file(tif_file, output_dir, progress=f"{idx}/{len(tif_files)}")
        except Exception as e:
            logger.error(f"Error processing {tif_file}: {e}")
            raise

    logger.info(f"Completed processing {len(tif_files)} files")


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python otsu_threshold.py <input_directory> [output_directory]")
        print("  input_directory: Directory containing TIF files (relative to BASE_DIR or absolute)")
        print("  output_directory: Optional output directory (default: auto-detect as 02_otsu)")
        sys.exit(1)

    # Setup logging
    setup_logging()

    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        process_directory(input_dir, output_dir)
        logger.info("Processing completed")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()


# python preprocess/otsu_threshold.py preprocess_output/20251117_171518/
