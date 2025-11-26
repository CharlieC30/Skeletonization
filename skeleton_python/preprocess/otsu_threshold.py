"""
Otsu thresholding for TIF stacks.
Applies stack histogram mode (single threshold for entire 3D volume).
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import tifffile
from natsort import natsorted
from skimage.filters import threshold_otsu

from pipeline.utils import auto_detect_subdir, setup_logging

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.resolve()


def compute_stack_otsu_threshold(image: np.ndarray) -> float:
    """Compute single Otsu threshold from entire 3D stack.

    Args:
        image: 3D numpy array with shape (Z, Y, X).

    Returns:
        Otsu threshold value.

    Raises:
        ValueError: If image is not 3D.
    """
    if image.ndim != 3:
        raise ValueError(
            f"Expected 3D image, got {image.ndim}D. "
            "Run check_tif_format.py first."
        )

    # Determine histogram range based on dtype
    if image.dtype == np.uint8:
        hist_range = (0, 256)
        bins = 256
    else:
        logger.warning(
            f"Input dtype is {image.dtype}, expected uint8. "
            "Consider running check_tif_format.py first."
        )
        if np.issubdtype(image.dtype, np.integer):
            info = np.iinfo(image.dtype)
            hist_range = (info.min, info.max + 1)
            bins = min(256, info.max - info.min + 1)
        else:
            # Float dtype: use actual data range
            hist_range = (float(image.min()), float(image.max()))
            bins = 256

    # Compute histogram to avoid high memory usage from ravel()
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

    # Find all TIF files (naturally sorted)
    tif_files = natsorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(('.tif', '.tiff')) and '_otsu' not in f
    ])

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
    parser = argparse.ArgumentParser(
        description='Otsu thresholding for TIF stacks.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python otsu_threshold.py preprocess_output/20251117_171518/
  python otsu_threshold.py preprocess_output/20251117_171518/01_format/ -o output/02_otsu/
        """
    )
    parser.add_argument('input_dir',
                        help='Directory containing TIF files (relative to BASE_DIR or absolute)')
    parser.add_argument('-o', '--output-dir',
                        help='Output directory (default: auto-detect as 02_otsu)')
    parser.add_argument('--log-file',
                        help='Log file path')

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_file=args.log_file)

    try:
        process_directory(args.input_dir, args.output_dir)
        logger.info("Processing completed")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()


# python preprocess/otsu_threshold.py preprocess_output/20251126_165434
