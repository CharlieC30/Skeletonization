"""
Binary mask cleaning with 3D morphological filters.
Workflow: Remove Small Objects -> Opening -> Closing -> Fill Holes
"""

import os
import sys
import argparse
import time
import logging
from pathlib import Path

import numpy as np
import tifffile
from natsort import natsorted
from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes
from skimage.morphology import remove_small_objects

from pipeline.utils import auto_detect_subdir, setup_logging, load_config, PREPROCESS_SCHEMA

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.resolve()


def get_structure_3d(radius: int) -> np.ndarray:
    """Generate 3D cubic structure element.

    Args:
        radius: Radius of structure element.

    Returns:
        3D array of ones with shape (2*radius+1, 2*radius+1, 2*radius+1).
    """
    size = 2 * radius + 1
    return np.ones((size, size, size))


def clean_mask(
    image: np.ndarray,
    opening_radius: int = 1,
    closing_radius: int = 2,
    min_size: int = 64,
    skip_remove_small: bool = False,
    skip_fill_holes: bool = False,
) -> np.ndarray:
    """Clean binary mask with morphological operations.

    Workflow: Remove Small Objects -> Opening -> Closing -> Fill Holes

    Args:
        image: Input 3D binary mask with shape (Z, Y, X).
        opening_radius: Radius for opening operation (0 to skip).
        closing_radius: Radius for closing operation (0 to skip).
        min_size: Min object size in voxels.
        skip_remove_small: Skip remove small objects step.
        skip_fill_holes: Skip fill holes step.

    Returns:
        Cleaned binary mask (uint8, values 0 or 255).

    Raises:
        ValueError: If image is not 3D.
    """
    if image.ndim != 3:
        raise ValueError(
            f"Expected 3D image, got {image.ndim}D. "
            "Run check_tif_format.py first."
        )

    binary = image > 0

    if not skip_remove_small:
        logger.debug(f"Removing small objects (min_size={min_size})")
        t0 = time.time()
        binary = remove_small_objects(binary, min_size=min_size)
        elapsed = time.time() - t0
        logger.debug(f"  Completed in {elapsed:.1f}s")

    if opening_radius > 0:
        struct = get_structure_3d(opening_radius)
        logger.debug(f"Applying binary opening (radius={opening_radius})")
        t0 = time.time()
        binary = binary_opening(binary, structure=struct)
        elapsed = time.time() - t0
        logger.debug(f"  Completed in {elapsed:.1f}s")

    if closing_radius > 0:
        struct = get_structure_3d(closing_radius)
        logger.debug(f"Applying binary closing (radius={closing_radius})")
        t0 = time.time()
        binary = binary_closing(binary, structure=struct)
        elapsed = time.time() - t0
        logger.debug(f"  Completed in {elapsed:.1f}s")

    if not skip_fill_holes:
        logger.debug("Filling holes")
        t0 = time.time()
        binary = binary_fill_holes(binary, structure=np.ones((3, 3, 3)))
        elapsed = time.time() - t0
        logger.debug(f"  Completed in {elapsed:.1f}s")

    return binary.astype(np.uint8) * 255


def process_single_file(
    input_path: str,
    output_dir: str,
    progress: str = "",
    **kwargs,
) -> str:
    """Process single TIF file with cleaning operations.

    Args:
        input_path: Path to input TIF file.
        output_dir: Output directory path.
        progress: Optional progress string (e.g., "1/10").
        **kwargs: Arguments passed to clean_mask().

    Returns:
        Path to output file.
    """
    filename = os.path.basename(input_path)
    progress_prefix = f"[{progress}] " if progress else ""
    logger.info(f"{progress_prefix}Processing: {filename}")

    image = tifffile.imread(input_path)
    logger.debug(f"  Shape: {image.shape}, dtype: {image.dtype}")

    cleaned = clean_mask(image, **kwargs)

    input_name = Path(input_path).stem
    output_path = os.path.join(output_dir, f"{input_name}_cleaned.tif")

    tifffile.imwrite(output_path, cleaned, imagej=True, metadata={'axes': 'ZYX'})
    logger.debug(f"  Saved to: {output_path}")

    return output_path


def process_directory(
    input_dir: str,
    output_dir: str = None,
    continue_on_error: bool = False,
    **kwargs
) -> None:
    """Process all *_otsu.tif files in directory.

    Args:
        input_dir: Input directory containing *_otsu.tif files.
        output_dir: Output directory (default: auto-detect as 03_cleaned).
        continue_on_error: If True, continue processing on file errors.
        **kwargs: Arguments passed to clean_mask().

    Raises:
        FileNotFoundError: If input directory does not exist.
        ValueError: If no *_otsu.tif files found.
    """

    input_dir_obj = Path(input_dir)
    if not input_dir_obj.is_absolute():
        input_dir_obj = BASE_DIR / input_dir
    input_dir = str(input_dir_obj.resolve())

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if not os.path.isdir(input_dir):
        raise ValueError(f"Input path is not a directory: {input_dir}")

    # Auto-detect 02_otsu subdirectory if current dir has no TIF files
    detected_dir = auto_detect_subdir(input_dir, '02_otsu')
    if detected_dir != input_dir:
        input_dir = detected_dir
        logger.info(f"Auto-detected input directory: {input_dir}")

    # Auto-detect output directory based on input path structure
    if output_dir is None:
        input_path = Path(input_dir)
        if input_path.name == '02_otsu' and input_path.parent.parent.name == 'preprocess_output':
            # Input is preprocess_output/YYYYMMDD_HHMMSS/02_otsu/ -> output to 03_cleaned/
            output_dir = str(input_path.parent / '03_cleaned')
        else:
            # Create 03_cleaned/ in parent directory
            output_dir = str(input_path.parent / '03_cleaned')

    tif_files = natsorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(('.tif', '.tiff'))
        and '_otsu' in f
        and '_cleaned' not in f
    ])

    if not tif_files:
        raise ValueError(f"No *_otsu.tif files found in directory: {input_dir}")

    logger.info(f"Found {len(tif_files)} TIF files in directory")
    logger.info(f"Output directory: {output_dir}")

    # Create output directory before processing
    os.makedirs(output_dir, exist_ok=True)

    failed_files = []
    for idx, tif_file in enumerate(tif_files, start=1):
        try:
            process_single_file(tif_file, output_dir, progress=f"{idx}/{len(tif_files)}", **kwargs)
        except Exception as e:
            logger.error(f"Failed to process {tif_file}: {e}")
            failed_files.append((tif_file, str(e)))
            if not continue_on_error:
                raise

    # Report results
    successful = len(tif_files) - len(failed_files)
    logger.info(f"Completed processing {successful}/{len(tif_files)} files")

    if failed_files:
        logger.warning(f"Failed files ({len(failed_files)}):")
        for path, error in failed_files:
            logger.warning(f"  {path}: {error}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Clean binary masks with morphological filters.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python clean_masks.py preprocess_output/20251117_171518/
  python clean_masks.py preprocess_output/20251117_171518/ --opening-radius 2 --closing-radius 3
  python clean_masks.py preprocess_output/20251117_171518/ --min-size 100
        """
    )
    parser.add_argument(
        'input_dir',
        help='Directory containing *_otsu.tif files (relative to BASE_DIR or absolute)',
    )
    parser.add_argument(
        '-o', '--output-dir',
        help='Output directory (default: auto-detect as 03_cleaned)',
    )
    parser.add_argument(
        '-c', '--config',
        help='Config file path (default: pipeline/preprocess_config.yaml)',
    )
    parser.add_argument(
        '--opening-radius',
        type=int,
        help='Opening radius (0 to skip, overrides config)',
    )
    parser.add_argument(
        '--closing-radius',
        type=int,
        help='Closing radius (0 to skip, overrides config)',
    )
    parser.add_argument(
        '--min-size',
        type=int,
        help='Min object size in voxels (overrides config)',
    )
    parser.add_argument(
        '--skip-remove-small',
        action='store_true',
        help='Skip remove small objects step',
    )
    parser.add_argument(
        '--skip-fill-holes',
        action='store_true',
        help='Skip fill holes step',
    )
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue processing other files if one fails',
    )
    parser.add_argument(
        '--log-file',
        help='Path to log file',
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_file=args.log_file)

    # Load config
    config_path = args.config or (BASE_DIR / 'pipeline' / 'preprocess_config.yaml')
    try:
        config = load_config(config_path, PREPROCESS_SCHEMA)
        clean_config = config.get('clean_masks', {})
    except FileNotFoundError:
        logger.warning(f"Config not found: {config_path}, using defaults")
        clean_config = {}

    # Get parameters (CLI overrides config)
    opening_radius = args.opening_radius if args.opening_radius is not None else clean_config.get('opening_radius', 1)
    closing_radius = args.closing_radius if args.closing_radius is not None else clean_config.get('closing_radius', 2)
    min_size = args.min_size if args.min_size is not None else clean_config.get('min_size', 64)

    logger.info(f"Parameters: opening_radius={opening_radius}, closing_radius={closing_radius}, min_size={min_size}")

    kwargs = {
        'opening_radius': opening_radius,
        'closing_radius': closing_radius,
        'min_size': min_size,
        'skip_remove_small': args.skip_remove_small,
        'skip_fill_holes': args.skip_fill_holes,
    }

    try:
        process_directory(
            args.input_dir,
            output_dir=args.output_dir,
            continue_on_error=args.continue_on_error,
            **kwargs
        )
        logger.info("Processing completed")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()


# python preprocess/clean_masks.py preprocess_output/20251117_171518/
# python preprocess/clean_masks.py preprocess_output/20251117_171518/ --opening-radius 2 --closing-radius 3
# python preprocess/clean_masks.py preprocess_output/20251117_171518/ --min-size 100
# python preprocess/clean_masks.py preprocess_output/20251117_171518/ --skip-remove-small