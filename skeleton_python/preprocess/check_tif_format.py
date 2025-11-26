"""
TIF format validation and conversion utility.

Handles ImageJ virtual stack issues, normalizes values, and converts to uint8.
Supports single 3D TIF files or directories of 2D TIF sequences.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile
from natsort import natsorted

from pipeline.utils import setup_logging, load_config, PREPROCESS_SCHEMA

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.resolve()


def load_and_check_tif(path: str) -> np.ndarray:
    """Load TIF file and handle ImageJ virtual stack format.

    Args:
        path: Path to TIF file.

    Returns:
        3D numpy array with shape (Z, Y, X).

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file cannot be read or has invalid dimensions.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    try:
        with tifffile.TiffFile(path) as tif:
            num_pages = len(tif.pages)
            is_imagej = tif.is_imagej

            if is_imagej:
                metadata = tif.imagej_metadata or {}
                expected_slices = metadata.get('slices', 1)

                # ImageJ virtual stack: single page but metadata indicates multiple slices
                if num_pages == 1 and expected_slices > 1:
                    logger.debug(f"Detected ImageJ virtual stack ({expected_slices} slices)")

                    page = tif.pages[0]
                    height, width = page.shape
                    base_dtype = page.dtype
                    data_offset = page.dataoffsets[0]

                    # Map to big-endian dtype for ImageJ compatibility
                    dtype_map = {
                        np.dtype('float32'): np.dtype('>f4'),
                        np.dtype('float64'): np.dtype('>f8'),
                        np.dtype('uint16'): np.dtype('>u2'),
                        np.dtype('int16'): np.dtype('>i2'),
                    }
                    read_dtype = dtype_map.get(base_dtype, base_dtype)

                    with open(path, 'rb') as f:
                        f.seek(data_offset)
                        all_data = np.fromfile(f, dtype=read_dtype)

                    pixels_per_slice = height * width
                    total_complete_slices = len(all_data) // pixels_per_slice

                    if total_complete_slices == 0:
                        raise ValueError("Insufficient data for even one complete slice")

                    complete_data = all_data[:total_complete_slices * pixels_per_slice]
                    image = complete_data.reshape((total_complete_slices, height, width))

                    if image.dtype.byteorder == '>':
                        image = image.astype(image.dtype.newbyteorder('='))

                    logger.debug(f"Loaded shape {image.shape}, dtype {image.dtype}")
                    return image

        image = tifffile.imread(path)

        if image.ndim == 2:
            raise ValueError(
                f"Input is 2D ({image.shape}). "
                "Skeletonization requires 3D data. "
                "For 2D slice sequence, provide folder path instead."
            )
        elif image.ndim != 3:
            raise ValueError(f"Expected 3D array, got {image.ndim}D")

        logger.debug(f"Loaded shape {image.shape}, dtype {image.dtype}")
        return image

    except Exception as e:
        raise ValueError(f"Failed to read TIF file {path}: {e}")


def normalize(array: np.ndarray,
              method: str = 'minmax',
              percentile_low: float = 0.0,
              percentile_high: float = 100.0) -> np.ndarray:
    """Normalize array values to [0, 1] range.

    Args:
        array: Input numpy array of any numeric dtype.
        method: Normalization method, 'minmax' or 'percentile'.
        percentile_low: Lower percentile for clipping (only for 'percentile').
        percentile_high: Upper percentile for clipping (only for 'percentile').

    Returns:
        Normalized array with values in [0, 1] range.

    Raises:
        ValueError: If unknown method is specified.
    """
    if method == 'minmax':
        arr_min, arr_max = array.min(), array.max()
    elif method == 'percentile':
        arr_min = np.percentile(array, percentile_low)
        arr_max = np.percentile(array, percentile_high)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    if arr_min == arr_max:
        logger.debug("Array has constant value, returning zeros")
        return np.zeros_like(array, dtype=np.float64)

    clipped = np.clip(array, arr_min, arr_max)
    normalized = (clipped.astype(np.float64) - arr_min) / (arr_max - arr_min)
    logger.debug(f"Normalized using {method} (range [{arr_min:.2f}, {arr_max:.2f}])")
    return normalized


def convert_to_uint8(array: np.ndarray) -> np.ndarray:
    """Convert normalized [0, 1] array to uint8 [0, 255].

    Args:
        array: Normalized array with values in [0, 1] range.

    Returns:
        uint8 array with values in [0, 255].
    """
    if array.dtype == np.uint8:
        return array

    return (array * 255).astype(np.uint8)


def load_2d_sequence(folder_path: str) -> np.ndarray:
    """Load folder of 2D TIFs as 3D stack, sorted by numeric order.

    Args:
        folder_path: Path to folder containing 2D TIF files.

    Returns:
        3D numpy array with shape (Z, Y, X), or None if not a 2D sequence.

    Raises:
        ValueError: If no TIF files found, mixed dimensions, or single 2D file.
    """
    tif_files = [f for f in os.listdir(folder_path)
                 if f.lower().endswith(('.tif', '.tiff'))]
    tif_files = natsorted(tif_files)

    if not tif_files:
        raise ValueError(f"No TIF files found in {folder_path}")

    # Check first file using load_and_check_tif to handle virtual stacks
    first_path = os.path.join(folder_path, tif_files[0])
    try:
        first = load_and_check_tif(first_path)
        # If load_and_check_tif succeeds, it's 3D (not a 2D sequence)
        return None
    except ValueError as e:
        # Check if error is due to 2D input
        if "2D" in str(e):
            # It's a 2D file, continue with 2D sequence logic
            first = tifffile.imread(first_path)
        else:
            raise

    # Single 2D file is not valid
    if len(tif_files) == 1:
        raise ValueError(
            f"Single 2D file found ({first.shape}). "
            "Skeletonization requires 3D data with multiple slices."
        )

    logger.debug(f"Detected 2D sequence ({len(tif_files)} files)")

    # Stack all slices
    slices = [first]
    for f in tif_files[1:]:
        img = tifffile.imread(os.path.join(folder_path, f))
        if img.ndim != 2:
            raise ValueError(f"Mixed dimensions in sequence: {f}")
        if img.shape != first.shape:
            raise ValueError(f"Shape mismatch: {f} has {img.shape}, expected {first.shape}")
        slices.append(img)

    stack = np.stack(slices, axis=0)
    logger.debug(f"Stacked {len(slices)} slices, shape {stack.shape}, dtype {stack.dtype}")
    return stack


def process_single_file(input_path: str,
                        output_dir: str,
                        progress: str = "",
                        normalize_method: str = 'minmax',
                        percentile_low: float = 0.0,
                        percentile_high: float = 100.0) -> str:
    """Process single TIF file: load, normalize, convert to uint8, and save.

    Args:
        input_path: Path to input TIF file.
        output_dir: Output directory path.
        progress: Optional progress string (e.g., "1/10").
        normalize_method: Normalization method ('minmax' or 'percentile').
        percentile_low: Lower percentile for clipping.
        percentile_high: Upper percentile for clipping.

    Returns:
        Path to output file.
    """
    filename = os.path.basename(input_path)
    progress_prefix = f"[{progress}] " if progress else ""
    logger.info(f"{progress_prefix}Processing: {filename}")

    image = load_and_check_tif(input_path)
    logger.debug(f"  Shape: {image.shape}, dtype: {image.dtype}")

    normalized = normalize(image,
                           method=normalize_method,
                           percentile_low=percentile_low,
                           percentile_high=percentile_high)
    image_uint8 = convert_to_uint8(normalized)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    tifffile.imwrite(output_path, image_uint8, imagej=True, metadata={'axes': 'ZYX'})
    logger.debug(f"  Saved to: {output_path}")

    return output_path


def process_path(input_path: str,
                 output_dir: str = None,
                 normalize_method: str = 'minmax',
                 percentile_low: float = 0.0,
                 percentile_high: float = 100.0) -> None:
    """Process input path (file or directory) and output corrected TIF files.

    Handles three input types:
    - Single TIF file (3D only, 2D raises error)
    - Directory of 3D TIF files (processed individually)
    - Directory of 2D TIF files (combined into single 3D stack)

    Args:
        input_path: Path to TIF file or directory.
        output_dir: Output directory (default: preprocess_output/TIMESTAMP/01_format).
        normalize_method: Normalization method ('minmax' or 'percentile').
        percentile_low: Lower percentile for clipping.
        percentile_high: Upper percentile for clipping.

    Raises:
        FileNotFoundError: If input path does not exist.
        ValueError: If no TIF files found or invalid input.
    """
    input_path_obj = Path(input_path)
    if not input_path_obj.is_absolute():
        input_path_obj = BASE_DIR / input_path
    input_path = str(input_path_obj.resolve())

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = str(BASE_DIR / 'preprocess_output' / timestamp / '01_format')

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    # Common normalize kwargs
    norm_kwargs = {
        'normalize_method': normalize_method,
        'percentile_low': percentile_low,
        'percentile_high': percentile_high,
    }

    if os.path.isfile(input_path):
        if not input_path.lower().endswith(('.tif', '.tiff')):
            raise ValueError(f"Input file is not a TIF file: {input_path}")
        process_single_file(input_path, output_dir, **norm_kwargs)

    elif os.path.isdir(input_path):
        # Try to load as 2D sequence first
        stack = load_2d_sequence(input_path)

        if stack is not None:
            # 2D sequence detected, save as single 3D TIF
            logger.info(f"Processing 2D sequence from: {input_path}")
            normalized = normalize(stack,
                                   method=normalize_method,
                                   percentile_low=percentile_low,
                                   percentile_high=percentile_high)
            image_uint8 = convert_to_uint8(normalized)

            os.makedirs(output_dir, exist_ok=True)
            folder_name = os.path.basename(input_path.rstrip('/\\'))
            output_path = os.path.join(output_dir, f"{folder_name}.tif")

            tifffile.imwrite(output_path, image_uint8, imagej=True, metadata={'axes': 'ZYX'})
            logger.debug(f"  Saved to: {output_path}")
        else:
            # Not a 2D sequence, process each 3D TIF separately
            tif_files = natsorted([
                os.path.join(input_path, f)
                for f in os.listdir(input_path)
                if f.lower().endswith(('.tif', '.tiff'))
            ])

            if not tif_files:
                raise ValueError(f"No TIF files found in directory: {input_path}")

            logger.info(f"Found {len(tif_files)} TIF files in directory")

            for idx, tif_file in enumerate(tif_files, start=1):
                try:
                    process_single_file(tif_file, output_dir,
                                        progress=f"{idx}/{len(tif_files)}",
                                        **norm_kwargs)
                except Exception as e:
                    logger.error(f"Error processing {tif_file}: {e}")
                    raise

            logger.info(f"Completed processing {len(tif_files)} files")
    else:
        raise ValueError(f"Input path is neither file nor directory: {input_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='TIF format validation and conversion utility.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check_tif_format.py DATA/ori_image/
  python check_tif_format.py DATA/ori_image/sample.tif
  python check_tif_format.py DATA/ori_image/ --normalize-method percentile --percentile-low 1 --percentile-high 99
        """
    )
    parser.add_argument('input_path',
                        help='TIF file, directory of 3D TIFs, or directory of 2D TIFs (sequence)')
    parser.add_argument('-o', '--output-dir',
                        help='Output directory (default: preprocess_output/TIMESTAMP/01_format)')
    parser.add_argument('-c', '--config',
                        help='Config file path (default: pipeline/preprocess_config.yaml)')
    parser.add_argument('--normalize-method', choices=['minmax', 'percentile'],
                        help='Normalization method (overrides config)')
    parser.add_argument('--percentile-low', type=float,
                        help='Lower percentile for clipping (overrides config)')
    parser.add_argument('--percentile-high', type=float,
                        help='Upper percentile for clipping (overrides config)')
    parser.add_argument('--log-file',
                        help='Log file path')

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_file=args.log_file)

    # Load config
    config_path = args.config or (BASE_DIR / 'pipeline' / 'preprocess_config.yaml')
    try:
        config = load_config(config_path, PREPROCESS_SCHEMA)
        format_config = config.get('format_conversion', {})
    except FileNotFoundError:
        logger.warning(f"Config not found: {config_path}, using defaults")
        format_config = {}

    # Get normalize parameters (CLI overrides config)
    normalize_method = args.normalize_method or format_config.get('normalize_method', 'minmax')
    percentile_low = args.percentile_low if args.percentile_low is not None else format_config.get('percentile_low', 0.0)
    percentile_high = args.percentile_high if args.percentile_high is not None else format_config.get('percentile_high', 100.0)

    logger.info(f"Normalize: method={normalize_method}, percentile=[{percentile_low}, {percentile_high}]")

    try:
        process_path(
            args.input_path,
            args.output_dir,
            normalize_method=normalize_method,
            percentile_low=percentile_low,
            percentile_high=percentile_high,
        )
        logger.info("Processing completed")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()


# python preprocess/check_tif_format.py DATA/ori_image/
# python preprocess/check_tif_format.py DATA/ori_image/skeleton_roi_8bit_z60-630_crop
# python preprocess/check_tif_format.py DATA/ori_image/skeleton_roi_8bit_z60-630_crop.tif