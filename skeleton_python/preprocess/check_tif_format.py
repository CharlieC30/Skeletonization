"""
TIF format validation and correction utility.
Handles ImageJ virtual stack issues, converts to uint8.
"""

import os
import re
import sys
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile

from pipeline.utils import ensure_3d, setup_logging

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
        image = ensure_3d(image)

        logger.debug(f"Loaded shape {image.shape}, dtype {image.dtype}")
        return image

    except Exception as e:
        raise ValueError(f"Failed to read TIF file {path}: {e}")


def normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    """Convert array to uint8 with linear scaling.

    Args:
        array: Input numpy array of any numeric dtype.

    Returns:
        uint8 numpy array scaled to [0, 255].
    """
    if array.dtype == np.uint8:
        return array

    arr_min, arr_max = array.min(), array.max()

    if arr_min == arr_max:
        return np.zeros_like(array, dtype=np.uint8)

    scaled = (array.astype(np.float64) - arr_min) / (arr_max - arr_min) * 255
    logger.debug(f"Converted to uint8 (scaled from range [{arr_min:.2f}, {arr_max:.2f}])")
    return scaled.astype(np.uint8)


def natural_sort_key(filename: str) -> list:
    """Generate sort key for natural numeric ordering.

    Args:
        filename: Filename string to generate key for.

    Returns:
        List of string and int parts for sorting.
    """
    return [int(c) if c.isdigit() else c.lower()
            for c in re.split(r'(\d+)', filename)]


def load_2d_sequence(folder_path: str) -> np.ndarray:
    """Load folder of 2D TIFs as 3D stack, sorted by numeric order.

    Args:
        folder_path: Path to folder containing 2D TIF files.

    Returns:
        3D numpy array with shape (Z, Y, X), or None if not a 2D sequence.

    Raises:
        ValueError: If no TIF files found or mixed dimensions detected.
    """
    tif_files = [f for f in os.listdir(folder_path)
                 if f.lower().endswith(('.tif', '.tiff'))]
    tif_files = sorted(tif_files, key=natural_sort_key)

    if not tif_files:
        raise ValueError(f"No TIF files found in {folder_path}")

    # Check if first file is 2D
    first_path = os.path.join(folder_path, tif_files[0])
    first = tifffile.imread(first_path)
    if first.ndim != 2:
        return None  # Not a 2D sequence, use existing logic

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
    logger.debug(f"Loaded shape {stack.shape}, dtype {stack.dtype}")
    return stack


def process_single_file(input_path: str, output_dir: str, progress: str = "") -> str:
    """Process single TIF file: load, convert to uint8, and save.

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

    image = load_and_check_tif(input_path)
    logger.debug(f"  Shape: {image.shape}, dtype: {image.dtype}")

    image_uint8 = normalize_to_uint8(image)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    tifffile.imwrite(output_path, image_uint8, imagej=True, metadata={'axes': 'ZYX'})
    logger.debug(f"  Saved to: {output_path}")

    return output_path


def process_path(input_path: str, output_dir: str = None) -> None:
    """Process input path (file or directory) and output corrected TIF files.

    Handles three input types:
    - Single TIF file (2D or 3D)
    - Directory of 3D TIF files (processed individually)
    - Directory of 2D TIF files (combined into single 3D stack)

    Args:
        input_path: Path to TIF file or directory.
        output_dir: Output directory (default: preprocess_output/TIMESTAMP/01_format).

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

    if os.path.isfile(input_path):
        if not input_path.lower().endswith(('.tif', '.tiff')):
            raise ValueError(f"Input file is not a TIF file: {input_path}")
        process_single_file(input_path, output_dir)

    elif os.path.isdir(input_path):
        # Try to load as 2D sequence first
        stack = load_2d_sequence(input_path)

        if stack is not None:
            # 2D sequence detected, save as single 3D TIF
            logger.info(f"Processing 2D sequence from: {input_path}")
            image_uint8 = normalize_to_uint8(stack)

            os.makedirs(output_dir, exist_ok=True)
            folder_name = os.path.basename(input_path.rstrip('/\\'))
            output_path = os.path.join(output_dir, f"{folder_name}.tif")

            tifffile.imwrite(output_path, image_uint8, imagej=True, metadata={'axes': 'ZYX'})
            logger.debug(f"  Saved to: {output_path}")
        else:
            # Not a 2D sequence, process each 3D TIF separately
            tif_files = sorted(
                [os.path.join(input_path, f)
                 for f in os.listdir(input_path)
                 if f.lower().endswith(('.tif', '.tiff'))],
                key=lambda x: natural_sort_key(os.path.basename(x))
            )

            if not tif_files:
                raise ValueError(f"No TIF files found in directory: {input_path}")

            logger.info(f"Found {len(tif_files)} TIF files in directory")

            for idx, tif_file in enumerate(tif_files, start=1):
                try:
                    process_single_file(tif_file, output_dir, progress=f"{idx}/{len(tif_files)}")
                except Exception as e:
                    logger.error(f"Error processing {tif_file}: {e}")
                    raise

            logger.info(f"Completed processing {len(tif_files)} files")
    else:
        raise ValueError(f"Input path is neither file nor directory: {input_path}")


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python check_tif_format.py <input_path> [output_dir]")
        print("  input_path: TIF file, directory of 3D TIFs, or directory of 2D TIFs (sequence)")
        print("  output_dir: Optional output directory (default: preprocess_output/YYYYMMDD_HHMMSS/01_format)")
        sys.exit(1)

    # Setup logging
    setup_logging()

    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        process_path(input_path, output_dir)
        logger.info("Processing completed")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()


# python preprocess/check_tif_format.py DATA/ori_image/
# python preprocess/check_tif_format.py DATA/ori_image/skeleton_roi_32bit.tif
# python preprocess/check_tif_format.py DATA/2d_sequence_folder/