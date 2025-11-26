"""
Binary mask cleaning with 3D/2D adaptive filters.
Recommended workflow: Remove Small Objects → Opening → Closing → Fill Holes
"""

import os
import sys
import argparse
import time
from pathlib import Path
import numpy as np
import tifffile
from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes
from skimage.morphology import remove_small_objects

BASE_DIR = Path(__file__).parent.parent.resolve()


def get_structure(image: np.ndarray, radius: int) -> np.ndarray:
    """Generate structure element based on image dimensions.

    Args:
        image: Input array (used to determine 2D or 3D).
        radius: Radius of structure element.

    Returns:
        2D or 3D array of ones with size (2*radius+1).

    Raises:
        ValueError: If image is not 2D or 3D.
    """
    size = 2 * radius + 1
    if image.ndim == 2:
        return np.ones((size, size))
    elif image.ndim == 3:
        return np.ones((size, size, size))
    else:
        raise ValueError(f"Unsupported dimensions: {image.ndim}")


def get_min_size(image: np.ndarray, min_size_3d: int, min_size_2d: int) -> int:
    """Select appropriate min_size threshold based on dimensions.

    Args:
        image: Input array (used to determine 2D or 3D).
        min_size_3d: Min size for 3D images (voxels).
        min_size_2d: Min size for 2D images (pixels).

    Returns:
        Appropriate min_size value.

    Raises:
        ValueError: If image is not 2D or 3D.
    """
    if image.ndim == 2:
        return min_size_2d
    elif image.ndim == 3:
        return min_size_3d
    else:
        raise ValueError(f"Unsupported dimensions: {image.ndim}")


def clean_mask(
    image: np.ndarray,
    opening_radius: int = 1,
    closing_radius: int = 2,
    min_size_3d: int = 64,
    min_size_2d: int = 15,
    skip_remove_small: bool = False,
    skip_fill_holes: bool = False,
) -> np.ndarray:
    """Clean binary mask with morphological operations.

    Workflow: Remove Small Objects -> Opening -> Closing -> Fill Holes

    Args:
        image: Input binary mask.
        opening_radius: Radius for opening operation (0 to skip).
        closing_radius: Radius for closing operation (0 to skip).
        min_size_3d: Min object size for 3D images (voxels).
        min_size_2d: Min object size for 2D images (pixels).
        skip_remove_small: Skip remove small objects step.
        skip_fill_holes: Skip fill holes step.

    Returns:
        Cleaned binary mask (uint8, values 0 or 255).
    """
    binary = image > 0

    if not skip_remove_small:
        min_size = get_min_size(binary, min_size_3d, min_size_2d)
        print(f"  Removing small objects (min_size={min_size}, ndim={binary.ndim})")
        t0 = time.time()
        binary = remove_small_objects(binary, min_size=min_size)
        elapsed = time.time() - t0
        print(f"  └─ Completed in {elapsed:.1f}s")

    if opening_radius > 0:
        struct = get_structure(binary, opening_radius)
        print(f"  Applying binary opening (radius={opening_radius}, ndim={binary.ndim})")
        t0 = time.time()
        binary = binary_opening(binary, structure=struct)
        elapsed = time.time() - t0
        print(f"  └─ Completed in {elapsed:.1f}s")

    if closing_radius > 0:
        struct = get_structure(binary, closing_radius)
        print(f"  Applying binary closing (radius={closing_radius}, ndim={binary.ndim})")
        t0 = time.time()
        binary = binary_closing(binary, structure=struct)
        elapsed = time.time() - t0
        print(f"  └─ Completed in {elapsed:.1f}s")

    if not skip_fill_holes:
        print(f"  Filling holes (ndim={binary.ndim})")
        t0 = time.time()
        if binary.ndim == 2:
            binary = binary_fill_holes(binary)
        else:
            binary = binary_fill_holes(binary, structure=np.ones((3, 3, 3)))
        elapsed = time.time() - t0
        print(f"  └─ Completed in {elapsed:.1f}s")

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
    progress_prefix = f"{progress}: " if progress else ""
    print(f"Processing {progress_prefix}{input_path}")

    image = tifffile.imread(input_path)

    if image.ndim not in (2, 3):
        raise ValueError(f"Unexpected dimensions: {image.ndim}. Expected 2D or 3D.")

    print(f"  Loaded shape {image.shape}, dtype {image.dtype}")

    cleaned = clean_mask(image, **kwargs)

    input_name = Path(input_path).stem
    output_path = os.path.join(output_dir, f"{input_name}_cleaned.tif")

    if cleaned.ndim == 3:
        tifffile.imwrite(output_path, cleaned, imagej=True, metadata={'axes': 'ZYX'})
    else:
        tifffile.imwrite(output_path, cleaned)

    print(f"  Saved to: {output_path}")

    return output_path


def process_directory(input_dir: str, output_dir: str = None, **kwargs) -> None:
    """Process all *_otsu.tif files in directory.

    Args:
        input_dir: Input directory containing *_otsu.tif files.
        output_dir: Output directory (default: auto-detect as 03_cleaned).
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

    # Auto-detect 02_otsu subdirectory if current dir has no *_otsu.tif files
    potential_otsu_dir = os.path.join(input_dir, '02_otsu')
    if os.path.exists(potential_otsu_dir) and os.path.isdir(potential_otsu_dir):
        current_otsu_tifs = [f for f in os.listdir(input_dir)
                            if f.lower().endswith(('.tif', '.tiff')) and '_otsu' in f
                            and os.path.isfile(os.path.join(input_dir, f))]
        if not current_otsu_tifs:
            input_dir = potential_otsu_dir
            print(f"Auto-detected input directory: {input_dir}")

    # Auto-detect output directory based on input path structure
    if output_dir is None:
        input_path = Path(input_dir)
        if input_path.name == '02_otsu' and input_path.parent.parent.name == 'preprocess_output':
            # Input is preprocess_output/YYYYMMDD_HHMMSS/02_otsu/ -> output to 03_cleaned/
            output_dir = str(input_path.parent / '03_cleaned')
        else:
            # Create 03_cleaned/ in parent directory
            output_dir = str(input_path.parent / '03_cleaned')

    tif_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(('.tif', '.tiff'))
        and '_otsu' in f
        and '_cleaned' not in f
    ]

    if not tif_files:
        raise ValueError(f"No *_otsu.tif files found in directory: {input_dir}")

    print(f"Found {len(tif_files)} TIF files in directory")
    print(f"Output directory: {output_dir}")

    # Create output directory before processing
    os.makedirs(output_dir, exist_ok=True)

    for idx, tif_file in enumerate(tif_files, start=1):
        try:
            process_single_file(tif_file, output_dir, progress=f"{idx}/{len(tif_files)}", **kwargs)
        except Exception as e:
            print(f"Error processing {tif_file}: {e}")
            raise

    print(f"Completed processing {len(tif_files)} files")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Clean binary masks with adaptive 2D/3D filters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'input_dir',
        help='Directory containing *_otsu.tif files (relative to BASE_DIR or absolute)',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory (default: auto-detect as 03_cleaned)',
    )
    parser.add_argument(
        '--opening-radius',
        type=int,
        default=1,
        help='Opening radius (0 to skip)',
    )
    parser.add_argument(
        '--closing-radius',
        type=int,
        default=2,
        help='Closing radius (0 to skip)',
    )
    parser.add_argument(
        '--min-size-3d',
        type=int,
        default=64,
        help='Min object size for 3D images (voxels)',
    )
    parser.add_argument(
        '--min-size-2d',
        type=int,
        default=15,
        help='Min object size for 2D images (pixels)',
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

    args = parser.parse_args()

    kwargs = {
        'opening_radius': args.opening_radius,
        'closing_radius': args.closing_radius,
        'min_size_3d': args.min_size_3d,
        'min_size_2d': args.min_size_2d,
        'skip_remove_small': args.skip_remove_small,
        'skip_fill_holes': args.skip_fill_holes,
    }

    try:
        process_directory(args.input_dir, output_dir=args.output_dir, **kwargs)
        print("Processing completed")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()


# python preprocess/clean_masks.py preprocess_output/20251117_171518/
# python preprocess/clean_masks.py preprocess_output/20251117_171518/ --opening-radius 2 --closing-radius 3
# python preprocess/clean_masks.py preprocess_output/20251117_171518/ --min-size-3d 100 --min-size-2d 20
# python preprocess/clean_masks.py preprocess_output/20251117_171518/ --skip-remove-small