"""3D skeletonization using Kimimaro algorithm."""
import os
import sys
import argparse
import time
import logging
from pathlib import Path

import numpy as np
import tifffile
import kimimaro
from natsort import natsorted

from pipeline.utils import (
    auto_detect_subdir,
    extract_timestamp_from_path,
    get_output_dir,
    load_config,
    update_config_from_args,
    setup_logging,
    SKELETON_SCHEMA,
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.resolve()
DEFAULT_CONFIG = BASE_DIR / 'pipeline' / 'skeleton_config.yaml'


def load_config_file(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = DEFAULT_CONFIG
    return load_config(config_path, schema=SKELETON_SCHEMA)


def skeletonize_mask(
    image: np.ndarray,
    teasar_params: dict = None,
    dust_threshold: int = 500,
    anisotropy: tuple = (1, 1, 1),
    fix_branching: bool = True,
    fix_borders: bool = True,
    progress: bool = True,
    parallel: int = 1,
    postprocess_dust_threshold: int = 1000,
    postprocess_tick_threshold: int = 0,
    keep_largest_component_only: bool = True,
) -> dict:
    """
    Skeletonize binary mask using Kimimaro.

    Returns:
        Dictionary of skeleton objects keyed by label ID.
    """
    # Default TEASAR parameters
    if teasar_params is None:
        teasar_params = {
            'scale': 0.3,
            'const': 50,
            'pdrf_scale': 100000,
            'pdrf_exponent': 4,
            'soma_detection_threshold': 999999,
            'soma_acceptance_threshold': 999999,
            'soma_invalidation_scale': 2,
            'soma_invalidation_const': 300,
            'max_paths': None,
        }

    # Convert to binary labels (values > 0 are considered mask)
    labels = (image > 0).astype(np.uint8)

    # Run Kimimaro skeletonization
    t0 = time.time()
    skels = kimimaro.skeletonize(
        labels,
        teasar_params=teasar_params,
        dust_threshold=dust_threshold,
        anisotropy=anisotropy,
        fix_branching=fix_branching,
        fix_borders=fix_borders,
        progress=progress,
        parallel=parallel,
    )
    elapsed = time.time() - t0
    logger.debug(f"Skeletonization completed in {elapsed:.1f}s")

    if len(skels) == 0:
        return {}

    # Postprocessing
    logger.debug("Postprocessing skeletons")
    t0 = time.time()
    skels_filtered = {}
    for label_id, skel in skels.items():
        original_vertices = len(skel.vertices)

        # Apply kimimaro postprocessing
        skel = kimimaro.postprocess(
            skel,
            dust_threshold=postprocess_dust_threshold,
            tick_threshold=postprocess_tick_threshold,
        )

        # Optionally keep only largest component
        if keep_largest_component_only:
            components = skel.components()
            if len(components) > 1:
                logger.debug(f"  Label {label_id}: {len(components)} components, keeping largest")
                skel = max(components, key=lambda c: c.cable_length())
            elif len(components) == 1:
                skel = components[0]

        filtered_vertices = len(skel.vertices)
        if filtered_vertices < original_vertices:
            logger.debug(f"  Label {label_id}: filtered {original_vertices} -> {filtered_vertices} vertices")

        skels_filtered[label_id] = skel

    elapsed = time.time() - t0
    logger.debug(f"Postprocessing completed in {elapsed:.1f}s")

    return skels_filtered


def process_single_file(
    input_path: str,
    output_dir: str,
    progress: str = "",
    **kwargs,
) -> str:
    """Process single TIF file with Kimimaro skeletonization."""
    # Extract kimimaro's progress parameter (passed as 'kimimaro_progress' to avoid conflict)
    kimimaro_progress = kwargs.pop('kimimaro_progress', True)

    filename = os.path.basename(input_path)
    progress_prefix = f"[{progress}] " if progress else ""
    logger.info(f"{progress_prefix}Processing: {filename}")

    image = tifffile.imread(input_path)
    if image.ndim != 3:
        raise ValueError(
            f"Expected 3D image, got {image.ndim}D. "
            "Run check_tif_format.py first."
        )

    logger.debug(f"  Shape: {image.shape}, dtype: {image.dtype}")

    logger.debug("Running Kimimaro skeletonization")
    skels = skeletonize_mask(image, progress=kimimaro_progress, **kwargs)

    if len(skels) == 0:
        logger.warning("No skeletons generated")
        return output_dir

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    input_name = Path(input_path).stem
    total_vertices = 0
    total_edges = 0

    for label_id, skel in skels.items():
        total_vertices += len(skel.vertices)
        total_edges += len(skel.edges)

        # Save SWC file
        swc_file = os.path.join(output_dir, f'{input_name}_label_{label_id}.swc')
        try:
            swc_content = skel.to_swc()
            with open(swc_file, 'w') as f:
                f.write(swc_content)
        except Exception as e:
            logger.warning(f"Failed to save SWC: {e}")

        # Save TIF visualization
        tif_file = os.path.join(output_dir, f'{input_name}_label_{label_id}.tif')
        skeleton_img = np.zeros(image.shape, dtype=np.uint8)

        # Convert to voxel space (coordinates are in z,y,x order)
        skel_voxel = skel.voxel_space()
        vertices_int = skel_voxel.vertices.astype(int)

        for v in vertices_int:
            z, y, x = v
            if 0 <= z < image.shape[0] and 0 <= y < image.shape[1] and 0 <= x < image.shape[2]:
                skeleton_img[z, y, x] = 255

        if skeleton_img.ndim == 3:
            tifffile.imwrite(tif_file, skeleton_img, imagej=True, metadata={'axes': 'ZYX'})
        else:
            tifffile.imwrite(tif_file, skeleton_img)

    logger.debug(f"  Generated {len(skels)} skeleton(s), {total_vertices} vertices, {total_edges} edges")
    logger.debug(f"  Saved to: {output_dir}")

    return output_dir


def process_directory(
    input_dir: str,
    output_dir: str = None,
    continue_on_error: bool = False,
    **kwargs
) -> None:
    """Process all *_cleaned.tif files in directory.

    Args:
        input_dir: Directory containing *_cleaned.tif files.
        output_dir: Output directory (auto-detected if None).
        continue_on_error: If True, continue processing on file errors.
        **kwargs: Arguments passed to process_single_file.
    """

    input_dir_obj = Path(input_dir)
    if not input_dir_obj.is_absolute():
        input_dir_obj = BASE_DIR / input_dir
    input_dir = str(input_dir_obj.resolve())

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if not os.path.isdir(input_dir):
        raise ValueError(f"Input path is not a directory: {input_dir}")

    # Auto-detect 03_cleaned subdirectory if current dir has no TIF files
    detected_dir = auto_detect_subdir(input_dir, '03_cleaned')
    if detected_dir != input_dir:
        input_dir = detected_dir
        logger.info(f"Auto-detected input directory: {input_dir}")

    # Auto-detect output directory using timestamp extraction
    if output_dir is None:
        output_dir = get_output_dir(input_dir, str(BASE_DIR / 'output'), '04_skeleton')

    tif_files = natsorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(('.tif', '.tiff'))
        and '_cleaned' in f
    ])

    if not tif_files:
        raise ValueError(f"No *_cleaned.tif files found in directory: {input_dir}")

    logger.info(f"Found {len(tif_files)} TIF files in directory")
    logger.info(f"Output directory: {output_dir}")

    # Create output directory before processing
    os.makedirs(output_dir, exist_ok=True)

    failed_files = []
    for idx, tif_file in enumerate(tif_files, start=1):
        try:
            # Extract 'progress' from kwargs and pass with different key to avoid conflict
            file_kwargs = {k: v for k, v in kwargs.items() if k != 'progress'}
            file_kwargs['kimimaro_progress'] = kwargs.get('progress', True)

            process_single_file(tif_file, output_dir, progress=f"{idx}/{len(tif_files)}", **file_kwargs)
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
        description='Skeletonize cleaned binary masks using Kimimaro',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'input_dir',
        help='Directory containing *_cleaned.tif files (relative to BASE_DIR or absolute)',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory (default: auto-detect with timestamp)',
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config file (YAML)',
    )

    # Kimimaro TEASAR parameters
    parser.add_argument(
        '--scale',
        type=float,
        help='TEASAR scale parameter',
    )
    parser.add_argument(
        '--const',
        type=int,
        help='TEASAR const parameter',
    )
    parser.add_argument(
        '--pdrf-scale',
        type=float,
        help='PDRF scale parameter',
    )
    parser.add_argument(
        '--pdrf-exponent',
        type=float,
        help='PDRF exponent parameter',
    )

    # Other Kimimaro parameters
    parser.add_argument(
        '--dust-threshold',
        type=int,
        help='Remove components smaller than this before skeletonization (voxels)',
    )
    parser.add_argument(
        '--anisotropy',
        type=float,
        nargs=3,
        help='Voxel anisotropy (Z Y X)',
    )
    parser.add_argument(
        '--parallel',
        type=int,
        help='Number of parallel processes',
    )
    parser.add_argument(
        '--postprocess-dust-threshold',
        type=int,
        help='Remove skeleton fragments shorter than this (physical distance)',
    )
    parser.add_argument(
        '--postprocess-tick-threshold',
        type=int,
        help='Remove terminal branches shorter than this',
    )
    parser.add_argument(
        '--keep-largest-only',
        action='store_true',
        help='Keep only the largest connected component',
    )
    parser.add_argument(
        '--no-fix-branching',
        action='store_true',
        help='Disable branching correction',
    )
    parser.add_argument(
        '--no-fix-borders',
        action='store_true',
        help='Disable border correction',
    )
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bar',
    )
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue processing other files if one fails',
    )
    parser.add_argument(
        '--log-file',
        type=str,
        help='Path to log file',
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_file=args.log_file)

    # Load config file (fallback defaults)
    config = load_config_file(args.config)

    # CLI arg to config key mapping
    key_mapping = {
        'scale': ('teasar_params', 'scale'),
        'const': ('teasar_params', 'const'),
        'pdrf_scale': ('teasar_params', 'pdrf_scale'),
        'pdrf_exponent': ('teasar_params', 'pdrf_exponent'),
        'keep_largest_only': 'keep_largest_component_only',
    }

    # Apply CLI overrides to config
    config = update_config_from_args(config, args, key_mapping)

    # Handle inverted boolean flags
    if args.no_fix_branching:
        config['fix_branching'] = False
    if args.no_fix_borders:
        config['fix_borders'] = False
    if args.no_progress:
        config['progress'] = False

    # Ensure anisotropy is tuple
    if 'anisotropy' in config and isinstance(config['anisotropy'], list):
        config['anisotropy'] = tuple(config['anisotropy'])

    try:
        process_directory(
            args.input_dir,
            output_dir=args.output_dir,
            continue_on_error=args.continue_on_error,
            **config
        )
        logger.info("Processing completed")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()


# python skeletonize/kimimaro_runner.py preprocess_output/20251127_171506
# python skeletonize/kimimaro_runner.py preprocess_output/20251117_171518/ --dust-threshold 1000
# python skeletonize/kimimaro_runner.py preprocess_output/20251117_171518/ --config pipeline/skeleton_config.yaml
# python skeletonize/kimimaro_runner.py preprocess_output/20251117_171518/03_cleaned/ --parallel 2
