"""Complete pipeline orchestrator."""
import os
import sys
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

from preprocess import check_tif_format, otsu_threshold, clean_masks
from skeletonize import kimimaro_runner
from pipeline.utils import (
    load_config,
    setup_logging,
    log_config,
    format_duration,
    PREPROCESS_SCHEMA,
    SKELETON_SCHEMA,
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.resolve()
DEFAULT_PREPROCESS_CONFIG = Path(__file__).parent / 'preprocess_config.yaml'
DEFAULT_SKELETON_CONFIG = Path(__file__).parent / 'skeleton_config.yaml'


def run_pipeline(
    input_path: str,
    preprocess_config_path: str = None,
    skeleton_config_path: str = None,
    output_base: str = None,
    **cli_overrides
) -> str:
    """Run complete pipeline: preprocessing + skeletonization.

    Args:
        input_path: Input TIF file or directory.
        preprocess_config_path: Path to preprocessing config.
        skeleton_config_path: Path to skeletonization config.
        output_base: Base output directory.
        **cli_overrides: CLI arguments that override config values.

    Returns:
        Path to output directory (04_skeleton).
    """
    pipeline_start = time.time()
    step_times = {}

    # Load configs
    if preprocess_config_path is None:
        preprocess_config_path = DEFAULT_PREPROCESS_CONFIG
    if skeleton_config_path is None:
        skeleton_config_path = DEFAULT_SKELETON_CONFIG

    preprocess_config = load_config(preprocess_config_path, schema=PREPROCESS_SCHEMA)
    skeleton_config = load_config(skeleton_config_path, schema=SKELETON_SCHEMA)

    # Apply CLI overrides
    if cli_overrides:
        preprocess_keys = ['opening_radius', 'closing_radius', 'min_size']
        for key, value in cli_overrides.items():
            if key in preprocess_keys:
                preprocess_config['clean_masks'][key] = value
            else:
                skeleton_config[key] = value

    # Setup output directory
    if output_base is None:
        output_base = str(BASE_DIR / 'output')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base, timestamp)

    logger.info("=" * 60)
    logger.info("Complete Pipeline: Preprocessing + Skeletonization")
    logger.info("=" * 60)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Preprocess config: {preprocess_config_path}")
    logger.info(f"Skeleton config: {skeleton_config_path}")
    log_config(preprocess_config, logger, "Preprocess Configuration")
    log_config(skeleton_config, logger, "Skeleton Configuration")
    logger.info("=" * 60)

    # Step 1: Format conversion
    logger.info("[Step 1/4] Format conversion")
    t0 = time.time()
    format_output = os.path.join(output_dir, '01_format')
    format_config = preprocess_config.get('format_conversion', {})
    check_tif_format.process_path(
        input_path,
        format_output,
        normalize_method=format_config.get('normalize_method', 'minmax'),
        percentile_low=format_config.get('percentile_low', 0.0),
        percentile_high=format_config.get('percentile_high', 100.0),
    )
    step_times['Format'] = time.time() - t0

    # Step 2: Otsu thresholding
    logger.info("[Step 2/4] Otsu thresholding")
    t0 = time.time()
    otsu_output = os.path.join(output_dir, '02_otsu')
    otsu_threshold.process_directory(format_output, otsu_output)
    step_times['Otsu'] = time.time() - t0

    # Step 3: Mask cleaning
    logger.info("[Step 3/4] Mask cleaning")
    t0 = time.time()
    cleaned_output = os.path.join(output_dir, '03_cleaned')
    clean_masks.process_directory(
        otsu_output,
        output_dir=cleaned_output,
        **preprocess_config['clean_masks']
    )
    step_times['Clean'] = time.time() - t0

    # Step 4: Skeletonization
    logger.info("[Step 4/4] Skeletonization")
    t0 = time.time()
    skeleton_output = os.path.join(output_dir, '04_skeleton')
    kimimaro_runner.process_directory(
        cleaned_output,
        skeleton_output,
        **skeleton_config
    )
    step_times['Skeleton'] = time.time() - t0

    # Pipeline summary
    total_time = time.time() - pipeline_start
    logger.info("")
    logger.info("=" * 60)
    logger.info("Pipeline Summary")
    logger.info("=" * 60)
    logger.info(f"Total time: {format_duration(total_time)}")
    for step_name, step_time in step_times.items():
        logger.info(f"  {step_name}: {format_duration(step_time)}")
    logger.info(f"Output: {output_dir}")

    return skeleton_output


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Run complete pipeline: preprocessing + skeletonization',
    )
    parser.add_argument(
        'input_path',
        help='Input TIF file or directory',
    )
    parser.add_argument(
        '--preprocess-config',
        type=str,
        help='Path to preprocessing config (YAML)',
    )
    parser.add_argument(
        '--skeleton-config',
        type=str,
        help='Path to skeletonization config (YAML)',
    )
    parser.add_argument(
        '--output-base',
        type=str,
        help='Base output directory',
    )
    parser.add_argument(
        '--log-file',
        type=str,
        help='Path to log file',
    )

    # CLI overrides
    parser.add_argument('--opening-radius', type=int, help='Morphological opening radius')
    parser.add_argument('--closing-radius', type=int, help='Morphological closing radius')
    parser.add_argument('--min-size', type=int, help='Min object size in voxels')
    parser.add_argument('--dust-threshold', type=int, help='Skeleton dust threshold')
    parser.add_argument('--parallel', type=int, help='Number of parallel processes')
    parser.add_argument('--keep-largest-only', action='store_true', help='Keep only largest component')

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_file=args.log_file)

    # Build CLI overrides
    cli_overrides = {}
    if args.opening_radius is not None:
        cli_overrides['opening_radius'] = args.opening_radius
    if args.closing_radius is not None:
        cli_overrides['closing_radius'] = args.closing_radius
    if args.min_size is not None:
        cli_overrides['min_size'] = args.min_size
    if args.dust_threshold is not None:
        cli_overrides['dust_threshold'] = args.dust_threshold
    if args.parallel is not None:
        cli_overrides['parallel'] = args.parallel
    if args.keep_largest_only:
        cli_overrides['keep_largest_component_only'] = True

    try:
        run_pipeline(
            args.input_path,
            preprocess_config_path=args.preprocess_config,
            skeleton_config_path=args.skeleton_config,
            output_base=args.output_base,
            **cli_overrides
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
