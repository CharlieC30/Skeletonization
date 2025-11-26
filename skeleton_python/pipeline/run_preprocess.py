"""Preprocessing pipeline orchestrator."""
import os
import sys
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

from preprocess import check_tif_format, otsu_threshold, clean_masks
from pipeline.utils import (
    load_config,
    setup_logging,
    log_config,
    format_duration,
    PREPROCESS_SCHEMA,
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.resolve()
DEFAULT_CONFIG = Path(__file__).parent / 'preprocess_config.yaml'


def run_pipeline(
    input_path: str,
    config_path: str = None,
    output_base: str = None,
) -> str:
    """Run preprocessing pipeline.

    Args:
        input_path: Input TIF file or directory.
        config_path: Path to config file.
        output_base: Base output directory.

    Returns:
        Path to output directory (03_cleaned).
    """
    pipeline_start = time.time()
    step_times = {}

    if config_path is None:
        config_path = DEFAULT_CONFIG

    config = load_config(config_path, schema=PREPROCESS_SCHEMA)

    if output_base is None:
        output_base = str(BASE_DIR / 'preprocess_output')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base, timestamp)

    logger.info("=" * 60)
    logger.info("Preprocessing Pipeline")
    logger.info("=" * 60)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Config: {config_path}")
    log_config(config, logger)
    logger.info("=" * 60)

    # Step 1: Format conversion
    logger.info("[Step 1/3] Format conversion")
    t0 = time.time()
    format_output = os.path.join(output_dir, '01_format')
    format_config = config.get('format_conversion', {})
    check_tif_format.process_path(
        input_path,
        format_output,
        normalize_method=format_config.get('normalize_method', 'minmax'),
        percentile_low=format_config.get('percentile_low', 0.0),
        percentile_high=format_config.get('percentile_high', 100.0),
    )
    step_times['Format'] = time.time() - t0

    # Step 2: Otsu thresholding
    logger.info("[Step 2/3] Otsu thresholding")
    t0 = time.time()
    otsu_output = os.path.join(output_dir, '02_otsu')
    otsu_threshold.process_directory(format_output, otsu_output)
    step_times['Otsu'] = time.time() - t0

    # Step 3: Mask cleaning
    logger.info("[Step 3/3] Mask cleaning")
    t0 = time.time()
    cleaned_output = os.path.join(output_dir, '03_cleaned')
    clean_masks.process_directory(
        otsu_output,
        output_dir=cleaned_output,
        **config['clean_masks']
    )
    step_times['Clean'] = time.time() - t0

    # Pipeline summary
    total_time = time.time() - pipeline_start
    logger.info("")
    logger.info("=" * 60)
    logger.info("Pipeline Summary")
    logger.info("=" * 60)
    logger.info(f"Total time: {format_duration(total_time)}")
    for step_name, step_time in step_times.items():
        logger.info(f"  {step_name}: {format_duration(step_time)}")
    logger.info(f"Output: {cleaned_output}")

    return cleaned_output


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Run preprocessing pipeline',
    )
    parser.add_argument(
        'input_path',
        help='Input TIF file or directory',
    )
    parser.add_argument(
        '-c', '--config',
        type=str,
        help='Path to config file (YAML)',
    )
    parser.add_argument(
        '-o', '--output-base',
        type=str,
        help='Base output directory',
    )
    parser.add_argument(
        '--log-file',
        type=str,
        help='Path to log file',
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_file=args.log_file)

    try:
        run_pipeline(
            args.input_path,
            config_path=args.config,
            output_base=args.output_base,
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()


# python pipeline/run_preprocess.py DATA/ori_image/skeleton_roi_8bit_z60-630.tif
# python pipeline/run_preprocess.py DATA/ori_image/ -c pipeline/preprocess_config.yaml
