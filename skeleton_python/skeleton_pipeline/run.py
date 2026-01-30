#!/usr/bin/env python3
"""Skeleton Pipeline - Main Entry Point.

A complete pipeline for 3D image skeletonization and analysis.

Usage:
    python run.py --input ../DATA/input.tif
    python run.py --input ../DATA/input.tif --output /custom/path
    python run.py --input ../DATA/input.tif --step 3
    python run.py --input ../DATA/input.tif --from 2 --to 4
    python run.py --help
"""
import argparse
import os
import shutil
import sys
import time
import logging
from pathlib import Path

from utils import (
    load_config,
    get_output_dir,
    setup_logging,
    format_duration,
    resolve_input_path,
)

from scripts import (
    step01_format,
    step02_otsu,
    step03_clean,
    step04_skeleton,
    step05_analyze,
)


# Step registry
STEPS = [
    (1, step01_format, "Format conversion"),
    (2, step02_otsu, "Otsu thresholding"),
    (3, step03_clean, "Mask cleaning"),
    (4, step04_skeleton, "Skeletonization"),
    (5, step05_analyze, "Length analysis"),
]


def run_pipeline(
    input_path: str,
    output_dir: str,
    config: dict,
    logger: logging.Logger,
    step_from: int = 1,
    step_to: int = 5,
) -> None:
    """Run the skeleton pipeline.

    Args:
        input_path: Path to input TIF file or directory.
        output_dir: Output directory.
        config: Configuration dictionary.
        logger: Logger instance.
        step_from: Starting step number (1-5).
        step_to: Ending step number (1-5).
    """
    logger.info("=" * 60)
    logger.info("Skeleton Pipeline")
    logger.info("=" * 60)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Steps: {step_from} to {step_to}")
    logger.info("-" * 60)

    # Filter steps to run
    steps_to_run = []
    
    for num, module, desc in STEPS:
        if step_from <= num <= step_to:
            steps_to_run.append((num, module, desc))

    if not steps_to_run:
        logger.error(f"No valid steps in range {step_from}-{step_to}")
        return

    total_start = time.time()
    step_times = []
    current_input = input_path

    for step_num, step_module, step_desc in steps_to_run:
        logger.info("")
        logger.info(f"[Step {step_num}/5] {step_desc}")
        logger.info("-" * 40)

        step_start = time.time()
        try:
            # For first step, use original input; otherwise use previous output
            if step_num == step_from:
                step_input = input_path
            else:
                step_input = current_input

            current_input = step_module.run(
                input_path=step_input,
                output_dir=output_dir,
                config=config,
                logger=logger,
            )
            step_elapsed = time.time() - step_start
            step_times.append((step_num, step_desc, step_elapsed))
            logger.info(f"Step {step_num} completed in {format_duration(step_elapsed)}")

        except Exception as e:
            logger.error(f"Step {step_num} failed: {e}")
            raise

    # Summary
    total_elapsed = time.time() - total_start
    logger.info("")
    logger.info("=" * 60)
    logger.info("Pipeline Summary")
    logger.info("=" * 60)
    for step_num, step_desc, elapsed in step_times:
        logger.info(f"  Step {step_num} ({step_desc}): {format_duration(elapsed)}")
    logger.info("-" * 60)
    logger.info(f"Total time: {format_duration(total_elapsed)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)


def main():
    """CLI entry point for the skeleton pipeline.

    Parses command line arguments and runs the pipeline.
    Run with --help to see all options.
    """
    parser = argparse.ArgumentParser(
        description='Skeleton Pipeline - 3D image skeletonization and analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python run.py --input ../DATA/input.tif

  # Specify output directory
  python run.py --input ../DATA/input.tif --output /path/to/output

  # Run only step 3
  python run.py --input ../DATA/input.tif --step 3

  # Run steps 2 to 4
  python run.py --input ../DATA/input.tif --from 2 --to 4

  # Use custom config
  python run.py --input ../DATA/input.tif --config custom_config.yaml

Steps:
  1. Format conversion   - Normalize and convert TIF to uint8
  2. Otsu thresholding   - Binarize using Otsu's method
  3. Mask cleaning       - Morphological operations
  4. Skeletonization     - Extract skeleton using Kimimaro
  5. Length analysis     - Analyze trunk and branches
        """
    )

    # Required arguments
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input TIF file or directory',
    )

    # Optional arguments
    parser.add_argument(
        '--output', '-o',
        help='Output directory (default: ../output/TIMESTAMP/)',
    )
    parser.add_argument(
        '--config', '-c',
        help='Configuration file (default: config_sample.yaml)',
    )
    parser.add_argument(
        '--step', '-s',
        type=int,
        choices=[1, 2, 3, 4, 5],
        help='Run only this step',
    )
    parser.add_argument(
        '--from',
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=1,
        dest='step_from',
        help='Starting step (default: 1)',
    )
    parser.add_argument(
        '--to',
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=5,
        dest='step_to',
        help='Ending step (default: 5)',
    )
    parser.add_argument(
        '--log-file',
        help='Log file path',
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)',
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(level=args.log_level, log_file=args.log_file)

    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        logger.error(f"Config file not found: {e}")
        sys.exit(1)

    # Resolve input path
    try:
        input_path = str(resolve_input_path(args.input))
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # Get output directory
    output_dir = str(get_output_dir(config, args.output))

    # Create output directory and save config backup
    os.makedirs(output_dir, exist_ok=True)
    config_backup_path = os.path.join(output_dir, "config_used.yaml")
    if args.config:
        shutil.copy(args.config, config_backup_path)
    else:
        # Copy default config
        default_config = Path(__file__).parent / "config" / "config_sample.yaml"
        shutil.copy(default_config, config_backup_path)
    logger.info(f"Config saved to: {config_backup_path}")

    # Determine step range
    if args.step is not None:
        step_from = args.step
        step_to = args.step
    else:
        step_from = args.step_from
        step_to = args.step_to

    if step_from > step_to:
        logger.error(f"Invalid step range: {step_from} to {step_to}")
        sys.exit(1)

    # Run pipeline
    try:
        run_pipeline(
            input_path=input_path,
            output_dir=output_dir,
            config=config,
            logger=logger,
            step_from=step_from,
            step_to=step_to,
        )
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

    logger.info("Done!")


if __name__ == '__main__':
    main()
