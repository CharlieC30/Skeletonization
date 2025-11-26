"""Complete pipeline orchestrator."""
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent.resolve()
DEFAULT_PREPROCESS_CONFIG = Path(__file__).parent / 'preprocess_config.yaml'
DEFAULT_SKELETON_CONFIG = Path(__file__).parent / 'skeleton_config.yaml'
sys.path.insert(0, str(BASE_DIR))

from preprocess import check_tif_format, otsu_threshold, clean_masks
from skeletonize import kimimaro_runner
from pipeline.config_utils import load_config


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
    # Load configs
    if preprocess_config_path is None:
        preprocess_config_path = DEFAULT_PREPROCESS_CONFIG
    if skeleton_config_path is None:
        skeleton_config_path = DEFAULT_SKELETON_CONFIG

    preprocess_config = load_config(preprocess_config_path)
    skeleton_config = load_config(skeleton_config_path)

    # Apply CLI overrides
    if cli_overrides:
        preprocess_keys = ['opening_radius', 'closing_radius', 'min_size_3d', 'min_size_2d']
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

    print("=" * 60)
    print("Complete Pipeline: Preprocessing + Skeletonization")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Preprocess config: {preprocess_config_path}")
    print(f"Skeleton config: {skeleton_config_path}")
    print("=" * 60)

    # Step 1: Format conversion
    print("\n[Step 1/4] Format conversion")
    format_output = os.path.join(output_dir, '01_format')
    check_tif_format.process_path(input_path, format_output)

    # Step 2: Otsu thresholding
    print("\n[Step 2/4] Otsu thresholding")
    otsu_output = os.path.join(output_dir, '02_otsu')
    otsu_threshold.process_directory(format_output, otsu_output)

    # Step 3: Mask cleaning
    print("\n[Step 3/4] Mask cleaning")
    cleaned_output = os.path.join(output_dir, '03_cleaned')
    clean_masks.process_directory(
        otsu_output,
        output_dir=cleaned_output,
        **preprocess_config['clean_masks']
    )

    # Step 4: Skeletonization
    print("\n[Step 4/4] Skeletonization")
    skeleton_output = os.path.join(output_dir, '04_skeleton')
    kimimaro_runner.process_directory(
        cleaned_output,
        skeleton_output,
        **skeleton_config
    )

    print("\n" + "=" * 60)
    print("Pipeline completed")
    print("=" * 60)
    print(f"Output: {output_dir}")

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

    # CLI overrides
    parser.add_argument('--opening-radius', type=int)
    parser.add_argument('--closing-radius', type=int)
    parser.add_argument('--min-size-3d', type=int)
    parser.add_argument('--min-size-2d', type=int)
    parser.add_argument('--dust-threshold', type=int)
    parser.add_argument('--parallel', type=int)
    parser.add_argument('--keep-largest-only', action='store_true')

    args = parser.parse_args()

    # Build CLI overrides
    cli_overrides = {}
    if args.opening_radius is not None:
        cli_overrides['opening_radius'] = args.opening_radius
    if args.closing_radius is not None:
        cli_overrides['closing_radius'] = args.closing_radius
    if args.min_size_3d is not None:
        cli_overrides['min_size_3d'] = args.min_size_3d
    if args.min_size_2d is not None:
        cli_overrides['min_size_2d'] = args.min_size_2d
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
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
