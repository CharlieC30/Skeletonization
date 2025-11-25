"""
Complete pipeline orchestrator.
Executes check_tif_format → otsu_threshold → clean_masks → kimimaro skeletonization.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
BASE_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(BASE_DIR))

from preprocess import check_tif_format, otsu_threshold, clean_masks
from skeletonize import kimimaro_runner


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def run_pipeline(
    input_path: str,
    preprocess_config: dict = None,
    skeleton_config: dict = None,
    output_base: str = None,
    **cli_overrides
) -> str:
    """
    Run complete pipeline: preprocessing + skeletonization.

    Args:
        input_path: Input TIF file or directory
        preprocess_config: Preprocessing configuration dictionary
        skeleton_config: Skeletonization configuration dictionary
        output_base: Base output directory (default: BASE_DIR/preprocess_output)
        **cli_overrides: CLI arguments that override config values

    Returns:
        Path to final output directory (04_skeleton)
    """
    # Default preprocessing config
    default_preprocess_config = {
        "clean_masks": {
            "opening_radius": 1,
            "closing_radius": 2,
            "min_size_3d": 64,
            "min_size_2d": 15
        }
    }

    # Default skeletonization config
    default_skeleton_config = {
        "teasar_params": {
            "scale": 0.3,
            "const": 50,
            "pdrf_scale": 100000,
            "pdrf_exponent": 4,
            "soma_detection_threshold": 999999,
            "soma_acceptance_threshold": 999999,
            "soma_invalidation_scale": 2,
            "soma_invalidation_const": 300,
            "max_paths": None
        },
        "dust_threshold": 500,
        "anisotropy": [1, 1, 1],
        "fix_branching": True,
        "fix_borders": True,
        "progress": True,
        "parallel": 1,
        "postprocess_dust_threshold": 1000,
        "postprocess_tick_threshold": 0,
        "keep_largest_component_only": True
    }

    # Merge configs with defaults
    if preprocess_config is None:
        preprocess_config = default_preprocess_config
    else:
        for key in default_preprocess_config:
            if key not in preprocess_config:
                preprocess_config[key] = default_preprocess_config[key]
            else:
                preprocess_config[key] = {**default_preprocess_config[key], **preprocess_config.get(key, {})}

    if skeleton_config is None:
        skeleton_config = default_skeleton_config
    else:
        for key in default_skeleton_config:
            if key not in skeleton_config:
                skeleton_config[key] = default_skeleton_config[key]
            elif key == 'teasar_params' and isinstance(skeleton_config[key], dict):
                skeleton_config[key] = {**default_skeleton_config[key], **skeleton_config[key]}

    # Apply CLI overrides to preprocessing config
    if cli_overrides:
        preprocess_overrides = {}
        skeleton_overrides = {}

        # Separate preprocessing and skeletonization overrides
        preprocess_keys = ['opening_radius', 'closing_radius', 'min_size_3d', 'min_size_2d']
        for key, value in cli_overrides.items():
            if key in preprocess_keys:
                preprocess_overrides[key] = value
            else:
                skeleton_overrides[key] = value

        if preprocess_overrides:
            preprocess_config['clean_masks'] = {**preprocess_config['clean_masks'], **preprocess_overrides}
        if skeleton_overrides:
            skeleton_config = {**skeleton_config, **skeleton_overrides}

    # Setup output directory
    if output_base is None:
        output_base = str(BASE_DIR / 'preprocess_output')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base, timestamp)

    print("=" * 60)
    print("Complete Pipeline: Preprocessing + Skeletonization")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Output base: {output_dir}")
    print(f"Timestamp: {timestamp}")
    print("=" * 60)

    # Step 1: Format checking and conversion
    print("\n[Step 1/4] Format checking and uint8 conversion")
    print("-" * 60)
    format_output = os.path.join(output_dir, '01_format')
    check_tif_format.process_path(input_path, format_output)
    print(f"Step 1 completed: {format_output}")

    # Step 2: Otsu thresholding
    print("\n[Step 2/4] Otsu thresholding")
    print("-" * 60)
    otsu_output = os.path.join(output_dir, '02_otsu')
    otsu_threshold.process_directory(format_output, otsu_output)
    print(f"Step 2 completed: {otsu_output}")

    # Step 3: Mask cleaning
    print("\n[Step 3/4] Mask cleaning")
    print("-" * 60)
    cleaned_output = os.path.join(output_dir, '03_cleaned')
    clean_masks.process_directory(
        otsu_output,
        output_dir=cleaned_output,
        **preprocess_config['clean_masks']
    )
    print(f"Step 3 completed: {cleaned_output}")

    # Step 4: Skeletonization
    print("\n[Step 4/4] Skeletonization")
    print("-" * 60)
    skeleton_base = BASE_DIR / 'skeletonize_output' / timestamp
    skeleton_output = str(skeleton_base / '04_skeleton')
    kimimaro_runner.process_directory(
        cleaned_output,
        skeleton_output,
        **skeleton_config
    )
    print(f"Step 4 completed: {skeleton_output}")

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    print(f"Preprocessing output: {output_dir}")
    print(f"  - 01_format: format conversion")
    print(f"  - 02_otsu: binary thresholding")
    print(f"  - 03_cleaned: mask cleaning")
    print(f"Skeletonization output: {str(skeleton_base)}")
    print(f"  - 04_skeleton: skeleton files (SWC + TIF)")
    print("=" * 60)

    return skeleton_output


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Run complete pipeline: preprocessing + skeletonization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'input_path',
        help='Input TIF file or directory',
    )
    parser.add_argument(
        '--preprocess-config',
        type=str,
        help='Path to preprocessing JSON config file',
    )
    parser.add_argument(
        '--skeleton-config',
        type=str,
        help='Path to skeletonization JSON config file',
    )
    parser.add_argument(
        '--output-base',
        type=str,
        help='Base output directory (default: BASE_DIR/preprocess_output)',
    )

    # Preprocessing parameters (optional CLI overrides)
    parser.add_argument(
        '--opening-radius',
        type=int,
        help='Opening radius for mask cleaning',
    )
    parser.add_argument(
        '--closing-radius',
        type=int,
        help='Closing radius for mask cleaning',
    )
    parser.add_argument(
        '--min-size-3d',
        type=int,
        help='Min object size for 3D mask cleaning',
    )
    parser.add_argument(
        '--min-size-2d',
        type=int,
        help='Min object size for 2D mask cleaning',
    )

    # Skeletonization parameters (optional CLI overrides)
    parser.add_argument(
        '--dust-threshold',
        type=int,
        help='Remove components smaller than this before skeletonization',
    )
    parser.add_argument(
        '--parallel',
        type=int,
        help='Number of parallel processes (1 = single-threaded)',
    )
    parser.add_argument(
        '--keep-largest-only',
        action='store_true',
        help='Keep only the largest connected component',
    )

    args = parser.parse_args()

    # Load preprocessing config if provided
    preprocess_config = None
    if args.preprocess_config:
        try:
            preprocess_config = load_config(args.preprocess_config)
            print(f"Loaded preprocessing config from: {args.preprocess_config}")
        except Exception as e:
            print(f"Warning: Failed to load preprocessing config: {e}")

    # Load skeletonization config if provided
    skeleton_config = None
    if args.skeleton_config:
        try:
            skeleton_config = load_config(args.skeleton_config)
            print(f"Loaded skeletonization config from: {args.skeleton_config}")
        except Exception as e:
            print(f"Warning: Failed to load skeletonization config: {e}")

    # Build CLI overrides dict
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
            preprocess_config=preprocess_config,
            skeleton_config=skeleton_config,
            output_base=args.output_base,
            **cli_overrides
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
