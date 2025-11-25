"""
Preprocessing pipeline orchestrator.
Executes check_tif_format → otsu_threshold → clean_masks in sequence.
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


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def run_pipeline(
    input_path: str,
    config: dict = None,
    output_base: str = None,
    **cli_overrides
) -> str:
    """
    Run complete preprocessing pipeline.

    Args:
        input_path: Input TIF file or directory
        config: Configuration dictionary (from JSON or dict)
        output_base: Base output directory (default: BASE_DIR/preprocess_output)
        **cli_overrides: CLI arguments that override config values

    Returns:
        Path to final output directory (03_cleaned)
    """
    # Default config
    default_config = {
        "clean_masks": {
            "opening_radius": 1,
            "closing_radius": 2,
            "min_size_3d": 64,
            "min_size_2d": 15
        }
    }

    # Merge configs: default < loaded config < CLI overrides
    if config is None:
        config = default_config
    else:
        # Merge with defaults
        for key in default_config:
            if key not in config:
                config[key] = default_config[key]
            else:
                config[key] = {**default_config[key], **config.get(key, {})}

    # Apply CLI overrides to clean_masks config
    if cli_overrides:
        config['clean_masks'] = {**config['clean_masks'], **cli_overrides}

    # Setup output directory
    if output_base is None:
        output_base = str(BASE_DIR / 'preprocess_output')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base, timestamp)

    print("=" * 60)
    print("Preprocessing Pipeline")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Output base: {output_dir}")
    print(f"Timestamp: {timestamp}")
    print(f"Config: {json.dumps(config, indent=2)}")
    print("=" * 60)

    # Step 1: Format checking and conversion
    print("\n[Step 1/3] Format checking and uint8 conversion")
    print("-" * 60)
    format_output = os.path.join(output_dir, '01_format')
    check_tif_format.process_path(input_path, format_output)
    print(f"Step 1 completed: {format_output}")

    # Step 2: Otsu thresholding
    print("\n[Step 2/3] Otsu thresholding")
    print("-" * 60)
    otsu_output = os.path.join(output_dir, '02_otsu')
    otsu_threshold.process_directory(format_output, otsu_output)
    print(f"Step 2 completed: {otsu_output}")

    # Step 3: Mask cleaning
    print("\n[Step 3/3] Mask cleaning")
    print("-" * 60)
    cleaned_output = os.path.join(output_dir, '03_cleaned')
    clean_masks.process_directory(
        otsu_output,
        output_dir=cleaned_output,
        **config['clean_masks']
    )
    print(f"Step 3 completed: {cleaned_output}")

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    print(f"Final output: {cleaned_output}")
    print("=" * 60)

    return cleaned_output


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Run complete preprocessing pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'input_path',
        help='Input TIF file or directory',
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to JSON config file (default: use built-in defaults)',
    )
    parser.add_argument(
        '--output-base',
        type=str,
        help='Base output directory (default: BASE_DIR/preprocess_output)',
    )

    # Clean masks parameters (optional CLI overrides)
    parser.add_argument(
        '--opening-radius',
        type=int,
        help='Opening radius (overrides config)',
    )
    parser.add_argument(
        '--closing-radius',
        type=int,
        help='Closing radius (overrides config)',
    )
    parser.add_argument(
        '--min-size-3d',
        type=int,
        help='Min object size for 3D (overrides config)',
    )
    parser.add_argument(
        '--min-size-2d',
        type=int,
        help='Min object size for 2D (overrides config)',
    )

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        try:
            config = load_config(args.config)
            print(f"Loaded config from: {args.config}")
        except Exception as e:
            print(f"Warning: Failed to load config file: {e}")
            print("Using default configuration")

    # Build CLI overrides dict (only include specified parameters)
    cli_overrides = {}
    if args.opening_radius is not None:
        cli_overrides['opening_radius'] = args.opening_radius
    if args.closing_radius is not None:
        cli_overrides['closing_radius'] = args.closing_radius
    if args.min_size_3d is not None:
        cli_overrides['min_size_3d'] = args.min_size_3d
    if args.min_size_2d is not None:
        cli_overrides['min_size_2d'] = args.min_size_2d

    try:
        run_pipeline(
            args.input_path,
            config=config,
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


# python pipeline/run_preprocess.py DATA/ori_image/skeleton_roi_8bit_z60-630.tif --config pipeline/preprocess_config.json
# python pipeline/run_preprocess.py DATA/ori_image/ --config pipeline/preprocess_config.json
