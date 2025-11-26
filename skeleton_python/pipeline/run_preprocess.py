"""Preprocessing pipeline orchestrator."""
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent.resolve()
DEFAULT_CONFIG = Path(__file__).parent / 'preprocess_config.yaml'
sys.path.insert(0, str(BASE_DIR))

from preprocess import check_tif_format, otsu_threshold, clean_masks
from pipeline.config_utils import load_config


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
    if config_path is None:
        config_path = DEFAULT_CONFIG

    config = load_config(config_path)

    if output_base is None:
        output_base = str(BASE_DIR / 'preprocess_output')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base, timestamp)

    print("=" * 60)
    print("Preprocessing Pipeline")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Config: {config_path}")
    print("=" * 60)

    # Step 1: Format conversion
    print("\n[Step 1/3] Format conversion")
    format_output = os.path.join(output_dir, '01_format')
    check_tif_format.process_path(input_path, format_output)

    # Step 2: Otsu thresholding
    print("\n[Step 2/3] Otsu thresholding")
    otsu_output = os.path.join(output_dir, '02_otsu')
    otsu_threshold.process_directory(format_output, otsu_output)

    # Step 3: Mask cleaning
    print("\n[Step 3/3] Mask cleaning")
    cleaned_output = os.path.join(output_dir, '03_cleaned')
    clean_masks.process_directory(
        otsu_output,
        output_dir=cleaned_output,
        **config['clean_masks']
    )

    print("\n" + "=" * 60)
    print("Pipeline completed")
    print("=" * 60)
    print(f"Output: {cleaned_output}")

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

    args = parser.parse_args()

    try:
        run_pipeline(
            args.input_path,
            config_path=args.config,
            output_base=args.output_base,
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()


# python pipeline/run_preprocess.py DATA/ori_image/skeleton_roi_8bit_z60-630.tif
# python pipeline/run_preprocess.py DATA/ori_image/ -c pipeline/preprocess_config.yaml
