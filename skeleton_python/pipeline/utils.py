"""Pipeline utilities."""
import os
import re
import logging
from datetime import datetime
from pathlib import Path

import yaml
import numpy as np


# Schema definitions for config validation
PREPROCESS_SCHEMA = {
    'clean_masks': {
        'opening_radius': int,
        'closing_radius': int,
        'min_size_3d': int,
        'min_size_2d': int,
    },
}

SKELETON_SCHEMA = {
    'teasar_params': {
        'scale': (int, float),
        'const': int,
        'pdrf_scale': (int, float),
        'pdrf_exponent': (int, float),
    },
    'dust_threshold': int,
    'anisotropy': list,
    'parallel': int,
}


def load_config(config_path: Path, schema: dict = None) -> dict:
    """Load YAML configuration file with optional schema validation.

    Args:
        config_path: Path to config file.
        schema: Optional dict defining expected keys and types.
                Format: {'key': type, 'nested': {'subkey': type}}

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: Config file not found.
        ValueError: Empty config or schema validation failed.
        yaml.YAMLError: Invalid YAML format.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Empty config file: {config_path}")

    if schema is not None:
        validate_schema(config, schema)

    return config


def validate_schema(config: dict, schema: dict, path: str = "") -> None:
    """Validate config against schema.

    Args:
        config: Configuration dict to validate.
        schema: Schema dict with expected keys and types.
        path: Current path for error messages.

    Raises:
        ValueError: If validation fails.
    """
    for key, expected in schema.items():
        full_key = f"{path}.{key}" if path else key

        if key not in config:
            raise ValueError(f"Missing required key: {full_key}")

        if isinstance(expected, dict):
            if not isinstance(config[key], dict):
                raise ValueError(f"Expected dict for {full_key}, got {type(config[key]).__name__}")
            validate_schema(config[key], expected, full_key)
        elif isinstance(expected, tuple):
            # Multiple allowed types
            if not isinstance(config[key], expected) and config[key] is not None:
                type_names = "/".join(t.__name__ for t in expected)
                raise ValueError(f"Invalid type for {full_key}: expected {type_names}, got {type(config[key]).__name__}")
        elif expected is not None and not isinstance(config[key], expected):
            if config[key] is not None:
                raise ValueError(f"Invalid type for {full_key}: expected {expected.__name__}, got {type(config[key]).__name__}")


def ensure_3d(image: np.ndarray) -> np.ndarray:
    """Ensure image is 3D (Z, Y, X).

    Args:
        image: Input 2D or 3D array.

    Returns:
        3D array with shape (Z, Y, X).

    Raises:
        ValueError: If image is not 2D or 3D.
    """
    if image.ndim == 2:
        return image[np.newaxis, ...]
    elif image.ndim == 3:
        return image
    raise ValueError(f"Expected 2D or 3D, got {image.ndim}D")


def auto_detect_subdir(input_dir: str, subdir_name: str) -> str:
    """Auto-detect subdirectory if input_dir has no TIF files.

    Args:
        input_dir: Input directory path.
        subdir_name: Expected subdirectory name (e.g., '01_format').

    Returns:
        Subdirectory path if detected, otherwise original input_dir.
    """
    potential = os.path.join(input_dir, subdir_name)
    if os.path.isdir(potential):
        tifs = [f for f in os.listdir(input_dir)
                if f.lower().endswith(('.tif', '.tiff'))
                and os.path.isfile(os.path.join(input_dir, f))]
        if not tifs:
            return potential
    return input_dir


def extract_timestamp_from_path(input_path: str) -> str:
    """Extract timestamp from path if exists, otherwise create new one.

    Looks for YYYYMMDD_HHMMSS pattern in path components.

    Args:
        input_path: Input file or directory path.

    Returns:
        Extracted or newly generated timestamp string.
    """
    path = Path(input_path)
    for part in path.parts:
        if re.match(r'\d{8}_\d{6}', part):
            return part
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_output_dir(input_path: str, output_base: str, stage: str) -> str:
    """Get output directory based on input path's experiment timestamp.

    Args:
        input_path: Input file or directory path.
        output_base: Base output directory (e.g., 'output').
        stage: Stage subdirectory (e.g., '04_skeleton').

    Returns:
        Output directory path preserving experiment timestamp.
    """
    timestamp = extract_timestamp_from_path(input_path)
    return str(Path(output_base) / timestamp / stage)


def update_config_from_args(config: dict, args, key_mapping: dict = None) -> dict:
    """Update config dict with non-None CLI arguments.

    Args:
        config: Base configuration dict.
        args: argparse.Namespace object.
        key_mapping: Optional mapping from arg names to config keys.
                     Supports nested keys as tuples: ('parent', 'child').
                     Supports value inversion for boolean flags.

    Returns:
        Updated config dict.
    """
    if key_mapping is None:
        key_mapping = {}

    for arg_name, value in vars(args).items():
        if value is None:
            continue

        if arg_name in key_mapping:
            target = key_mapping[arg_name]
            if isinstance(target, tuple):
                # Nested key like ('teasar_params', 'scale')
                parent, child = target
                if parent not in config:
                    config[parent] = {}
                config[parent][child] = value
            else:
                config[target] = value
        elif arg_name in config:
            config[arg_name] = value

    return config


def setup_logging(log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """Setup logging to console and optionally to file.

    Args:
        log_file: Optional path to log file.
        level: Logging level (default: INFO).

    Returns:
        Configured logger instance.
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers,
        force=True,
    )
    return logging.getLogger(__name__)
