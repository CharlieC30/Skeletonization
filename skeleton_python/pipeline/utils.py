"""Pipeline utilities."""
import os
import yaml
import numpy as np
from pathlib import Path


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to config file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: Config file not found.
        yaml.YAMLError: Invalid YAML format.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


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
