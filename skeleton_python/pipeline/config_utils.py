"""Configuration loading utilities."""
import yaml
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
