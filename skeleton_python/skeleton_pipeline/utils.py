"""Skeleton pipeline utilities."""
import os
import time
import logging
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

import yaml


# Base directory (skeleton_pipeline/)
BASE_DIR = Path(__file__).parent.resolve()
# Project root (skeleton_python/)
PROJECT_ROOT = BASE_DIR.parent


def load_config(config_path: Path = None) -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to config file. If None, uses default config.yaml.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: Config file not found.
        ValueError: Empty config file.
    """
    if config_path is None:
        config_path = BASE_DIR / "config" / "config_sample.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Empty config file: {config_path}")

    return config


def get_output_dir(config: dict, custom_output: str = None) -> Path:
    """Get output directory based on config or custom path.

    Args:
        config: Configuration dictionary.
        custom_output: Optional custom output path (overrides config).

    Returns:
        Output directory path.
    """
    if custom_output:
        output_dir = Path(custom_output)
    else:
        output_config = config.get('output', {})
        base_dir = output_config.get('base_dir', '../output')

        # Resolve relative to BASE_DIR
        if not os.path.isabs(base_dir):
            output_dir = BASE_DIR / base_dir
        else:
            output_dir = Path(base_dir)

        # Add timestamp subdirectory if configured
        if output_config.get('use_timestamp', True):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = output_dir / timestamp

    return output_dir.resolve()


def setup_logging(level: str = "INFO", log_file: str = None) -> logging.Logger:
    """Setup logging to console and optionally to file.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to log file.

    Returns:
        Configured logger instance.
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    log_level = level_map.get(level.upper(), logging.INFO)

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers,
        force=True,
    )
    return logging.getLogger(__name__)


@contextmanager
def timer(description: str, logger: logging.Logger = None):
    """Context manager for timing code blocks.

    Args:
        description: Description of the operation being timed.
        logger: Optional logger instance. If None, uses print.

    Yields:
        None
    """
    start = time.time()
    yield
    elapsed = time.time() - start
    msg = f"{description}: {format_duration(elapsed)}"
    if logger:
        logger.info(msg)
    else:
        print(msg)


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string like "1m 23s" or "45.2s".
    """
    if seconds >= 60:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    return f"{seconds:.1f}s"


def log_config(config: dict, logger: logging.Logger, title: str = "Configuration"):
    """Log configuration parameters in formatted output.

    Args:
        config: Configuration dictionary to log.
        logger: Logger instance.
        title: Title for the configuration block.
    """
    logger.info(f"{title}:")
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for k, v in value.items():
                logger.info(f"    {k}: {v}")
        else:
            logger.info(f"  {key}: {value}")


def resolve_input_path(input_path: str) -> Path:
    """Resolve input path to absolute path.

    Args:
        input_path: Input file or directory path.

    Returns:
        Resolved absolute path.

    Raises:
        FileNotFoundError: If path doesn't exist.
    """
    path = Path(input_path)

    # If relative, try resolving from current directory first
    if not path.is_absolute():
        # Try from current directory
        if path.exists():
            return path.resolve()
        # Try from project root
        project_path = PROJECT_ROOT / path
        if project_path.exists():
            return project_path.resolve()

    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    return path.resolve()
