"""
Configuration loader for the melanoma detection project.
Loads settings from environment variables with safe defaults.
"""

import os
from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parent.parent


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable with optional default."""
    return os.environ.get(key, default)


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    val = os.environ.get(key, str(default)).lower()
    return val in ('true', '1', 'yes', 'on')


def get_env_int(key: str, default: int) -> int:
    """Get integer environment variable."""
    try:
        return int(os.environ.get(key, default))
    except ValueError:
        return default


class Config:
    """Application configuration."""
    
    # Project paths
    PROJECT_ROOT = get_project_root()
    
    # Server settings
    GRADIO_SERVER_NAME = get_env('GRADIO_SERVER_NAME', '0.0.0.0')
    GRADIO_SERVER_PORT = get_env_int('GRADIO_SERVER_PORT', 7860)
    GRADIO_SHARE = get_env_bool('GRADIO_SHARE', False)
    
    # Model paths (relative to project root)
    WEIGHTS_PATH = PROJECT_ROOT / get_env('WEIGHTS_PATH', 'models/checkpoints/melanoma_resnet50_nb.pth')
    LABEL_MAP_PATH = PROJECT_ROOT / get_env('LABEL_MAP_PATH', 'models/label_maps/label_map_nb.json')
    TEMPERATURE_JSON_PATH = PROJECT_ROOT / get_env('TEMPERATURE_JSON_PATH', 'models/checkpoints/temperature.json')
    OPERATING_JSON_PATH = PROJECT_ROOT / get_env('OPERATING_JSON_PATH', 'models/checkpoints/operating_points.json')
    
    # Authentication (optional)
    GRADIO_USERNAME = get_env('GRADIO_USERNAME')
    GRADIO_PASSWORD = get_env('GRADIO_PASSWORD')
    
    # Logging
    LOG_LEVEL = get_env('LOG_LEVEL', 'INFO')
    
    @classmethod
    def validate(cls) -> None:
        """Validate critical paths exist."""
        if not cls.WEIGHTS_PATH.exists():
            raise FileNotFoundError(f"Model weights not found at {cls.WEIGHTS_PATH}")
        if not cls.LABEL_MAP_PATH.exists():
            raise FileNotFoundError(f"Label map not found at {cls.LABEL_MAP_PATH}")
    
    @classmethod
    def has_auth(cls) -> bool:
        """Check if authentication is configured."""
        return bool(cls.GRADIO_USERNAME and cls.GRADIO_PASSWORD)
    
    @classmethod
    def get_auth(cls) -> Optional[tuple]:
        """Get authentication tuple for Gradio."""
        if cls.has_auth():
            return (cls.GRADIO_USERNAME, cls.GRADIO_PASSWORD)
        return None


# Try to load .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    env_path = get_project_root() / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, will use system env vars only
