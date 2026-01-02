"""Configuration management for Reachy Mini Gemini App."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

# Config file location - use XDG_CONFIG_HOME or fallback to ~/.config
CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "reachy-mini-gemini"
CONFIG_FILE = CONFIG_DIR / "settings.json"

# Default settings
DEFAULT_SETTINGS = {
    "api_key": None,
    "robot_audio": False,
    "use_camera": True,
    "holiday_cheer": False,
    "mic_gain": 3.0,
    "chunk_size": 512,
    "send_queue_size": 5,
    "recv_queue_size": 8,
    "camera_fps": 1.0,
    "jpeg_quality": 50,
    "camera_width": 640,
}


def ensure_config_dir() -> None:
    """Ensure the config directory exists."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_settings() -> Dict[str, Any]:
    """Load settings from the config file."""
    settings = DEFAULT_SETTINGS.copy()

    # First check environment variable for API key
    env_api_key = os.environ.get("GOOGLE_API_KEY")
    if env_api_key:
        settings["api_key"] = env_api_key

    # Then load from config file (file settings override defaults but not env)
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                file_settings = json.load(f)
                # Don't override API key from env
                if env_api_key and "api_key" in file_settings:
                    del file_settings["api_key"]
                settings.update(file_settings)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load settings from {CONFIG_FILE}: {e}")

    return settings


def save_settings(settings: Dict[str, Any]) -> bool:
    """Save settings to the config file."""
    try:
        ensure_config_dir()

        # Load existing settings to preserve any we're not updating
        existing = {}
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        # Update with new settings
        existing.update(settings)

        # Write back
        with open(CONFIG_FILE, "w") as f:
            json.dump(existing, f, indent=2)

        return True
    except IOError as e:
        print(f"Error saving settings: {e}")
        return False


def get_api_key() -> Optional[str]:
    """Get the API key from environment or config file."""
    # Environment variable takes precedence
    env_key = os.environ.get("GOOGLE_API_KEY")
    if env_key:
        return env_key

    # Fall back to config file
    settings = load_settings()
    return settings.get("api_key")


def get_settings_for_api() -> Dict[str, Any]:
    """Get settings formatted for the API response (hides actual API key)."""
    settings = load_settings()
    return {
        "api_key_set": bool(settings.get("api_key")),
        "robot_audio": settings.get("robot_audio", False),
        "use_camera": settings.get("use_camera", True),
        "holiday_cheer": settings.get("holiday_cheer", False),
        "mic_gain": settings.get("mic_gain", 3.0),
        "chunk_size": settings.get("chunk_size", 512),
        "send_queue_size": settings.get("send_queue_size", 5),
        "recv_queue_size": settings.get("recv_queue_size", 8),
        "camera_fps": settings.get("camera_fps", 1.0),
        "jpeg_quality": settings.get("jpeg_quality", 50),
        "camera_width": settings.get("camera_width", 640),
    }
