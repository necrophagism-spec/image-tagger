"""
Configuration Manager Module
Handles persistent storage of application settings.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
import base64


def get_config_dir() -> Path:
    """Get the configuration directory path."""
    if os.name == 'nt':  # Windows
        base = os.environ.get('APPDATA', os.path.expanduser('~'))
    else:  # Linux/Mac
        base = os.path.expanduser('~/.config')
    
    config_dir = Path(base) / 'ImageTagger'
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_config_path() -> Path:
    """Get the config file path."""
    return get_config_dir() / 'config.json'


# Default configuration values
DEFAULT_CONFIG = {
    "last_folder": "",
    "model_type": "gemini",
    "gemini_model": "gemini-2.5-flash",
    "api_key": "",
    "local_model_path": "",
    "local_mmproj_path": "",
    "temperature": 0.4,
    "top_k": 40,
    "top_p": 0.9,
    "min_p": 0.05,
    "repeat_penalty": 1.1,
    "output_format": "captioning",
    "window_geometry": "1400x900",
    "selected_template": "default",
    "system_prompt": "You are an expert image tagger for anime, illustrations, and photographs.",
}


class ConfigManager:
    """Manages application configuration persistence."""
    
    def __init__(self):
        self._config: Dict[str, Any] = DEFAULT_CONFIG.copy()
        self._config_path = get_config_path()
        self.load()
    
    def load(self) -> bool:
        """Load configuration from file."""
        try:
            if self._config_path.exists():
                with open(self._config_path, 'r', encoding='utf-8') as f:
                    saved = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    for key, value in saved.items():
                        if key in self._config:
                            # Decrypt API key if present
                            if key == 'api_key' and value:
                                try:
                                    value = base64.b64decode(value.encode()).decode()
                                except:
                                    pass
                            self._config[key] = value
                return True
        except Exception as e:
            print(f"Error loading config: {e}")
        return False
    
    def save(self) -> bool:
        """Save configuration to file."""
        try:
            # Create a copy for saving
            to_save = self._config.copy()
            
            # Simple obfuscation for API key (not secure, just basic)
            if to_save.get('api_key'):
                to_save['api_key'] = base64.b64encode(
                    to_save['api_key'].encode()
                ).decode()
            
            with open(self._config_path, 'w', encoding='utf-8') as f:
                json.dump(to_save, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
        return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self._config.get(key, default if default is not None else DEFAULT_CONFIG.get(key))
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        self._config[key] = value
    
    def update(self, values: Dict[str, Any]) -> None:
        """Update multiple configuration values."""
        self._config.update(values)
    
    def reset(self) -> None:
        """Reset configuration to defaults."""
        self._config = DEFAULT_CONFIG.copy()
    
    @property
    def all(self) -> Dict[str, Any]:
        """Get all configuration values."""
        return self._config.copy()


# Global instance
_config_manager: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get or create the global ConfigManager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
