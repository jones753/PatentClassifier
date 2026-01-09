"""Configuration management for the patent classifier."""

import os
from pathlib import Path
from typing import Any, Dict

import yaml


class Config:
    """Configuration manager that loads settings from config.yaml."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration from YAML file.

        Args:
            config_path: Path to the configuration file
        """
        self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key using dot notation.

        Args:
            key: Configuration key (e.g., 'data.raw_path')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    @property
    def data(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self._config.get('data', {})

    @property
    def preprocessing(self) -> Dict[str, Any]:
        """Get preprocessing configuration."""
        return self._config.get('preprocessing', {})

    @property
    def tfidf(self) -> Dict[str, Any]:
        """Get TF-IDF configuration."""
        return self._config.get('tfidf', {})

    @property
    def embeddings(self) -> Dict[str, Any]:
        """Get embeddings configuration."""
        return self._config.get('embeddings', {})

    @property
    def models(self) -> Dict[str, Any]:
        """Get model paths configuration."""
        return self._config.get('models', {})


# Global configuration instance
_config = None


def get_config(config_path: str = "config.yaml") -> Config:
    """
    Get or create global configuration instance.

    Args:
        config_path: Path to configuration file

    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config


def reload_config(config_path: str = "config.yaml") -> Config:
    """
    Reload configuration from file.

    Args:
        config_path: Path to configuration file

    Returns:
        New Config instance
    """
    global _config
    _config = Config(config_path)
    return _config
