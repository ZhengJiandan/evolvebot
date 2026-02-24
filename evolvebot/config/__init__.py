"""Configuration module for evolvebot."""

from evolvebot.config.loader import load_config, get_config_path
from evolvebot.config.schema import Config

__all__ = ["Config", "load_config", "get_config_path"]
