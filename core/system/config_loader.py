# config_loader.py (S:/Snowball/core)

import json
import logging
import os
from typing import Optional, Dict, Any

class ConfigLoader:
    _cache = {}
    DEFAULT_CONFIG_DIR = os.environ.get("CONFIG_DIR", "S:/Snowball/config")

    def __init__(self):
        self.logger = SnowballLogger()

    def load_config(self, file_name: str, force_reload: bool = False, config_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Loads the JSON configuration file and returns the data as a dictionary.
        Caches the file data to prevent reloading it multiple times, unless forced.
        """
        config_dir = config_dir or ConfigLoader.DEFAULT_CONFIG_DIR
        file_path = os.path.join(config_dir, file_name)

        if not os.path.exists(file_path):
            self.logger.log_error(f"Configuration file {file_name} not found in {config_dir}")
            raise FileNotFoundError(f"Configuration file {file_name} not found in {config_dir}")

        if file_name in ConfigLoader._cache and not force_reload:
            self.logger.logger.info(f"Loading configuration from cache: {file_name}")
            return ConfigLoader._cache[file_name]

        try:
            with open(file_path, 'r') as f:
                config_data = json.load(f)
                ConfigLoader._cache[file_name] = config_data  # Cache the loaded config
                self.logger.logger.info(f"Successfully loaded configuration: {file_name}")
                return config_data
        except json.JSONDecodeError as e:
            self.logger.log_error(f"Error decoding JSON from file {file_name}: {e}")
            raise
        except IOError as e:
            self.logger.log_error(f"IOError while opening file {file_name}: {e}")
            raise

    def get_default_settings(self, section_name: str) -> Dict[str, Any]:
        """
        Return the default settings for a given section.
        """
        default_settings_map = {
            "ai_settings": {
                "enabled": True,
                "learning_rate": 0.01,
                "dynamic_learning_rate": True,
                "learning_rate_decay": 0.001,
                "daily_training_sessions": 5,
                "training_mode": "Supervised",
                "epoch_count": 50,
                "batch_size": 32,
                "training_data_path": "S:/Snowball/data/training_dataset",
                "optimizer": "Adam",
                "model_type": "Neural Network",
                "performance_tracking": True,
                "save_training_logs": True,
                "evaluation_frequency": "Daily",
                "personality_mode": "Friendly",
                "allow_casual_conversation": True,
                "response_speed": "Instant",
                "knowledge_expansion": True,
                "memory_retention_limit": 1000,
                "auto_learning": True,
                "safe_mode": True,
                "max_consecutive_failures": 3,
                "privacy_protection": True
            },
            "interface_settings": {
                "theme": "Light",
                "language": "en-US",
                "volume": 70,
                "speech_rate": 1.0
            },
            "game_preferences": {
                "difficulty": "Normal",
                "graphics_quality": "High",
                "sound_effects": True,
                "music_volume": 50
            }
        }

        return default_settings_map.get(section_name.lower(), {})

    def save_config(self, file_name: str, config_data: dict, config_dir: Optional[str] = None):
        """
        Saves the configuration data to the specified JSON file.
        """
        config_dir = config_dir or ConfigLoader.DEFAULT_CONFIG_DIR
        file_path = os.path.join(config_dir, file_name)

        try:
            with open(file_path, 'w') as f:
                json.dump(config_data, f, indent=4)
                ConfigLoader._cache[file_name] = config_data  # Update the cache
                self.logger.logger.info(f"Configuration saved to {file_path}")
        except IOError as e:
            self.logger.log_error(f"Failed to write configuration to {file_path}: {e}")
            raise

    def clear_cache(self):
        """
        Clears the configuration cache for all files, forcing a reload on the next load.
        """
        ConfigLoader._cache.clear()
        self.logger.logger.info("Configuration cache cleared.")

    def remove_from_cache(self, file_name: str):
        """
        Removes a specific configuration file from the cache.
        """
        if file_name in ConfigLoader._cache:
            del ConfigLoader._cache[file_name]
            self.logger.logger.info(f"Configuration for {file_name} removed from cache.")
        else:
            self.logger.logger.info(f"No cached configuration found for {file_name}.")

    def cache_status(self) -> list:
        """
        Returns the current status of the cache (which files are cached).
        """
        self.logger.logger.info(f"Current cache status: {list(ConfigLoader._cache.keys())}")
        return list(ConfigLoader._cache.keys())
    
    def save_config(self, file_name: str, config_data: dict, config_dir: Optional[str] = None):
        """
        Saves the configuration data to the specified JSON file.

        :param file_name: The name of the JSON file (with extension).
        :param config_data: The dictionary data to save as JSON.
        :param config_dir: Optionally specify a custom config directory, otherwise uses default.
        :raises IOError: If the file cannot be written.
        """
        config_dir = config_dir or ConfigLoader.DEFAULT_CONFIG_DIR
        file_path = os.path.join(config_dir, file_name)

        try:
            with open(file_path, 'w') as f:
                json.dump(config_data, f, indent=4)
                ConfigLoader._cache[file_name] = config_data  # Update the cache
                self.logger.logger.info(f"Configuration saved to {file_path}")
        except IOError as e:
            self.logger.log_error(f"Failed to write configuration to {file_path}: {e}")
            raise

    def refresh_config(self, file_name: str, config_dir: Optional[str] = None) -> dict:
        """
        Forces a cache refresh by reloading the configuration file from disk.

        :param file_name: The name of the JSON file (with extension).
        :param config_dir: Optionally specify a custom config directory, otherwise uses default.
        :return: Dictionary of the reloaded configuration.
        :raises FileNotFoundError: If the configuration file does not exist.
        :raises json.JSONDecodeError: If the configuration file contains invalid JSON.
        """
        self.logger.logger.info(f"Refreshing configuration for {file_name}")
        return self.load_config(file_name, force_reload=True, config_dir=config_dir)

    def validate_config(self, config_data: dict, required_keys: list) -> bool:
        """
        Validates that the configuration data contains all required keys.

        :param config_data: The configuration data to validate.
        :param required_keys: A list of keys that must be present in the configuration.
        :return: True if all required keys are present, False otherwise.
        """
        missing_keys = [key for key in required_keys if key not in config_data]
        if missing_keys:
            self.logger.log_warning(f"Missing keys in configuration: {missing_keys}")
            return False
        return True

# Example usage:
if __name__ == "__main__":
    config_loader = ConfigLoader()
    try:
        # Load the config file, forcing a reload from disk
        config = config_loader.load_config("user_customizations.json", force_reload=True)
        print(config)

        # Check the current cache status
        print("Cached files:", config_loader.cache_status())

        # Load the config file without forcing reload (should come from cache)
        config_cached = config_loader.load_config("user_customizations.json")
        print(config_cached)

        # Clear the entire cache
        config_loader.clear_cache()

        # Check the cache status after clearing
        print("Cached files after clearing:", config_loader.cache_status())

        # Save new configuration data to a file
        new_config_data = {"new_setting": "example_value"}
        config_loader.save_config("new_config.json", new_config_data)

        # Force refresh and load new configuration
        new_config = config_loader.refresh_config("new_config.json")
        print(new_config)

    except Exception as e:
        config_loader.logger.log_error(f"Error: {e}")