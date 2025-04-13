# src/utils/config.py
"""
Configuration Utilities for Poker AI

This module provides utilities for handling configuration settings.
"""

import os
import yaml
from typing import Dict, Any, Optional

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the config file
        
    Returns:
        Configuration dictionary
    """
    # Default configuration
    default_config = {
        "capture": {
            "interval": 0.5,
            "detect_changes": True
        },
        "strategy": {
            "profile": "balanced",
            "risk_tolerance": 1.0,
            "use_position": True,
            "bluff_factor": 0.1
        },
        "llm": {
            "api_base": "http://localhost:11434/api",
            "model_name": "lava",
            "temperature": 0.7,
            "max_tokens": 1024
        },
        "ui": {
            "human_like": True,
            "action_delay": 0.5
        },
        "execution": {
            "loop_delay": 1.0,
            "max_hands": 0,
            "autonomous_mode": False
        },
        "logging": {
            "level": "INFO",
            "log_to_file": True
        }
    }
    
    # Try to load user config
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                
            # Merge with default config
            if user_config:
                deep_update(default_config, user_config)
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Using default configuration")
    
    return default_config

def deep_update(base_dict: Dict, update_dict: Dict) -> None:
    """
    Deeply update a dictionary with values from another dictionary.
    
    Args:
        base_dict: Dictionary to update
        update_dict: Dictionary with new values
    """
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            deep_update(base_dict[key], value)
        else:
            base_dict[key] = value

def save_config(config: Dict, config_path: str) -> bool:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to the config file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        config_dir = os.path.dirname(config_path)
        if config_dir and not os.path.exists(config_dir):
            os.makedirs(config_dir)
            
        # Save config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False