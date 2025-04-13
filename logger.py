# src/utils/logger.py
"""
Logger Module for Poker AI

This module provides centralized logging configuration and utilities.
"""

import logging
import os
import time
from typing import Optional

# Default log format
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def setup_logger(
    name: str, 
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_str: str = DEFAULT_FORMAT
) -> logging.Logger:
    """
    Set up a logger with specified configuration.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Path to log file (None for no file logging)
        format_str: Log message format
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(format_str)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if specified
    if log_file:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_timestamp_str() -> str:
    """
    Get a formatted timestamp string.
    
    Returns:
        Formatted timestamp
    """
    return time.strftime("%Y%m%d_%H%M%S")