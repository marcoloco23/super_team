"""
Logging configuration for the Superteam application.
"""

import logging
import os
from datetime import datetime


def setup_logger(name: str = "superteam", level: str = None) -> logging.Logger:
    """
    Set up and return a configured logger.

    Args:
        name: Logger name (default: "superteam")
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
               Falls back to LOG_LEVEL env var, then INFO

    Returns:
        Configured logger instance
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Format with timestamp, level, and message
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler (if LOG_FILE env var is set)
    log_file = os.getenv("LOG_FILE")
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Default logger instance
logger = setup_logger()
