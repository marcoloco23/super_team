"""
Superteam - NBA Basketball Analytics Framework

A machine learning framework for predicting team performance and identifying
competitive NBA team compositions using XGBoost regression models.
"""

__version__ = "1.0.0"

from .constants import MONGO_PW, MONGO_DB, MONGO_NAME
from .logger import setup_logger, logger

__all__ = [
    "MONGO_PW",
    "MONGO_DB",
    "MONGO_NAME",
    "setup_logger",
    "logger",
]
