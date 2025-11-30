"""
Configuration constants loaded from environment variables.

For local development, create a .env file with the following variables:
- MONGO_PW: MongoDB password
- MONGO_DB: MongoDB database name (default: 'dev')
- MONGO_NAME: MongoDB username (default: 'superteam')
"""

import os
from dotenv import load_dotenv

load_dotenv()

MONGO_PW = os.getenv("MONGO_PW", "")
MONGO_DB = os.getenv("MONGO_DB", "dev")
MONGO_NAME = os.getenv("MONGO_NAME", "superteam")

# Validate required credentials
if not MONGO_PW:
    import warnings
    warnings.warn(
        "MONGO_PW environment variable not set. "
        "Database operations will fail. "
        "Create a .env file with your MongoDB credentials."
    )
