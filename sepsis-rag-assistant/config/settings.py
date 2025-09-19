"""Configuration settings for sepsis-rag-assistant."""

import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
GUIDELINES_PATH = os.getenv("GUIDELINES_PATH", os.path.join(BASE_DIR, "data", "guidelines")) + os.sep
PROCESSED_PATH = os.getenv("PROCESSED_PATH", os.path.join(BASE_DIR, "data", "processed"))

# Optional: OpenAI configuration via .env or environment
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
