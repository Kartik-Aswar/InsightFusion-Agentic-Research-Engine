"""
Centralized LLM configuration for all agents.
All agents import `llm` from this module for consistent behavior.
"""

import os
import sys

from crewai import LLM
from dotenv import load_dotenv

load_dotenv()


def _validate_api_keys() -> None:
    """Validate required API keys are present before running."""

    required_keys = {
        "GEMINI_API_KEY": "Gemini (LLM)",
        "SERPER_API_KEY": "Serper (Web Search)",
    }

    missing = []

    for key, name in required_keys.items():
        if not os.environ.get(key):
            missing.append(f"  - {key} ({name})")

    if missing:
        print("\n⚠ MISSING API KEYS:")
        print("\n".join(missing))
        print("\nPlease set them in your .env file.\n")
        sys.exit(1)


# Validate on import
_validate_api_keys()

# Ensure GOOGLE_API_KEY is set for litellm / CrewAI compatibility
if os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]


# Centralized LLM configuration
# Using the same model string that was working before
llm = LLM(
    model="gemini/gemini-2.5-flash",
    temperature=0.1,  # Lower temp for research accuracy
    max_tokens=4000,  # Keep at 4000 for free tier quota safety
)
