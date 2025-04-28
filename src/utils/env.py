"""
Environment variable utilities.
"""

import os
from typing import Optional
from dotenv import load_dotenv


def load_env_vars() -> None:
    """
    Load environment variables from .env file.
    """
    # Load environment variables from .env file
    load_dotenv()

def get_openai_api_key() -> Optional[str]:
    """
    Get the OpenAI API key from environment variables.
    Returns None if not found.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # Instead of raising error, return None
        # logger.warning("OPENAI_API_KEY environment variable not set.") 
        return None
        # raise ValueError(
        #     "OPENAI_API_KEY environment variable not set. "
        #     "Please set it in your environment or in a .env file."
        # )
    return api_key



def get_gemini_api_key() -> str:
    """
    Get the Gemini API key from environment variables.
    
    Returns:
        str: The Gemini API key
        
    Raises:
        ValueError: If the GEMINI_API_KEY environment variable is not set
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable not set. "
            "Please set it in your environment or in a .env file."
        )
    return api_key
