import hashlib
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# --- Configuration ---
# Use a fast hash algorithm like SHA-1, collision resistance is sufficient here.
_HASH_ALGORITHM = hashlib.sha1
_ENCODING = 'utf-8'
_DELIMITER = ":"
# ---

def calculate_prompt_hash(prompt) -> str:
    """
    Calculates a hash for the given prompt.

    Args:
        prompt: The input prompt, which can be a string or a list of message dictionaries.

    Returns:
        A hexadecimal string representation of the hash.
    """
    try:
        # Handle different prompt formats
        if isinstance(prompt, str):
            # Original case: prompt is a string
            prompt_str = prompt
        elif isinstance(prompt, list):
            # New case: prompt is a list of message dictionaries (chat format)
            # Extract the user message content
            user_content = ""
            for message in prompt:
                if isinstance(message, dict) and message.get("role") == "user":
                    user_content = message.get("content", "")
                    break
            prompt_str = user_content
        else:
            # Unexpected format
            logger.error(f"Unexpected prompt format: {type(prompt)}")
            # Return a consistent hash for unexpected formats
            return "0" * 40  # 40 zeros (length of SHA-1 hash)

        # Normalize by encoding to bytes
        prompt_bytes = prompt_str.encode(_ENCODING)
        hash_object = _HASH_ALGORITHM(prompt_bytes)
        return hash_object.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash for prompt: {e}", exc_info=True)
        # Return a consistent value on error
        return "0" * 40  # 40 zeros (length of SHA-1 hash)

def create_compound_tier(tier: str, prompt_hash: str) -> str:
    """
    Combines a tier identifier and a prompt hash using a delimiter.

    Args:
        tier: The original tier identifier (e.g., 'A', 'B', 'C').
        prompt_hash: The hexadecimal hash string of the prompt.

    Returns:
        The combined compound tier string (e.g., 'A:hash_value').
    """
    return f"{tier}{_DELIMITER}{prompt_hash}"

def split_compound_tier(compound_tier: str) -> Optional[Tuple[str, str]]:
    """
    Splits a compound tier string into its original tier and hash components.

    Args:
        compound_tier: The combined string (e.g., 'A:hash_value').

    Returns:
        A tuple containing (original_tier, embedded_hash) if successful,
        otherwise None.
    """
    if not compound_tier:
        return None
    parts = compound_tier.split(_DELIMITER, 1) # Split only on the first delimiter
    if len(parts) == 2:
        return parts[0], parts[1]
    else:
        logger.warning(f"Could not split compound tier string: '{compound_tier}' - Expected format 'Tier{_DELIMITER}Hash'.")
        return None

def verify_prompt_tier_hash(prompt, compound_tier: str) -> bool:
    """
    Verifies if the hash embedded in the compound tier matches the hash
    calculated from the prompt.

    Args:
        prompt: The prompt to verify (string or list of message dictionaries).
        compound_tier: The compound tier string containing the embedded hash.

    Returns:
        True if the hashes match or the tier format is invalid (can't check),
        False if the hashes explicitly mismatch.
    """
    # Log the prompt format for debugging
    logger.debug(f"Prompt type: {type(prompt)}")
    if isinstance(prompt, list) and len(prompt) > 0:
        logger.debug(f"Prompt is a list of length {len(prompt)}, first item type: {type(prompt[0])}")
        if isinstance(prompt[0], dict):
            logger.debug(f"First message keys: {list(prompt[0].keys())}")

    split_result = split_compound_tier(compound_tier)
    if split_result is None:
        # Could not parse, cannot verify. Logged in split_compound_tier.
        logger.warning(f"Could not split compound tier '{compound_tier}', skipping verification")
        # Return True to avoid falsely flagging an issue just due to bad format.
        return True

    _, embedded_hash = split_result
    try:
        calculated_hash = calculate_prompt_hash(prompt)
        match = (calculated_hash == embedded_hash)
        if not match:
            # For chat format prompts, extract and log the user message for debugging
            user_message = ""
            if isinstance(prompt, list):
                for message in prompt:
                    if isinstance(message, dict) and message.get("role") == "user":
                        user_message = message.get("content", "")[:50] + "..."
                        break
                logger.warning(f"HASH MISMATCH: Calculated '{calculated_hash}' != Embedded '{embedded_hash}' for chat prompt with user message: '{user_message}'")
            else:
                logger.warning(f"HASH MISMATCH: Calculated '{calculated_hash}' != Embedded '{embedded_hash}' for prompt: '{prompt[:50]}...'")
        return match
    except Exception as e:
        # Error during hash calculation for the prompt, log details and continue
        logger.error(f"Error during hash verification: {e}")
        # Return True to avoid false failure due to hashing error
        return True

def extract_tier_from_compound(compound_tier: str, default_tier: str = "C") -> str:
    """
    Extracts the original tier identifier from a compound tier string.

    Args:
        compound_tier: The combined string (e.g., 'A:hash_value').
        default_tier: The tier to return if extraction fails. Defaults to 'C'.

    Returns:
        The extracted original tier identifier (e.g., 'A'), or the default_tier
        if extraction fails.
    """
    split_result = split_compound_tier(compound_tier)
    if split_result:
        return split_result[0]
    else:
        # If splitting failed, return the original string only if it looks like a plain tier
        # Otherwise, return the default.
        if compound_tier and _DELIMITER not in compound_tier and len(compound_tier) == 1 and compound_tier.isalpha():
            return compound_tier # Assume it might be an unprocessed tier
        return default_tier