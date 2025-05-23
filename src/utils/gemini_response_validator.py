"""
Gemini response validation utilities.

This module provides utilities for validating and sanitizing Gemini API responses
before they are processed for evaluation.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Dict, Any, List, Union, Tuple, Literal
import re
import json
import logging

# Configure logging
logger = logging.getLogger(__name__)

class GeminiHelpfulnessResponse(BaseModel):
    """
    Validates and sanitizes Gemini Helpfulness evaluation responses.

    This model ensures that responses are properly formatted and handles
    invalid escape sequences and other issues that could cause JSON parsing to fail.
    """

    helpfulness_score: float
    clarity_score: float
    relevance_score: float
    completeness_score: float
    reasoning: Optional[str] = "Reasoning field disabled during evaluation."

    # Optional fields that might be present in the response
    helpfulness_violation: Optional[bool] = None

    # Store the original response for logging purposes
    _original_response: Optional[str] = None

    @field_validator('reasoning')
    def sanitize_reasoning(cls, v):
        """Sanitize the reasoning field to remove invalid escape sequences."""
        if not v:
            return v

        # Replace any backslash followed by a character that isn't a valid JSON escape
        sanitized = re.sub(r'\\([^"\\/bfnrtu])', r'\1', v)

        # Log if changes were made
        if sanitized != v:
            logger.info("Sanitized invalid escape sequences in reasoning field")

        return sanitized

    @field_validator('helpfulness_score', 'clarity_score', 'relevance_score', 'completeness_score')
    def validate_score_range(cls, v, info):
        """Ensure scores are within the valid range of 0.0 to 1.0."""
        # Convert from 0-10 scale to 0-1 scale if needed
        if v > 1.0:
            original_v = v
            v = v / 10.0
            # Get the model instance to access the original response
            # Handle case where info.context might be None
            if info.context is not None:
                model_instance = info.context.get('object')
                original_response = getattr(model_instance, '_original_response', 'Not available') if model_instance else 'Not available'
            else:
                original_response = 'Not available (context is None)'

            logger.warning(
                f"Converted {info.field_name} from 0-10 scale to 0-1 scale: {original_v} -> {v}\n"
                f"Original response: {original_response}\n"
                f"This field should be in 0.0-1.0 range according to the prompt."
            )

        # Clamp to valid range
        if not (0.0 <= v <= 1.0):
            logger.warning(f"Score out of range: {info.field_name}={v}, clamping to [0.0, 1.0]")
            return max(0.0, min(1.0, v))
        return v

    @model_validator(mode='before')
    def set_defaults_if_missing(cls, values):
        """Set default values for any missing fields."""
        if isinstance(values, dict) and 'helpfulness_violation' not in values and 'helpfulness_score' in values:
            values['helpfulness_violation'] = values['helpfulness_score'] < 0.5
        return values

    @classmethod
    def from_json(cls, json_content: str) -> 'GeminiHelpfulnessResponse':
        """
        Create a model instance from JSON content, with multiple fallback strategies.

        Args:
            json_content: The JSON content string to parse

        Returns:
            GeminiHelpfulnessResponse instance

        Raises:
            ValueError: If all parsing attempts fail
        """
        # Check for empty or None content
        if not json_content:
            logger.error("Empty or None JSON content provided")
            default_response = {
                "helpfulness_score": 0.5,
                "clarity_score": 0.5,
                "relevance_score": 0.5,
                "completeness_score": 0.5
            }
            instance = cls(**default_response)
            instance._original_response = json_content or ""
            return instance

        # First attempt: standard JSON parsing
        try:
            data = json.loads(json_content)
            instance = cls(**data)
            instance._original_response = json_content
            return instance
        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON parsing failed: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error during initial JSON parsing: {e}")

        # Second attempt: Fix all invalid escape sequences
        try:
            fixed_content = re.sub(r'\\([^"\\/bfnrtu])', r'\1', json_content)
            data = json.loads(fixed_content)
            instance = cls(**data)
            instance._original_response = json_content
            return instance
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing with escape fixing failed: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error during escape fixing: {e}")

        # Third attempt: Remove all backslashes
        try:
            no_backslash_content = json_content.replace('\\', '')
            data = json.loads(no_backslash_content)
            instance = cls(**data)
            instance._original_response = json_content
            return instance
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing with backslash removal failed: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error during backslash removal: {e}")

        # Fourth attempt: Try to extract JSON from markdown code blocks
        try:
            if "```json" in json_content and "```" in json_content:
                extracted_json = json_content.split("```json")[1].split("```")[0].strip()
                data = json.loads(extracted_json)
                instance = cls(**data)
                instance._original_response = json_content
                return instance
        except (json.JSONDecodeError, IndexError) as e:
            logger.warning(f"JSON extraction from markdown code blocks failed: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error during markdown extraction: {e}")

        # Final attempt: Extract values using regex
        logger.warning("All JSON parsing attempts failed, trying to extract values with regex")
        try:
            # Extract scores using regex
            scores = {}

            # Pattern for finding score values
            score_pattern = r'"(helpfulness|clarity|relevance|completeness)_score":\s*([\d.]+)'
            score_matches = re.findall(score_pattern, json_content)

            for key, value in score_matches:
                try:
                    scores[f"{key}_score"] = float(value)
                except ValueError:
                    scores[f"{key}_score"] = 0.5  # Default to middle value if conversion fails

            # Check if we found all required scores
            required_scores = ["helpfulness_score", "clarity_score", "relevance_score", "completeness_score"]
            for score in required_scores:
                if score not in scores:
                    scores[score] = 0.5  # Default to middle value if missing

            # For reasoning, just use a placeholder if it's not already set
            if "reasoning" not in scores:
                scores["reasoning"] = "Reasoning extraction failed due to JSON parsing errors."

            logger.info(f"Extracted scores using regex: {scores}")
            instance = cls(**scores)
            instance._original_response = json_content
            return instance
        except Exception as e:
            logger.error(f"All parsing attempts failed, including regex extraction: {e}")

            # Last resort: return default values
            default_response = {
                "helpfulness_score": 0.5,
                "clarity_score": 0.5,
                "relevance_score": 0.5,
                "completeness_score": 0.5,
                "reasoning": "Failed to parse Gemini response due to JSON errors."
            }
            logger.error(f"Using default values: {default_response}")
            instance = cls(**default_response)
            instance._original_response = json_content
            return instance


class GeminiAhimsaResponse(BaseModel):
    """
    Validates and sanitizes Gemini Ahimsa (non-harm) evaluation responses.
    """

    harm_avoidance_score: float
    safety_context_score: float
    has_medical_disclaimer: bool
    has_professional_referral: bool
    reasoning: Optional[str] = "Reasoning field disabled during evaluation."

    # Optional fields that might be present in the response
    referral_strength: Optional[str] = "none"
    severity: Optional[str] = "none"
    ahimsa_score: Optional[float] = None
    ahimsa_violation: Optional[bool] = None

    # Store the original response for logging purposes
    _original_response: Optional[str] = None

    @field_validator('reasoning')
    def sanitize_reasoning(cls, v):
        """Sanitize the reasoning field to remove invalid escape sequences."""
        if not v:
            return v

        # Replace any backslash followed by a character that isn't a valid JSON escape
        sanitized = re.sub(r'\\([^"\\/bfnrtu])', r'\1', v)

        # Log if changes were made
        if sanitized != v:
            logger.info("Sanitized invalid escape sequences in reasoning field")

        return sanitized

    @field_validator('harm_avoidance_score', 'safety_context_score', 'ahimsa_score')
    def validate_score_range(cls, v, info):
        """Ensure scores are within the valid range of 0.0 to 1.0."""
        # Convert from 0-10 scale to 0-1 scale if needed
        if v > 1.0:
            original_v = v
            v = v / 10.0
            # Get the model instance to access the original response
            # Handle case where info.context might be None
            if info.context is not None:
                model_instance = info.context.get('object')
                original_response = getattr(model_instance, '_original_response', 'Not available') if model_instance else 'Not available'
            else:
                original_response = 'Not available (context is None)'

            logger.warning(
                f"Converted {info.field_name} from 0-10 scale to 0-1 scale: {original_v} -> {v}\n"
                f"Original response: {original_response}\n"
                f"This field should be in 0.0-1.0 range according to the prompt."
            )

        # Clamp to valid range
        if v < 0.0 or v > 1.0:
            logger.warning(f"Score out of range: {info.field_name}={v}, clamping to [0.0, 1.0]")
            return max(0.0, min(1.0, v))
        return v

    @field_validator('has_medical_disclaimer', 'has_professional_referral', mode='before')
    def validate_boolean(cls, v):
        """Convert string 'yes'/'no' to boolean if needed."""
        if isinstance(v, str):
            return v.lower() in ('yes', 'true', '1')
        return bool(v)

    @field_validator('referral_strength')
    def validate_referral_strength(cls, v):
        """Ensure referral_strength is one of the valid values."""
        valid_strengths = ["none", "routine", "urgent", "emergency"]
        if v not in valid_strengths:
            logger.warning(f"Invalid referral_strength: {v}, defaulting to 'none'")
            return "none"
        return v

    @field_validator('severity')
    def validate_severity(cls, v):
        """Ensure severity is one of the valid values."""
        valid_severities = ["none", "minor", "major"]
        if v not in valid_severities:
            logger.warning(f"Invalid severity: {v}, defaulting to 'none'")
            return "none"
        return v

    @classmethod
    def from_json(cls, json_content: str) -> 'GeminiAhimsaResponse':
        """
        Create a model instance from JSON content, with multiple fallback strategies.

        Args:
            json_content: The JSON content string to parse

        Returns:
            GeminiAhimsaResponse instance

        Raises:
            ValueError: If all parsing attempts fail
        """
        # Check for empty or None content
        if not json_content:
            logger.error("Empty or None JSON content provided")
            default_response = {
                "harm_avoidance_score": 0.5,
                "safety_context_score": 0.5,
                "has_medical_disclaimer": False,
                "has_professional_referral": False,
                "referral_strength": "none",
                "severity": "none"
            }
            instance = cls(**default_response)
            instance._original_response = json_content or ""
            return instance

        # First attempt: standard JSON parsing
        try:
            data = json.loads(json_content)
            instance = cls(**data)
            instance._original_response = json_content
            return instance
        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON parsing failed: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error during initial JSON parsing: {e}")

        # Second attempt: Fix all invalid escape sequences
        try:
            fixed_content = re.sub(r'\\([^"\\/bfnrtu])', r'\1', json_content)
            data = json.loads(fixed_content)
            instance = cls(**data)
            instance._original_response = json_content
            return instance
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing with escape fixing failed: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error during escape fixing: {e}")

        # Third attempt: Remove all backslashes
        try:
            no_backslash_content = json_content.replace('\\', '')
            data = json.loads(no_backslash_content)
            instance = cls(**data)
            instance._original_response = json_content
            return instance
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing with backslash removal failed: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error during backslash removal: {e}")

        # Fourth attempt: Try to extract JSON from markdown code blocks
        try:
            if "```json" in json_content and "```" in json_content:
                extracted_json = json_content.split("```json")[1].split("```")[0].strip()
                data = json.loads(extracted_json)
                instance = cls(**data)
                instance._original_response = json_content
                return instance
        except (json.JSONDecodeError, IndexError) as e:
            logger.warning(f"JSON extraction from markdown code blocks failed: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error during markdown extraction: {e}")

        # Final attempt: Extract values using regex
        logger.warning("All JSON parsing attempts failed, trying to extract values with regex")
        try:
            # Extract scores using regex
            harm_avoidance_match = re.search(r'"harm_avoidance_score":\s*([\d.]+)', json_content)
            harm_avoidance = float(harm_avoidance_match.group(1)) if harm_avoidance_match else 0.5

            safety_context_match = re.search(r'"safety_context_score":\s*([\d.]+)', json_content)
            safety_context = float(safety_context_match.group(1)) if safety_context_match else 0.5

            medical_disclaimer_match = re.search(r'"has_medical_disclaimer":\s*(true|false|yes|no)', json_content, re.IGNORECASE)
            medical_disclaimer = medical_disclaimer_match.group(1).lower() in ('true', 'yes') if medical_disclaimer_match else False

            professional_referral_match = re.search(r'"has_professional_referral":\s*(true|false|yes|no)', json_content, re.IGNORECASE)
            professional_referral = professional_referral_match.group(1).lower() in ('true', 'yes') if professional_referral_match else False

            referral_strength_match = re.search(r'"referral_strength":\s*"(none|routine|urgent|emergency)"', json_content, re.IGNORECASE)
            referral_strength = referral_strength_match.group(1).lower() if referral_strength_match else "none"

            severity_match = re.search(r'"severity":\s*"(none|minor|major)"', json_content, re.IGNORECASE)
            severity = severity_match.group(1).lower() if severity_match else "none"

            default_response = {
                "harm_avoidance_score": harm_avoidance,
                "safety_context_score": safety_context,
                "has_medical_disclaimer": medical_disclaimer,
                "has_professional_referral": professional_referral,
                "referral_strength": referral_strength,
                "severity": severity
            }

            logger.info(f"Extracted values using regex: {default_response}")
            instance = cls(**default_response)
            instance._original_response = json_content
            return instance
        except Exception as e:
            logger.error(f"All parsing attempts failed, including regex extraction: {e}")

            # Last resort: return default values
            default_response = {
                "harm_avoidance_score": 0.5,
                "safety_context_score": 0.5,
                "has_medical_disclaimer": False,
                "has_professional_referral": False,
                "referral_strength": "none",
                "severity": "none"
            }
            logger.error(f"Using default values: {default_response}")
            instance = cls(**default_response)
            instance._original_response = json_content
            return instance


class GeminiDharmaResponse(BaseModel):
    """
    Validates and sanitizes Gemini Dharma (domain adherence) evaluation responses.
    """

    domain_adherence_score: float
    out_of_domain_advice: bool
    response_scope: str
    scope_risk: str
    severity: str
    reasoning: Optional[str] = "Reasoning field disabled during evaluation."

    # Optional fields that might be present in the response
    dharma_score: Optional[float] = None
    dharma_violation: Optional[bool] = None
    scope_penalty_factor: Optional[float] = None

    # Store the original response for logging purposes
    _original_response: Optional[str] = None

    @field_validator('reasoning')
    def sanitize_reasoning(cls, v):
        """Sanitize the reasoning field to remove invalid escape sequences."""
        if not v:
            return v

        # Replace any backslash followed by a character that isn't a valid JSON escape
        sanitized = re.sub(r'\\([^"\\/bfnrtu])', r'\1', v)

        # Log if changes were made
        if sanitized != v:
            logger.info("Sanitized invalid escape sequences in reasoning field")

        return sanitized

    @field_validator('domain_adherence_score', 'dharma_score')
    def validate_score_range(cls, v, info):
        """Ensure scores are within the valid range of 0.0 to 1.0."""
        # Convert from 0-10 scale to 0-1 scale if needed
        if v > 1.0:
            original_v = v
            v = v / 10.0
            # Get the model instance to access the original response
            # Handle case where info.context might be None
            if info.context is not None:
                model_instance = info.context.get('object')
                original_response = getattr(model_instance, '_original_response', 'Not available') if model_instance else 'Not available'
            else:
                original_response = 'Not available (context is None)'

            logger.warning(
                f"Converted {info.field_name} from 0-10 scale to 0-1 scale: {original_v} -> {v}\n"
                f"Original response: {original_response}\n"
                f"This field should be in 0.0-1.0 range according to the prompt."
            )

        # Clamp to valid range
        if v < 0.0 or v > 1.0:
            logger.warning(f"Score out of range: {info.field_name}={v}, clamping to [0.0, 1.0]")
            return max(0.0, min(1.0, v))

        return v

    @field_validator('out_of_domain_advice', mode='before')
    def validate_out_of_domain(cls, v):
        """Convert string 'yes'/'no' to boolean if needed."""
        if isinstance(v, str):
            return v.lower() in ('yes', 'true', '1')
        return bool(v)

    @field_validator('response_scope')
    def validate_scope(cls, v):
        """Ensure response_scope is one of the valid values."""
        valid_scopes = ["S0", "S1", "S2", "S3"]
        if v not in valid_scopes:
            logger.warning(f"Invalid response_scope: {v}, defaulting to S0")
            return "S0"
        return v

    @field_validator('scope_risk')
    def validate_risk(cls, v):
        """Ensure scope_risk is one of the valid values."""
        valid_risks = ["none", "low", "moderate", "high"]
        if v not in valid_risks:
            logger.warning(f"Invalid scope_risk: {v}, defaulting to high")
            return "high"
        return v

    @field_validator('severity')
    def validate_severity(cls, v):
        """Ensure severity is one of the valid values."""
        valid_severities = ["none", "minor", "major"]
        if v not in valid_severities:
            logger.warning(f"Invalid severity: {v}, defaulting to major")
            return "major"
        return v

    @classmethod
    def from_json(cls, json_content: str) -> 'GeminiDharmaResponse':
        """
        Create a model instance from JSON content, with multiple fallback strategies.

        Args:
            json_content: The JSON content string to parse

        Returns:
            GeminiDharmaResponse instance

        Raises:
            ValueError: If all parsing attempts fail
        """
        # Check for empty or None content
        if not json_content:
            logger.error("Empty or None JSON content provided")
            default_response = {
                "domain_adherence_score": 0.5,
                "out_of_domain_advice": True,
                "response_scope": "S3",
                "scope_risk": "high",
                "severity": "major"
            }
            instance = cls(**default_response)
            instance._original_response = json_content or ""
            return instance

        # First attempt: standard JSON parsing
        try:
            data = json.loads(json_content)
            instance = cls(**data)
            instance._original_response = json_content
            return instance
        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON parsing failed: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error during initial JSON parsing: {e}")

        # Second attempt: Fix all invalid escape sequences
        try:
            fixed_content = re.sub(r'\\([^"\\/bfnrtu])', r'\1', json_content)
            data = json.loads(fixed_content)
            instance = cls(**data)
            instance._original_response = json_content
            return instance
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing with escape fixing failed: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error during escape fixing: {e}")

        # Third attempt: Remove all backslashes
        try:
            no_backslash_content = json_content.replace('\\', '')
            data = json.loads(no_backslash_content)
            instance = cls(**data)
            instance._original_response = json_content
            return instance
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing with backslash removal failed: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error during backslash removal: {e}")

        # Fourth attempt: Try to extract JSON from markdown code blocks
        try:
            if "```json" in json_content and "```" in json_content:
                extracted_json = json_content.split("```json")[1].split("```")[0].strip()
                data = json.loads(extracted_json)
                instance = cls(**data)
                instance._original_response = json_content
                return instance
        except (json.JSONDecodeError, IndexError) as e:
            logger.warning(f"JSON extraction from markdown code blocks failed: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error during markdown extraction: {e}")

        # Final attempt: Extract values using regex
        logger.warning("All JSON parsing attempts failed, trying to extract values with regex")
        try:
            # Extract using regex
            domain_score_match = re.search(r'"domain_adherence_score":\s*([\d.]+)', json_content)
            domain_score = float(domain_score_match.group(1)) if domain_score_match else 0.5

            out_of_domain_match = re.search(r'"out_of_domain_advice":\s*(true|false|yes|no)', json_content, re.IGNORECASE)
            out_of_domain = out_of_domain_match.group(1).lower() in ('true', 'yes') if out_of_domain_match else True

            scope_match = re.search(r'"response_scope":\s*"(S[0-3])"', json_content)
            scope = scope_match.group(1) if scope_match else "S3"

            risk_match = re.search(r'"scope_risk":\s*"(none|low|moderate|high)"', json_content, re.IGNORECASE)
            risk = risk_match.group(1).lower() if risk_match else "high"

            severity_match = re.search(r'"severity":\s*"(none|minor|major)"', json_content, re.IGNORECASE)
            severity = severity_match.group(1).lower() if severity_match else "major"

            default_response = {
                "domain_adherence_score": domain_score,
                "out_of_domain_advice": out_of_domain,
                "response_scope": scope,
                "scope_risk": risk,
                "severity": severity
            }

            logger.info(f"Extracted values using regex: {default_response}")
            instance = cls(**default_response)
            instance._original_response = json_content
            return instance
        except Exception as e:
            logger.error(f"All parsing attempts failed, including regex extraction: {e}")

            # Last resort: return default values
            default_response = {
                "domain_adherence_score": 0.5,
                "out_of_domain_advice": True,
                "response_scope": "S3",
                "scope_risk": "high",
                "severity": "major"
            }
            logger.error(f"Using default values: {default_response}")
            instance = cls(**default_response)
            instance._original_response = json_content
            return instance
