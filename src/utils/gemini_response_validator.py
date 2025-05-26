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
from .json_extractor import extract_json_from_response, increment_default_score_count

# Configure logging
logger = logging.getLogger(__name__)

class GeminiHelpfulnessResponse(BaseModel):
    """
    Validates and sanitizes Gemini Helpfulness evaluation responses.

    This model ensures that responses are properly formatted and handles
    invalid escape sequences and other issues that could cause JSON parsing to fail.

    Note: helpfulness_score is no longer expected from Gemini - it will be calculated
    as an average of the four constituent scores in Python.
    """

    clarity_score: float = Field(..., ge=0.0, le=1.0)
    completeness_score: float = Field(..., ge=0.0, le=1.0)
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    empathy_score: float = Field(..., ge=0.0, le=1.0)
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

    @field_validator('clarity_score', 'relevance_score', 'completeness_score', 'empathy_score')
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
        # Note: helpfulness_violation will be calculated after Python computes the average helpfulness_score
        # No longer setting it here since we don't expect helpfulness_score from Gemini
        return values

    @classmethod
    def from_json(cls, json_content: str) -> 'GeminiHelpfulnessResponse':
        """
        Create a model instance from JSON content using centralized extraction.

        Args:
            json_content: The JSON content string to parse

        Returns:
            GeminiHelpfulnessResponse instance
        """
        # Use centralized JSON extraction
        parsed_data, success = extract_json_from_response(json_content, "helpfulness")

        if success and parsed_data:
            try:
                instance = cls(**parsed_data)
                instance._original_response = json_content
                return instance
            except Exception as e:
                logger.error(f"Failed to create GeminiHelpfulnessResponse from parsed data: {e}")
                # Fall through to default values

        # If extraction failed or data is invalid, use default values
        count = increment_default_score_count()
        logger.error(f"Using default helpfulness scores. Total default usage count: {count}")

        default_response = {
            "clarity_score": 0.5,
            "relevance_score": 0.5,
            "completeness_score": 0.5,
            "empathy_score": 0.5,
            "reasoning": "Failed to parse Gemini response due to JSON errors."
        }

        instance = cls(**default_response)
        instance._original_response = json_content or ""
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
        Create a model instance from JSON content using centralized extraction.

        Args:
            json_content: The JSON content string to parse

        Returns:
            GeminiAhimsaResponse instance
        """
        # Use centralized JSON extraction
        parsed_data, success = extract_json_from_response(json_content, "ahimsa")

        if success and parsed_data:
            try:
                instance = cls(**parsed_data)
                instance._original_response = json_content
                return instance
            except Exception as e:
                logger.error(f"Failed to create GeminiAhimsaResponse from parsed data: {e}")
                # Fall through to default values

        # If extraction failed or data is invalid, use default values
        count = increment_default_score_count()
        logger.error(f"Using default ahimsa scores. Total default usage count: {count}")

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
        Create a model instance from JSON content using centralized extraction.

        Args:
            json_content: The JSON content string to parse

        Returns:
            GeminiDharmaResponse instance
        """
        # Use centralized JSON extraction
        parsed_data, success = extract_json_from_response(json_content, "dharma")

        if success and parsed_data:
            try:
                instance = cls(**parsed_data)
                instance._original_response = json_content
                return instance
            except Exception as e:
                logger.error(f"Failed to create GeminiDharmaResponse from parsed data: {e}")
                # Fall through to default values

        # If extraction failed or data is invalid, use default values
        count = increment_default_score_count()
        logger.error(f"Using default dharma scores. Total default usage count: {count}")

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
