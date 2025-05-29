"""
Response validation utilities for model responses.

This module provides utilities for validating and sanitizing model responses
before they are sent to evaluators.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
import re
import logging

# Configure logging
logger = logging.getLogger(__name__)
VERBOSE_LOGGING = False
class ModelResponse(BaseModel):
    """
    Validates and sanitizes model responses before sending to evaluators.
    
    This model ensures that responses are properly formatted and don't contain
    problematic elements like LaTeX formatting, escape characters, or raw curly braces
    that could cause issues with JSON parsing.
    """
    
    raw_content: str
    sanitized_content: Optional[str] = None
    is_valid: bool = True
    sanitization_applied: bool = False
    error_message: Optional[str] = None
    
    @validator('sanitized_content', always=True)
    def sanitize_content(cls, v, values):
        """
        Sanitize the raw content to make it safe for embedding in prompts.
        
        This method removes LaTeX formatting, escape characters, and other problematic
        elements from the raw content.
        """
        raw = values.get('raw_content', '')
        if not raw:
            return ''
        
        sanitized = raw
        applied_sanitization = False
        
        # Remove LaTeX table formatting
        if '\\begin{tabular}' in sanitized:
            sanitized = re.sub(r'\\begin\{tabular\}.*?\\hline', '', sanitized)
            sanitized = re.sub(r'\\end\{tabular\}', '', sanitized)
            applied_sanitization = True
            if VERBOSE_LOGGING:
                logger.info("Removed LaTeX table formatting from response")
        
        # Remove other LaTeX commands
        if '\\' in sanitized:
            sanitized = re.sub(r'\\[a-zA-Z]+(\{.*?\})?', '', sanitized)
            applied_sanitization = True
            if VERBOSE_LOGGING:
                logger.info("Removed LaTeX commands from response")
        
        # Remove escape characters
        if any(c in sanitized for c in ['\n', '\t', '\r', '\b', '\f']):
            sanitized = sanitized.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ').replace('\b', ' ').replace('\f', ' ')
            applied_sanitization = True
            if VERBOSE_LOGGING:
                logger.info("Removed escape characters from response")
        
        # Handle curly braces that might confuse JSON parsing
        if '{' in sanitized or '}' in sanitized:
            # Replace only unescaped braces that aren't part of JSON
            sanitized = re.sub(r'(?<!\\)\{(?!\s*["\'a-zA-Z0-9_]+\s*:)', '(', sanitized)
            sanitized = re.sub(r'(?<!\\)\}(?!\s*,?\s*["\'a-zA-Z0-9_]+\s*:)', ')', sanitized)
            applied_sanitization = True
            if VERBOSE_LOGGING:
                logger.info("Replaced curly braces with parentheses in response")
        
        # Remove any control characters
        if re.search(r'[\x00-\x1F\x7F]', sanitized):
            sanitized = re.sub(r'[\x00-\x1F\x7F]', '', sanitized)
            applied_sanitization = True
            if VERBOSE_LOGGING:
                logger.info("Removed control characters from response")
        
        # Trim excessive whitespace
        original_length = len(sanitized)
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        if len(sanitized) < original_length:
            applied_sanitization = True
            if VERBOSE_LOGGING:
                logger.info("Trimmed excessive whitespace from response")
        
        # Update sanitization flag
        values['sanitization_applied'] = applied_sanitization
        
        return sanitized
    
    @validator('is_valid', always=True)
    def validate_content(cls, v, values):
        """
        Check if the content is valid for evaluation.
        
        This method checks for empty or too short responses, responses that are just
        formatting with no content, and responses where sanitization removed too much content.
        """
        raw = values.get('raw_content', '')
        sanitized = values.get('sanitized_content', '')
        
        # Check for empty or too short responses
        if not sanitized or len(sanitized) < 10:
            values['error_message'] = "Response too short or empty after sanitization"
            return False
        
        # Check for responses that are just formatting with no content
        if re.match(r'^[\s\\\{\}\[\]]*$', sanitized):
            values['error_message'] = "Response contains only formatting characters"
            return False
        
        # Check if sanitization removed too much content
        if len(sanitized) < len(raw) * 0.3:  # Lost more than 70% of content
            values['error_message'] = "Sanitization removed too much content"
            return False
        
        return True

class ModelResponseProcessor:
    """
    Processes model responses before evaluation.
    
    This class provides methods for validating and sanitizing model responses
    before they are sent to evaluators.
    """
    
    def __init__(self, fallback_penalty=0.5):
        """
        Initialize the ModelResponseProcessor.
        
        Args:
            fallback_penalty: Penalty to apply when using fallback responses.
        """
        self.fallback_penalty = fallback_penalty
        self.stats = {
            "processed": 0,
            "sanitized": 0,
            "invalid": 0,
            "fallback_used": 0
        }
    
    def process(self, model_response, prompt):
        """
        Process a model response and prepare it for evaluation.
        
        Args:
            model_response: The response generated by the model.
            prompt: The prompt that generated the response.
            
        Returns:
            The processed response ready for evaluation.
        """
        self.stats["processed"] += 1
        
        try:
            # Validate and sanitize the response
            response_obj = ModelResponse(raw_content=model_response)
            
            if response_obj.sanitization_applied:
                self.stats["sanitized"] += 1
                logger.info(f"Sanitized response for prompt: {prompt[:50]}...")
                logger.debug(f"Original: {model_response[:100]}...")
                logger.debug(f"Sanitized: {response_obj.sanitized_content[:100]}...")
            
            if not response_obj.is_valid:
                self.stats["invalid"] += 1
                logger.warning(f"Invalid response for prompt: {prompt[:50]}... Error: {response_obj.error_message}")
                return self._apply_fallback(prompt, model_response, response_obj.error_message)
            
            return response_obj.sanitized_content
            
        except Exception as e:
            logger.error(f"Error processing response: {str(e)}")
            return self._apply_fallback(prompt, model_response, str(e))
    
    def _apply_fallback(self, prompt, original_response, error):
        """
        Apply fallback strategy for invalid responses.
        
        Args:
            prompt: The prompt that generated the response.
            original_response: The original response from the model.
            error: The error message.
            
        Returns:
            A fallback response.
        """
        self.stats["fallback_used"] += 1
        
        # Option 1: Return a simplified version of the response
        if len(original_response) > 20:
            # Try to extract any plain text
            plain_text = re.sub(r'[^\w\s.,?!]', ' ', original_response)
            plain_text = re.sub(r'\s+', ' ', plain_text).strip()
            if len(plain_text) > 20:
                return plain_text
        
        # Option 2: Return a placeholder response with the error
        return f"[Invalid response that could not be processed. Error: {error}]"
    
    def get_stats(self):
        """
        Get statistics about processed responses.
        
        Returns:
            A dictionary with statistics.
        """
        return self.stats
