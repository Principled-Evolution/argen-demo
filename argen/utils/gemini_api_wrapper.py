"""
Wrapper classes for Gemini API that automatically track API calls.
"""

import functools
import logging
from typing import Any, Callable, Optional
import google.generativeai as genai
from google.generativeai.types.generation_types import BlockedPromptException

from .gemini_api_tracker import GeminiAPITracker

logger = logging.getLogger(__name__)

def track_gemini_api_call(model_name: Optional[str] = None):
    """Decorator to track Gemini API calls."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracker = GeminiAPITracker()
            
            # Extract model name from args if not provided
            actual_model_name = model_name
            if not actual_model_name:
                # Try to extract from self (for methods) or first arg
                if args and hasattr(args[0], 'model_name'):
                    actual_model_name = args[0].model_name
                elif args and hasattr(args[0], '_model_name'):
                    actual_model_name = args[0]._model_name
                else:
                    actual_model_name = "unknown"
            
            try:
                result = func(*args, **kwargs)
                # Check if result indicates success
                if hasattr(result, 'text') or hasattr(result, 'candidates'):
                    tracker.track_call(actual_model_name, success=True)
                else:
                    tracker.track_call(actual_model_name, success=False)
                return result
            except Exception as e:
                tracker.track_call(actual_model_name, success=False)
                raise
        
        return wrapper
    return decorator

class TrackedGenerativeModel:
    """Wrapper for genai.GenerativeModel that tracks API calls."""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self._model = genai.GenerativeModel(model_name, **kwargs)
        self._tracker = GeminiAPITracker()
        logger.debug(f"Created TrackedGenerativeModel for {model_name}")
    
    def generate_content(self, *args, **kwargs):
        """Tracked version of generate_content."""
        try:
            result = self._model.generate_content(*args, **kwargs)
            # Check if we got a valid response
            if hasattr(result, 'text') and result.text:
                self._tracker.track_call(self.model_name, success=True)
            elif hasattr(result, 'candidates') and result.candidates:
                self._tracker.track_call(self.model_name, success=True)
            else:
                self._tracker.track_call(self.model_name, success=False)
            return result
        except Exception as e:
            self._tracker.track_call(self.model_name, success=False)
            logger.error(f"Gemini API call failed for {self.model_name}: {e}")
            raise
    
    def start_chat(self, **kwargs):
        """Return a tracked chat session."""
        chat = self._model.start_chat(**kwargs)
        return TrackedChat(chat, self.model_name, self._tracker)
    
    def __getattr__(self, name):
        """Delegate other attributes to the underlying model."""
        return getattr(self._model, name)

class TrackedChat:
    """Wrapper for chat sessions that tracks send_message calls."""
    
    def __init__(self, chat, model_name: str, tracker: GeminiAPITracker):
        self._chat = chat
        self.model_name = model_name
        self._tracker = tracker
        logger.debug(f"Created TrackedChat for {model_name}")
    
    def send_message(self, *args, **kwargs):
        """Tracked version of send_message."""
        try:
            result = self._chat.send_message(*args, **kwargs)
            # Check if we got a valid response
            if hasattr(result, 'text') and result.text:
                self._tracker.track_call(self.model_name, success=True)
            elif hasattr(result, 'candidates') and result.candidates:
                self._tracker.track_call(self.model_name, success=True)
            else:
                self._tracker.track_call(self.model_name, success=False)
            return result
        except Exception as e:
            self._tracker.track_call(self.model_name, success=False)
            logger.error(f"Gemini chat API call failed for {self.model_name}: {e}")
            raise
    
    def __getattr__(self, name):
        """Delegate other attributes to the underlying chat."""
        return getattr(self._chat, name)
