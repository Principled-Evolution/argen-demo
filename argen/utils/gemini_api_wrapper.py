"""
Wrapper classes for Gemini API that automatically track API calls and support caching.

This module provides compatibility between the old google-generativeai SDK
and the new google-genai SDK, with integrated context caching support.
"""

import functools
import logging
from typing import Any, Callable, Optional
import google.generativeai as genai
from google.generativeai.types.generation_types import BlockedPromptException

from .gemini_api_tracker import GeminiAPITracker
from .gemini_cache_manager import cache_manager

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
    """Wrapper for genai.GenerativeModel that tracks API calls and supports caching."""

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

    async def start_cached_chat(self, system_prompt: str, cache_key: str, **kwargs):
        """
        Start a chat session with cached system prompt.

        Args:
            system_prompt: The system prompt to cache
            cache_key: Unique identifier for this cache entry
            **kwargs: Additional arguments for start_chat

        Returns:
            TrackedChat instance with cached content support
        """
        cache_manager.record_request()

        # Try to get cached content
        cached_content_name = cache_manager.get_cached_content_name(cache_key)

        if cached_content_name:
            logger.debug(f"Using cached content for {cache_key}: {cached_content_name}")
            # For now, fall back to regular chat since we need new SDK for cached content
            # TODO: Implement new SDK integration
            chat = self._model.start_chat(**kwargs)
            return TrackedCachedChat(chat, self.model_name, self._tracker, cache_key, cached_content_name)
        else:
            # Create cache entry for future use
            logger.debug(f"Creating cache entry for {cache_key}")
            cached_content_name = await cache_manager.create_cached_content(
                content=system_prompt,
                cache_key=cache_key,
                model_name=self.model_name
            )

            # Use regular chat for now
            chat = self._model.start_chat(**kwargs)
            return TrackedCachedChat(chat, self.model_name, self._tracker, cache_key, cached_content_name)

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


class TrackedCachedChat(TrackedChat):
    """Wrapper for chat sessions with caching support."""

    def __init__(self, chat, model_name: str, tracker: GeminiAPITracker, cache_key: str, cached_content_name: Optional[str]):
        super().__init__(chat, model_name, tracker)
        self.cache_key = cache_key
        self.cached_content_name = cached_content_name
        logger.debug(f"Created TrackedCachedChat for {model_name} with cache_key: {cache_key}")

    def send_message(self, *args, **kwargs):
        """Tracked version of send_message with cache awareness."""
        # For now, use the parent implementation
        # TODO: Integrate with new SDK for actual cached content usage
        if self.cached_content_name:
            logger.debug(f"Sending message with cached content: {self.cached_content_name}")

        return super().send_message(*args, **kwargs)
