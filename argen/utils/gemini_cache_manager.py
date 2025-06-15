"""
Gemini 2.0 Flash Context Caching Manager for ArGen Reward Functions.

This module provides centralized caching functionality for Gemini API calls,
specifically designed to cache system prompts, JSON schemas, and other
repetitive content to reduce input token costs.
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from threading import Lock
import json

# Import the new Google GenAI SDK
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    # Fallback for development/testing
    GENAI_AVAILABLE = False
    logging.warning("google-genai SDK not available. Caching will be disabled.")

from argen.utils.env import get_gemini_api_key, load_env_vars
from argen.config import get_gemini_caching_config

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Represents a cached content entry."""
    cache_name: str
    content_hash: str
    created_at: float
    ttl_hours: int = 24
    token_count: int = 0
    hit_count: int = 0
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return time.time() - self.created_at > (self.ttl_hours * 3600)

@dataclass
class CacheStats:
    """Statistics for cache performance monitoring."""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_errors: int = 0
    tokens_saved: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

class GeminiCacheManager:
    """
    Manages Gemini 2.0 Flash context caching for ArGen reward functions.
    
    This class provides:
    - System prompt caching
    - JSON schema caching  
    - Cache lifecycle management
    - Performance monitoring
    - Graceful fallback handling
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.client: Optional[Any] = None
            self.cache_entries: Dict[str, CacheEntry] = {}
            self.stats = CacheStats()
            self.lock = Lock()
            self.enabled = GENAI_AVAILABLE
            # Load configuration
            config = get_gemini_caching_config()
            self.min_tokens_for_caching = config.get("min_tokens_for_caching", 4096)
            self.default_ttl_hours = config.get("cache_ttl_hours", 24)
            self.enable_composite_caching = config.get("enable_composite_caching", False)
            self._initialized = True

            if self.enabled:
                self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Google GenAI client."""
        try:
            # Load environment variables first
            load_env_vars()

            api_key = get_gemini_api_key()
            if not api_key:
                logger.error("Gemini API key not found. Caching disabled.")
                self.enabled = False
                return

            if GENAI_AVAILABLE:
                # The new google-genai SDK expects the API key as a parameter
                # or in GOOGLE_API_KEY environment variable
                self.client = genai.Client(api_key=api_key)
                logger.info("Gemini cache manager initialized successfully")
            else:
                logger.warning("Google GenAI SDK not available, caching disabled")
                self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize Gemini cache manager: {e}")
            self.enabled = False
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate a hash for content to use as cache key."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
    
    def _estimate_token_count(self, content: str) -> int:
        """Rough estimation of token count for content."""
        # Simple heuristic: ~4 characters per token for English text
        return len(content) // 4

    async def create_composite_cached_content(
        self,
        reward_type: str,
        model_name: str = "gemini-2.0-flash-001",
        ttl_hours: int = None
    ) -> Optional[str]:
        """
        Create cached content using composite cache builder.

        Args:
            reward_type: Type of reward function ('helpfulness', 'dharma', 'ahimsa')
            model_name: Gemini model name
            ttl_hours: Time to live in hours (default: 24)

        Returns:
            Cache name if successful, None if failed
        """
        if not self.enabled or not self.enable_composite_caching:
            return None

        try:
            from argen.utils.composite_cache_builder import composite_builder

            # Get composite content based on reward type
            if reward_type == "helpfulness":
                content, cache_key = composite_builder.build_helpfulness_cache_content()
            elif reward_type == "dharma":
                content, cache_key = composite_builder.build_dharma_cache_content()
            elif reward_type == "ahimsa":
                content, cache_key = composite_builder.build_ahimsa_cache_content()
            else:
                logger.error(f"Unknown reward type for composite caching: {reward_type}")
                return None

            # Use the standard create_cached_content method
            return await self.create_cached_content(
                content=content,
                cache_key=cache_key,
                model_name=model_name,
                ttl_hours=ttl_hours
            )

        except Exception as e:
            logger.error(f"Failed to create composite cached content for {reward_type}: {e}")
            return None
    
    async def create_cached_content(
        self,
        content: str,
        cache_key: str,
        model_name: str = "gemini-2.0-flash-001",
        ttl_hours: int = None
    ) -> Optional[str]:
        """
        Create cached content in Gemini API.
        
        Args:
            content: The content to cache (system prompt, schema, etc.)
            cache_key: Unique identifier for this cache entry
            model_name: Gemini model name
            ttl_hours: Time to live in hours (default: 24)
            
        Returns:
            Cache name if successful, None if failed
        """
        if not self.enabled:
            return None
            
        ttl_hours = ttl_hours or self.default_ttl_hours
        content_hash = self._generate_content_hash(content)
        estimated_tokens = self._estimate_token_count(content)
        
        # Check if content meets minimum token requirement
        if estimated_tokens < self.min_tokens_for_caching:
            logger.debug(f"Content too short for caching: {estimated_tokens} tokens < {self.min_tokens_for_caching}")
            return None
        
        with self.lock:
            # Check if we already have this content cached
            existing_entry = self.cache_entries.get(cache_key)
            if existing_entry and not existing_entry.is_expired() and existing_entry.content_hash == content_hash:
                logger.debug(f"Using existing cache entry: {cache_key}")
                return existing_entry.cache_name
        
        try:
            # Create cached content using new SDK
            # Convert hours to seconds for TTL (API expects seconds with 's' suffix)
            ttl_seconds = ttl_hours * 3600
            cache_config = types.CreateCachedContentConfig(
                contents=[types.Content(parts=[types.Part(text=content)], role="user")],
                ttl=f"{ttl_seconds}s"
            )

            cached_content = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.caches.create(model=model_name, config=cache_config)
            )
            
            cache_name = cached_content.name
            
            # Store cache entry
            with self.lock:
                self.cache_entries[cache_key] = CacheEntry(
                    cache_name=cache_name,
                    content_hash=content_hash,
                    created_at=time.time(),
                    ttl_hours=ttl_hours,
                    token_count=estimated_tokens
                )
            
            logger.info(f"Created cache entry: {cache_key} -> {cache_name} ({estimated_tokens} tokens)")
            return cache_name
            
        except Exception as e:
            logger.error(f"Failed to create cached content for {cache_key}: {e}")
            with self.lock:
                self.stats.cache_errors += 1
            return None
    
    def get_cached_content_name(self, cache_key: str) -> Optional[str]:
        """
        Get the cache name for a given cache key.
        
        Args:
            cache_key: The cache key to lookup
            
        Returns:
            Cache name if found and valid, None otherwise
        """
        with self.lock:
            entry = self.cache_entries.get(cache_key)
            if entry and not entry.is_expired():
                entry.hit_count += 1
                self.stats.cache_hits += 1
                self.stats.tokens_saved += entry.token_count
                return entry.cache_name
            elif entry and entry.is_expired():
                # Remove expired entry
                del self.cache_entries[cache_key]
                logger.debug(f"Removed expired cache entry: {cache_key}")
            
            self.stats.cache_misses += 1
            return None
    
    def record_request(self):
        """Record a cache request for statistics."""
        with self.lock:
            self.stats.total_requests += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self.lock:
            return {
                "enabled": self.enabled,
                "total_requests": self.stats.total_requests,
                "cache_hits": self.stats.cache_hits,
                "cache_misses": self.stats.cache_misses,
                "cache_errors": self.stats.cache_errors,
                "hit_rate": self.stats.hit_rate,
                "tokens_saved": self.stats.tokens_saved,
                "active_cache_entries": len(self.cache_entries),
                "cache_entries": {
                    key: {
                        "cache_name": entry.cache_name,
                        "created_at": entry.created_at,
                        "ttl_hours": entry.ttl_hours,
                        "token_count": entry.token_count,
                        "hit_count": entry.hit_count,
                        "expired": entry.is_expired()
                    }
                    for key, entry in self.cache_entries.items()
                }
            }
    
    def cleanup_expired_entries(self):
        """Remove expired cache entries."""
        with self.lock:
            expired_keys = [
                key for key, entry in self.cache_entries.items() 
                if entry.is_expired()
            ]
            for key in expired_keys:
                del self.cache_entries[key]
                logger.debug(f"Cleaned up expired cache entry: {key}")
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def reset_stats(self):
        """Reset cache statistics."""
        with self.lock:
            self.stats = CacheStats()
            logger.info("Reset cache statistics")
    
    def print_summary(self):
        """Print a formatted summary of cache performance."""
        stats = self.get_stats()

        print(f"\n{'='*60}")
        print(f"GEMINI CACHE PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        print(f"Cache enabled: {stats['enabled']}")
        print(f"Total requests: {stats['total_requests']}")
        print(f"Cache hits: {stats['cache_hits']}")
        print(f"Cache misses: {stats['cache_misses']}")
        print(f"Cache errors: {stats['cache_errors']}")
        print(f"Hit rate: {stats['hit_rate']:.1%}")
        print(f"Tokens saved: {stats['tokens_saved']:,}")
        print(f"Active cache entries: {stats['active_cache_entries']}")

        if stats['cache_entries']:
            print(f"\nCache Entry Details:")
            for key, entry in stats['cache_entries'].items():
                status = "EXPIRED" if entry['expired'] else "ACTIVE"
                print(f"  {key}: {entry['hit_count']} hits, {entry['token_count']} tokens, {status}")

        print(f"{'='*60}")

# Global cache manager instance
cache_manager = GeminiCacheManager()
