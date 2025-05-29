"""
Centralized tracking for Gemini API calls with model-specific statistics.
"""

from threading import Lock
from typing import Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import logging

@dataclass
class APICallStats:
    """Statistics for API calls per model."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0

class GeminiAPITracker:
    """Thread-safe singleton for tracking Gemini API calls."""
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
            self.stats_by_model: Dict[str, APICallStats] = defaultdict(APICallStats)
            self.lock = Lock()
            self.logger = logging.getLogger(__name__)
            self._initialized = True
    
    def track_call(self, model_name: str, success: bool = True):
        """Track an API call for a specific model."""
        with self.lock:
            stats = self.stats_by_model[model_name]
            stats.total_calls += 1
            if success:
                stats.successful_calls += 1
            else:
                stats.failed_calls += 1
            
            self.logger.debug(f"Tracked {model_name} call: success={success}, total={stats.total_calls}")
    
    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """Get current statistics for all models."""
        with self.lock:
            return {
                model: {
                    "total_calls": stats.total_calls,
                    "successful_calls": stats.successful_calls,
                    "failed_calls": stats.failed_calls
                }
                for model, stats in self.stats_by_model.items()
            }
    
    def get_total_calls(self) -> int:
        """Get total calls across all models."""
        with self.lock:
            return sum(stats.total_calls for stats in self.stats_by_model.values())
    
    def get_total_successful_calls(self) -> int:
        """Get total successful calls across all models."""
        with self.lock:
            return sum(stats.successful_calls for stats in self.stats_by_model.values())
    
    def get_total_failed_calls(self) -> int:
        """Get total failed calls across all models."""
        with self.lock:
            return sum(stats.failed_calls for stats in self.stats_by_model.values())
    
    def get_overall_success_rate(self) -> float:
        """Get overall success rate as a percentage."""
        total = self.get_total_calls()
        if total == 0:
            return 0.0
        return (self.get_total_successful_calls() / total) * 100
    
    def reset(self):
        """Reset all counters."""
        with self.lock:
            self.stats_by_model.clear()
            self.logger.info("Reset all Gemini API call counters")
    
    def print_summary(self):
        """Print a formatted summary of API usage."""
        stats = self.get_stats()
        total = self.get_total_calls()
        overall_success_rate = self.get_overall_success_rate()
        
        print(f"\n{'='*60}")
        print(f"GEMINI API USAGE SUMMARY")
        print(f"{'='*60}")
        print(f"Total API calls: {total}")
        print(f"Overall success rate: {overall_success_rate:.1f}%")
        
        if stats:
            print(f"\nBreakdown by model:")
            for model, model_stats in stats.items():
                success_rate = (model_stats['successful_calls'] / model_stats['total_calls'] * 100) if model_stats['total_calls'] > 0 else 0
                print(f"  {model}:")
                print(f"    Total: {model_stats['total_calls']}")
                print(f"    Successful: {model_stats['successful_calls']}")
                print(f"    Failed: {model_stats['failed_calls']}")
                print(f"    Success rate: {success_rate:.1f}%")
        else:
            print("No API calls recorded.")
        print(f"{'='*60}")
