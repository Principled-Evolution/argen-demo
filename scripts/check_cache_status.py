#!/usr/bin/env python3
"""
Quick script to check Gemini cache status and performance.
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from argen.utils.gemini_cache_manager import cache_manager
from argen.config import is_gemini_caching_enabled, get_gemini_caching_config

def main():
    print("="*60)
    print("GEMINI CACHE STATUS CHECK")
    print("="*60)
    
    print(f"Caching enabled in config: {is_gemini_caching_enabled()}")
    print(f"Cache manager enabled: {cache_manager.enabled}")
    
    # Print current cache statistics
    cache_manager.print_summary()
    
    # Get detailed stats
    stats = cache_manager.get_stats()
    
    if stats['total_requests'] > 0:
        print(f"\n🎯 CACHE PERFORMANCE:")
        print(f"   • Total requests: {stats['total_requests']}")
        print(f"   • Cache hits: {stats['cache_hits']}")
        print(f"   • Cache misses: {stats['cache_misses']}")
        print(f"   • Hit rate: {stats['hit_rate']:.1%}")
        print(f"   • Tokens saved: {stats['tokens_saved']:,}")
        
        if stats['tokens_saved'] > 0:
            print(f"\n💰 ESTIMATED COST SAVINGS:")
            # Rough estimate: $0.000125 per 1K input tokens for Gemini 2.0 Flash
            cost_per_1k_tokens = 0.000125
            estimated_savings = (stats['tokens_saved'] / 1000) * cost_per_1k_tokens
            print(f"   • Estimated cost savings: ${estimated_savings:.4f}")
    else:
        print(f"\n📊 No cache requests recorded yet.")
        print(f"   This could mean:")
        print(f"   • Caching is disabled")
        print(f"   • No evaluations have been run yet")
        config = get_gemini_caching_config()
        min_tokens = config.get("min_tokens_for_caching", 4096)
        print(f"   • System prompts are too short for caching (< {min_tokens} tokens)")

if __name__ == "__main__":
    main()
