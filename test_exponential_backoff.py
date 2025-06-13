#!/usr/bin/env python3
"""
Test script to verify exponential backoff implementation in OpenAI rewards module.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'argen'))

from argen.reward_functions.openai_rewards import calculate_exponential_backoff_delay

def test_exponential_backoff():
    """Test the exponential backoff delay calculation."""
    print("Testing exponential backoff delay calculation:")
    print("=" * 50)
    
    # Test with default parameters
    print("Default parameters (base_delay=2.0, max_delay=60.0, jitter=True):")
    for attempt in range(5):
        delay = calculate_exponential_backoff_delay(attempt)
        expected_base = 2.0 * (2 ** attempt)
        expected_capped = min(expected_base, 60.0)
        print(f"  Attempt {attempt}: {delay:.2f}s (expected base: {expected_base:.2f}s, capped: {expected_capped:.2f}s)")
    
    print()
    
    # Test without jitter
    print("Without jitter (jitter=False):")
    for attempt in range(5):
        delay = calculate_exponential_backoff_delay(attempt, jitter=False)
        expected = min(2.0 * (2 ** attempt), 60.0)
        print(f"  Attempt {attempt}: {delay:.2f}s (expected: {expected:.2f}s)")
    
    print()
    
    # Test with custom parameters
    print("Custom parameters (base_delay=1.0, max_delay=30.0):")
    for attempt in range(5):
        delay = calculate_exponential_backoff_delay(attempt, base_delay=1.0, max_delay=30.0, jitter=False)
        expected = min(1.0 * (2 ** attempt), 30.0)
        print(f"  Attempt {attempt}: {delay:.2f}s (expected: {expected:.2f}s)")

if __name__ == "__main__":
    test_exponential_backoff()
