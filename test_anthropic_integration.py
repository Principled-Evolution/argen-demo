#!/usr/bin/env python3
"""
Simple test script for Anthropic integration.
This script tests the basic functionality without requiring an actual API key.
"""

import asyncio
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from argen.reward_functions.anthropic_rewards import (
    evaluate_ahimsa_with_anthropic,
    evaluate_dharma_with_anthropic,
    evaluate_helpfulness_with_anthropic,
    batch_evaluate_with_anthropic,
    DEFAULT_EVAL_RESPONSE
)

async def test_anthropic_integration():
    """Test Anthropic integration with mock data."""
    print("Testing Anthropic integration...")
    
    # Test data
    test_prompt = "What should I do if I have a headache?"
    test_response = "For a headache, you can try rest, hydration, and over-the-counter pain relievers like ibuprofen. If headaches persist or worsen, consult a healthcare professional."
    
    # Test without API key (should return error response)
    print("\n1. Testing Ahimsa evaluation without API key...")
    result = await evaluate_ahimsa_with_anthropic(test_prompt, test_response, None)
    print(f"Result: {result.get('error', 'No error')}")
    assert result.get('error') == 'API Key Missing', "Should return API Key Missing error"
    
    print("\n2. Testing Dharma evaluation without API key...")
    result = await evaluate_dharma_with_anthropic(test_prompt, test_response, None)
    print(f"Result: {result.get('error', 'No error')}")
    assert result.get('error') == 'API Key Missing', "Should return API Key Missing error"
    
    print("\n3. Testing Helpfulness evaluation without API key...")
    result = await evaluate_helpfulness_with_anthropic(test_prompt, test_response, None)
    print(f"Result: {result.get('error', 'No error')}")
    assert result.get('error') == 'API Key Missing', "Should return API Key Missing error"
    
    print("\n4. Testing batch evaluation without API key...")
    results = await batch_evaluate_with_anthropic([test_prompt], [test_response], None)
    print(f"Batch result count: {len(results)}")
    assert len(results) == 1, "Should return one result"
    assert results[0].get('error') == 'Evaluation failed', "Should return default error response"
    
    print("\n5. Testing default response structure...")
    required_keys = [
        'ahimsa_score', 'dharma_score', 'helpfulness_score', 'combined_score',
        'ahimsa_violation', 'dharma_violation', 'helpfulness_violation',
        'harm_avoidance_score', 'safety_consideration_score', 'domain_adherence_score',
        'clarity_score', 'relevance_score', 'completeness_score', 'empathy_score',
        'has_medical_disclaimer', 'has_professional_referral', 'out_of_domain_advice',
        'response_scope', 'scope_risk', 'scope_penalty_factor'
    ]
    
    for key in required_keys:
        assert key in DEFAULT_EVAL_RESPONSE, f"Missing key in DEFAULT_EVAL_RESPONSE: {key}"
    
    print(f"✅ All required keys present in DEFAULT_EVAL_RESPONSE: {len(required_keys)} keys")
    
    print("\n6. Testing model name parameter...")
    result = await evaluate_ahimsa_with_anthropic(
        test_prompt, test_response, None, model_name="claude-3-5-sonnet"
    )
    assert result.get('error') == 'API Key Missing', "Should still return API Key Missing error"
    
    print("\n✅ All tests passed! Anthropic integration is working correctly.")
    print("\nTo test with actual API calls, set the ANTHROPIC_API_KEY environment variable.")

if __name__ == "__main__":
    asyncio.run(test_anthropic_integration())
