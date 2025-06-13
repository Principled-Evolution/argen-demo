#!/usr/bin/env python3
"""
Test script to verify JSON parsing robustness improvements for OpenAI and Anthropic evaluators.

This script tests the new robust JSON extraction functionality with various problematic
JSON responses that were causing failures in the original implementation.
"""

import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from argen.utils.json_extractor import extract_json_from_response


def test_json_extraction():
    """Test the robust JSON extraction with various problematic inputs."""
    
    print("Testing JSON extraction robustness...")
    print("=" * 50)
    
    # Test cases based on the error patterns mentioned in issue #26
    test_cases = [
        {
            "name": "Unterminated string",
            "content": '''{\n  "domain_adherence_score": 5,\n  "out_of_domain_advice": true,\n  "response_scope": "S3",\n  "scope_risk": "high",\n  "severity": "major",\n  "reasoning": "Valid mental health advice provided; however, financial''',
            "expected_success": False  # This should fail gracefully
        },
        {
            "name": "Empty content",
            "content": "",
            "expected_success": False
        },
        {
            "name": "Malformed JSON - missing property name",
            "content": '''{\n  "domain_adherence_score": 0.8,\n  "out_of_domain_advice": false,\n  "response_scope": "S1",\n  "scope_risk": "low",\n  : "minor",\n  "reasoning": "Good healthcare advice"\n}''',
            "expected_success": False
        },
        {
            "name": "Valid JSON in markdown",
            "content": '''```json\n{\n  "domain_adherence_score": 0.8,\n  "out_of_domain_advice": false,\n  "response_scope": "S1",\n  "scope_risk": "low",\n  "severity": "minor",\n  "reasoning": "Good healthcare advice"\n}\n```''',
            "expected_success": True
        },
        {
            "name": "JSON with trailing comma",
            "content": '''{\n  "domain_adherence_score": 0.8,\n  "out_of_domain_advice": false,\n  "response_scope": "S1",\n  "scope_risk": "low",\n  "severity": "minor",\n  "reasoning": "Good healthcare advice",\n}''',
            "expected_success": True  # Should be fixed by preprocessing
        },
        {
            "name": "JSON with yes/no instead of true/false",
            "content": '''{\n  "domain_adherence_score": 0.8,\n  "out_of_domain_advice": no,\n  "response_scope": "S1",\n  "scope_risk": "low",\n  "severity": "minor",\n  "reasoning": "Good healthcare advice"\n}''',
            "expected_success": True  # Should be fixed by preprocessing
        },
        {
            "name": "Valid JSON without markdown",
            "content": '''{\n  "harm_avoidance_score": 0.9,\n  "safety_context_score": 0.8,\n  "has_medical_disclaimer": true,\n  "has_professional_referral": false,\n  "reasoning": "Safe medical advice provided"\n}''',
            "expected_success": True
        },
        {
            "name": "Truncated JSON with valid fields (real example)",
            "content": '''{\n  "domain_adherence_score": 0.5,\n  "out_of_domain_advice": true,\n  "response_scope": "S2",\n  "scope_risk": "moderate",\n  "severity": "minor",\n  "reasoning": "Mixes appropriate medical anxiety advice with non-medical''',
            "expected_success": True  # Should extract valid fields
        },
        {
            "name": "Truncated JSON mid-string (real example)",
            "content": '''{\n  "clarity_score": 0.9,\n  "completeness_score": 0.9,\n  "relevance_score": 1.0,\n  "empathy_score": 0.75,\n  "''',
            "expected_success": True  # Should extract the numeric fields
        },
        {
            "name": "Truncated JSON with incomplete reasoning",
            "content": '''{\n  "harm_avoidance_score": 0.9,\n  "safety_context_score": 0.9,\n  "has_medical_disclaimer": true,\n  "has_professional_referral": true,\n  "referral_strength": "urgent",\n  "severity": "minor",\n  "reasoning": "The response offers clear advice to seek professional evaluation''',
            "expected_success": True  # Should extract all the complete fields
        }
    ]
    
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print("-" * 30)
        
        try:
            result, success = extract_json_from_response(test_case['content'], "test")
            
            if success == test_case['expected_success']:
                if success:
                    print(f"✅ PASS - Successfully extracted JSON: {result}")
                else:
                    print(f"✅ PASS - Correctly failed to extract JSON (as expected)")
                passed += 1
            else:
                print(f"❌ FAIL - Expected success={test_case['expected_success']}, got success={success}")
                if result:
                    print(f"   Result: {result}")
                failed += 1
                
        except Exception as e:
            print(f"❌ FAIL - Exception occurred: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return failed == 0


async def test_openai_evaluator():
    """Test OpenAI evaluator with mock data (if API key available)."""
    
    print("\nTesting OpenAI evaluator integration...")
    print("=" * 50)
    
    # Check if OpenAI API key is available
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        print("⚠️  SKIP - No OPENAI_API_KEY environment variable found")
        return True
    
    try:
        from argen.reward_functions.openai_rewards import evaluate_ahimsa_with_openai
        
        # Test with a simple prompt
        test_prompt = "What should I do if I have a headache?"
        test_response = "You should rest, drink water, and consider taking an over-the-counter pain reliever like ibuprofen. If the headache persists or worsens, please consult a healthcare professional."
        
        print(f"Testing with prompt: {test_prompt}")
        print(f"Testing with response: {test_response}")
        
        result = await evaluate_ahimsa_with_openai(
            original_prompt=test_prompt,
            model_response=test_response,
            openai_api_key=openai_api_key,
            model_name="gpt-4o-mini"  # Use a reliable model
        )
        
        if "error" not in result:
            print("✅ PASS - OpenAI evaluator completed successfully")
            print(f"   Ahimsa score: {result.get('ahimsa_score', 'N/A')}")
            return True
        else:
            print(f"❌ FAIL - OpenAI evaluator returned error: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ FAIL - Exception in OpenAI evaluator test: {e}")
        return False


async def test_anthropic_evaluator():
    """Test Anthropic evaluator with mock data (if API key available)."""
    
    print("\nTesting Anthropic evaluator integration...")
    print("=" * 50)
    
    # Check if Anthropic API key is available
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    if not anthropic_api_key:
        print("⚠️  SKIP - No ANTHROPIC_API_KEY environment variable found")
        return True
    
    try:
        from argen.reward_functions.anthropic_rewards import evaluate_ahimsa_with_anthropic
        
        # Test with a simple prompt
        test_prompt = "What should I do if I have a headache?"
        test_response = "You should rest, drink water, and consider taking an over-the-counter pain reliever like ibuprofen. If the headache persists or worsens, please consult a healthcare professional."
        
        print(f"Testing with prompt: {test_prompt}")
        print(f"Testing with response: {test_response}")
        
        result = await evaluate_ahimsa_with_anthropic(
            original_prompt=test_prompt,
            model_response=test_response,
            anthropic_api_key=anthropic_api_key,
            model_name="claude-3-5-haiku-20241022"  # Use a reliable model
        )
        
        if "error" not in result:
            print("✅ PASS - Anthropic evaluator completed successfully")
            print(f"   Ahimsa score: {result.get('ahimsa_score', 'N/A')}")
            return True
        else:
            print(f"❌ FAIL - Anthropic evaluator returned error: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ FAIL - Exception in Anthropic evaluator test: {e}")
        return False


async def main():
    """Run all tests."""
    
    print("JSON Parsing Robustness Test Suite")
    print("=" * 50)
    print("Testing improvements for Issue #26")
    print()
    
    # Test 1: JSON extraction robustness
    json_test_passed = test_json_extraction()
    
    # Test 2: OpenAI evaluator integration
    openai_test_passed = await test_openai_evaluator()
    
    # Test 3: Anthropic evaluator integration
    anthropic_test_passed = await test_anthropic_evaluator()
    
    # Summary
    print("\n" + "=" * 50)
    print("FINAL TEST SUMMARY")
    print("=" * 50)
    
    all_passed = json_test_passed and openai_test_passed and anthropic_test_passed
    
    print(f"JSON Extraction Tests: {'✅ PASS' if json_test_passed else '❌ FAIL'}")
    print(f"OpenAI Integration Tests: {'✅ PASS' if openai_test_passed else '❌ FAIL'}")
    print(f"Anthropic Integration Tests: {'✅ PASS' if anthropic_test_passed else '❌ FAIL'}")
    print()
    print(f"Overall Result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
