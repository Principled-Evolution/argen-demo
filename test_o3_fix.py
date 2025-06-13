#!/usr/bin/env python3
"""
Test script to verify that the o3-mini model parameter fix works correctly.
This script tests that the OpenAI API calls use the correct parameter names
for different model types.
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, patch, MagicMock

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from argen.reward_functions.openai_rewards import (
    evaluate_ahimsa_with_openai,
    evaluate_dharma_with_openai,
    evaluate_helpfulness_with_openai
)


async def test_o3_model_parameter_handling():
    """Test that o3 models use max_completion_tokens instead of max_tokens."""
    
    # Mock the OpenAI client and response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"score": 0.8, "reasoning": "test"}'
    
    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_response
    
    # Test data
    test_prompt = "What is the weather like?"
    test_response = "I don't have access to current weather data."
    test_api_key = "test-key"
    
    print("Testing o3-mini model parameter handling...")
    
    # Test with o3-mini model
    with patch('argen.reward_functions.openai_rewards.AsyncOpenAI', return_value=mock_client):
        await evaluate_ahimsa_with_openai(
            test_prompt, 
            test_response, 
            test_api_key, 
            model_name="o3-mini"
        )
    
    # Check that the API was called with max_completion_tokens and no temperature
    call_args = mock_client.chat.completions.create.call_args
    assert 'max_completion_tokens' in call_args.kwargs, "o3-mini should use max_completion_tokens"
    assert 'max_tokens' not in call_args.kwargs, "o3-mini should not use max_tokens"
    assert 'temperature' not in call_args.kwargs, "o3-mini should not use temperature"
    print("✓ o3-mini correctly uses max_completion_tokens and excludes temperature")
    
    # Reset mock
    mock_client.reset_mock()
    
    # Test with gpt-4o-mini model
    with patch('argen.reward_functions.openai_rewards.AsyncOpenAI', return_value=mock_client):
        await evaluate_ahimsa_with_openai(
            test_prompt, 
            test_response, 
            test_api_key, 
            model_name="gpt-4o-mini"
        )
    
    # Check that the API was called with max_tokens and temperature
    call_args = mock_client.chat.completions.create.call_args
    assert 'max_tokens' in call_args.kwargs, "gpt-4o-mini should use max_tokens"
    assert 'max_completion_tokens' not in call_args.kwargs, "gpt-4o-mini should not use max_completion_tokens"
    assert 'temperature' in call_args.kwargs, "gpt-4o-mini should use temperature"
    print("✓ gpt-4o-mini correctly uses max_tokens and temperature")
    
    print("All tests passed! The o3-mini parameter fix is working correctly.")


async def test_all_evaluation_functions():
    """Test that all three evaluation functions handle o3 models correctly."""
    
    # Mock the OpenAI client and response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"score": 0.8, "reasoning": "test"}'
    
    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_response
    
    # Test data
    test_prompt = "What is the weather like?"
    test_response = "I don't have access to current weather data."
    test_api_key = "test-key"
    
    evaluation_functions = [
        ("Ahimsa", evaluate_ahimsa_with_openai),
        ("Dharma", evaluate_dharma_with_openai),
        ("Helpfulness", evaluate_helpfulness_with_openai)
    ]
    
    print("\nTesting all evaluation functions with o3 model...")
    
    for func_name, func in evaluation_functions:
        print(f"Testing {func_name} evaluation...")
        
        # Reset mock
        mock_client.reset_mock()
        
        with patch('argen.reward_functions.openai_rewards.AsyncOpenAI', return_value=mock_client):
            await func(
                test_prompt, 
                test_response, 
                test_api_key, 
                model_name="o3-mini"
            )
        
        # Check that the API was called with max_completion_tokens
        call_args = mock_client.chat.completions.create.call_args
        assert 'max_completion_tokens' in call_args.kwargs, f"{func_name} should use max_completion_tokens for o3-mini"
        assert 'max_tokens' not in call_args.kwargs, f"{func_name} should not use max_tokens for o3-mini"
        print(f"✓ {func_name} evaluation correctly uses max_completion_tokens for o3-mini")
    
    print("All evaluation functions handle o3 models correctly!")


if __name__ == "__main__":
    print("Running o3-mini parameter fix tests...")
    asyncio.run(test_o3_model_parameter_handling())
    asyncio.run(test_all_evaluation_functions())
    print("\n🎉 All tests completed successfully!")
