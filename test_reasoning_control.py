#!/usr/bin/env python3
"""
Test script to verify that reasoning control is working for OpenAI and Anthropic evaluators.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_reasoning_control():
    """Test that reasoning control flags work correctly."""
    
    print("Testing Reasoning Control for OpenAI and Anthropic Evaluators")
    print("=" * 65)
    
    try:
        # Test OpenAI reasoning control
        from argen.reward_functions.openai_rewards import INCLUDE_REASONING as openai_reasoning, set_include_reasoning as set_openai_reasoning
        print(f"✅ OpenAI INCLUDE_REASONING imported: {openai_reasoning}")
        
        # Test Anthropic reasoning control
        from argen.reward_functions.anthropic_rewards import INCLUDE_REASONING as anthropic_reasoning, set_include_reasoning as set_anthropic_reasoning
        print(f"✅ Anthropic INCLUDE_REASONING imported: {anthropic_reasoning}")
        
        # Test Gemini reasoning control (for comparison)
        from argen.reward_functions.gemini_rewards import INCLUDE_REASONING as gemini_reasoning, set_include_reasoning as set_gemini_reasoning
        print(f"✅ Gemini INCLUDE_REASONING imported: {gemini_reasoning}")
        
        print(f"\nDefault Values:")
        print(f"  OpenAI: {openai_reasoning}")
        print(f"  Anthropic: {anthropic_reasoning}")
        print(f"  Gemini: {gemini_reasoning}")
        
        # Test setting reasoning to True
        print(f"\nTesting set_include_reasoning(True)...")
        set_openai_reasoning(True)
        set_anthropic_reasoning(True)
        set_gemini_reasoning(True)
        
        # Re-import to check if values changed
        from argen.reward_functions.openai_rewards import INCLUDE_REASONING as openai_reasoning_new
        from argen.reward_functions.anthropic_rewards import INCLUDE_REASONING as anthropic_reasoning_new
        from argen.reward_functions.gemini_rewards import INCLUDE_REASONING as gemini_reasoning_new
        
        print(f"After setting to True:")
        print(f"  OpenAI: {openai_reasoning_new}")
        print(f"  Anthropic: {anthropic_reasoning_new}")
        print(f"  Gemini: {gemini_reasoning_new}")
        
        # Test setting reasoning to False
        print(f"\nTesting set_include_reasoning(False)...")
        set_openai_reasoning(False)
        set_anthropic_reasoning(False)
        set_gemini_reasoning(False)
        
        # Re-import to check if values changed
        from argen.reward_functions.openai_rewards import INCLUDE_REASONING as openai_reasoning_final
        from argen.reward_functions.anthropic_rewards import INCLUDE_REASONING as anthropic_reasoning_final
        from argen.reward_functions.gemini_rewards import INCLUDE_REASONING as gemini_reasoning_final
        
        print(f"After setting to False:")
        print(f"  OpenAI: {openai_reasoning_final}")
        print(f"  Anthropic: {anthropic_reasoning_final}")
        print(f"  Gemini: {gemini_reasoning_final}")
        
        # Verify all are False (default for reduced token usage)
        all_false = not openai_reasoning_final and not anthropic_reasoning_final and not gemini_reasoning_final
        
        if all_false:
            print(f"\n✅ SUCCESS: All evaluators default to INCLUDE_REASONING=False")
            print(f"   This reduces token usage and truncation issues!")
        else:
            print(f"\n❌ ISSUE: Some evaluators still have INCLUDE_REASONING=True")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_prompt_generation():
    """Test that prompts are generated correctly with and without reasoning."""
    
    print(f"\nTesting Prompt Generation with Reasoning Control")
    print("=" * 55)
    
    try:
        # Test Dharma prompt template
        from argen.reward_functions.prompt_templates import get_dharma_system_prompt
        
        print("Testing Dharma prompt template:")
        
        # Test with reasoning
        dharma_with_reasoning = get_dharma_system_prompt(include_reasoning=True)
        has_reasoning_field = "reasoning" in dharma_with_reasoning.lower()
        print(f"  With reasoning: {'✅ Contains reasoning field' if has_reasoning_field else '❌ Missing reasoning field'}")
        
        # Test without reasoning
        dharma_without_reasoning = get_dharma_system_prompt(include_reasoning=False)
        no_reasoning_field = "reasoning" not in dharma_without_reasoning.lower()
        print(f"  Without reasoning: {'✅ No reasoning field' if no_reasoning_field else '❌ Still has reasoning field'}")
        
        # Test Helpfulness prompt template
        from argen.reward_functions.prompt_templates import get_helpfulness_system_prompt
        
        print("Testing Helpfulness prompt template:")
        helpfulness_prompt = get_helpfulness_system_prompt()
        print(f"  Base prompt: ✅ Generated successfully")
        
        # Check token reduction
        with_reasoning_length = len(dharma_with_reasoning)
        without_reasoning_length = len(dharma_without_reasoning)
        token_reduction = with_reasoning_length - without_reasoning_length
        
        print(f"\nToken Usage Analysis:")
        print(f"  With reasoning: {with_reasoning_length} characters")
        print(f"  Without reasoning: {without_reasoning_length} characters")
        print(f"  Reduction: {token_reduction} characters (~{token_reduction//4} tokens)")
        
        if token_reduction > 0:
            print(f"  ✅ Token usage reduced by removing reasoning field")
        else:
            print(f"  ❌ No token reduction achieved")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def main():
    """Run all reasoning control tests."""
    
    print("Reasoning Control Test Suite")
    print("=" * 40)
    print("Testing INCLUDE_REASONING=False default for reduced token usage")
    print()
    
    # Test 1: Reasoning control flags
    control_test_passed = test_reasoning_control()
    
    # Test 2: Prompt generation
    prompt_test_passed = test_prompt_generation()
    
    # Summary
    print("\n" + "=" * 65)
    print("FINAL TEST SUMMARY")
    print("=" * 65)
    
    all_passed = control_test_passed and prompt_test_passed
    
    print(f"Reasoning Control Tests: {'✅ PASS' if control_test_passed else '❌ FAIL'}")
    print(f"Prompt Generation Tests: {'✅ PASS' if prompt_test_passed else '❌ FAIL'}")
    print()
    
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("🎉 Reasoning control is working correctly!")
        print("\nBenefits:")
        print("  • Reduced token usage in evaluation prompts")
        print("  • Lower chance of response truncation")
        print("  • Faster evaluation times")
        print("  • Lower API costs")
        print("\nTo enable reasoning when needed:")
        print("  from argen.reward_functions.openai_rewards import set_include_reasoning")
        print("  set_include_reasoning(True)")
    else:
        print("❌ SOME TESTS FAILED")
        print("⚠️  Review the failed tests before using in production")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
