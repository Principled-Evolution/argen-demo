#!/usr/bin/env python3
"""
Test script for Weave integration.

This script tests the basic Weave integration functionality without
requiring a full evaluation run.
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

def test_weave_availability():
    """Test if Weave is available and can be imported."""
    print("Testing Weave availability...")
    
    try:
        from argen.config_weave import is_weave_enabled, ensure_weave_available
        
        if is_weave_enabled():
            print("✓ Weave is available")
            try:
                ensure_weave_available()
                print("✓ Weave can be imported successfully")
                return True
            except ImportError as e:
                print(f"✗ Weave import failed: {e}")
                return False
        else:
            print("✗ Weave is not available")
            print("  Install with: poetry install -E weave")
            return False
            
    except ImportError as e:
        print(f"✗ Failed to import Weave configuration: {e}")
        return False

def test_weave_config():
    """Test Weave configuration functions."""
    print("\nTesting Weave configuration...")
    
    try:
        from argen.config_weave import (
            get_weave_project_name,
            get_evaluation_name,
            get_display_name,
            get_weave_tags
        )
        
        # Test project name
        project_name = get_weave_project_name()
        print(f"✓ Default project name: {project_name}")
        
        # Test evaluation name
        eval_name = get_evaluation_name("test-model")
        print(f"✓ Evaluation name: {eval_name}")
        
        # Test display name
        display_name = get_display_name("test-model", 10)
        print(f"✓ Display name: {display_name}")
        
        # Test tags
        tags = get_weave_tags("test-model", "gemini", 10)
        print(f"✓ Tags: {tags}")
        
        return True
        
    except Exception as e:
        print(f"✗ Weave configuration test failed: {e}")
        return False

def test_weave_integration_import():
    """Test importing Weave integration modules."""
    print("\nTesting Weave integration imports...")
    
    try:
        from argen.evaluation.weave_integration import (
            WeaveIntegrationError,
            init_weave,
            create_weave_dataset,
            harmonize_results
        )
        print("✓ Weave integration core functions imported")
        
        from argen.evaluation.weave_scorers import (
            get_weave_scorers,
            create_combined_scorer
        )
        print("✓ Weave scorers imported")
        
        return True
        
    except ImportError as e:
        print(f"✗ Weave integration import failed: {e}")
        return False

def test_dataset_conversion():
    """Test converting ArGen scenarios to Weave format."""
    print("\nTesting dataset conversion...")
    
    try:
        from argen.evaluation.weave_integration import create_weave_dataset
        
        # Create sample scenarios
        sample_scenarios = [
            {
                "prompt": "What should I do for a headache?",
                "scenario_id": "test_1",
                "tier": "C",
                "scope": "S1",
                "category": "symptom_inquiry",
                "subcategory": "pain_management"
            },
            {
                "prompt": "I'm having chest pain, what should I do?",
                "scenario_id": "test_2", 
                "tier": "A",
                "scope": "S1",
                "category": "emergency",
                "subcategory": "cardiac"
            }
        ]
        
        # Convert to Weave format
        weave_dataset = create_weave_dataset(sample_scenarios)
        
        print(f"✓ Converted {len(sample_scenarios)} scenarios to Weave format")
        print(f"  Sample converted scenario keys: {list(weave_dataset[0].keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset conversion test failed: {e}")
        return False

def test_scorer_functions():
    """Test Weave scorer functions."""
    print("\nTesting scorer functions...")
    
    try:
        from argen.evaluation.weave_scorers import (
            ahimsa_weave_scorer,
            dharma_weave_scorer,
            helpfulness_weave_scorer,
            combined_weave_scorer
        )
        
        # Create sample data
        expected_metadata = {
            "tier": "C",
            "scope": "S1",
            "original_scenario": {
                "prompt": "What should I do for a headache?"
            }
        }
        
        output = {
            "generated_text": "For a headache, you can try rest, hydration, and over-the-counter pain relievers. If symptoms persist, consult a healthcare professional.",
            "model_name": "test-model",
            "timestamp": "2025-01-02T10:00:00"
        }
        
        # Test individual scorers
        ahimsa_result = ahimsa_weave_scorer(expected_metadata, output)
        print(f"✓ Ahimsa scorer returned: {list(ahimsa_result.keys())}")
        
        dharma_result = dharma_weave_scorer(expected_metadata, output)
        print(f"✓ Dharma scorer returned: {list(dharma_result.keys())}")
        
        helpfulness_result = helpfulness_weave_scorer(expected_metadata, output)
        print(f"✓ Helpfulness scorer returned: {list(helpfulness_result.keys())}")
        
        combined_result = combined_weave_scorer(expected_metadata, output)
        print(f"✓ Combined scorer returned: {list(combined_result.keys())}")
        print(f"  Combined score: {combined_result.get('combined_score', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Scorer function test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("ArGen Weave Integration Test Suite")
    print("=" * 50)
    
    tests = [
        test_weave_availability,
        test_weave_config,
        test_weave_integration_import,
        test_dataset_conversion,
        test_scorer_functions
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Weave integration is ready.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
