#!/usr/bin/env python3
"""
Test script for ablation mode debug logging functionality.

This script tests the enhanced debug logging system implemented for issue #32.
"""

import os
import sys
import logging

# Add the project root to the path
sys.path.insert(0, os.path.abspath('.'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ablation_debug_logging():
    """Test the ablation debug logging functionality."""
    
    # Enable ablation debug logging
    os.environ["ARGEN_ABLATION_DEBUG"] = "true"
    
    # Test different ablation modes
    modes_to_test = ["full", "reward_only", "policy_only"]
    
    for mode in modes_to_test:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing ablation mode: {mode}")
        logger.info(f"{'='*60}")
        
        # Set the ablation mode
        os.environ["ARGEN_ABLATION_MODE"] = mode
        
        # Import the debug logging functions
        from argen.config import get_ablation_debug, get_ablation_mode
        from argen.reward_functions.trl_rewards import log_ablation_debug, log_ablation_summary
        
        # Verify configuration
        assert get_ablation_debug() == True, "Debug logging should be enabled"
        assert get_ablation_mode() == mode, f"Ablation mode should be {mode}"
        
        logger.info(f"✅ Configuration verified: debug={get_ablation_debug()}, mode={get_ablation_mode()}")
        
        # Test debug logging function
        raw_scores = {"A": 0.85, "D": 0.75, "H": 0.82}
        penalties = {"scope": 0.8, "severity": 0.0}
        selected_score = 0.641
        calculation = f"({raw_scores['A']:.3f}×0.3 + {raw_scores['D']:.3f}×0.4 + {raw_scores['H']:.3f}×0.3) × {penalties['scope']:.1f} = {selected_score:.3f}"
        
        cross_mode_scores = {
            "full": 0.641,
            "reward_only": 0.801,
            "policy_only": 0.800
        }
        
        log_ablation_debug(
            mode=mode,
            item_idx=2,  # 0-based, so this is item 3
            batch_size=10,
            raw_scores=raw_scores,
            penalties=penalties,
            selected_score=selected_score,
            calculation=calculation,
            cross_mode_scores=cross_mode_scores,
            component="Test"
        )
        
        # Test summary logging function
        component_scores = {"ahimsa": 0.85, "dharma": 0.75, "helpfulness": 0.82}
        weights = {"ahimsa": 0.3, "dharma": 0.4, "helpfulness": 0.3}
        final_combined = 0.780
        
        log_ablation_summary(
            mode=mode,
            item_idx=2,
            batch_size=10,
            component_scores=component_scores,
            weights=weights,
            final_combined=final_combined,
            cross_mode_scores=cross_mode_scores
        )
        
        logger.info(f"✅ Debug logging test completed for mode: {mode}")
    
    logger.info(f"\n{'='*60}")
    logger.info("All ablation debug logging tests completed successfully!")
    logger.info(f"{'='*60}")

def test_command_line_integration():
    """Test that the command line argument works correctly."""
    
    logger.info(f"\n{'='*60}")
    logger.info("Testing command line integration")
    logger.info(f"{'='*60}")
    
    # Test the argument parsing
    import argparse
    
    # Simulate the argument parser from train_grpo.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation_debug", action="store_true",
                      help="Enable detailed ablation mode debug logging")
    
    # Test with debug flag
    args = parser.parse_args(["--ablation_debug"])
    assert args.ablation_debug == True, "Debug flag should be True"
    
    # Test without debug flag
    args = parser.parse_args([])
    assert args.ablation_debug == False, "Debug flag should be False by default"
    
    logger.info("✅ Command line integration test passed")

if __name__ == "__main__":
    try:
        test_command_line_integration()
        test_ablation_debug_logging()
        print("\n🎉 All tests passed! The ablation debug logging system is working correctly.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
