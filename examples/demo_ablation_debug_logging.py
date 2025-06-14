#!/usr/bin/env python3
"""
Demo script showing ablation mode debug logging in action.

This script demonstrates the enhanced debug logging system implemented for GitHub issue #32.
It shows how to use the --ablation_debug flag and what the output looks like for different
ablation modes.

Usage:
    # Test with combined rewards mode (default)
    python examples/demo_ablation_debug_logging.py --ablation_mode full --ablation_debug
    
    # Test with separate rewards mode
    python examples/demo_ablation_debug_logging.py --ablation_mode reward_only --ablation_debug --use_separate_rewards
    
    # Test without debug logging (normal operation)
    python examples/demo_ablation_debug_logging.py --ablation_mode policy_only
"""

import os
import sys
import logging
import argparse

# Add the project root to the path
sys.path.insert(0, os.path.abspath('.'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_mock_data():
    """Create mock training data for demonstration."""
    # Create proper tier hashes for the prompts
    from argen.utils.data_integrity import calculate_prompt_hash, create_compound_tier

    prompts = [
        "What should I do if I have chest pain?",
        "How can I treat my diabetes at home?",
        "What are the side effects of aspirin?"
    ]

    # Generate proper tier hashes
    tiers = []
    for i, prompt in enumerate(prompts):
        tier_letter = ["A", "B", "C"][i]
        prompt_hash = calculate_prompt_hash(prompt)
        compound_tier = create_compound_tier(tier_letter, prompt_hash)
        tiers.append(compound_tier)

    return {
        "prompts": prompts,
        "completions": [
            "If you're experiencing chest pain, it's important to seek immediate medical attention. Call emergency services or go to the nearest emergency room.",
            "For diabetes management, it's crucial to work with your healthcare provider. Monitor your blood sugar regularly and follow your prescribed treatment plan.",
            "Aspirin can have side effects including stomach irritation, bleeding, and allergic reactions. Consult your doctor before taking aspirin regularly."
        ],
        "tiers": tiers,
        "scopes": ["S1", "S0", "S2"]  # Mock scope data
    }

def demo_combined_rewards_mode(ablation_mode: str, debug_enabled: bool):
    """Demonstrate debug logging with combined rewards mode."""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"DEMO: Combined Rewards Mode - {ablation_mode.upper()}")
    logger.info(f"Debug Logging: {'ENABLED' if debug_enabled else 'DISABLED'}")
    logger.info(f"{'='*80}")
    
    # Set environment variables
    os.environ["ARGEN_ABLATION_MODE"] = ablation_mode
    if debug_enabled:
        os.environ["ARGEN_ABLATION_DEBUG"] = "true"
    else:
        os.environ.pop("ARGEN_ABLATION_DEBUG", None)
    
    # Import after setting environment variables
    from argen.reward_functions.trl_rewards import combined_reward_trl
    
    # Create mock data
    data = create_mock_data()
    
    logger.info(f"Processing {len(data['prompts'])} prompt-completion pairs...")
    
    # Call the combined reward function
    try:
        rewards = combined_reward_trl(
            prompts=data["prompts"],
            completions=data["completions"],
            tier=data["tiers"],
            scope=data["scopes"]
        )
        
        logger.info(f"✅ Combined rewards calculated: {rewards}")
        logger.info(f"   Average reward: {sum(rewards)/len(rewards):.3f}")
        logger.info(f"   Reward range: [{min(rewards):.3f}, {max(rewards):.3f}]")
        
    except Exception as e:
        logger.error(f"❌ Error in combined rewards mode: {e}")
        import traceback
        traceback.print_exc()

def demo_separate_rewards_mode(ablation_mode: str, debug_enabled: bool):
    """Demonstrate debug logging with separate rewards mode."""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"DEMO: Separate Rewards Mode - {ablation_mode.upper()}")
    logger.info(f"Debug Logging: {'ENABLED' if debug_enabled else 'DISABLED'}")
    logger.info(f"{'='*80}")
    
    # Set environment variables
    os.environ["ARGEN_ABLATION_MODE"] = ablation_mode
    if debug_enabled:
        os.environ["ARGEN_ABLATION_DEBUG"] = "true"
    else:
        os.environ.pop("ARGEN_ABLATION_DEBUG", None)
    
    # Enable separate rewards mode
    from argen.reward_functions.shared_evaluation_coordinator import set_separate_rewards_mode
    set_separate_rewards_mode(True)
    
    # Import reward functions
    from argen.reward_functions.trl_rewards import ahimsa_reward_trl, dharma_reward_trl, helpfulness_reward_trl
    from argen.config import REWARD_WEIGHTS
    
    # Create mock data
    data = create_mock_data()
    
    logger.info(f"Processing {len(data['prompts'])} prompt-completion pairs with separate reward functions...")
    
    try:
        # Call individual reward functions
        ahimsa_scores = ahimsa_reward_trl(
            prompts=data["prompts"],
            completions=data["completions"],
            tier=data["tiers"],
            scope=data["scopes"]
        )
        
        dharma_scores = dharma_reward_trl(
            prompts=data["prompts"],
            completions=data["completions"],
            tier=data["tiers"],
            scope=data["scopes"]
        )
        
        helpfulness_scores = helpfulness_reward_trl(
            prompts=data["prompts"],
            completions=data["completions"],
            tier=data["tiers"],
            scope=data["scopes"]
        )
        
        # Calculate combined scores manually (as GRPO would do)
        combined_scores = []
        for a, d, h in zip(ahimsa_scores, dharma_scores, helpfulness_scores):
            combined = (a * REWARD_WEIGHTS["ahimsa"] + 
                       d * REWARD_WEIGHTS["dharma"] + 
                       h * REWARD_WEIGHTS["helpfulness"])
            combined_scores.append(combined)
        
        logger.info(f"✅ Individual scores calculated:")
        logger.info(f"   Ahimsa: {ahimsa_scores}")
        logger.info(f"   Dharma: {dharma_scores}")
        logger.info(f"   Helpfulness: {helpfulness_scores}")
        logger.info(f"   Combined: {combined_scores}")
        logger.info(f"   Average combined: {sum(combined_scores)/len(combined_scores):.3f}")
        
    except Exception as e:
        logger.error(f"❌ Error in separate rewards mode: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Reset separate rewards mode
        set_separate_rewards_mode(False)

def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Demo ablation mode debug logging")
    parser.add_argument("--ablation_mode", type=str, choices=["full", "reward_only", "policy_only"], 
                       default="full", help="Ablation study mode")
    parser.add_argument("--ablation_debug", action="store_true",
                       help="Enable detailed ablation mode debug logging")
    parser.add_argument("--use_separate_rewards", action="store_true",
                       help="Use separate reward functions instead of combined")
    
    args = parser.parse_args()
    
    logger.info("🧪 Ablation Mode Debug Logging Demo")
    logger.info(f"Mode: {args.ablation_mode}")
    logger.info(f"Debug: {'Enabled' if args.ablation_debug else 'Disabled'}")
    logger.info(f"Rewards: {'Separate' if args.use_separate_rewards else 'Combined'}")
    
    if args.use_separate_rewards:
        demo_separate_rewards_mode(args.ablation_mode, args.ablation_debug)
    else:
        demo_combined_rewards_mode(args.ablation_mode, args.ablation_debug)
    
    logger.info(f"\n{'='*80}")
    logger.info("Demo completed! 🎉")
    logger.info("To see debug logging in action, run with --ablation_debug flag")
    logger.info("Try different ablation modes: --ablation_mode full|reward_only|policy_only")
    logger.info(f"{'='*80}")

if __name__ == "__main__":
    main()
