#!/usr/bin/env python3
import json
from pathlib import Path

print("Starting test...")

# Test paths
base_path = Path("../")
claude_path = base_path / ".." / "reward-policy-ablations-claude"
gemini_path = base_path / "reward-policy-ablations-gemini"

print(f"Claude path: {claude_path}")
print(f"Claude exists: {claude_path.exists()}")
print(f"Gemini path: {gemini_path}")
print(f"Gemini exists: {gemini_path.exists()}")

# Test file loading
claude_reward_file = claude_path / "eval_seed_3_108_ablation_reward_only_benchmarking_20250510_135534-cleanprep-hashprompt.json"
print(f"Claude reward file exists: {claude_reward_file.exists()}")

if claude_reward_file.exists():
    try:
        with open(claude_reward_file, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded Claude reward file")
        print(f"Combined score: {data['summary_metrics']['average_combined_score']}")
    except Exception as e:
        print(f"Error loading file: {e}")

print("Test complete.")
