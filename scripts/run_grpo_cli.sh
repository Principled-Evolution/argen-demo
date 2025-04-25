#!/bin/bash
# Script to run a GRPO job using the Predibase CLI

# Set variables
MODEL="llama-3-2-1b-instruct"
DATASET="argen_combined_dataset"
REPO="argen-gemini-opa"
DESCRIPTION="ArGen GRPO fine-tuning with Gemini-OPA reward functions"

# Create a temporary JSON file for the reward functions
cat > reward_functions.json << EOL
{
  "code": "def gemini_opa_ahimsa_reward(prompt: str, completion: str, example: dict) -> float:\n    # Simple reward function that always returns 1.0\n    print(f\"Processing prompt: {prompt[:50]}...\")\n    print(f\"Completion: {completion[:50]}...\")\n    return 1.0\n",
  "functions": {
    "ahimsa": "gemini_opa_ahimsa_reward"
  },
  "runtime": {
    "packages": ["google-generativeai", "python-dotenv"],
    "env_vars": {
      "GEMINI_API_KEY": "${GEMINI_API_KEY}"
    }
  }
}
EOL

# Create a temporary JSON file for the GRPO configuration
cat > grpo_config.json << EOL
{
  "base_model": "${MODEL}",
  "learning_rate": 5e-5,
  "num_train_epochs": 1,
  "per_device_train_batch_size": 4,
  "reward_fns": $(cat reward_functions.json)
}
EOL

# Print the configuration
echo "GRPO Configuration:"
cat grpo_config.json

# Run the GRPO job
echo "Running GRPO job..."
predibase finetuning jobs create --config grpo_config.json --dataset ${DATASET} --repo ${REPO} --description "${DESCRIPTION}"

# Clean up temporary files
rm reward_functions.json grpo_config.json

echo "Job submitted successfully!"
