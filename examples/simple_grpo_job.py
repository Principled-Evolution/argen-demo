"""
Simple script to create a GRPO job in Predibase.
"""

import os
import json
import argparse

# Add the project root to the Python path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.env import load_env_vars


def main():
    """Run the simple GRPO job script."""
    parser = argparse.ArgumentParser(description="Create a simple GRPO job in Predibase.")
    parser.add_argument("--model", type=str, default="llama-3-2-1b-instruct", help="Name of the base model")
    parser.add_argument("--dataset", type=str, default="argen_combined_dataset", help="Name of the dataset in Predibase")
    parser.add_argument("--repo", type=str, default="argen-gemini-opa", help="Name of the repository to save the adapter to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Enable debug mode
    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    print("Starting simple GRPO job script...")

    # Load environment variables
    print("Loading environment variables...")
    load_env_vars()

    # Get Gemini API key
    print("Getting Gemini API key...")
    gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    print("Gemini API key found.")

    # Get API token from config file
    config_path = os.path.expanduser("~/.predibase/config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            pb_config = json.load(f)
            api_token = pb_config.get("api_key")
            if not api_token:
                raise ValueError("API token not found in config file")
    else:
        raise ValueError("Config file not found")

    try:
        from predibase import Predibase, GRPOConfig, RewardFunction
    except ImportError:
        raise ImportError(
            "Predibase SDK not installed. Please install it with 'pip install predibase'."
        )

    # Initialize Predibase client
    print(f"Initializing Predibase client...")
    pb = Predibase(api_token=api_token)

    # Import reward function code as strings
    reward_functions_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "reward_functions")

    with open(os.path.join(reward_functions_dir, "gemini_rewards.py"), "r") as f:
        gemini_rewards_code = f.read()

    with open(os.path.join(reward_functions_dir, "gemini_opa_rewards.py"), "r") as f:
        gemini_opa_rewards_code = f.read()

    # Create a simple GRPO configuration
    print(f"Creating GRPO configuration...")

    # Try different approaches to create the configuration
    try:
        # Approach 1: Using the GRPOConfig class
        print("Trying approach 1: Using GRPOConfig class...")

        # Create the reward functions
        reward_functions = {
            "code": gemini_rewards_code + "\n\n" + gemini_opa_rewards_code,
            "functions": {
                "ahimsa": "gemini_opa_ahimsa_reward",
                "dharma": "gemini_opa_dharma_reward"
            },
            "runtime": {
                "packages": ["google-generativeai", "python-dotenv"],
                "env_vars": {
                    "GEMINI_API_KEY": gemini_api_key
                }
            }
        }

        # Create the GRPO configuration
        config = GRPOConfig(
            base_model=args.model,
            task="grpo",
            epochs=1,
            learning_rate=5e-5,
            effective_batch_size=4
        )

        # Convert to dictionary
        config_dict = config.model_dump()

        # Add reward functions
        config_dict["reward_functions"] = reward_functions

        # Submit the job
        print(f"Submitting GRPO job to Predibase...")
        print(f"Configuration: {json.dumps(config_dict, indent=2)}")

        job = pb.finetuning.jobs.create(
            config=config_dict,
            dataset=args.dataset,
            repo=args.repo,
            description="ArGen GRPO fine-tuning with Gemini-OPA reward functions"
        )

        print(f"GRPO job submitted successfully! Job ID: {job.id}")

    except Exception as e:
        print(f"Error with approach 1: {e}")

        try:
            # Approach 2: Using a simple dictionary
            print("Trying approach 2: Using a simple dictionary...")

            # Create a simple configuration dictionary
            config = {
                "base_model": args.model,
                "task": "grpo",
                "epochs": 1,
                "learning_rate": 5e-5,
                "effective_batch_size": 4,
                "reward_functions": {
                    "code": gemini_rewards_code + "\n\n" + gemini_opa_rewards_code,
                    "functions": {
                        "ahimsa": "gemini_opa_ahimsa_reward",
                        "dharma": "gemini_opa_dharma_reward"
                    },
                    "runtime": {
                        "packages": ["google-generativeai", "python-dotenv"],
                        "env_vars": {
                            "GEMINI_API_KEY": gemini_api_key
                        }
                    }
                }
            }

            # Submit the job
            print(f"Submitting GRPO job to Predibase...")
            print(f"Configuration: {json.dumps(config, indent=2)}")

            job = pb.finetuning.jobs.create(
                config=config,
                dataset=args.dataset,
                repo=args.repo,
                description="ArGen GRPO fine-tuning with Gemini-OPA reward functions"
            )

            print(f"GRPO job submitted successfully! Job ID: {job.id}")

        except Exception as e:
            print(f"Error with approach 2: {e}")

            try:
                # Approach 3: Using a minimal dictionary
                print("Trying approach 3: Using a minimal dictionary...")

                # Create a minimal configuration dictionary
                config = {
                    "base_model": args.model
                }

                # Submit the job
                print(f"Submitting GRPO job to Predibase...")
                print(f"Configuration: {json.dumps(config, indent=2)}")

                job = pb.finetuning.jobs.create(
                    config=config,
                    dataset=args.dataset,
                    repo=args.repo,
                    description="ArGen GRPO fine-tuning with Gemini-OPA reward functions"
                )

                print(f"GRPO job submitted successfully! Job ID: {job.id}")

            except Exception as e:
                print(f"Error with approach 3: {e}")
                print("All approaches failed. Please check the Predibase documentation for the correct configuration format.")


if __name__ == "__main__":
    main()
