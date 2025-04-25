#!/usr/bin/env python3
"""
Script to evaluate a GRPO-trained model in Predibase.
"""

import os
import sys
import json
import logging
import argparse
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluate_grpo_model.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('evaluate_grpo_model')

def get_api_token():
    """Get the Predibase API token from the config file."""
    config_path = os.path.expanduser("~/.predibase/config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            pb_config = json.load(f)
            api_token = pb_config.get("api_key")
            if not api_token:
                logger.error("API token not found in config file")
                sys.exit(1)
            return api_token
    else:
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

def evaluate_model(adapter_id, deployment_name, test_data_path, output_path):
    """
    Evaluate a GRPO-trained model in Predibase.
    
    Args:
        adapter_id: The ID of the adapter (e.g., "argen-gemini-opa/4")
        deployment_name: The name of the deployment to use
        test_data_path: Path to the test data file
        output_path: Path to save the evaluation results
    """
    logger.info(f"Evaluating GRPO-trained model {adapter_id} using deployment {deployment_name}...")
    
    # Get API token
    api_token = get_api_token()
    
    # Import Predibase
    try:
        from predibase import Predibase, DeploymentConfig
    except ImportError as e:
        logger.error(f"Error importing Predibase: {e}")
        sys.exit(1)
    
    # Initialize Predibase client
    try:
        pb = Predibase(api_token=api_token)
        logger.info("Predibase client initialized")
    except Exception as e:
        logger.error(f"Error initializing Predibase client: {e}")
        sys.exit(1)
    
    # Check if deployment exists, create if not
    try:
        deployments = pb.deployments.list()
        deployment_exists = any(d.name == deployment_name for d in deployments)
        
        if not deployment_exists:
            logger.info(f"Creating deployment {deployment_name}...")
            
            # Get the base model from the adapter
            adapter = pb.adapters.get(adapter_id)
            base_model = adapter.base_model
            
            # Create the deployment
            pb.deployments.create(
                name=deployment_name,
                config=DeploymentConfig(
                    base_model=base_model,
                    cooldown_time=600,
                    min_replicas=0,
                    max_replicas=1
                ),
                description=f"Deployment for evaluating {adapter_id}"
            )
            logger.info(f"Deployment {deployment_name} created")
        else:
            logger.info(f"Deployment {deployment_name} already exists")
    except Exception as e:
        logger.error(f"Error checking/creating deployment: {e}")
        sys.exit(1)
    
    # Load test data
    try:
        logger.info(f"Loading test data from {test_data_path}...")
        if test_data_path.endswith('.json') or test_data_path.endswith('.jsonl'):
            test_data = pd.read_json(test_data_path, lines=True)
        elif test_data_path.endswith('.csv'):
            test_data = pd.read_csv(test_data_path)
        else:
            logger.error(f"Unsupported file format: {test_data_path}")
            sys.exit(1)
        
        logger.info(f"Loaded {len(test_data)} test examples")
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        sys.exit(1)
    
    # Get client for the deployment
    try:
        logger.info(f"Getting client for deployment {deployment_name}...")
        client = pb.deployments.client(deployment_name)
        logger.info("Client initialized")
    except Exception as e:
        logger.error(f"Error getting client: {e}")
        sys.exit(1)
    
    # Evaluate the model
    try:
        logger.info(f"Evaluating model on {len(test_data)} examples...")
        results = []
        
        for i, row in test_data.iterrows():
            prompt = row["prompt"]
            logger.info(f"Processing example {i+1}/{len(test_data)}")
            
            # Generate completion
            completion = client.generate(prompt, adapter_id=adapter_id).generated_text
            
            # Save result
            result = {
                "prompt": prompt,
                "completion": completion
            }
            
            # Add any additional fields from the test data
            for col in test_data.columns:
                if col != "prompt" and col != "completion":
                    result[col] = row[col]
            
            results.append(result)
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_json(output_path, orient="records", lines=True)
        logger.info(f"Evaluation results saved to {output_path}")
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        sys.exit(1)
    
    logger.info("Evaluation completed successfully")

def main():
    """Main function to evaluate a GRPO-trained model."""
    parser = argparse.ArgumentParser(description="Evaluate a GRPO-trained model in Predibase.")
    parser.add_argument("--adapter-id", type=str, required=True, help="ID of the adapter (e.g., 'argen-gemini-opa/4')")
    parser.add_argument("--deployment-name", type=str, default="argen-llama-3-deployment", help="Name of the deployment to use")
    parser.add_argument("--test-data", type=str, required=True, help="Path to the test data file")
    parser.add_argument("--output", type=str, default="evaluation_results.jsonl", help="Path to save the evaluation results")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Evaluate the model
    evaluate_model(
        adapter_id=args.adapter_id,
        deployment_name=args.deployment_name,
        test_data_path=args.test_data,
        output_path=args.output
    )

if __name__ == "__main__":
    main()
