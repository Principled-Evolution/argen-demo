#!/usr/bin/env python3
"""
Debug script for Predibase.
"""

import os
import sys
import json
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Gemini API key
gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
if not gemini_api_key:
    print("GEMINI_API_KEY environment variable not set")
    sys.exit(1)
print("Gemini API key found")

# Get API token from config file
config_path = os.path.expanduser("~/.predibase/config.json")
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        pb_config = json.load(f)
        api_token = pb_config.get("api_key")
        if not api_token:
            print("API token not found in config file")
            sys.exit(1)
else:
    print("Config file not found")
    sys.exit(1)

try:
    print("Importing Predibase...")
    from predibase import Predibase, GRPOConfig, RewardFunctionsConfig, RewardFunctionsRuntimeConfig
    print("Predibase imported successfully")
    
    print("Initializing Predibase client...")
    pb = Predibase(api_token=api_token)
    print("Predibase client initialized")
    
    print("Getting datasets...")
    datasets = pb.datasets.list()
    print(f"Found {len(datasets)} datasets")
    
    print("Getting repositories...")
    repos = pb.repos.list()
    print(f"Found {len(repos)} repositories")
    
    print("Script completed successfully")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
    sys.exit(1)
