#!/usr/bin/env python3
"""
Debug script for Predibase with verbose output.
"""

import os
import sys
import json
import traceback
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('debug_predibase')

# Load environment variables
logger.info("Loading environment variables...")
load_dotenv()

# Get Gemini API key
logger.info("Getting Gemini API key...")
gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
if not gemini_api_key:
    logger.error("GEMINI_API_KEY environment variable not set")
    sys.exit(1)
logger.info("Gemini API key found")

# Get API token from config file
logger.info("Getting API token from config file...")
config_path = os.path.expanduser("~/.predibase/config.json")
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        pb_config = json.load(f)
        api_token = pb_config.get("api_key")
        if not api_token:
            logger.error("API token not found in config file")
            sys.exit(1)
        logger.info("API token found")
else:
    logger.error(f"Config file not found: {config_path}")
    sys.exit(1)

try:
    logger.info("Importing Predibase...")
    from predibase import Predibase
    logger.info("Predibase imported successfully")
    
    logger.info("Initializing Predibase client...")
    pb = Predibase(api_token=api_token)
    logger.info("Predibase client initialized")
    
    logger.info("Getting datasets...")
    try:
        datasets = pb.datasets.list()
        logger.info(f"Found {len(datasets)} datasets")
        for dataset in datasets:
            logger.info(f"  - {dataset.name}")
    except Exception as e:
        logger.error(f"Error getting datasets: {e}")
        traceback.print_exc()
    
    logger.info("Getting repositories...")
    try:
        repos = pb.repos.list()
        logger.info(f"Found {len(repos)} repositories")
        for repo in repos:
            logger.info(f"  - {repo.name}")
    except Exception as e:
        logger.error(f"Error getting repositories: {e}")
        traceback.print_exc()
    
    logger.info("Getting models...")
    try:
        models = pb.models.list()
        logger.info(f"Found {len(models)} models")
        for model in models:
            logger.info(f"  - {model.name}")
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        traceback.print_exc()
    
    logger.info("Script completed successfully")
except Exception as e:
    logger.error(f"Error: {e}")
    traceback.print_exc()
    sys.exit(1)
