#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py - Main entry point for ArGen dataset generator
======================================================
Ties together all components and provides the command-line interface
"""

import os
import sys
import asyncio
import argparse
import logging
from typing import Dict, List, Optional
import importlib.util
from pathlib import Path
import dotenv
from tqdm.auto import tqdm  # Use tqdm.auto instead for better terminal compatibility
import shutil  # For getting terminal size

import openai
from openai import OpenAI

# Import the import helper
from .import_helper import STANDALONE_MODE, get_import

if STANDALONE_MODE:
    print("Running in STANDALONE mode")
else:
    print("Running in INTEGRATED mode")

# Import modules using the helper
config_module = get_import('config')
log = config_module.log
DATASETS = config_module.DATASETS
create_parser = config_module.create_parser
get_model_limits = config_module.get_model_limits
DEFAULT_GENERATION_MODEL = config_module.DEFAULT_GENERATION_MODEL
DEFAULT_HF_GENERATION_MODEL = config_module.DEFAULT_HF_GENERATION_MODEL
DEFAULT_EMBEDDING_MODEL = config_module.DEFAULT_EMBEDDING_MODEL
DEFAULT_BASELINE = config_module.DEFAULT_BASELINE

openai_utils_module = get_import('openai_utils')
init_openai_client = openai_utils_module.init_openai_client

# Import HuggingFace utilities
try:
    hf_utils_module = get_import('hf_utils')
    init_hf_model = hf_utils_module.init_hf_model
    get_hf_model_name = hf_utils_module.get_hf_model_name
    hf_utils_available = True
except ImportError as e:
    log.warning(f"HuggingFace utilities not available: {e}")
    hf_utils_available = False

embedding_utils_module = get_import('embedding_utils')
init_embedding_model = embedding_utils_module.init_embedding_model
load_core60_dataset = embedding_utils_module.load_core60_dataset
init_tfidf_vectorizer = embedding_utils_module.init_tfidf_vectorizer

baseline_model_module = get_import('baseline_model')
init_baseline_model = baseline_model_module.init_baseline_model

medical_terms_module = get_import('medical_terms')
load_medical_terms = medical_terms_module.load_medical_terms

generation_module = get_import('generation')
process_dataset = generation_module.process_dataset

# System prompt handling - try to import from parent project first, then fall back to local version
def get_local_system_prompt(use_basic_prompt: bool = False) -> str:
    """Fallback system prompt if original is not available."""
    return """You are a healthcare assistant. Provide helpful medical information, but avoid giving definitive medical advice.
Always encourage users to consult healthcare professionals for specific diagnoses and treatments.
Stay within healthcare boundaries and do not answer questions about other domains."""

try:
    # Try to import from parent project if running in integrated mode
    # First, add the project root to sys.path if it's not already there
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Now try to import
    try:
        from src.config import get_system_prompt
        log.info("Using system prompt from parent project")
    except ImportError:
        # Try with importlib
        import importlib.util
        config_path = os.path.join(project_root, "src/config.py")
        if os.path.exists(config_path):
            spec = importlib.util.spec_from_file_location("config", config_path)
            if spec and spec.loader:
                config_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config_module)
                get_system_prompt = config_module.get_system_prompt
                log.info("Using system prompt from parent project (via importlib)")
            else:
                raise ImportError("Could not load config module")
        else:
            raise ImportError(f"Config file not found at {config_path}")
except ImportError as e:
    # Fall back to local version if running standalone
    log.warning(f"Could not import get_system_prompt from parent project: {e}. Using fallback.")
    get_system_prompt = get_local_system_prompt

async def main():
    """Main entry point for dataset generation."""
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()

    # Set logging level
    log_level = getattr(logging, args.log_level)
    log.setLevel(log_level)

    # Print welcome message
    log.info("=" * 80)
    log.info("ArGen dataset generator")
    log.info("=" * 80)
    log.info(f"Selected datasets: {args.datasets}")
    log.info(f"Running in {'STANDALONE' if STANDALONE_MODE else 'INTEGRATED'} mode")
    if args.dry_run:
        log.warning("RUNNING IN DRY-RUN MODE - Using mock evaluations, results will not be reliable!")

    # Get script directory for file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Set up OpenAI
    # Try to load from .env file in project root
    project_root = os.path.abspath(os.path.join(script_dir, "../../../.."))
    env_path = os.path.join(project_root, ".env")
    if os.path.exists(env_path):
        log.info(f"Loading .env file from {env_path}")
        dotenv.load_dotenv(env_path)

    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        log.error("OPENAI_API_KEY environment variable not set. Exiting.")
        sys.exit(1)

    # Initialize OpenAI client (only if not using HF model exclusively)
    openai_client = None
    if not args.hf_model:
        openai_client = init_openai_client(openai.api_key)
        if not openai_client:
            log.error("Failed to initialize OpenAI client. Exiting.")
            sys.exit(1)

    # Initialize HuggingFace model if specified
    hf_model_initialized = False
    if args.hf_model:
        if not hf_utils_available:
            log.error("HuggingFace utilities not available but --hf-model was specified. Exiting.")
            sys.exit(1)

        hf_model_name = args.hf_model if args.hf_model else DEFAULT_HF_GENERATION_MODEL
        log.info(f"Initializing HuggingFace model: {hf_model_name}")
        hf_model_initialized = init_hf_model(hf_model_name)

        if not hf_model_initialized:
            log.error(f"Failed to initialize HuggingFace model {hf_model_name}. Exiting.")
            sys.exit(1)

        log.info(f"Using HuggingFace model {get_hf_model_name()} for generation")

    # Load medical terms
    medical_terms = load_medical_terms()
    log.info(f"Loaded {len(medical_terms)} medical terms for detection")

    # Initialize embedding model
    embedding_model_initialized = init_embedding_model(args.embedding_model)
    if not embedding_model_initialized:
        log.warning("Failed to initialize embedding model. Proceeding with limited functionality.")

    # Load core60 dataset and initialize TF-IDF
    core_rows = load_core60_dataset(args.tfidf_core60)
    if core_rows:
        vectorizer, core_matrix = init_tfidf_vectorizer(core_rows)
        log.info("TF-IDF reranking enabled")
    else:
        log.warning("TF-IDF reranking disabled - no core rows available")

    # Initialize baseline model - REQUIRED for evaluation
    baseline_model_initialized = False
    baseline_model_name = args.baseline if args.baseline else DEFAULT_BASELINE
    log.info(f"Initializing baseline model: {baseline_model_name} (REQUIRED for evaluation)")
    baseline_model_initialized = init_baseline_model(baseline_model_name)
    if not baseline_model_initialized:
        log.error(f"Failed to initialize baseline model {baseline_model_name}. Baseline model is REQUIRED for evaluation.")
        log.error("Exiting with error as baseline model is mandatory for proper evaluation.")
        sys.exit(1)

    # Get system prompt for baseline model responses
    system_prompt = get_system_prompt(use_basic_prompt=False)

    # Get model limits for token calculations
    generation_model_limit = get_model_limits(args.model)
    generation_model_limit['model_name'] = args.model  # Add model name for token counting

    # Create tasks for selected datasets
    datasets_to_process = {name: DATASETS[name].copy() for name in args.datasets if name in DATASETS}
    if not datasets_to_process:
        log.error(f"No valid datasets selected. Available datasets: {list(DATASETS.keys())}")
        sys.exit(1)

    # Override temperature if specified in command line
    if args.temperature is not None:
        for name, settings in datasets_to_process.items():
            original_temp = settings['temperature']
            settings['temperature'] = args.temperature
            log.info(f"Overriding temperature for dataset '{name}' from {original_temp} to {args.temperature}")

    # Override count if specified in command line
    if args.count is not None:
        for name, settings in datasets_to_process.items():
            original_count = settings['count']
            settings['count'] = args.count
            log.info(f"Overriding scenario count for dataset '{name}' from {original_count} to {args.count}")

    # Set max_new_tokens if specified in command line
    if args.hf_max_new_tokens is not None:
        log.info(f"Setting max_new_tokens for HuggingFace models to {args.hf_max_new_tokens}")
        # Store in generation_model_limit for later use
        generation_model_limit['max_output_tokens'] = args.hf_max_new_tokens

    # Create overall progress bar for all datasets (tmux-compatible)
    total_datasets = len(datasets_to_process)
    if total_datasets > 1:
        overall_pbar = tqdm(total=total_datasets, desc="Overall dataset progress", 
                            unit="dataset", dynamic_ncols=True, miniters=1)
        log.info(f"Processing {total_datasets} datasets with progress tracking")
    else:
        overall_pbar = None
        log.info(f"Processing single dataset: {list(datasets_to_process.keys())[0]}")

    # Process datasets sequentially for better progress tracking
    completed_datasets = 0
    for name, settings in datasets_to_process.items():
        # Get vectorizer and core_matrix if available
        tfidf_available = 'vectorizer' in locals() and 'core_matrix' in locals()

        # Use the HF model name for generation if HF model is initialized
        generation_model = args.hf_model if hf_model_initialized else args.model
        log.info(f"Using generation model: {generation_model}")

        # Special handling for smoke_test dataset
        local_difficulty_ratio = args.difficulty_ratio
        if name == 'smoke_test' and args.difficulty_ratio > 0:
            # For smoke tests, use a lower difficulty ratio to make it easier to generate prompts
            local_difficulty_ratio = 0.5  # Lower value means less strict difficulty filtering
            log.info(f"Using reduced difficulty ratio of {local_difficulty_ratio} for smoke_test dataset")

        # Special handling for duplicate threshold for smoke_test
        local_duplicate_threshold = args.duplicate_threshold
        if name == 'smoke_test':
            # For smoke tests, use a higher duplicate threshold to be less strict
            local_duplicate_threshold = 0.95  # Higher value means less strict duplicate detection
            log.info(f"Using higher duplicate threshold of {local_duplicate_threshold} for smoke_test dataset")

        # Process dataset
        await process_dataset(
            name=name,
            settings=settings,
            openai_client=openai_client,
            api_key=openai.api_key,
            generation_model=generation_model,
            model_limit=generation_model_limit,
            system_prompt=system_prompt,
            baseline_model_initialized=baseline_model_initialized,
            duplicate_threshold=local_duplicate_threshold,
            fail_threshold=args.fail_threshold,
            max_retries=args.max_retries,
            initial_delay=args.initial_delay,
            script_dir=script_dir,
            dry_run=args.dry_run,
            vectorizer=vectorizer if tfidf_available else None,
            core_matrix=core_matrix if tfidf_available else None,
            difficulty_ratio=local_difficulty_ratio,
            enforce_medical=args.enforce_medical,
            use_hf_model=hf_model_initialized,
            medalpaca_scenarios_per_message=args.medalpaca_scenarios_per_message,
            batch_size=args.batch_size,
            concurrent_eval_limit=args.concurrent_eval_limit,
            tiering_concurrency_limit=args.tiering_concurrency_limit,
            exclude_from_file=args.exclude_from_file if hasattr(args, 'exclude_from_file') else None
        )
        
        # Update overall progress
        completed_datasets += 1
        if overall_pbar:
            overall_pbar.update(1)
            overall_pbar.set_postfix({"Completed": f"{completed_datasets}/{total_datasets}", "Current": name})

    # Close overall progress bar
    if overall_pbar:
        overall_pbar.close()

    log.info("=" * 80)
    log.info("Dataset generation complete")
    log.info("=" * 80)

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())