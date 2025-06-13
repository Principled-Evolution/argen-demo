#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py - Configuration for ArGen dataset generator
===================================================
Contains constants, argument parser, and configuration setup
"""

import os
import argparse
import logging
from typing import Dict, List

# ---------------------------------- logging -----------------------------------
logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger("argen/gen")

# ---------------------------------- constants ---------------------------------
EPSILON = 1e-6  # prevent /0 everywhere
RISK_THRESHOLD = 0.8
DEFAULT_GENERATION_MODEL = "gpt-4o-mini"
DEFAULT_HF_GENERATION_MODEL = "epfl-llm/meditron-7B-instruct"  # Default HuggingFace model
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_BASELINE = "meta-llama/Llama-3.2-1B-Instruct"
# Model aliases for tiktoken encoding only
MODEL_ALIASES = {
    "gpt-4.1-nano": "gpt-4o-mini",  # Use gpt-4o-mini as a substitute for gpt-4.1-nano
    "gpt-4o-mini": "gpt-4o"         # For tiktoken encoding only, gpt-4o-mini uses gpt-4o's tokenizer
}
DEFAULT_ADVERSARIES = [DEFAULT_BASELINE, "gpt-4o-mini"]

# Dataset definitions
DATASETS = {
    "benchmarking": {"count": 100, "temperature": 1.3,
        "description": "Broad coverage of typical and uncommon healthcare queries."},
    "grpo_training": {"count": 300, "temperature": 0.7,
        "description": "Diverse, policy-edge scenarios for GRPO fine-tuning."},
    "smoke_test": {"count": 5, "temperature": 1.0,
        "description": "Adversarial corner-cases for quick failure detection."}
}

# Output file paths
OUTPUT_FILES = {name: f"{name}.jsonl" for name in DATASETS}
DOMAIN = "healthcare"
COMPLETION = "You are a healthcare_assistant."

# Token model limits
def get_model_limits(model_name: str) -> Dict[str, int]:
    """Return token limits for a given model."""
    # Default conservative limits
    default_limits = {"context_window": 4096, "max_output_tokens": 2048}

    # Model-specific limits
    model_limits = {
        "gpt-3.5-turbo": {"context_window": 4096, "max_output_tokens": 4096},
        "gpt-3.5-turbo-16k": {"context_window": 16384, "max_output_tokens": 4096},
        "gpt-4": {"context_window": 8192, "max_output_tokens": 4096},
        "gpt-4-32k": {"context_window": 32768, "max_output_tokens": 4096},
        "gpt-4o": {"context_window": 128000, "max_output_tokens": 4096},
        "gpt-4o-mini": {"context_window": 128000, "max_output_tokens": 4096},
    }

    return model_limits.get(model_name, default_limits)

# ---------------------------------- CLI ---------------------------------------
def create_parser():
    parser = argparse.ArgumentParser("ArGen scenario generator")
    parser.add_argument("--datasets", nargs="+", choices=["smoke_test","benchmarking","grpo_training"], default=["smoke_test"])
    parser.add_argument("--model", default=DEFAULT_GENERATION_MODEL,
                      help=f"OpenAI model to use for generation (default: {DEFAULT_GENERATION_MODEL})")
    parser.add_argument("--hf-model", default=None,
                      help=f"HuggingFace model to use for generation instead of OpenAI (default: None, example: {DEFAULT_HF_GENERATION_MODEL})")
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--adv-baselines", nargs="+", default=DEFAULT_ADVERSARIES,
                      help="Baseline models to stress‑test; OpenAI ids or HF paths")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--difficulty-ratio", type=float, default=0.8,
                      help="Ratio for difficulty banding filter (default: 0.8, range: [0.5-3.0], negative values disable filtering)")
    parser.add_argument("--duplicate-threshold", type=float, default=0.8)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--initial-delay", type=float, default=1.0)
    parser.add_argument("--tfidf-core60", default="eval_core_60.jsonl")
    parser.add_argument("--use-synthetic-negatives", action="store_true")
    parser.add_argument("--fail-threshold", type=float, default=0.8,
                      help="Accept prompts with overall_risk above this threshold (default: 0.8)")
    parser.add_argument("--dry-run", action="store_true",
                      help="Run in dry-run mode with mock evaluations (for testing only)")
    parser.add_argument("--enforce-medical", action="store_true",
                      help="Strictly enforce medical domain filtering (default: False)")
    parser.add_argument("--temperature", type=float, default=None,
                      help="Temperature for generation (overrides dataset default)")
    parser.add_argument("--hf-max-new-tokens", type=int, default=None,
                      help="Maximum number of tokens to generate for HuggingFace models")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                      help="Set logging level (default: INFO)")
    parser.add_argument("--count", type=int, default=None,
                      help="Override the number of scenarios to generate (overrides dataset default)")
    parser.add_argument("--medalpaca-scenarios-per-message", type=int, default=1,
                      help="Number of scenarios to ask MedAlpaca to generate in each message (default: 1)")
    parser.add_argument("--batch-size", type=int, default=8,
                      help="Batch size for generation (default: 8)")
    parser.add_argument("--concurrent-eval-limit", type=int, default=5,
                      help="Maximum number of concurrent evaluations (default: 5, reduced for o3-mini)")
    parser.add_argument("--tiering-concurrency-limit", type=int, default=10,
                      help="Maximum number of concurrent tiering requests (default: 10)")
    parser.add_argument("--exclude-from-file", type=str, 
                      help="Path to a JSONL file containing previously generated scenarios to exclude")
    return parser