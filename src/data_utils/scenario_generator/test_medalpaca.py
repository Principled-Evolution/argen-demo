#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_medalpaca.py - Test script for medalpaca model
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer

# Add the current directory to the path
package_dir = os.path.dirname(os.path.abspath(__file__))
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)

# Import the logging module
from config import log

def test_medalpaca():
    """Test loading the medalpaca model."""
    model_name = "medalpaca/medalpaca-7b"
    
    log.info(f"Testing medalpaca model: {model_name}")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")
    
    # Try loading with LlamaTokenizer
    log.info("Trying to load with LlamaTokenizer...")
    try:
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        log.info("Successfully loaded tokenizer with LlamaTokenizer")
        
        # Set pad token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            log.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")
        
        # Load model
        log.info(f"Loading model {model_name} on {device}...")
        model_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=model_dtype,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        log.info("Model loaded successfully!")
        return True
    except Exception as e:
        log.error(f"Failed to load medalpaca model: {e}")
        return False

if __name__ == "__main__":
    success = test_medalpaca()
    if success:
        print("✅ Medalpaca model test passed!")
        sys.exit(0)
    else:
        print("❌ Medalpaca model test failed!")
        sys.exit(1)
