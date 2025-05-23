#!/usr/bin/env python3
"""
Script to test chat template handling for Llama 3.2 models.
"""

import os
import sys
import logging
import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
log_dir = Path("debug_logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "chat_template_test.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def test_chat_template(model_name, system_prompt, user_prompt):
    """Test chat template formatting for a specific model."""
    logger.info(f"Testing chat template for model: {model_name}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        logger.info(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
        logger.info(f"Chat template: {tokenizer.chat_template}")
        
        # Create messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Format with chat template
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        logger.info(f"Formatted prompt:\n{formatted}")
        
        # Save to file
        output_file = log_dir / f"{model_name.replace('/', '_')}_formatted.txt"
        with open(output_file, "w") as f:
            f.write(formatted)
        logger.info(f"Saved formatted prompt to {output_file}")
        
        # Tokenize and check token IDs
        tokens = tokenizer.encode(formatted)
        logger.info(f"Token count: {len(tokens)}")
        logger.info(f"First 10 tokens: {tokens[:10]}")
        logger.info(f"Last 10 tokens: {tokens[-10:]}")
        
        # Decode back to verify
        decoded = tokenizer.decode(tokens)
        logger.info(f"Decoded tokens:\n{decoded}")
        
        # Test generation
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            logger.info(f"Model loaded: {model.__class__.__name__}")
            
            # Generate a response
            input_ids = tokenizer.encode(formatted, return_tensors="pt").to(model.device)
            
            # Log the input shape
            logger.info(f"Input shape: {input_ids.shape}")
            
            # Generate with different parameters
            for temp in [0.0, 0.7]:
                for max_new in [50, 100]:
                    logger.info(f"Generating with temperature={temp}, max_new_tokens={max_new}")
                    
                    output = model.generate(
                        input_ids,
                        max_new_tokens=max_new,
                        temperature=temp if temp > 0 else None,
                        do_sample=temp > 0,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    
                    # Decode the output
                    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                    logger.info(f"Generated text (temp={temp}, max_new={max_new}):\n{generated_text}")
                    
                    # Save to file
                    gen_file = log_dir / f"{model_name.replace('/', '_')}_gen_temp{temp}_max{max_new}.txt"
                    with open(gen_file, "w") as f:
                        f.write(generated_text)
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error testing chat template: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_multiple_formats(model_name):
    """Test different message formats for the same model."""
    logger.info(f"Testing multiple formats for model: {model_name}")
    
    system_prompt = "You are a helpful assistant."
    user_prompt = "Hello, how are you?"
    
    # Format 1: Standard messages list
    logger.info("Testing standard messages list format")
    messages1 = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Format 2: Just user message
    logger.info("Testing user-only message format")
    messages2 = [
        {"role": "user", "content": user_prompt}
    ]
    
    # Format 3: Raw string (should be handled by preprocessing)
    logger.info("Testing raw string format")
    messages3 = user_prompt
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Test Format 1
        formatted1 = tokenizer.apply_chat_template(messages1, tokenize=False, add_generation_prompt=True)
        logger.info(f"Format 1 result:\n{formatted1}")
        
        # Test Format 2
        formatted2 = tokenizer.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
        logger.info(f"Format 2 result:\n{formatted2}")
        
        # Test Format 3 (with preprocessing)
        if isinstance(messages3, str):
            messages3_processed = [{"role": "user", "content": messages3}]
            formatted3 = tokenizer.apply_chat_template(messages3_processed, tokenize=False, add_generation_prompt=True)
            logger.info(f"Format 3 result (after preprocessing):\n{formatted3}")
        
        # Save all formats to files
        with open(log_dir / f"{model_name.replace('/', '_')}_format1.txt", "w") as f:
            f.write(formatted1)
        
        with open(log_dir / f"{model_name.replace('/', '_')}_format2.txt", "w") as f:
            f.write(formatted2)
        
        with open(log_dir / f"{model_name.replace('/', '_')}_format3.txt", "w") as f:
            f.write(formatted3)
        
    except Exception as e:
        logger.error(f"Error testing multiple formats: {e}")

def main():
    logger.info("Starting chat template test")
    
    # Test with Llama 3.2 1B model
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    
    # Test with different system prompts
    system_prompts = [
        "You are a helpful assistant.",
        "You are a concise AI assistant. Answer accurately or say you do not know. If the user asks for medical advice, recommend seeing a qualified professional.",
        ""  # Empty system prompt
    ]
    
    # Test with different user prompts
    user_prompts = [
        "Hello, how are you?",
        "I've had this weird mole on my back for years, but lately it's been itching like crazy. I also have a job interview next week, what color shirt should I wear to make a good impression?",
        "My water just broke, but it's green. I'm only 34 weeks pregnant. What does this mean and what do I do?"
    ]
    
    # Run tests with different combinations
    for i, system_prompt in enumerate(system_prompts):
        for j, user_prompt in enumerate(user_prompts):
            logger.info(f"=== Test {i+1}.{j+1}: System prompt {i+1}, User prompt {j+1} ===")
            test_chat_template(model_name, system_prompt, user_prompt)
    
    # Test multiple formats
    test_multiple_formats(model_name)
    
    logger.info("Chat template test completed")

if __name__ == "__main__":
    main()
