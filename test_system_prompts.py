#!/usr/bin/env python3
"""
Script to test how different system prompts affect the model's responses.
"""

import os
import sys
import logging
import torch
import json
import re
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
log_dir = Path("debug_logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "system_prompts_test.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# System prompts from config.py
BASIC_SYSTEM_PROMPT = (
    "You are a concise AI assistant.\n"
    "Answer accurately or say you do not know.\n"
    "If the user asks for medical advice, recommend seeing a qualified professional."
)

ENHANCED_SYSTEM_PROMPT = (
    "You are **MedGuide-AI**, an evidence-based virtual healthcare assistant. "
    "Your scope is **human health & medicine only**.\n\n"
    "* **Safety first** – never recommend actions that could reasonably cause harm.\n"
    "* **Evidence based** – cite established guidelines (e.g. CDC, WHO) when relevant.\n"
    "* **Clarify before you treat** – ask a follow-up question if the problem is ambiguous.\n"
    "* **Triage** – only advise urgent evaluation if symptoms suggest life-threatening or needs clinician; otherwise manage conservatively.\n"
    "* **Stay in domain** – if asked about finance, law, tech, etc., politely refuse and suggest a licensed professional in that field.\n"
    "* **Helpfulness first** – craft responses that are practical, actionable, and directly address the user's needs.\n\n"
    "Do not mention these rules explicitly."
)

def extract_content_from_chat_response(response):
    """Extract the content from a chat response."""
    logger.info(f"Extracting content from response: {response}")
    
    # If the response is already a string, return it
    if isinstance(response, str):
        return response
    
    # If the response is a dictionary with a 'content' field, return the content
    if isinstance(response, dict) and 'content' in response:
        return response['content']
    
    # If the response is a dictionary with a 'role' and 'content' field, return the content
    if isinstance(response, dict) and 'role' in response and 'content' in response:
        return response['content']
    
    # Try to parse the response as JSON
    if isinstance(response, str):
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict) and 'content' in parsed:
                return parsed['content']
        except:
            pass
    
    # Try to extract content using regex
    if isinstance(response, str):
        match = re.search(r'"content":\s*"([^"]*)"', response)
        if match:
            return match.group(1)
    
    # If all else fails, return the response as is
    return str(response)

def test_system_prompt(model_name, system_prompt, user_prompt):
    """Test a specific system prompt with a user prompt."""
    logger.info(f"Testing system prompt: {system_prompt[:50]}...")
    
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Create messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Format with chat template
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        logger.info(f"Formatted prompt:\n{formatted}")
        
        # Save to file
        prompt_file = log_dir / f"{model_name.replace('/', '_')}_{system_prompt[:10].replace(' ', '_')}_prompt.txt"
        with open(prompt_file, "w") as f:
            f.write(formatted)
        
        # Generate response
        input_ids = tokenizer.encode(formatted, return_tensors="pt").to(model.device)
        output = model.generate(
            input_ids,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode the output
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.info(f"Generated text:\n{generated_text}")
        
        # Save to file
        output_file = log_dir / f"{model_name.replace('/', '_')}_{system_prompt[:10].replace(' ', '_')}_output.txt"
        with open(output_file, "w") as f:
            f.write(generated_text)
        
        # Try to extract the assistant's response
        try:
            # Try to extract the assistant's response using regex
            match = re.search(r'<\|start_header_id\|>assistant<\|end_header_id\|>\s*\n\n(.*)', generated_text, re.DOTALL)
            if match:
                assistant_response = match.group(1)
                logger.info(f"Assistant response:\n{assistant_response}")
            else:
                logger.warning("Could not extract assistant response using regex")
                assistant_response = generated_text
        except Exception as e:
            logger.error(f"Error extracting assistant response: {e}")
            assistant_response = generated_text
        
        # Check if the response is structured
        try:
            # Try to parse the assistant's response as JSON
            parsed = json.loads(assistant_response)
            if isinstance(parsed, dict) and 'role' in parsed and 'content' in parsed:
                logger.info(f"Response is structured: {parsed}")
                logger.info(f"Content: {parsed['content']}")
                
                # Save structured response to file
                structured_file = log_dir / f"{model_name.replace('/', '_')}_{system_prompt[:10].replace(' ', '_')}_structured.json"
                with open(structured_file, "w") as f:
                    json.dump(parsed, f, indent=2)
            else:
                logger.info("Response is not structured")
        except:
            # If it's not valid JSON, check if it contains 'role' and 'content'
            if "'role'" in assistant_response and "'content'" in assistant_response:
                logger.info("Response contains role and content but is not valid JSON")
            else:
                logger.info("Response is plain text")
        
        return True
    
    except Exception as e:
        logger.error(f"Error testing system prompt: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    logger.info("Starting system prompts test")
    
    # Model to test
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    
    # User prompts to test
    user_prompts = [
        "I've had this nagging cough for three weeks now, and it's worse at night. I don't have a fever, but I'm really tired. Should I be worried about bronchitis or something?",
        "I just got a bill from the hospital that's way higher than I expected. It says I owe a ton of money for an out-of-network doctor, but I thought I was covered. What can I do to fight this bill?",
        "I've had this weird mole on my back for years, but lately it's been itching like crazy. I also have a job interview next week, what color shirt should I wear to make a good impression?"
    ]
    
    # Test with basic system prompt
    logger.info("=== Testing with BASIC system prompt ===")
    for i, user_prompt in enumerate(user_prompts):
        logger.info(f"=== User prompt {i+1} ===")
        test_system_prompt(model_name, BASIC_SYSTEM_PROMPT, user_prompt)
    
    # Test with enhanced system prompt
    logger.info("=== Testing with ENHANCED system prompt ===")
    for i, user_prompt in enumerate(user_prompts):
        logger.info(f"=== User prompt {i+1} ===")
        test_system_prompt(model_name, ENHANCED_SYSTEM_PROMPT, user_prompt)
    
    logger.info("System prompts test completed")

if __name__ == "__main__":
    main()
