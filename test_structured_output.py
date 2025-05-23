#!/usr/bin/env python3
"""
Script to test if the model always returns structured responses or if it depends on the prompt.
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
log_file = log_dir / "structured_output_test.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def test_with_chat_template(model_name, system_prompt, user_prompt, use_chat_template=True):
    """Test with or without chat template."""
    logger.info(f"Testing with {'chat template' if use_chat_template else 'raw prompt'}")
    
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        if use_chat_template:
            # Create messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Format with chat template
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # Create a raw prompt without chat template
            formatted = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
        
        logger.info(f"Formatted prompt:\n{formatted}")
        
        # Save to file
        prompt_file = log_dir / f"{model_name.replace('/', '_')}_{'chat' if use_chat_template else 'raw'}_prompt.txt"
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
        output_file = log_dir / f"{model_name.replace('/', '_')}_{'chat' if use_chat_template else 'raw'}_output.txt"
        with open(output_file, "w") as f:
            f.write(generated_text)
        
        # Check if the response is structured
        try:
            # Try to parse the response as JSON
            if use_chat_template:
                # Try to extract the assistant's response using regex
                match = re.search(r'<\|start_header_id\|>assistant<\|end_header_id\|>\s*\n\n(.*)', generated_text, re.DOTALL)
                if match:
                    assistant_response = match.group(1)
                else:
                    assistant_response = generated_text
            else:
                # Try to extract the assistant's response after "Assistant:"
                match = re.search(r'Assistant:(.*)', generated_text, re.DOTALL)
                if match:
                    assistant_response = match.group(1).strip()
                else:
                    assistant_response = generated_text
            
            logger.info(f"Assistant response:\n{assistant_response}")
            
            # Check if the response is structured
            try:
                parsed = json.loads(assistant_response)
                if isinstance(parsed, dict) and 'role' in parsed and 'content' in parsed:
                    logger.info(f"Response is structured: {parsed}")
                    return True
                else:
                    logger.info("Response is not structured JSON")
                    return False
            except:
                # If it's not valid JSON, check if it contains 'role' and 'content'
                if "'role'" in assistant_response and "'content'" in assistant_response:
                    logger.info("Response contains role and content but is not valid JSON")
                    return True
                else:
                    logger.info("Response is plain text")
                    return False
        except Exception as e:
            logger.error(f"Error checking if response is structured: {e}")
            return False
    
    except Exception as e:
        logger.error(f"Error testing with {'chat template' if use_chat_template else 'raw prompt'}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_with_direct_generation(model_name, system_prompt, user_prompt):
    """Test with direct generation without tokenizer.generate."""
    logger.info(f"Testing with direct generation")
    
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Create a raw prompt
        raw_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
        logger.info(f"Raw prompt:\n{raw_prompt}")
        
        # Save to file
        prompt_file = log_dir / f"{model_name.replace('/', '_')}_direct_prompt.txt"
        with open(prompt_file, "w") as f:
            f.write(raw_prompt)
        
        # Tokenize the prompt
        input_ids = tokenizer.encode(raw_prompt, return_tensors="pt").to(model.device)
        
        # Generate with model.forward instead of model.generate
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
            
            # Get the next token prediction
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()
            next_token = tokenizer.decode([next_token_id])
            
            logger.info(f"Next token prediction: {next_token}")
            
            # Generate a few more tokens manually
            generated_ids = input_ids.clone()
            for _ in range(20):
                outputs = model(generated_ids)
                logits = outputs.logits
                next_token_logits = logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).item()
                generated_ids = torch.cat([generated_ids, torch.tensor([[next_token_id]]).to(model.device)], dim=1)
            
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            logger.info(f"Generated text:\n{generated_text}")
            
            # Save to file
            output_file = log_dir / f"{model_name.replace('/', '_')}_direct_output.txt"
            with open(output_file, "w") as f:
                f.write(generated_text)
            
            # Check if the response is structured
            try:
                # Try to extract the assistant's response after "Assistant:"
                match = re.search(r'Assistant:(.*)', generated_text, re.DOTALL)
                if match:
                    assistant_response = match.group(1).strip()
                else:
                    assistant_response = generated_text
                
                logger.info(f"Assistant response:\n{assistant_response}")
                
                # Check if the response is structured
                try:
                    parsed = json.loads(assistant_response)
                    if isinstance(parsed, dict) and 'role' in parsed and 'content' in parsed:
                        logger.info(f"Response is structured: {parsed}")
                        return True
                    else:
                        logger.info("Response is not structured JSON")
                        return False
                except:
                    # If it's not valid JSON, check if it contains 'role' and 'content'
                    if "'role'" in assistant_response and "'content'" in assistant_response:
                        logger.info("Response contains role and content but is not valid JSON")
                        return True
                    else:
                        logger.info("Response is plain text")
                        return False
            except Exception as e:
                logger.error(f"Error checking if response is structured: {e}")
                return False
    
    except Exception as e:
        logger.error(f"Error testing with direct generation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    logger.info("Starting structured output test")
    
    # Model to test
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    
    # System prompt
    system_prompt = "You are a helpful assistant."
    
    # User prompt
    user_prompt = "Hello, how are you?"
    
    # Test with chat template
    is_structured_with_chat_template = test_with_chat_template(model_name, system_prompt, user_prompt, use_chat_template=True)
    
    # Test without chat template
    is_structured_without_chat_template = test_with_chat_template(model_name, system_prompt, user_prompt, use_chat_template=False)
    
    # Test with direct generation
    is_structured_with_direct_generation = test_with_direct_generation(model_name, system_prompt, user_prompt)
    
    # Log results
    logger.info(f"Is structured with chat template: {is_structured_with_chat_template}")
    logger.info(f"Is structured without chat template: {is_structured_without_chat_template}")
    logger.info(f"Is structured with direct generation: {is_structured_with_direct_generation}")
    
    logger.info("Structured output test completed")

if __name__ == "__main__":
    main()
