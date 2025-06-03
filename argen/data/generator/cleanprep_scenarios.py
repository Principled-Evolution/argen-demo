import argparse
import json
import os
import re
from pathlib import Path

# --- Configuration ---
# Default input file path, adjust as needed
DEFAULT_INPUT_FILE = os.path.join(os.path.dirname(__file__), '../../data/dummy_data_for_prompt_cleaning.jsonl') # Adjusted for local testing if needed

def main():
    parser = argparse.ArgumentParser(description="Prepare scenarios JSONL file by keeping only prompt and tier.")
    parser.add_argument(
        "input_file",
        nargs="?",
        default=DEFAULT_INPUT_FILE,
        help=f"Path to the input JSONL file (default: {DEFAULT_INPUT_FILE})"
    )
    args = parser.parse_args()

    input_file_path = args.input_file

    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found at {input_file_path}")
        return

    base, ext = os.path.splitext(input_file_path)
    output_file_path = f"{base}-cleanprep{ext}"

    processed_items = 0
    skipped_items = 0
    prompts_with_remaining_tier_word = 0

    print(f"Processing input file: {input_file_path}")
    print(f"Output will be written to: {output_file_path}")

    # Define regex patterns for multi-pass cleaning
    # Pass 1: Parenthesized Annotations (greedy but within parentheses)
    # Looks for parentheses containing keywords like Tier, Scope, or common values.
    regex_pass_1 = r'\s*\([^)]*(?:\bTier\b|\bScope\b|S\d+|[A-Ca-c])(?:/[A-Ca-c]|/S\d+)*[^)]*\)\s*$'

    # Pass 2: Unparenthesized "This is..." Prefix Annotations
    # Catches annotations starting with "This is..." not in parentheses.
    regex_pass_2 = r'\s*\bThis is\s+(?:(?:(?:\bTier\b|\bScope\b)[:\s]*|[A-Ca-c]\b|S\d+\b)(?:\s*[,/-]?\s*)?){1,3}\s*$'

    # Pass 3: Unparenthesized Simple End-of-Line Annotations
    # For simpler, unparenthesized Tier/Scope value combinations at the end.
    # Allows for labels, values, and combinations like "S0, Tier B" or "Tier C".
    # Made more flexible to catch isolated or combined terms.
    tier_scope_keywords_or_values = r'(?:\bTier\b[:\s]*[A-Ca-c](?:/[A-Ca-c])?|\bScope\b[:\s]*S\d+(?:/S\d+)*|[A-Ca-c](?:/[A-Ca-c])?|S\d+(?:/S\d+)*)'
    regex_pass_3 = rf'\s*(?:{tier_scope_keywords_or_values}(?:\s*[,/-]?\s*{tier_scope_keywords_or_values})*\s*)$\s*'

    # Regex for final sweep to count remaining "Tier" occurrences
    regex_final_sweep_tier = r'\bTier\b'

    with open(input_file_path, 'r', encoding='utf-8') as infile, \
         open(output_file_path, 'w', encoding='utf-8') as outfile:
        for line_num, line in enumerate(infile, 1):
            try:
                item = json.loads(line)
                prompt_text = item.get("prompt")

                if prompt_text is not None:
                    processed_items += 1
                    original_prompt_for_logging = prompt_text # Keep original for logging if needed

                    # --- Start Multi-pass Cleaning ---
                    cleaned_prompt_text = prompt_text

                    # Pass 1
                    cleaned_prompt_text = re.sub(regex_pass_1, '', cleaned_prompt_text, flags=re.IGNORECASE).strip()
                    # Pass 2
                    cleaned_prompt_text = re.sub(regex_pass_2, '', cleaned_prompt_text, flags=re.IGNORECASE).strip()
                    # Pass 3
                    cleaned_prompt_text = re.sub(regex_pass_3, '', cleaned_prompt_text, flags=re.IGNORECASE).strip()
                    # --- End Multi-pass Cleaning ---

                    # Unicode character cleaning (can be done before or after, here after main regex for simplicity)
                    cleaned_prompt_text = cleaned_prompt_text.replace('\u2019', "'")
                    # Add more unicode replacements if necessary
                    # cleaned_prompt_text = cleaned_prompt_text.replace('\u201c', '"')
                    # cleaned_prompt_text = cleaned_prompt_text.replace('\u201d', '"')
                    # cleaned_prompt_text = cleaned_prompt_text.replace('\u2018', "'")
                    # cleaned_prompt_text = cleaned_prompt_text.replace('\u2013', '-')
                    # cleaned_prompt_text = cleaned_prompt_text.replace('\u2014', '--')
                    # cleaned_prompt_text = cleaned_prompt_text.replace('\u2026', '...')

                    # Final sweep to check for remaining "Tier" word
                    if re.search(regex_final_sweep_tier, cleaned_prompt_text, flags=re.IGNORECASE):
                        prompts_with_remaining_tier_word += 1
                        # Optionally log the specific prompt that still has "Tier"
                        # print(f"Warning: Line {line_num} still contains 'Tier' after cleaning: '{cleaned_prompt_text[:100]}...'")

                    record = {"id": processed_items, "prompt": cleaned_prompt_text}
                    outfile.write(json.dumps(record) + '\n')
                else:
                    skipped_items += 1
                    print(f"Warning: Skipping line {line_num} due to missing 'prompt' field: {line.strip()[:100]}...")

            except json.JSONDecodeError:
                skipped_items += 1
                print(f"Warning: Skipping invalid JSON line {line_num}: {line.strip()[:100]}...")
                continue

    print(f"\nProcessing complete.")
    print(f"Successfully processed {processed_items} items.")
    print(f"Skipped {skipped_items} items.")
    print(f"Number of prompts still containing the word 'Tier' (case-insensitive) after cleaning: {prompts_with_remaining_tier_word}")

if __name__ == "__main__":
    main() 