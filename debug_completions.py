#!/usr/bin/env python3
"""
Debug script to check if completions in GRPO training are different.
"""

import sys
import re
import json
from collections import defaultdict

def extract_completions(log_file):
    """
    Extract completions from the log file and check if they are different.
    """
    with open(log_file, 'r') as f:
        log_content = f.read()

    # Find all completion logs
    completion_pattern = r"Reward function \d+ - Completions: (\[\[.*?\]\])"
    completion_matches = re.findall(completion_pattern, log_content, re.DOTALL)

    if not completion_matches:
        print("No completion logs found.")
        return

    print(f"Found {len(completion_matches)} completion log entries.")

    identical_count = 0
    different_count = 0

    for i, match in enumerate(completion_matches):
        try:
            # Manual parsing since the JSON is complex and has single quotes
            completions = []
            # Extract content from each completion
            content_pattern = r"'content': ['\"]([^'\"]+(?:\\.[^'\"]+)*)['\"]"
            content_matches = re.findall(content_pattern, match, re.DOTALL)

            if len(content_matches) < 2:
                print(f"Entry {i+1}: Not enough completions found")
                continue

            # Compare the first 50 chars of each completion
            first_content = content_matches[0][:50]
            all_identical = all(content[:50] == first_content for content in content_matches)

            if all_identical:
                identical_count += 1
                print(f"Entry {i+1}: All completions appear to be IDENTICAL")
            else:
                different_count += 1
                print(f"Entry {i+1}: Completions are DIFFERENT")

                # Print the first 50 chars of each completion for comparison
                for j, content in enumerate(content_matches):
                    print(f"  Completion {j+1}: {content[:50]}...")
        except Exception as e:
            print(f"Error processing entry {i+1}: {e}")

    print("\nSummary:")
    print(f"Total completion entries: {len(completion_matches)}")
    print(f"Identical completion groups: {identical_count}")
    print(f"Different completion groups: {different_count}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python debug_completions.py <log_file>")
        sys.exit(1)

    log_file = sys.argv[1]
    extract_completions(log_file)
