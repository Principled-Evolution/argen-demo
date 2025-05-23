#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_medical_detection.py - Simple verification script for medical detection
=============================================================================
Tests the medical detection functionality in the scenario generator
"""

import time
import sys
import os

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import the modules with absolute imports
from medical_terms import (
    looks_medical,
    looks_medical_trie,
    looks_medical_ner,
    medical_entity_score,
    get_nlp,
    get_medical_entities
)
from config import log

def main():
    """Run verification tests for medical detection."""
    log.info("Starting medical detection verification")

    # Test sentences
    medical_sentences = [
        "The patient has hypertension and diabetes.",
        "What are the side effects of metformin?",
        "How do I manage my blood pressure?",
        "What is the recommended dosage of insulin for type 2 diabetes?",
        "Can you explain what an angioplasty procedure involves?"
    ]

    non_medical_sentences = [
        "What's the weather like today?",
        "Can you help me with my homework?",
        "What's the best restaurant in town?",
        "How do I reset my password?",
        "Tell me a joke."
    ]

    # Check if scispaCy is available
    nlp = get_nlp()
    if nlp is not None:
        log.info("scispaCy model is available - testing NER-based detection")
    else:
        log.warning("scispaCy model is not available - testing will use trie-based fallback")

    # Test trie-based detection
    log.info("\nTesting trie-based detection:")
    trie_medical_count = 0
    for i, sentence in enumerate(medical_sentences):
        is_medical = looks_medical_trie(sentence)
        trie_medical_count += int(is_medical)
        log.info(f"Medical sentence {i+1}: {is_medical}")

    trie_non_medical_count = 0
    for i, sentence in enumerate(non_medical_sentences):
        is_medical = looks_medical_trie(sentence)
        trie_non_medical_count += int(is_medical)
        log.info(f"Non-medical sentence {i+1}: {is_medical}")

    log.info(f"Trie-based detection: {trie_medical_count}/5 medical sentences detected")
    log.info(f"Trie-based detection: {trie_non_medical_count}/5 non-medical sentences incorrectly detected")

    # Test combined detection (the actual looks_medical function)
    log.info("\nTesting combined detection:")
    combined_medical_count = 0
    for i, sentence in enumerate(medical_sentences):
        is_medical = looks_medical(sentence)
        combined_medical_count += int(is_medical)
        log.info(f"Medical sentence {i+1}: {is_medical}")

    combined_non_medical_count = 0
    for i, sentence in enumerate(non_medical_sentences):
        is_medical = looks_medical(sentence)
        combined_non_medical_count += int(is_medical)
        log.info(f"Non-medical sentence {i+1}: {is_medical}")

    log.info(f"Combined detection: {combined_medical_count}/5 medical sentences detected")
    log.info(f"Combined detection: {combined_non_medical_count}/5 non-medical sentences incorrectly detected")

    # Test medical entity score
    log.info("\nTesting medical entity score:")
    for i, sentence in enumerate(medical_sentences):
        score = medical_entity_score(sentence)
        log.info(f"Medical sentence {i+1} score: {score:.2f}")

    for i, sentence in enumerate(non_medical_sentences):
        score = medical_entity_score(sentence)
        log.info(f"Non-medical sentence {i+1} score: {score:.2f}")

    # Performance test
    log.info("\nPerformance test:")
    start_time = time.time()
    for _ in range(10):  # Process each sentence 10 times
        for sentence in medical_sentences + non_medical_sentences:
            _ = looks_medical(sentence)
    end_time = time.time()
    log.info(f"Processed 100 sentences in {end_time - start_time:.2f} seconds")
    log.info(f"Average time per sentence: {(end_time - start_time) / 100 * 1000:.2f} ms")

if __name__ == "__main__":
    main()
