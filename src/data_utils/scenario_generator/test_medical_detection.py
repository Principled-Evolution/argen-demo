#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_medical_detection.py - Test script for medical term detection
=================================================================
Tests the medical term detection functionality in medical_terms.py
"""

import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger("medical_detection_test")

# Import directly from the local modules
try:
    from medical_terms import (
        looks_medical,
        looks_medical_trie,
        looks_medical_ner,
        medical_entity_score,
        get_nlp,
        get_medical_entities
    )
    log.info("Successfully imported medical_terms module")
except ImportError as e:
    log.error(f"Error importing medical_terms: {e}")
    log.info("Trying alternative import path...")
    try:
        from .medical_terms import (
            looks_medical,
            looks_medical_trie,
            looks_medical_ner,
            medical_entity_score,
            get_nlp,
            get_medical_entities
        )
        log.info("Successfully imported medical_terms module using relative import")
    except ImportError as e:
        log.error(f"Error importing medical_terms with relative import: {e}")
        raise

def test_medical_detection():
    """Test the medical term detection functionality."""
    # Test sentences
    medical_sentences = [
        "The patient has hypertension and diabetes.",
        "What are the side effects of metformin?",
        "How do I manage my blood pressure?",
        "What is the recommended dosage of insulin for type 2 diabetes?",
        "Can you explain what an angioplasty procedure involves?",
        "My doctor diagnosed me with psoriasis, what treatments are available?",
        "I'm experiencing chest pain and shortness of breath.",
        "What are the symptoms of a stroke?",
        "How often should I monitor my blood glucose levels?",
        "What are the contraindications for statins?"
    ]

    non_medical_sentences = [
        "What's the weather like today?",
        "Can you help me with my homework?",
        "What's the best restaurant in town?",
        "How do I reset my password?",
        "Tell me a joke.",
        "What's the capital of France?",
        "How do I cook pasta?",
        "What's the latest news?",
        "Can you recommend a good book?",
        "What's the meaning of life?"
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

    log.info(f"Trie-based detection: {trie_medical_count}/10 medical sentences detected")
    log.info(f"Trie-based detection: {trie_non_medical_count}/10 non-medical sentences incorrectly detected")

    # Test NER-based detection if available
    if nlp is not None:
        log.info("\nTesting NER-based detection:")
        ner_medical_count = 0
        for i, sentence in enumerate(medical_sentences):
            is_medical = looks_medical_ner(sentence)
            ner_medical_count += int(is_medical)
            log.info(f"Medical sentence {i+1}: {is_medical}")
            entities = get_medical_entities(sentence)
            if entities:
                log.info(f"  Entities: {entities}")

        ner_non_medical_count = 0
        for i, sentence in enumerate(non_medical_sentences):
            is_medical = looks_medical_ner(sentence)
            ner_non_medical_count += int(is_medical)
            log.info(f"Non-medical sentence {i+1}: {is_medical}")
            if is_medical:
                entities = get_medical_entities(sentence)
                log.info(f"  Entities (false positive): {entities}")

        log.info(f"NER-based detection: {ner_medical_count}/10 medical sentences detected")
        log.info(f"NER-based detection: {ner_non_medical_count}/10 non-medical sentences incorrectly detected")

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

    log.info(f"Combined detection: {combined_medical_count}/10 medical sentences detected")
    log.info(f"Combined detection: {combined_non_medical_count}/10 non-medical sentences incorrectly detected")

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
    log.info(f"Processed 200 sentences in {end_time - start_time:.2f} seconds")
    log.info(f"Average time per sentence: {(end_time - start_time) / 200 * 1000:.2f} ms")

if __name__ == "__main__":
    test_medical_detection()
