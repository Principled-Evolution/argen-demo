#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
medical_terms.py - Medical term detection for ArGen dataset generator
====================================================================
Contains the trie structure and functions for medical term detection
with enhanced scispaCy NER-based detection
"""

import os
import re
import time
from typing import Dict, Set, List, Tuple, Optional, Any

import sys
import os

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from .config import log
except ImportError:
    # Fallback for standalone mode
    try:
        import config
        log = config.log
    except ImportError:
        # Last resort: create a basic logger
        import logging
        logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
        log = logging.getLogger("medical_terms")

# ---------------------------------- medical term trie -------------------------
class Trie(dict):
    """Trie data structure for prefix-based term lookup."""
    def __init__(self):
        super().__init__()
        self.end = False

# Basic medical terms
MEDICAL_TERMS: Set[str] = {
    "hypertension", "diabetes", "insulin", "fracture", "oncology", "chemotherapy",
    "bmi", "blood pressure", "bp", "statin", "metformin", "obstetrics", "gynecology",
    "cardiology", "stroke", "angioplasty", "ceftriaxone", "asthma", "bronchodilator",
    "dialysis", "hemodialysis", "psoriasis"
}

def load_medical_terms(file_path: str = "data/umls_5k_terms.txt") -> Set[str]:
    """Load medical terms from a file and add them to the MEDICAL_TERMS set."""
    global MEDICAL_TERMS

    # Try the local symbolic link first
    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/umls_5k_terms.txt")

    if os.path.exists(local_path):
        try:
            with open(local_path) as fh:
                for t in fh:
                    term = t.strip().lower()
                    if term:
                        MEDICAL_TERMS.add(term)
            log.info(f"Loaded additional medical terms from {local_path}")
            return MEDICAL_TERMS
        except Exception as e:
            log.warning(f"Error loading UMLS terms from {local_path}: {e}")

    # If local link doesn't work, try other possible locations
    possible_paths = [
        file_path,  # Original path
        os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path),  # Relative to this file
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../data/umls_5k_terms.txt"),  # Project root data folder
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/umls_5k_terms.txt"),  # Parent data folder
    ]

    for path in possible_paths:
        if path == local_path:  # Skip if we already tried this path
            continue
        try:
            with open(path) as fh:
                for t in fh:
                    term = t.strip().lower()
                    if term:
                        MEDICAL_TERMS.add(term)
            log.info(f"Loaded additional medical terms from {path}")
            return MEDICAL_TERMS
        except FileNotFoundError:
            continue

    log.warning(f"UMLS term file missing at all checked locations – basic seed list only (recall ↓)")
    return MEDICAL_TERMS

# Build trie from medical terms
def build_medical_trie() -> Trie:
    """Build a trie data structure from the MEDICAL_TERMS set."""
    trie = Trie()
    for word in MEDICAL_TERMS:
        node = trie
        for ch in word:
            node = node.setdefault(ch, Trie())
        node.end = True
    return trie

# Initialize the trie
TRIE = build_medical_trie()

def trie_has_prefix(text: str) -> bool:
    """Check if the trie contains a prefix of the given text."""
    node = TRIE
    for ch in text:
        node = node.get(ch)
        if node is None:
            return False
        if node.end:
            return True
    return False

def looks_medical_trie(sentence: str) -> bool:
    """Check if a sentence contains medical terms using trie-based approach."""
    for chunk in re.split(r"\W+", sentence.lower()):
        if trie_has_prefix(chunk):
            return True
    return False

# ---------------------------------- scispaCy NER-based detection -------------------------
# Global variables for caching the NLP pipeline
_nlp = None
_nlp_load_attempted = False
_nlp_load_time = 0
_medical_entity_cache = {}  # Simple cache for medical entity detection

# Medical entity types from scispaCy that strongly indicate medical content
MEDICAL_ENTITY_TYPES = {
    'DISEASE', 'CHEMICAL', 'DRUG', 'PROCEDURE', 'ANATOMY', 'PROBLEM',
    'TEST', 'TREATMENT', 'FINDING', 'BODY_PART', 'SIGN', 'SYMPTOM'
}

def get_nlp():
    """Get or initialize the spaCy NLP pipeline with caching."""
    global _nlp, _nlp_load_attempted, _nlp_load_time

    # Return cached pipeline if available
    if _nlp is not None:
        return _nlp

    # Don't retry loading if we've already failed and it's been less than 5 minutes
    if _nlp_load_attempted and (time.time() - _nlp_load_time) < 300:
        return None

    _nlp_load_time = time.time()
    _nlp_load_attempted = True

    try:
        import spacy
        import scispacy

        # Load the model with minimal components for speed
        log.info("Loading scispaCy NER model (en_core_sci_sm)...")
        start_time = time.time()
        _nlp = spacy.load("en_core_sci_sm", disable=["parser", "tagger", "lemmatizer"])
        load_time = time.time() - start_time
        log.info(f"scispaCy model loaded successfully in {load_time:.2f} seconds")
        return _nlp
    except Exception as e:
        log.warning(f"Failed to load scispaCy model: {e}")
        return None

def get_medical_entities(text: str) -> List[Tuple[str, str]]:
    """Extract medical entities from text using scispaCy NER."""
    # Check cache first
    if text in _medical_entity_cache:
        return _medical_entity_cache[text]

    nlp = get_nlp()
    if nlp is None:
        return []

    try:
        # Process the text and extract entities
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        # Cache the result
        _medical_entity_cache[text] = entities

        # Limit cache size to prevent memory issues
        if len(_medical_entity_cache) > 1000:
            # Remove a random 20% of entries when cache gets too large
            import random
            keys_to_remove = random.sample(list(_medical_entity_cache.keys()),
                                          k=int(len(_medical_entity_cache) * 0.2))
            for key in keys_to_remove:
                _medical_entity_cache.pop(key, None)

        return entities
    except Exception as e:
        log.warning(f"Error extracting medical entities: {e}")
        return []

def looks_medical_ner(text: str) -> bool:
    """Check if text contains medical entities using scispaCy NER."""
    entities = get_medical_entities(text)
    return len(entities) > 0

def looks_medical(sentence: str) -> bool:
    """
    Check if a sentence contains medical terms using scispaCy NER if available,
    falling back to trie-based detection if scispaCy is not available.
    """
    # Try NER-based detection first
    nlp = get_nlp()
    if nlp is not None:
        return looks_medical_ner(sentence)

    # Fall back to trie-based detection
    return looks_medical_trie(sentence)

def medical_entity_score(text: str) -> float:
    """
    Calculate medical entity score using scispaCy NER.
    Returns a score between 0.0 and 1.0 indicating the medical relevance.
    """
    # Try NER-based scoring
    entities = get_medical_entities(text)
    if entities:
        # Cap the score at 1.0 (4 or more entities)
        return min(len(entities) / 4.0, 1.0)

    # Fall back to trie-based detection if no entities found
    return 1.0 if looks_medical_trie(text) else 0.0

# Ensure UMLS terms are loaded
_ = load_medical_terms()