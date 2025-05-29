#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
embedding_utils.py - Embedding and similarity utilities for ArGen dataset generator
=================================================================================
Contains functions for generating embeddings and detecting similar prompts
"""

import os
import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
from sentence_transformers import SentenceTransformer, util

# Import the import helper
try:
    # Try relative import first (for integrated mode)
    from .import_helper import STANDALONE_MODE, get_import
except ImportError:
    # Fall back to direct import (for standalone mode)
    try:
        from import_helper import STANDALONE_MODE, get_import
    except ImportError:
        # Last resort: define our own standalone mode detection
        import sys
        try:
            import argen.data.utils
            STANDALONE_MODE = False
        except ImportError:
            STANDALONE_MODE = True

        # Define a simple get_import function
        import importlib
        def get_import(module_name):
            if STANDALONE_MODE:
                return importlib.import_module(module_name)
            else:
                return importlib.import_module(f".{module_name}", package="argen.data.generator")

# Import config module using the helper
try:
    config_module = get_import('config')
    log = config_module.log
    DEFAULT_EMBEDDING_MODEL = config_module.DEFAULT_EMBEDDING_MODEL
except Exception as e:
    # Fallback to direct import if helper fails
    if STANDALONE_MODE:
        from config import log, DEFAULT_EMBEDDING_MODEL
    else:
        from .config import log, DEFAULT_EMBEDDING_MODEL

# Global embedding model and device
_embedding_model: Optional[SentenceTransformer] = None
_device: Optional[str] = None

def init_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> bool:
    """Initialize the sentence transformer embedding model."""
    global _embedding_model, _device

    if _embedding_model is not None:
        log.info(f"Embedding model already initialized: {model_name}")
        return True

    try:
        # Determine device based on GPU availability
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Initializing embedding model {model_name} on {_device}")

        # Load the model
        _embedding_model = SentenceTransformer(model_name)
        _embedding_model.to(_device)

        # Test the model with a simple text
        test_text = "Test embedding generation"
        log.info(f"Generating embeddings for batch of 1 texts")
        # Directly use the model to avoid circular dependency with get_embeddings_batch
        _embedding_model.encode([test_text], convert_to_tensor=True, device=_device, show_progress_bar=False)

        log.info(f"Successfully initialized embedding model: {model_name}")
        return True
    except Exception as e:
        log.error(f"Failed to initialize embedding model: {e}")
        _embedding_model = None
        _device = None
        return False

def get_embeddings_batch(texts: List[str]) -> Optional[np.ndarray]:
    """Generates embeddings for a batch of texts using the local model."""
    global _embedding_model, _device

    # Double-check if model is initialized, and try to reinitialize if not
    if _embedding_model is None or _device is None:
        log.warning("Embedding model not initialized. Attempting to initialize with default model...")
        if not init_embedding_model(DEFAULT_EMBEDDING_MODEL):
            log.error("Failed to initialize embedding model. Cannot generate embeddings.")
            return None

    if not texts:
        log.warning("Empty text list provided to get_embeddings_batch")
        return np.array([])

    log.info(f"Generating embeddings for batch of {len(texts)} texts")
    try:
        # Ensure model is on the correct device
        _embedding_model.to(_device)
        embeddings = _embedding_model.encode(texts, convert_to_tensor=True, device=_device, show_progress_bar=False)
        log.debug(f"Generated embeddings batch shape: {embeddings.shape}")
        # Detach from GPU and move to CPU for numpy conversion if needed later
        return embeddings.cpu().numpy()
    except Exception as e:
        log.error(f"Failed to generate embeddings batch: {e}")
        # Try to reinitialize the model in case it's in a bad state
        _embedding_model = None
        _device = None
        init_embedding_model(DEFAULT_EMBEDDING_MODEL)
        return None

def find_near_duplicates_batch(
    new_embeddings: np.ndarray,
    existing_embeddings: Optional[np.ndarray],
    threshold: float
) -> Tuple[List[int], np.ndarray]:
    """
    Identifies near-duplicates between new and existing embeddings using batched cosine similarity.

    Args:
        new_embeddings: Numpy array of embeddings for the new batch of prompts.
        existing_embeddings: Numpy array of embeddings for existing unique prompts, or None if empty.
        threshold: Cosine similarity threshold for duplication.

    Returns:
        A tuple containing:
        - List of indices (relative to new_embeddings) that are NOT duplicates.
        - Numpy array of the unique embeddings identified in this batch.
    """
    global _device

    if existing_embeddings is None or existing_embeddings.shape[0] == 0:
        log.debug("No existing embeddings. All new embeddings are unique.")
        return list(range(new_embeddings.shape[0])), new_embeddings

    log.debug(f"Comparing {new_embeddings.shape[0]} new embeddings against {existing_embeddings.shape[0]} existing.")

    # Convert numpy arrays to PyTorch tensors and move to GPU if available
    new_tensor = torch.tensor(new_embeddings).to(_device)
    existing_tensor = torch.tensor(existing_embeddings).to(_device)

    # Calculate cosine similarity between all new and all existing embeddings
    # Shape: (num_new, num_existing)
    cos_sim = util.pytorch_cos_sim(new_tensor, existing_tensor)

    # Find the maximum similarity for each new embedding against all existing ones
    # Shape: (num_new,)
    max_sim, _ = torch.max(cos_sim, dim=1)

    # Identify indices where max similarity is below the threshold
    # These are the unique ones relative to the existing set
    unique_indices_mask = max_sim < threshold
    unique_indices_batch = torch.where(unique_indices_mask)[0]  # Indices within the current batch

    log.debug(f"Found {len(unique_indices_batch)} unique embeddings in this batch (max similarity < {threshold}).")

    # Also need to check for duplicates *within* the new batch itself
    # We only consider the ones deemed unique relative to the existing set
    if len(unique_indices_batch) > 1:
        potentially_unique_embeddings = new_tensor[unique_indices_batch]
        # Compare potentially unique embeddings against each other
        intra_batch_sim = util.pytorch_cos_sim(potentially_unique_embeddings, potentially_unique_embeddings)
        # Set diagonal to 0 to ignore self-similarity
        intra_batch_sim.fill_diagonal_(0)
        # Find max similarity for each *within* the potentially unique set
        intra_max_sim, _ = torch.max(intra_batch_sim, dim=1)
        # Keep only those whose max similarity *within the batch* is also below threshold
        intra_unique_mask = intra_max_sim < threshold
        final_unique_indices_in_batch = unique_indices_batch[intra_unique_mask]
        log.debug(f"Intra-batch check: {len(final_unique_indices_in_batch)} embeddings are unique within the batch.")
    elif len(unique_indices_batch) == 1:
        final_unique_indices_in_batch = unique_indices_batch  # Single item is always unique within batch
    else:
        final_unique_indices_in_batch = torch.tensor([], dtype=torch.long, device=_device)  # Empty tensor

    # Get the actual embeddings that are unique
    unique_embeddings_batch = new_tensor[final_unique_indices_in_batch].cpu().numpy()

    return final_unique_indices_in_batch.cpu().tolist(), unique_embeddings_batch

# TF-IDF based reranking functions
def load_core60_dataset(file_path: str) -> List[str]:
    """Load the core60 dataset for TF-IDF reranking."""
    import json

    log.info(f"Loading core60 dataset from {file_path}")

    # Try the local symbolic link first
    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/eval_core_60.jsonl")

    if os.path.exists(local_path):
        try:
            with open(local_path) as fh:
                core_rows = [json.loads(l)["prompt"] for l in fh if "prompt" in json.loads(l)]

            # Validate core rows count
            if len(core_rows) >= 40:
                log.info(f"Loaded {len(core_rows)} core60 rows for TF-IDF ranking from symbolic link")
                return core_rows
            else:
                log.warning(f"Core60 file at {local_path} has only {len(core_rows)} rows - will try other paths")
        except Exception as e:
            log.warning(f"Error loading core60 file at {local_path}: {e}")

    # Try other possible locations for the core60 file
    possible_paths = [
        file_path,  # Original path
        os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path),  # Relative to this file
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../data/eval_core_60.jsonl"),  # Project root data folder
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/eval_core_60.jsonl"),  # Parent data folder
    ]

    # Try each path
    for path in possible_paths:
        if path == local_path:  # Skip if we already tried this path
            continue
        if os.path.exists(path):
            log.info(f"Found core60 file at {path}")
            try:
                with open(path) as fh:
                    core_rows = [json.loads(l)["prompt"] for l in fh if "prompt" in json.loads(l)]

                # Validate core rows count
                if len(core_rows) >= 40:
                    log.info(f"Loaded {len(core_rows)} core60 rows for TF-IDF ranking")
                    return core_rows
                else:
                    log.warning(f"Core60 file at {path} has only {len(core_rows)} rows - trying next path")
            except Exception as e:
                log.warning(f"Error loading core60 file at {path}: {e} - trying next path")

    # If we get here, we didn't find a valid file, so create a default one
    log.warning(f"Core60 file not found at any checked location, creating default at {file_path}")
    _write_default_core60(file_path)

    # Load the default core rows
    try:
        with open(file_path) as fh:
            core_rows = [json.loads(l)["prompt"] for l in fh if "prompt" in json.loads(l)]

        log.info(f"Loaded {len(core_rows)} core60 rows from default file")
        return core_rows
    except Exception as e:
        log.error(f"Error loading default core60 file: {e}")
        return []

def _write_default_core60(path: str) -> None:
    """Write a default core60 file with placeholder prompts."""
    import json

    sample = [{"prompt": f"Q{i} placeholder medical question"} for i in range(60)]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "w", encoding="utf-8") as fh:
        for row in sample:
            fh.write(json.dumps(row) + "\n")

    log.info(f"Created default core60 file at {path}")

def init_tfidf_vectorizer(core_rows: List[str]):
    """Initialize the TF-IDF vectorizer with core rows."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(min_df=1, stop_words="english")
    vectorizer.fit(core_rows)
    core_matrix = vectorizer.transform(core_rows)

    log.info(f"TF-IDF vectorizer initialized with {len(core_rows)} reference prompts")
    return vectorizer, core_matrix

def too_close_to_core(prompt: str, vectorizer, core_matrix, threshold: float = 0.8) -> bool:
    """Check if a prompt is too similar to core60 prompts using TF-IDF."""
    v = vectorizer.transform([prompt])
    similarity = (v @ core_matrix.T).max()
    return similarity >= threshold