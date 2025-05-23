#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ArGen Scenario Generator Package
===============================
A modular implementation of the ArGen dataset generator that creates
healthcare scenario prompts for testing and training language models.

This package provides functionality to generate diverse, challenging prompts
that test an AI's ability to stay within healthcare domain boundaries.
"""

import importlib.util
import os
import sys

# Determine if we're running as a standalone package or as part of the parent project
try:
    import src.data_utils
    STANDALONE_MODE = False
except ImportError:
    STANDALONE_MODE = True

# For standalone mode, ensure the package directory is in the path
if STANDALONE_MODE:
    package_dir = os.path.dirname(os.path.abspath(__file__))
    if package_dir not in sys.path:
        sys.path.insert(0, package_dir)

# Import with the appropriate style based on mode
if STANDALONE_MODE:
    # Standalone mode - use direct imports
    from config import (
        DATASETS, OUTPUT_FILES, DOMAIN, COMPLETION,
        RISK_THRESHOLD, DEFAULT_GENERATION_MODEL,
        DEFAULT_EMBEDDING_MODEL, DEFAULT_BASELINE,
        get_model_limits
    )
    from main import main
else:
    # Integrated mode - use relative imports
    from .config import (
        DATASETS, OUTPUT_FILES, DOMAIN, COMPLETION,
        RISK_THRESHOLD, DEFAULT_GENERATION_MODEL,
        DEFAULT_EMBEDDING_MODEL, DEFAULT_BASELINE,
        get_model_limits
    )
    from .main import main

__version__ = "0.1.0"

__all__ = [
    'main',
    'DATASETS',
    'OUTPUT_FILES',
    'DOMAIN',
    'COMPLETION',
    'RISK_THRESHOLD',
    'DEFAULT_GENERATION_MODEL',
    'DEFAULT_EMBEDDING_MODEL',
    'DEFAULT_BASELINE',
    'get_model_limits',
    'STANDALONE_MODE',
]