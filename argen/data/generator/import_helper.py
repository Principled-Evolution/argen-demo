#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
import_helper.py - Helper module for consistent imports
=======================================================
This module provides a consistent way to handle imports in both standalone and integrated modes.
"""

import os
import sys
import importlib.util

# Determine if we're running as a standalone package or as part of the parent project
try:
    import argen.data.utils
    STANDALONE_MODE = False
except ImportError:
    STANDALONE_MODE = True

# For standalone mode, ensure the package directory is in the path
if STANDALONE_MODE:
    package_dir = os.path.dirname(os.path.abspath(__file__))
    if package_dir not in sys.path:
        sys.path.insert(0, package_dir)

def get_import(module_name):
    """
    Get the appropriate import for a module based on the current mode.

    Args:
        module_name: The name of the module to import

    Returns:
        The imported module
    """
    # Try multiple import strategies to be more robust
    errors = []

    # Get the current directory for file-based imports
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Strategy 1: Direct import (works when the module is in sys.path)
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        errors.append(f"Direct import failed: {e}")

    # Strategy 2: File-based import from current directory
    try:
        module_path = os.path.join(current_dir, f"{module_name}.py")
        if os.path.exists(module_path):
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
    except Exception as e:
        errors.append(f"File-based import failed: {e}")

    # Strategy 3: Relative import (works when called from within the package)
    try:
        return importlib.import_module(f".{module_name}", package="argen.data.generator")
    except ImportError as e:
        errors.append(f"Relative import failed: {e}")

    # Strategy 4: Absolute import with full package path
    try:
        return importlib.import_module(f"argen.data.generator.{module_name}")
    except ImportError as e:
        errors.append(f"Absolute import failed: {e}")

    # If all strategies fail, raise an error with details
    raise ImportError(f"Could not import module '{module_name}'. Tried multiple strategies:\n" + "\n".join(errors))

# Pre-import all modules to avoid circular imports
def preload_modules():
    """Preload all modules to avoid circular imports."""
    modules = {}

    # First, import config which is needed by most modules
    modules['config'] = get_import('config')

    # Then import the rest of the modules
    for module_name in [
        'openai_utils',
        'embedding_utils',
        'baseline_model',
        'medical_terms',
        'evaluation',
        'generation',
        'main',
    ]:
        try:
            modules[module_name] = get_import(module_name)
        except ImportError as e:
            print(f"Warning: Could not import {module_name}: {e}")

    return modules

# Preload modules
MODULES = {}
