#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify the installation of the scenario_generator package
"""

import sys
import importlib.util

def check_import(module_name):
    spec = importlib.util.find_spec(module_name)
    if spec is not None:
        print(f"✅ {module_name} can be imported")
        return True
    else:
        print(f"❌ {module_name} cannot be imported")
        return False

print("--- Testing ArGen Scenario Generator Installation ---")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print("---")

# Check if we can import the package and its modules
modules_to_check = [
    "config",
    "main",
    "openai_utils",
    "embedding_utils",
    "medical_terms",
    "evaluation",
    "baseline_model",
    "generation",
]

# Add the current directory to the path
import os
package_dir = os.path.dirname(os.path.abspath(__file__))
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)

all_imports_ok = True
for module in modules_to_check:
    if not check_import(module):
        all_imports_ok = False

# Print summary
print("---")
if all_imports_ok:
    print("✅ All modules can be imported - installation looks good!")
else:
    print("❌ Some modules cannot be imported - check installation issues")

print("---")
print("To run the scenario generator:")
print("  poetry run python run.py --datasets smoke_test")
print("---")