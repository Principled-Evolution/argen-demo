#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_generator.py - Simple wrapper script to run the scenario generator
=====================================================================
This script provides a simple way to run the scenario generator in standalone mode.
"""

import os
import sys
import asyncio

# Add the current directory to the path
package_dir = os.path.dirname(os.path.abspath(__file__))
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)

# Set PYTHONPATH to include the parent directory
parent_dir = os.path.abspath(os.path.join(package_dir, '../../..'))
os.environ['PYTHONPATH'] = parent_dir

# Import the main function directly
from main import main

if __name__ == "__main__":
    print("Running ArGen Scenario Generator")
    # Run the generator
    asyncio.run(main())
