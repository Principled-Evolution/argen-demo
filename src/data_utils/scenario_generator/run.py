#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run.py - Standalone entry point for ArGen scenario generator
===========================================================
This script allows running the scenario generator without requiring
the parent project's environment or imports.
"""

import os
import sys
import asyncio

# Add the current directory to the path
package_dir = os.path.dirname(os.path.abspath(__file__))
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)

# Import the import helper
from import_helper import get_import

# Import main module using the helper
main_module = get_import('main')
main = main_module.main

if __name__ == "__main__":
    print("Running ArGen Scenario Generator in standalone mode")
    # Run the generator
    asyncio.run(main())