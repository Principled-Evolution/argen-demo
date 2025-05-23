#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script to test imports
"""

import os
import sys

# Add the current directory to the path
package_dir = os.path.dirname(os.path.abspath(__file__))
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)

print(f"sys.path: {sys.path}")

try:
    import config
    print("✅ config imported successfully")
except ImportError as e:
    print(f"❌ Failed to import config: {e}")

try:
    import main
    print("✅ main imported successfully")
except ImportError as e:
    print(f"❌ Failed to import main: {e}")

try:
    import openai_utils
    print("✅ openai_utils imported successfully")
except ImportError as e:
    print(f"❌ Failed to import openai_utils: {e}")
