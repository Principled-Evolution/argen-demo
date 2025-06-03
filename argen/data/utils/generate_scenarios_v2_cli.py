#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_scenarios_v2_cli.py  ·  ArGen dataset generator CLI (2025‑05‑02)
=======================================================================
A command-line wrapper for the modularized ArGen dataset generator. This script
provides the same functionality as the original generate_scenarios_v2.py but
uses the modularized implementation from the scenario_generator package.

```bash
python src/data_utils/generate_scenarios_v2_cli.py --datasets smoke_test benchmarking grpo_training \
       --use-synthetic-negatives
```
"""

import os
import sys
import asyncio
import argparse

# Ensure the project root is in the path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Add the scenario_generator directory to the path
SCENARIO_GENERATOR_DIR = os.path.join(SCRIPT_DIR, 'scenario_generator')
if SCENARIO_GENERATOR_DIR not in sys.path:
    sys.path.insert(0, SCENARIO_GENERATOR_DIR)

# Import main function from the modular package
from argen.data.generator.main import main

if __name__ == "__main__":
    # Run the modularized generator
    asyncio.run(main())