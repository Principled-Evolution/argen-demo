#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_imports.py - Script to update imports in all files
=========================================================
This script updates all Python files in the scenario_generator package
to use the import_helper module for consistent imports.
"""

import os
import re

# Files to update
files_to_update = [
    'baseline_model.py',
    'config.py',
    'embedding_utils.py',
    'evaluation.py',
    'generation.py',
    'medical_terms.py',
    'openai_utils.py',
]

# Import helper template
import_helper_template = """
# Import the import helper
from import_helper import STANDALONE_MODE, get_import

# Import with the appropriate style based on mode
if STANDALONE_MODE:
    # Standalone mode - use direct imports
    from {module} import {imports}
else:
    # Integrated mode - use relative imports
    from .{module} import {imports}
"""

# Function to update imports in a file
def update_imports(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all relative imports
    relative_imports = re.findall(r'from \.([\w_]+) import ([\w_, ]+)', content)
    
    # Replace each relative import with the import helper
    for module, imports in relative_imports:
        old_import = f"from .{module} import {imports}"
        new_import = import_helper_template.format(module=module, imports=imports)
        content = content.replace(old_import, new_import)
    
    # Write the updated content back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Updated imports in {file_path}")

# Update imports in all files
for file in files_to_update:
    file_path = os.path.join(os.path.dirname(__file__), file)
    if os.path.exists(file_path):
        update_imports(file_path)
    else:
        print(f"File not found: {file_path}")

print("Done updating imports!")
