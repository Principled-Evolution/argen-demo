#!/usr/bin/env python3
"""
Script to check the available parameters for GRPOConfig.
"""

import inspect
from trl import GRPOConfig

# Get the signature of the GRPOConfig constructor
signature = inspect.signature(GRPOConfig.__init__)

# Print the parameters
print("Available parameters for GRPOConfig:")
for param_name, param in signature.parameters.items():
    if param_name != 'self':
        if param.default != inspect.Parameter.empty:
            print(f"  {param_name}: default={param.default}")
        else:
            print(f"  {param_name}")

# Create a minimal config to check the attributes
config = GRPOConfig(output_dir="test")

# Print the attributes
print("\nAttributes of GRPOConfig instance:")
for attr_name in dir(config):
    if not attr_name.startswith('_') and not callable(getattr(config, attr_name)):
        print(f"  {attr_name}: {getattr(config, attr_name)}")
