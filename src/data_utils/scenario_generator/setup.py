#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for the scenario_generator package.
This allows for installing the package with pip directly.
"""

from setuptools import setup, find_packages

setup(
    name="argen-scenario-generator",
    version="0.1.0",
    packages=find_packages(),
    description="A modular scenario generator for ArGen healthcare assistant training",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    install_requires=[
        "openai>=1.0.0",
        "tiktoken>=0.5.0",
        "torch>=2.0.0",
        "sentence-transformers>=2.2.2",
        "transformers>=4.30.0",
        "scikit-learn>=1.2.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
)