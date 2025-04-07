from setuptools import setup, find_packages

setup(
    name="argen-grpo",
    version="0.1.0",
    description="ArGen GRPO Fine-Tuning Implementation",
    author="Principled Evolution",
    author_email="info@principled-evolution.com",
    url="https://github.com/Principled-Evolution/argen-demo",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "predibase>=0.1.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "jupyter>=1.0.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
