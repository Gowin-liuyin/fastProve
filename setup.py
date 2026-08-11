"""Legacy setuptools entry point for older pip versions.

The project metadata remains canonical in ``pyproject.toml``.  This tiny
shim keeps legacy metadata/build paths usable on the Python 3.9/macOS
environment used by the prototype, whose system pip predates PEP 660 editable
installs. A modern virtualenv may still use editable mode; the system Xcode
Python should use a regular user install instead.
"""

from setuptools import find_packages, setup


setup(
    name="fastprove",
    version="0.1.0",
    description="Reference prototype for augmented covariant Transformer obfuscation",
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.8",
        "numpy>=2.0",
        "PyYAML>=6.0",
        "matplotlib>=3.9",
    ],
    extras_require={
        "test": ["pytest>=8.0"],
        "pretrained-lite": [
            "transformers>=4.57",
            "safetensors>=0.7",
        ],
        "pretrained": [
            "transformers>=4.57",
            "datasets>=4.0",
            "safetensors>=0.7",
        ],
    },
)
