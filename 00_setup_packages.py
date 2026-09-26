"""
00_setup_packages.py

Single responsibility:
    Ensure every third-party package used across the Problem 1 workflow
    is installed, once, before any of the other cells run. Run this
    cell first and only once per kernel session.
"""

import importlib
import subprocess
import sys

REQUIRED_PACKAGES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "torchmetrics": "torchmetrics",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
}


def ensure_packages(packages=REQUIRED_PACKAGES):
    """Import each required package, pip-installing it first if missing."""
    for import_name, pip_name in packages.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            print(f"'{import_name}' not found -- installing '{pip_name}'...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", pip_name])


ensure_packages()
print("All required packages are available.")