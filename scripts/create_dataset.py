"""
Standalone script to generate the dataset.
Run: python scripts/create_dataset.py
(from the manu/ directory)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.generate_dataset import main

if __name__ == "__main__":
    main()
