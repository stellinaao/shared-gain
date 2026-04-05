"""
paths.py

...paths.

Author: Stellina X. Ao
Created: 2025-03-26 # actually sometime before, but lost the record
Last Modified: 2026-03-26
Python Version: 3.11.14
"""

from pathlib import Path

# please edit as needed if you do not have Stellina's file structure

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT.parents[0] / "data-np"

FIGURES_DIR = PROJECT_ROOT / "figs"
FIGURES_DIR.mkdir(exist_ok=True)
