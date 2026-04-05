"""
paths.py

...paths.

Author: Stellina X. Ao
Created: 2025-03-26 # actually sometime before, but lost the record
Last Modified: 2026-03-26
Python Version: 3.11.14
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FIGURES_DIR = PROJECT_ROOT / "figs"
FIGURES_DIR.mkdir(exist_ok=True)
