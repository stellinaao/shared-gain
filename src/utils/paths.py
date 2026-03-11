from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FIGURES_DIR = PROJECT_ROOT / "figs"
FIGURES_DIR.mkdir(exist_ok=True)
