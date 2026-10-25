"""Public test fixture entry point for the shared offline synthetic generator."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from crossskinnet.data.synthetic import DATASETS, make_synthetic_data

__all__ = ["DATASETS", "make_synthetic_data"]

if __name__ == "__main__":
    print(make_synthetic_data())
