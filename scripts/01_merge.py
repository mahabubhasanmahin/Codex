"""Thin CLI for schema-driven raw pool ingestion."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crossskinnet.data.merge import merge_datasets


def main() -> None:
    """Parse configuration paths and report actionable ingestion failures."""
    defaults = Path(__file__).resolve().parents[1] / "configs"
    parser = argparse.ArgumentParser(description="Merge datasets into separate raw train and external pools.")
    parser.add_argument("--config", type=Path, default=defaults / "default.yaml")
    parser.add_argument("--schemas", type=Path, default=defaults / "dataset_schemas.yaml")
    args = parser.parse_args()
    try:
        merge_datasets(args.config, args.schemas)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"Merge failed: {error}\n")


if __name__ == "__main__":
    main()
