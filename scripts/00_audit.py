"""Produce mapping-free dataset feasibility reports before real training."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from crossskinnet.data.audit import run


def main() -> None:
    """Merge actual configured inputs and audit labels, metadata, duplicates and IDs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/default.yaml'))
    parser.add_argument('--schemas', type=Path, default=Path('configs/dataset_schemas.yaml'))
    args = parser.parse_args()
    try:
        run(args.config, args.schemas)
    except (OSError, KeyError, ValueError) as error:
        parser.exit(2, f'Dataset audit failed: {error}\n')


if __name__ == '__main__':
    main()
