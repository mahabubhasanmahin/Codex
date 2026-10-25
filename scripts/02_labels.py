"""Thin CLI for label inventory and reviewed mapping application."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from crossskinnet.data.labels import apply, inventory


def main() -> None:
    """Select the explicit label operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['inventory', 'apply'])
    parser.add_argument('--config', type=Path, default=Path('configs/default.yaml'))
    parser.add_argument('--mapping', type=Path, default=Path('configs/label_mapping.csv'))
    args = parser.parse_args()
    try:
        (inventory if args.command == 'inventory' else apply)(args.config, args.mapping)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f'Label harmonization failed: {error}\n')


if __name__ == '__main__':
    main()
