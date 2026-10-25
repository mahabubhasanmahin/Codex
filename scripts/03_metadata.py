"""CLI for site inventory and missing-metadata normalization."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from crossskinnet.data.metadata import apply, inventory


def main() -> None:
    """Run the requested metadata operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['inventory', 'apply'])
    parser.add_argument('--config', type=Path, default=Path('configs/default.yaml'))
    parser.add_argument('--schemas', type=Path, default=Path('configs/dataset_schemas.yaml'))
    parser.add_argument('--mapping', type=Path, default=Path('configs/site_mapping.yaml'))
    args = parser.parse_args()
    try:
        if args.command == 'inventory':
            inventory(args.config, args.mapping)
        else:
            apply(args.config, args.schemas, args.mapping)
    except (KeyError, OSError, ValueError) as error:
        parser.exit(2, f'Metadata failed: {error}\n')


if __name__ == '__main__':
    main()
