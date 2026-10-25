"""CLI for group splits and the portable unified dataset."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from crossskinnet.data.build_unified import run


def main() -> None:
    """Build the dataset with configured output and reviewed labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/default.yaml'))
    parser.add_argument('--mapping', type=Path, default=Path('configs/label_mapping.csv'))
    parser.add_argument('--schemas', type=Path, help='Dataset schemas, including identifier scope; defaults beside --config')
    args = parser.parse_args()
    try:
        run(args.config, args.mapping, args.schemas)
    except (KeyError, OSError, ValueError) as error:
        parser.exit(2, f'Unified dataset failed: {error}\n')


if __name__ == '__main__':
    main()
