"""CLI for duplicate hashing, removal and manual-audit outputs."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from crossskinnet.data.dedup import run


def main() -> None:
    """Run deduplication with the configured seed and threshold."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/default.yaml'))
    args = parser.parse_args()
    try:
        run(args.config)
    except (KeyError, OSError, ValueError) as error:
        parser.exit(2, f'Deduplication failed: {error}\n')


if __name__ == '__main__':
    main()
