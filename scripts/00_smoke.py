"""Run the entire pipeline with generated fake data and no weight downloads."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main() -> None:
    """Keep the demonstration isolated under configured output_root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/default.yaml'))
    args = parser.parse_args()
    from crossskinnet.demo import run
    try:
        run(args.config)
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.exit(2, f'Synthetic demo failed: {error}\n')


if __name__ == '__main__':
    main()
