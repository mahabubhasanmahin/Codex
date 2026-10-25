"""Review exact/perceptual evidence and sample non-ISIC overlap components."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from crossskinnet.data.audit import review_overlap


def main() -> None:
    """Read an existing audit without rescanning or redownloading datasets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/default.yaml'))
    args = parser.parse_args()
    try:
        print(review_overlap(args.config))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f'Overlap review failed: {error}\n')


if __name__ == '__main__':
    main()
