"""CLI for reproducible CrossSkinNet training and validation calibration."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from crossskinnet.train.engine import run


def main() -> None:
    """Select a configured backbone and optional epoch/resume override."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/default.yaml'))
    parser.add_argument('--backbone', choices=['tiny', 'efficientnet', 'swin', 'dinov2', 'dinov3'], default='tiny')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--frozen', action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    try:
        print(run(args.config, args.backbone, args.epochs, args.resume, args.frozen))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f'Training failed: {error}\n')


if __name__ == '__main__':
    main()
