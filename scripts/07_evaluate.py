"""CLI for held-out, zero-shot, robustness and explainability evaluation."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from crossskinnet.eval.evaluate import run
from crossskinnet.io import config


def main() -> None:
    """Evaluate one checkpoint; comparison CSV accumulates completed backbone runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/default.yaml'))
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--explainability', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint or config(args.config)['output'] / 'runs/tiny_finetuned/best_calibrated.pt'
    try:
        print(run(args.config, checkpoint, args.explainability))
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.exit(2, f'Evaluation failed: {error}\n')


if __name__ == '__main__':
    main()
