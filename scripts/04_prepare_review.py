"""Prepare duplicate candidates for selected clinical labels before site normalization."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from crossskinnet.data.review import prepare


def main() -> None:
    """Apply the mapping and create a raw preview without approving any duplicate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/default.yaml'))
    parser.add_argument('--mapping',type=Path,default=Path('configs/label_mapping.csv'))
    args = parser.parse_args()
    try:
        print(prepare(args.config,args.mapping))
    except (OSError,ValueError,KeyError) as error:
        parser.exit(2,f'Review preparation failed: {error}\n')


if __name__=='__main__':
    main()
