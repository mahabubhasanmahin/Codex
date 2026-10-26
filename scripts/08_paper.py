"""Prepare, train, evaluate and summarize the complete configured paper workflow."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from crossskinnet.pipeline import prepare, train, evaluate, plan, summarize


def main() -> None:
    """Run one bounded phase/experiment; do not launch every GPU experiment silently."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','train','evaluate','run','plan','summarize'])
    parser.add_argument('--config',type=Path,default=Path('configs/paper.yaml'))
    parser.add_argument('--backbone',choices=['tiny','efficientnet','swin','dinov2','dinov3'],default='efficientnet')
    parser.add_argument('--architecture',choices=['crossskinnet','image_only','concat'],default='crossskinnet')
    parser.add_argument('--seed',type=int)
    parser.add_argument('--frozen',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--explainability',action=argparse.BooleanOptionalAction,default=True)
    args=parser.parse_args()
    try:
        if args.command in ('prepare','run'):
            print(prepare(args.config))
        if args.command in ('train','run'):
            print(train(args.config,args.backbone,args.architecture,args.seed,args.frozen,args.epochs))
        if args.command in ('evaluate','run'):
            print(evaluate(args.config,args.backbone,args.architecture,args.seed,args.frozen,args.explainability))
        if args.command=='plan':
            for row in plan(args.config):print(row)
        if args.command=='summarize':
            print(summarize(args.config))
    except (ValueError,KeyError,OSError,ImportError) as error:
        parser.exit(2,f'Paper workflow stopped: {error}\n')


if __name__=='__main__':
    main()
