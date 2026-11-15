"""Generate all robot plans locally with NumPy; no Franka or simulator needed."""
import argparse
import json
from pathlib import Path
if __package__:
    from .planner.robot_plans import generate
else:
    from planner.robot_plans import generate


def main():
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(root/'config.json'))
    parser.add_argument('--start-state',help='JSON captured by run.py --state')
    args=parser.parse_args()
    start=None if not args.start_state else json.loads(Path(args.start_state).read_text())['q']
    generate(args.config,root,start)


if __name__=='__main__': main()
