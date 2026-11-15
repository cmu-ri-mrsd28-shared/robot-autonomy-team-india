"""Generate joint plans and export a standalone robot-only folder offline."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lab3_cspace.kinematics import load_config
from lab3_cspace.plan import build


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def main():
    source=Path(__file__).resolve().parents[1]/'lab3_cspace_robot'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(source/'config.json'))
    parser.add_argument('--start-state',help='State JSON captured by robot run.py --state')
    parser.add_argument('--output',default=str(source))
    args=parser.parse_args(); cfg=load_config(args.config)
    start=None if not args.start_state else json.loads(Path(args.start_state).read_text())['q']
    # Complete every plan before modifying any operational plan files.
    patterns=build(cfg,'all',False,start)
    reset=build(cfg,'all',True,start)
    rotate=build(cfg,'rotate-place',False,start)
    plans={'patterns':patterns,'patterns-reset':reset,'rotate-place':rotate}
    dest=Path(args.output).resolve(); dest.mkdir(parents=True,exist_ok=True)
    (dest/'plans').mkdir(exist_ok=True)
    if dest!=source:
        for name in ('run.py','joint_executor.py','panda_model.py','requirements.txt','README.md'):
            shutil.copy2(source/name,dest/name)
    # Preserve the user's coordinate formatting/calibration file verbatim.
    text=Path(args.config).read_text(); (dest/'config.json').write_text(text)
    metadata={'schema':1,'geometry_hash':fingerprint({k:v for k,v in cfg.items() if k not in ('description','hardware_calibrated')}),'plans':{}}
    for name,plan in plans.items():
        path=dest/'plans'/(name+'.json')
        path.write_text(json.dumps(plan,indent=2)+'\n')
        metadata['plans'][name]={'file':'plans/'+path.name,'sha256':fingerprint(plan)}
    (dest/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Exported robot-only runtime and three joint plans to',dest)


if __name__=='__main__': main()
