"""Generate robot joint plans with NumPy geometry, without connecting to Franka."""
import hashlib
import json
from pathlib import Path
from lab3_cspace.plan import build


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def generate(config_path,destination,start=None):
    config_path=Path(config_path); destination=Path(destination)
    # Read the calibration once. Never acknowledge a concurrently edited config
    # using joint angles generated against different values.
    text=config_path.read_text()
    from lab3_cspace.kinematics import validate_config
    config=validate_config(json.loads(text))
    plans={'patterns':build(config,'all',False,start),
           'patterns-reset':build(config,'all',True,start),
           'rotate-place':build(config,'rotate-place',False,start)}
    metadata={'schema':1,'geometry_hash':fingerprint({k:v for k,v in config.items()
              if k not in ('description','hardware_calibrated')}),'plans':{}}
    # Nothing is replaced until all three searches succeed. Install the
    # manifest last so an interrupted export is rejected by the runner.
    (destination/'plans').mkdir(parents=True,exist_ok=True)
    for name,plan in plans.items():
        path=destination/'plans'/(name+'.json')
        temporary=path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(plan,indent=2)+'\n'); temporary.replace(path)
        metadata['plans'][name]={'file':'plans/'+path.name,'sha256':fingerprint(plan)}
    if destination.resolve()!=config_path.parent.resolve() or config_path.name!='config.json':
        (destination/'config.json').write_text(text)
    path=destination/'manifest.json'; temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(metadata,indent=2)+'\n'); temporary.replace(path)
    print('Generated three NumPy joint plans in',destination,flush=True)
    return metadata
