import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import numpy as np
from .core import Config, Throw, sample_throws

def main():
    p = argparse.ArgumentParser(description='Simulation-only Panda box catching')
    p.add_argument('--episodes',type=int,default=1)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--kind',choices=['drop','lob','mixed'],default='lob')
    p.add_argument('--position',type=float,nargs=3,help='Explicit world xyz metres')
    p.add_argument('--velocity',type=float,nargs=3,help='Explicit world xyz m/s')
    p.add_argument('--config',type=Path,help='JSON overrides for Config fields')
    p.add_argument('--throws-in',type=Path,help='Replay saved held-out throws exactly')
    p.add_argument('--out',type=Path,default=Path('results/latest'))
    p.add_argument('--view',action='store_true',help='Use mjpython on macOS')
    p.add_argument('--trace',action='store_true')
    a = p.parse_args()
    if a.episodes < 1:
        p.error('episodes must be positive')
    if (a.position is None) != (a.velocity is None):
        p.error('position and velocity must be supplied together')
    if a.throws_in and a.position:
        p.error('throws-in and explicit launch are mutually exclusive')
    cfg = Config(**(json.loads(a.config.read_text()) if a.config else {}))
    cfg.validate()
    from .sim import Simulation
    start = time.perf_counter()
    sim = Simulation(cfg)
    setup_s = time.perf_counter()-start
    if a.throws_in:
        saved = json.loads(a.throws_in.read_text())
        throws = [Throw(**t) for t in saved['throws']]
        a.seed = saved.get('seed', a.seed)
    elif a.position:
        throws = [Throw(a.position,a.velocity,'explicit') for _ in range(a.episodes)]
    else:
        throws = sample_throws(a.seed,a.episodes,a.kind,sim.nominal_opening)
    if not throws:
        p.error('throw list is empty')
    for throw in throws:
        if len(throw.position)!=3 or len(throw.velocity)!=3 or not np.isfinite(throw.position+throw.velocity).all():
            p.error('throws must contain finite three-dimensional position and velocity')
    a.out.mkdir(parents=True,exist_ok=True)
    payload = dict(seed=a.seed,throws=[asdict(t) for t in throws])
    (a.out/'throws.json').write_text(json.dumps(payload,indent=2))
    throw_hash = hashlib.sha256(json.dumps(payload['throws'],sort_keys=True).encode()).hexdigest()
    viewer = None
    if a.view:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(sim.model,sim.data)
        viewer.cam.lookat[:] = [.4,0,.45]
        viewer.cam.distance = 1.8
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        viewer.opt.geomgroup[0] = 0
        viewer.opt.sitegroup[:] = 0
    results = []
    start = time.perf_counter()
    try:
        with (a.out/'episodes.jsonl').open('w') as out:
            for i,throw in enumerate(throws):
                result,trace = sim.rollout(throw,viewer,a.trace)
                result.update(episode=i,throw=asdict(throw))
                # Convert numpy bool scalars to plain JSON types.
                result = json.loads(json.dumps(result,default=lambda x:x.item()))
                results.append(result)
                out.write(json.dumps(result)+'\n')
                out.flush()
                if a.trace:
                    (a.out/f'trace-{i:04d}.json').write_text(json.dumps(trace))
    finally:
        if viewer:
            viewer.close()
    elapsed = time.perf_counter()-start
    n = len(results)
    success = sum(r['retained'] for r in results)
    def group(rs):
        return dict(n=len(rs),entered=sum(r['entered'] for r in rs),catches=sum(r['retained'] for r in rs),
                    catch_rate=sum(r['retained'] for r in rs)/len(rs),outcomes=dict(Counter(r['outcome'] for r in rs)),
                    rim_hits=sum(r['rim_hit'] for r in rs),
                    held_before_crossing=sum(r['held_before_crossing'] for r in rs),
                    mean_intercept_error_m=float(np.mean([r['intercept_error_m'] for r in rs])))
    rate = success/n
    z = 1.96
    center = (rate+z*z/(2*n))/(1+z*z/n)
    delta = z*np.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/(1+z*z/n)
    summary = dict(**group(results),catch_rate_wilson95=[center-delta,center+delta],
                   by_kind={kind:group([r for r in results if r['throw']['kind']==kind]) for kind in sorted({t.kind for t in throws})},
                   setup_wall_s=setup_s,batch_wall_s=elapsed,episodes_per_s=n/elapsed,
                   simulated_seconds_per_wall_second=n*cfg.duration/elapsed,
                   control_transitions_per_s=n*cfg.duration/cfg.control_dt/elapsed,
                   physics_steps_per_s=sum(r['physics_steps'] for r in results)/elapsed,
                   timing_includes_reset_and_logging=True,viewer=a.view,
                   config=asdict(cfg),seed=a.seed,throws_sha256=throw_hash,
                   versions={x:importlib.metadata.version(x) for x in ['robosuite','mujoco','numpy']},
                   platform=platform.platform(),machine=platform.machine())
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    main()
