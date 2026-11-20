"""Scripted paddle and parked stock-gripper experiments; no hardware or learning."""
import argparse
from dataclasses import asdict
import json,time
from pathlib import Path
import mujoco
import numpy as np
from .core import Config,Throw,BallState,crossing,sample_throws
from .sim import Simulation

class Retention:
    def __init__(self):
        self.since=None;self.maximum=0.
    def update(self,t,inside):
        if inside:
            if self.since is None:self.since=t
            self.maximum=max(self.maximum,t-self.since)
        else:self.since=None
    @property
    def success(self):return self.maximum>=1.-1e-9

def rotation_for_normal(n):
    z=np.asarray(n,dtype=float);z/=np.linalg.norm(z)
    x=np.cross([0,1,0],z);x/=np.linalg.norm(x)
    return np.column_stack([x,np.cross(z,x),z])

def setup(tool,cfg):
    sim=Simulation(cfg,tool=tool)
    if tool=='gripper':
        # Park palm-up so a descending ball has access between the stock fingers.
        sim.home,_=sim.inverse_kinematics(np.array([.5,0,.45]))
        sim.reset();sim.nominal_opening=sim.data.site_xpos[sim.site].copy()
    return sim

def throws_for(sim,seed,n,kind,parked=False):
    throws=sample_throws(seed,n,kind,sim.nominal_opening+np.array([0,0,sim.cfg.radius if sim.tool=='paddle' else 0]))
    if parked:
        # Isolate securing first: all launches aimed at the fixed intercept,
        # with only 2 mm seeded perturbation. No moving-arm gripper claim.
        rng=np.random.default_rng(seed+10000)
        for throw in throws:
            state=BallState(0,np.array(throw.position),np.array(throw.velocity))
            _,p=crossing(state,sim.nominal_opening[2])
            delta=sim.nominal_opening[:2]-p[:2]+rng.uniform(-.002,.002,2)
            throw.position[:2]=(np.array(throw.position[:2])+delta).tolist()
    return throws

def rollout(sim,throw,mode='soft',opening=.07,closure_offset=-.04,jaw_speed=.10,delay=.03,record=False):
    """jaw_speed is total aperture target slew, m/s; offset relative to crossing.
    Actual jaw trajectories are dynamic, not prescribed or teleported.
    """
    sim.reset();d,m,c=sim.data,sim.model,sim.cfg
    if sim.tool=='gripper':
        d.qpos[7:9]=[opening/2,-opening/2]
    d.qpos[sim.ball_q:sim.ball_q+3]=throw.position
    d.qvel[sim.ball_v:sim.ball_v+3]=throw.velocity
    mujoco.mj_forward(m,d)
    height=sim.nominal_opening[2]+(c.radius if sim.tool=='paddle' else 0)
    pred=crossing(sim.read(),height)
    if pred is None:raise ValueError('No descending crossing')
    flight,target=pred
    target[2]-=c.radius if sim.tool=='paddle' else 0
    held=Retention();contact=False;entered=False;finger_strike=False;escaped=False
    first_contact=None;peak=0.;first_bounce_done=False;was_contact=False
    rows=[];goal=sim.home.copy();aperture=opening
    pads=[m.geom('gripper0_finger1_pad_collision').id,m.geom('gripper0_finger2_pad_collision').id]
    fingers=[{i for i in range(m.ngeom) if ('finger1' in m.geom(i).name)},
             {i for i in range(m.ngeom) if ('finger2' in m.geom(i).name)}]
    grip_rotation=d.site_xmat[sim.site].reshape(3,3).copy()
    start=time.perf_counter();previous_local=None;max_actual_speed=0.
    for k in range(round(c.duration/c.timestep)):
        t=float(d.time)
        if sim.tool=='paddle' and k%round(c.control_dt/c.timestep)==0:
            pos=target.copy();rot=np.eye(3)
            if mode=='soft':
                # 8 cm downward quintic stroke, then smooth stop.
                u=np.clip((t-(flight-.06))/.22,0,1)
                pos[2]-=.08*(10*u**3-15*u**4+6*u**5)
                if first_contact is not None:
                    ball=sim.read();center=d.site_xpos[sim.site]
                    error=ball.position[:2]-center[:2]
                    mujoco.mj_jacSite(m,d,sim.jacp,sim.jacr,sim.site)
                    relative=ball.velocity[:2]-(sim.jacp@d.qvel)[:2]
                    tilt=np.clip(-3*error-.15*relative,-.18,.18)
                    rot=rotation_for_normal([*tilt,1])
                    pos[:2]=np.clip(ball.position[:2]+.06*ball.velocity[:2],
                                    sim.nominal_opening[:2]-.2,sim.nominal_opening[:2]+.2)
            q,_=sim.inverse_kinematics(pos,rot)
            goal+=np.clip(q-goal,-1.5*c.control_dt,1.5*c.control_dt)
        sim.torque(goal)
        if sim.tool=='gripper':
            if t>=flight+closure_offset+delay:
                aperture=max(.034,aperture-jaw_speed*c.timestep)
            d.ctrl[7:9]=[aperture/2,-aperture/2]
        mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
        local=d.site_xmat[sim.site].reshape(3,3).T@(d.xpos[sim.ball_id]-d.site_xpos[sim.site])
        touch=False;both=[False,False];struck=False
        for con in d.contact:
            if sim.ball_geom not in (con.geom1,con.geom2):continue
            other=con.geom2 if con.geom1==sim.ball_geom else con.geom1
            if sim.tool=='paddle':touch|=other==m.geom('paddle').id
            else:
                for j in range(2):both[j]|=other in fingers[j]
                struck|=any(other in f for f in fingers)
        if sim.tool=='paddle':
            contact|=touch
            if touch and first_contact is None:first_contact=float(d.time)
            if first_contact is not None:
                if touch and not was_contact and d.time-first_contact>.01:first_bounce_done=True
                if not first_bounce_done:peak=max(peak,float(local[2]-c.radius))
            inside=(contact and np.linalg.norm(local[:2])+c.radius<=.075 and
                    c.radius-.003<=local[2]<=c.radius+.008)
        else:
            # Full-ball clearance across the measured jaw aperture; tip-plane entry.
            gap=abs(d.qpos[7])+abs(d.qpos[8])
            if previous_local is not None and previous_local[2]>.006>=local[2]:
                entered|=abs(local[0])+c.radius<=gap/2+.001 and abs(local[1])<.012
            finger_strike|=struck and not entered
            contact|=struck
            if struck and first_contact is None:first_contact=float(d.time)
            inside=entered and all(both) and abs(local[2])<.035 and abs(local[1])<.025
            max_actual_speed=max(max_actual_speed,float(abs(d.qvel[7])+abs(d.qvel[8])))
            escaped|=contact and np.linalg.norm(local)>.10
        held.update(float(d.time),inside)
        previous_local=local.copy();was_contact=touch
        if record and k%20==0:
            rows.append(dict(t=float(d.time),qpos=d.qpos.tolist(),ball=d.xpos[sim.ball_id].tolist(),
                             opening=d.site_xpos[sim.site].tolist(),jaw_aperture=float(abs(d.qpos[7])+abs(d.qpos[8]))))
    if any(w.number for w in d.warning) or not np.isfinite(d.qpos).all():raise RuntimeError('Numerical instability')
    outcome=('retained' if held.success else 'escaped_after_contact' if escaped else
             'finger_strike' if sim.tool=='gripper' and finger_strike else
             'contact_not_retained' if contact else 'miss')
    result=dict(tool=sim.tool,mode=mode,throw=asdict(throw),retained=held.success,
                max_retention_s=held.maximum,outcome=outcome,contact=contact,entered=entered,
                finger_strike=finger_strike,escaped=escaped,first_contact_s=first_contact,
                first_bounce_height_m=peak if first_contact is not None and sim.tool=='paddle' else None,
                first_bounce_complete=first_bounce_done,opening=opening,closure_offset=closure_offset,
                jaw_speed=jaw_speed,command_delay=delay,max_actual_jaw_speed=max_actual_speed,
                wall_s=time.perf_counter()-start)
    return json.loads(json.dumps(result,default=lambda x:x.item())),rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--tool',choices=['paddle','gripper'],required=True)
    p.add_argument('--episodes',type=int,default=20);p.add_argument('--seed',type=int,default=31415)
    p.add_argument('--kind',choices=['drop','lob'],default='lob');p.add_argument('--sweep',action='store_true')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--record',action='store_true')
    p.add_argument('--config',type=Path,help='JSON physics overrides, same Config as box')
    p.add_argument('--opening',type=float,default=.07);p.add_argument('--offset',type=float,default=-.04)
    p.add_argument('--jaw-speed',type=float,default=.10);p.add_argument('--delay',type=float,default=.03)
    a=p.parse_args()
    if not .04<=a.opening<=.08 or a.jaw_speed<=0 or a.delay<0 or a.episodes<1:p.error('Invalid jaw/episode parameters')
    options={'duration':3.0}
    if a.config:options.update(json.loads(a.config.read_text()))
    cfg=Config(**options);sim=setup(a.tool,cfg);a.out.mkdir(parents=True,exist_ok=True)
    throws=throws_for(sim,a.seed,a.episodes,a.kind,a.tool=='gripper')
    (a.out/'throws.json').write_text(json.dumps(dict(seed=a.seed,throws=[asdict(t) for t in throws]),indent=2))
    settings=([('stationary',a.opening,a.offset),('soft',a.opening,a.offset)] if a.tool=='paddle' else
              [('parked',o,t) for o in [.05,.065,.08] for t in [-.18,-.10,-.04,0,.04]] if a.sweep else [('parked',a.opening,a.offset)])
    results=[]
    for mode,opening,offset in settings:
        for i,throw in enumerate(throws):
            r,trace=rollout(sim,throw,mode,opening,offset,a.jaw_speed,a.delay,a.record)
            r['episode']=i;results.append(r)
            if a.record:(a.out/f'trace-{mode}-{opening}-{offset}-{i}.json').write_text(json.dumps(trace))
    (a.out/'episodes.json').write_text(json.dumps(results,indent=2))
    groups=[]
    for mode,opening,offset in settings:
        rs=[r for r in results if r['mode']==mode and r['opening']==opening and r['closure_offset']==offset]
        groups.append(dict(mode=mode,opening=opening,offset=offset,n=len(rs),retained=sum(r['retained'] for r in rs),
                           entered=sum(r['entered'] for r in rs),contact=sum(r['contact'] for r in rs),
                           mean_bounce_m=float(np.mean([r['first_bounce_height_m'] for r in rs if r['first_bounce_height_m'] is not None])) if a.tool=='paddle' and any(r['contact'] for r in rs) else None))
    summary=dict(config=asdict(cfg),seed=a.seed,kind=a.kind,groups=groups,
                 simulated_seconds_per_wall_second=len(results)*cfg.duration/sum(r['wall_s'] for r in results),
                 jaw_parameters_provisional=True)
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
