from dataclasses import dataclass, asdict
import numpy as np
import mujoco
from catchbox.core import Throw

WORKSPACE_RADIUS_M = 0.5

def in_workspace(position, center):
    return bool(np.linalg.norm(np.asarray(position) - np.asarray(center)) <= WORKSPACE_RADIUS_M + 1e-12)

def clamp_workspace(position, center):
    center = np.asarray(center, dtype=float)
    delta = np.asarray(position, dtype=float) - center
    distance = np.linalg.norm(delta)
    return center + delta * min(1.0, WORKSPACE_RADIUS_M / distance) if distance else center.copy()

@dataclass
class Prediction:
    time_to_contact: float
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    valid: bool

class Predictor:
    """World-frame constant-acceleration flight model; never extrapolate impact spikes."""
    def predict(self,state,height,in_contact=False):
        a=np.array(state.acceleration if state.acceleration is not None else [0,0,-9.81],dtype=float)
        if in_contact or not np.isfinite(a).all() or np.linalg.norm(a)>30:
            return Prediction(0.,state.position.copy(),state.velocity.copy(),a,False)
        p=np.asarray(state.position);v=np.asarray(state.velocity)
        if not state.valid or not np.isfinite(np.r_[p,v]).all():
            return Prediction(0.,p.copy(),v.copy(),a,False)
        z=p[2]-height
        roots=np.roots([.5*a[2],v[2],z]) if abs(a[2])>1e-8 else np.roots([v[2],z]) if abs(v[2])>1e-8 else []
        roots=[float(t.real) for t in roots if abs(t.imag)<1e-8 and t.real>0 and v[2]+a[2]*t.real<0]
        if not roots:return Prediction(0.,p.copy(),v.copy(),a,False)
        t=min(roots)
        return Prediction(t,p+v*t+.5*a*t*t,v+a*t,a,True)

def launch_prediction(sim, throw, height):
    """Predict using the actual simulator state at release, including contacts."""
    sim.data.qpos[sim.ball_q:sim.ball_q+3] = throw.position
    sim.data.qvel[sim.ball_v:sim.ball_v+3] = throw.velocity
    mujoco.mj_forward(sim.model, sim.data)
    return Predictor().predict(sim.read(), height)

@dataclass
class LaunchLimits:
    distance_min: float=.5
    distance_max: float=1.5
    angle_deg: float=80.
    flight_min: float=.50
    flight_max: float=.85
    displacement_min: float=.065
    displacement_max: float=.14
    speed_max: float=5.0

def varied_throw(rng,origin,level=1.,limits=None):
    """Jointly sample thrower and target, then solve a physically consistent launch.
    level changes speed/angle/distance breadth, never removes the movement requirement.
    No post-sampling velocity clipping, which would invalidate the sampled intercept.
    """
    lim=limits or LaunchLimits();level=float(np.clip(level,0,1))
    target=np.array(origin,dtype=float)
    az=rng.uniform(-np.pi,np.pi)
    displacement=rng.uniform(lim.displacement_min,lim.displacement_min+level*(lim.displacement_max-lim.displacement_min))
    target[:2]+=displacement*np.array([np.cos(az),np.sin(az)])
    # Front hemisphere: avoid throws launched through the robot base.
    angle=rng.uniform(-lim.angle_deg,lim.angle_deg)*(.25+.75*level)
    heading=np.deg2rad(angle)
    distance=rng.uniform(lim.distance_min,lim.distance_min+(.2+.8*level)*(lim.distance_max-lim.distance_min))
    flight=rng.uniform(lim.flight_min,lim.flight_max)
    height=rng.uniform(-.08,.12)
    launch=target+np.array([distance*np.cos(heading),distance*np.sin(heading),height])
    acceleration=np.array([0,0,-9.81])
    velocity=(target-launch-.5*acceleration*flight**2)/flight
    meta=dict(distance_m=distance,angle_deg=angle,speed_m_s=float(np.linalg.norm(velocity)),
              flight_s=flight,required_movement_m=displacement,target=target.tolist(),level=level)
    return Throw(launch.tolist(),velocity.tolist(),'varied_lob'),meta

def freeze_bank(sim,seed,count,path=None):
    """Reachability-filter before scoring; log every rejection and fixed candidate limits."""
    rng=np.random.default_rng(seed);throws=[];rejections=[]
    origin=sim.nominal_opening.copy()
    if sim.tool=='paddle':origin[2]+=sim.cfg.radius
    for candidate in range(count*100):
        throw,meta=varied_throw(rng,origin)
        target=np.array(meta['target']);target[2]-=sim.cfg.radius if sim.tool=='paddle' else 0
        if not in_workspace(meta['target'], sim.nominal_grip):
            rejections.append(dict(candidate=candidate,reason='landing_zone',**meta));continue
        prediction=launch_prediction(sim,throw,origin[2])
        if not prediction.valid or not in_workspace(prediction.position,sim.nominal_grip):
            rejections.append(dict(candidate=candidate,reason='predicted_landing_zone',**meta));continue
        q,error=sim.inverse_kinematics(target)
        move_time=np.max(abs(q-sim.home))*1.875/1.5
        reason=('speed' if meta['speed_m_s']>5 else 'ik' if error>.003 else 'deadline' if move_time+.04>meta['flight_s'] else None)
        if reason:rejections.append(dict(candidate=candidate,reason=reason,**meta));continue
        throws.append(dict(throw=asdict(throw),meta=meta))
        if len(throws)==count:break
    if len(throws)!=count:raise RuntimeError('Could not build bank within rejection budget')
    result=dict(seed=seed,limits=asdict(LaunchLimits()),workspace_radius_m=WORKSPACE_RADIUS_M,workspace_center=sim.nominal_grip.tolist(),throws=throws,rejections=rejections,
                proposed=count+len(rejections),accepted=count)
    if path:
        import json,hashlib
        result['sha256']=hashlib.sha256(json.dumps(throws,sort_keys=True).encode()).hexdigest()
        path.write_text(json.dumps(result,indent=2))
    return result
