# Generated from lab3_cspace/kinematics.py by export_robot.py.
"""Lab2 Panda URDF chain, damped numerical IK and joint trajectory timing.

NumPy only: the hardware executor can check the model without MuJoCo/ROS
imports leaking into the planner's mathematical primitives.
"""
import numpy as np

QMIN = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-.0175,-2.8973]) + .02
QMAX = np.array([2.8973,1.7628,2.8973,-.0698,2.8973,3.7525,2.8973]) - .02


def rz(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,-s,0],[s,c,0],[0,0,1.]])


def transform(xyz=(0,0,0), rotation=None):
    t=np.eye(4); t[:3,3]=xyz
    if rotation is not None: t[:3,:3]=rotation
    return t


def rotvec(r):
    # Project small numerical errors onto SO(3); handle the pi singularity.
    u,_,vt=np.linalg.svd(r); r=u@np.diag([1.,1.,np.linalg.det(u@vt)])@vt
    angle=np.arccos(np.clip((np.trace(r)-1)/2,-1,1))
    v=np.array([r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]])
    if angle<1e-7: return v/2
    if np.pi-angle<1e-5:
        values,vectors=np.linalg.eigh((r+np.eye(3))/2)
        axis=vectors[:,np.argmax(values)]
        if np.dot(axis,v)<0: axis=-axis
        return axis*angle
    return v*angle/(2*np.sin(angle))


class Panda:
    def __init__(self, config):
        # Same rigid transforms as Lab2/Franka.py, including the flange stub.
        rows=[(0,(0,0,.333)),(-np.pi/2,(0,0,0)),(np.pi/2,(0,-.316,0)),
              (np.pi/2,(.0825,0,0)),(-np.pi/2,(-.0825,.384,0)),
              (np.pi/2,(0,0,0)),(np.pi/2,(.088,0,0))]
        self.links=[]
        for a,xyz in rows:
            c,s=np.cos(a),np.sin(a)
            self.links.append(transform(xyz,np.array([[1,0,0],[0,c,-s],[0,s,c]])))
        tool=config['flange_to_tcp']
        self.tool=transform((0,0,.107))@transform(tool['translation'],rz(np.deg2rad(tool['yaw_deg'])))
        self.home=np.asarray(config['home_q'],dtype=float)

    def fk(self,q,jacobian=False):
        t=np.eye(4); origins=[]; axes=[]
        for link,angle in zip(self.links,q):
            t=t@link
            origins.append(t[:3,3].copy()); axes.append(t[:3,2].copy())
            t=t@transform(rotation=rz(angle))
        t=t@self.tool
        if not jacobian: return t
        j=np.column_stack([np.r_[np.cross(a,t[:3,3]-p),a] for a,p in zip(axes,origins)])
        return t,j

    def ik(self,target,seed,iterations=250):
        q=np.clip(np.asarray(seed,dtype=float).copy(),QMIN,QMAX)
        for _ in range(iterations):
            actual,j=self.fk(q,True)
            error=np.r_[target[:3,3]-actual[:3,3],rotvec(target[:3,:3]@actual[:3,:3].T)]
            if np.linalg.norm(error[:3])<.00015 and np.linalg.norm(error[3:])<.001:
                return q
            error[:3]*=min(1.,.025/max(np.linalg.norm(error[:3]),1e-12))
            error[3:]*=min(1.,.15/max(np.linalg.norm(error[3:]),1e-12))
            dq=j.T@np.linalg.solve(j@j.T+1e-5*np.eye(6),error)
            dq*=min(1.,.12/max(np.max(abs(dq)),1e-12))
            q=np.clip(q+dq,QMIN,QMAX)
        return None


def duration(qa,qb,settings):
    d=float(np.max(abs(np.asarray(qb)-qa)))
    # Exact extrema of a rest-to-rest quintic min-jerk segment.
    return max(settings['minimum_duration'],1.875*d/settings['joint_speed'],
               np.sqrt(5.773503*d/settings['joint_acceleration']),
               np.cbrt(60*d/settings['joint_jerk']))


def load_config(path):
    import json
    from pathlib import Path
    return validate_config(json.loads(Path(path).read_text()))


def validate_config(cfg):
    for key in ('cube_size','cube_mass','travel_z'):
        if not np.isfinite(cfg[key]) or cfg[key]<=0: raise ValueError('Invalid '+key)
    if cfg['travel_z']<=cfg['tcp_grasp_z']: raise ValueError('travel_z must exceed tcp_grasp_z')
    if set(cfg['waiting_xy'])!={f'{c}{i}' for c in 'RB' for i in range(4)}: raise ValueError('Need R0..R3 and B0..B3')
    for point in [*cfg['waiting_xy'].values(),cfg['grid_center_xy']]:
        if np.asarray(point).shape!=(2,) or not np.isfinite(point).all(): raise ValueError('Expected finite XY coordinates')
    if not np.isfinite([cfg['table_z'],cfg['tcp_grasp_z'],cfg['grid_spacing']]).all() or cfg['grid_spacing']<=cfg['cube_size']:
        raise ValueError('Invalid table/grasp height or grid spacing')
    if not cfg['cube_size']<cfg['gripper']['open_width']<=.08 or not 0<cfg['gripper']['force']<=60:
        raise ValueError('Invalid gripper configuration')
    for key in ('joint_speed','joint_acceleration','joint_jerk','minimum_duration','cartesian_step','edge_resolution'):
        if not np.isfinite(cfg['motion'][key]) or cfg['motion'][key]<=0: raise ValueError('Invalid motion.'+key)
    if not np.isfinite(cfg['motion']['clearance']) or cfg['motion']['clearance']<0: raise ValueError('Invalid collision clearance')
    q=np.asarray(cfg['home_q'],dtype=float)
    if q.shape!=(7,) or not np.isfinite(q).all() or np.any(q<QMIN) or np.any(q>QMAX): raise ValueError('Invalid home joints')
    tool=cfg['flange_to_tcp']; xyz=np.asarray(tool['translation'])
    if xyz.shape!=(3,) or not np.isfinite(xyz).all() or not np.isfinite(tool['yaw_deg']): raise ValueError('Invalid flange-to-TCP transform')
    return cfg


def task_config(cfg):
    return {'boxes':{k:[*xy,cfg['tcp_grasp_z']] for k,xy in cfg['waiting_xy'].items()},
            'grid':{'center':[*cfg['grid_center_xy'],cfg['tcp_grasp_z']],'spacing':cfg['grid_spacing']},
            'box_width':cfg['cube_size'],'table_height':cfg['table_z'],
            'gripper':{'opening':cfg['gripper']['open_width']}}
