# Generated from lab3_cspace/plan.py by export_robot.py.
"""Build a complete offline joint-space plan; never connects to a robot."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import sys
from .kinematics import Panda,load_config,task_config,transform
from .task import Planner,Pattern,Move


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()


class Builder:
    def __init__(self,cfg,start=None,seed=13):
        from .collision import CollisionScene
        from .motion import JointPlanner
        self.cfg=cfg; self.layout=Planner(task_config(cfg)); self.robot=Panda(cfg)
        self.scene=CollisionScene(cfg,self.layout); self.motion=JointPlanner(self.robot,self.scene,cfg,seed)
        self.locations=self.layout.initial_locations.copy()
        self.q=np.array(cfg['home_q'] if start is None else start,dtype=float)
        self.actions=[]; self.reports=[]
        if not self.scene.valid(self.q,self.locations,cfg['gripper']['open_width']):
            raise RuntimeError(f'Start configuration collides: {self.scene.last_collision}')

    def append_path(self,path,label,endpoint=None):
        if np.max(abs(path[0]-self.q))>.001: raise RuntimeError('Discontinuous planned path')
        self.actions.extend(self.motion.segments(path,label,endpoint)); self.q=np.array(path[-1])

    def valid(self,width,held=None,held_rotation=None,touch=None):
        # Capture a snapshot: planning actions mutate the occupancy dictionary.
        locations=self.locations.copy()
        return lambda q:self.scene.valid(q,locations,width,held,held_rotation,touch)

    def leg(self,location,axis,valid):
        xyz=self.layout.xyz(location).copy(); xyz[2]=self.cfg['tcp_grasp_z']
        rotation=self.layout.grasps.rotation(axis,self.robot.fk(self.q)[:3,:3])
        failures=[]
        for r in (rotation,rotation@np.diag([-1.,-1,1])):
            grasp=transform(xyz,r); pre=grasp.copy(); pre[2,3]=self.cfg['travel_z']
            try:
                down=self.motion.approach(pre,grasp,self.q,valid)
                travel=self.motion.transfer(self.q,down[0],valid)
                return travel,down,grasp
            except RuntimeError as error: failures.append(str(error))
        raise RuntimeError(f'{location} {axis}: '+ '; '.join(failures))

    def transfer(self,move):
        if self.locations[move.block]!=move.source or move.target in self.locations.values():
            raise ValueError('Invalid task transfer')
        cfg=self.cfg; opening=cfg['gripper']['open_width']
        print(f'Planning {move.block}: {move.source} -> {move.target} [{move.pick_axis}/{move.place_axis}]',flush=True)
        self.actions.append({'kind':'grip','label':'Open before pickup','width':opening,'block':None})
        pick_valid=self.valid(opening,touch=move.block)
        travel,down,grasp=self.leg(move.source,move.pick_axis,pick_valid)
        self.append_path(travel,'Approach '+move.source)
        self.append_path(down,'Descend to pick '+move.block,grasp)
        self.actions.append({'kind':'grip','label':'Grasp '+move.block,'width':0.,'block':move.block,
                             'location':move.source,'target_tcp':grasp.tolist()})
        self.locations[move.block]='held'
        held_r=self.robot.fk(self.q)[:3,:3].T
        held_valid=self.valid(cfg['cube_size']-.002,move.block,held_r)
        up=down[::-1]
        if not all(held_valid(q) for q in up): raise RuntimeError(f'Pickup lift collides: {self.scene.last_collision}')
        self.append_path(up,'Lift '+move.block)
        travel,down,grasp=self.leg(move.target,move.place_axis,held_valid)
        self.append_path(travel,'Transfer '+move.block)
        self.append_path(down,'Descend to place '+move.block,grasp)
        self.actions.append({'kind':'grip','label':'Release '+move.block,'width':opening,'block':move.block,
                             'location':move.target,'target_tcp':grasp.tolist()})
        self.locations[move.block]=move.target
        retract_valid=self.valid(opening,touch=move.block)
        if not all(retract_valid(q) for q in down): raise RuntimeError(f'Placement retract collides: {self.scene.last_collision}')
        self.append_path(down[::-1],'Retract after '+move.block)

    def reset(self):
        while any(v.startswith('grid:') for v in self.locations.values()):
            for block,source in self.locations.items():
                if not source.startswith('grid:'): continue
                picks=self.layout.orientations(source,self.locations,block)
                targets=['waiting:'+b for b in self.layout.blocks if b[0]==block[0] and 'waiting:'+b not in self.locations.values()]
                if not picks or not targets: continue
                target=targets[0]; places=self.layout.orientations(target,self.locations,block)
                common=[a for a in ('Y','X') if a in picks and a in places]
                pick=common[0] if common else picks[0]
                self.transfer(Move(block,source,target,pick,pick if pick in places else places[0])); break
            else: raise RuntimeError('No removable block during reset')


def build(cfg,test='all',final_reset=False,start=None,seed=13):
    b=Builder(cfg,start,seed)
    if test=='rotate-place':
        b.transfer(Move('B0','waiting:B0','grid:0','Y','X'))
        b.transfer(Move('B0','grid:0','waiting:B0','X','Y'))
    else:
        patterns=Pattern().get()
        names=list(patterns) if test=='all' else [test]
        for name in names:
            moves=b.layout.plan(patterns[name],b.locations)
            if moves is None:
                b.reports.append({'pattern':name,'skipped':True,'reason':'No feasible top-grasp ordering'})
                print('Skip',name,flush=True); continue
            for move in moves: b.transfer(move)
            expected={f'grid:{3*r+c}':color for r,row in enumerate(patterns[name]) for c,color in enumerate(row) if color!='.'}
            actual={location:block[0] for block,location in b.locations.items() if location.startswith('grid:')}
            if actual!=expected: raise RuntimeError('Task search did not produce the requested pattern')
            b.actions.append({'kind':'checkpoint','label':name,'locations':b.locations.copy(),'expected_grid':expected})
            b.reports.append({'pattern':name,'moves':len(moves),'skipped':False})
    if final_reset: b.reset()
    b.actions.append({'kind':'checkpoint','label':'final','locations':b.locations.copy()})
    return {'schema':1,'collision_backend':'numpy_gjk_v1','config':cfg,'config_hash':config_hash(cfg),
            'start_q':np.asarray(cfg['home_q'] if start is None else start).tolist(),
            'actions':b.actions,'reports':b.reports,'rrt_searches':b.motion.rrt_searches,
            'predicted_duration':sum(a.get('duration',1.) for a in b.actions if a['kind']!='checkpoint')}




