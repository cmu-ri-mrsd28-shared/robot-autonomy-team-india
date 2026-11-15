# Generated from lab3_cspace/collision.py by export_robot.py.
"""Geometric collision checks: OBB broad phase and convex GJK, NumPy only."""
import json
from itertools import combinations
from pathlib import Path
import numpy as np
from .kinematics import Panda,QMIN,QMAX,transform,rz


def overlaps(ca,ra,ha,cb,rb,hb,clearance=0.):
    """Batch separating-axis tests on all 15 OBB axes, with padded half sizes.

    Half of the clearance is added to each box. This is conservative padding
    along box axes, not an exact Euclidean distance computation.
    """
    ca,ra,ha,cb,rb,hb=map(np.asarray,(ca,ra,ha,cb,rb,hb))
    a=ha+clearance/2; b=hb+clearance/2
    r=np.einsum('...ji,...jk->...ik',ra,rb)
    absolute=np.abs(r)+1e-12
    t=np.einsum('...ji,...j->...i',ra,cb-ca)
    hit=np.all(np.abs(t)<=a+np.einsum('...ij,...j->...i',absolute,b),axis=-1)
    hit &= np.all(np.abs(np.einsum('...ji,...j->...i',r,t))<=b+np.einsum('...ji,...j->...i',absolute,a),axis=-1)
    for i in range(3):
        j,k=(i+1)%3,(i+2)%3
        for u in range(3):
            v,w=(u+1)%3,(u+2)%3
            distance=np.abs(t[...,k]*r[...,j,u]-t[...,j]*r[...,k,u])
            radius=a[...,j]*absolute[...,k,u]+a[...,k]*absolute[...,j,u]+b[...,v]*absolute[...,i,w]+b[...,w]*absolute[...,i,v]
            hit &= distance<=radius
    return hit


def closest_simplex(points):
    """Closest point to origin in a simplex, including all faces/edges."""
    points=np.asarray(points); best=None
    for size in range(1,len(points)+1):
        for indices in combinations(range(len(points)),size):
            face=points[list(indices)]
            if size==1: weights=np.ones(1)
            else:
                edges=(face[1:]-face[0]).T
                tail=np.linalg.lstsq(edges,-face[0],rcond=None)[0]
                weights=np.r_[1-tail.sum(),tail]
            if np.min(weights)<-1e-10: continue
            weights=np.maximum(weights,0); weights/=weights.sum()
            point=weights@face; distance=point@point
            if best is None or distance<best[0]:
                best=distance,point,face[weights>1e-10]
    return best[1],best[2]


def convex_collision(a,b,clearance=0.):
    """GJK intersection/clearance test on the convex hulls of vertex sets.

    A positive support-plane gap proves separation. A stalled or exhausted
    query is treated as collision, never accepted as a free configuration.
    """
    a,b=np.asarray(a),np.asarray(b)
    direction=a.mean(axis=0)-b.mean(axis=0)
    if np.linalg.norm(direction)<1e-12: direction=np.array([1.,0,0])
    def support(d): return a[np.argmax(a@d)]-b[np.argmin(b@d)]
    simplex=np.array([support(direction)])
    for _ in range(64):
        nearest,simplex=closest_simplex(simplex)
        distance=np.linalg.norm(nearest)
        if distance<=clearance+1e-9: return True
        point=support(-nearest)
        if nearest@point>clearance*distance+1e-12: return False
        if np.any(np.linalg.norm(simplex-point,axis=1)<1e-12): return True
        simplex=np.vstack([simplex,point])
    return True


SIGNS=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])


def box_vertices(center,rotation,half):
    return center+(SIGNS*half)@rotation.T


class CollisionScene:
    def __init__(self,cfg,layout):
        self.cfg,self.layout=cfg,layout; self.robot=Panda(cfg); self.last_collision=None
        geometry=json.loads(Path(__file__).with_name('panda_boxes.json').read_text())['boxes']
        self.names=[box['name'] for box in geometry]
        self.bodies=[box['body'] for box in geometry]
        self.local=np.array([box['center'] for box in geometry])
        self.half=np.array([box['half_size'] for box in geometry])
        self.vertices=[np.array(box['vertices'])-box['center'] for box in geometry]
        self.body_index={'link'+str(i):i for i in range(8)}
        self.body_index.update(hand=8,left_finger=9,right_finger=10)
        self.indices=np.array([self.body_index[b] for b in self.bodies])
        # Bodies joined by a joint overlap by design. Finger/finger checks are
        # retained; only their common palm is excluded from each finger.
        parents={i:i-1 for i in range(1,9)}; parents.update({9:8,10:8})
        pairs=[(i,j) for i in range(len(geometry)) for j in range(i+1,len(geometry))
               if self.indices[i]!=self.indices[j]
               and parents.get(self.indices[i])!=self.indices[j]
               and parents.get(self.indices[j])!=self.indices[i]]
        self.pairs=np.array(pairs,dtype=int)
        self.fingers=np.array([b.endswith('_finger') for b in self.bodies])
        self.moving=self.indices!=0
        self.environment=[('table',np.array([.46,0,cfg['table_z']-.025]),np.array([.275,.504,.025]))]
        self.environment += [(o['name'],np.asarray(o['center']),np.asarray(o['half_size'])) for o in cfg.get('obstacles',[])]
        for name,center,half in self.environment:
            if center.shape!=(3,) or half.shape!=(3,) or not np.isfinite([center,half]).all() or np.any(half<=0):
                raise ValueError('Invalid obstacle geometry: '+name)

    def robot_boxes(self,q,width):
        frames=[np.eye(4)]; t=np.eye(4)
        for link,angle in zip(self.robot.links,q):
            t=t@link@transform(rotation=rz(angle)); frames.append(t)
        # Physical hand geometry is fixed to the flange; calibrated TCP may
        # differ from the hand origin. Do not use the TCP as the hand origin.
        hand=t@transform((0,0,.107),rz(-np.pi/4)); frames.append(hand)
        frames.append(hand@transform((0,width/2,.0584)))
        frames.append(hand@transform((0,-width/2,.0584),rz(np.pi)))
        frames=np.array(frames)[self.indices]
        rotations=frames[:,:3,:3]
        centers=frames[:,:3,3]+np.einsum('nij,nj->ni',rotations,self.local)
        return centers,rotations,self.half

    def fail(self,a,b):
        self.last_collision=(a,b); return False

    def valid(self,q,locations,width,held=None,held_rotation=None,touch=None):
        self.last_collision=None
        q=np.asarray(q)
        if q.shape!=(7,) or not np.isfinite(q).all() or np.any(q<QMIN) or np.any(q>QMAX):
            return self.fail('joint limits','robot')
        c,r,h=self.robot_boxes(q,width); margin=self.cfg['motion']['clearance']
        world={}
        def points(index):
            if index not in world: world[index]=c[index]+self.vertices[index]@r[index].T
            return world[index]
        def confirmed(hits,other):
            return next((index for index in np.flatnonzero(hits)
                         if convex_collision(points(index),other,margin)),None)
        i,j=self.pairs.T
        hits=overlaps(c[i],r[i],h[i],c[j],r[j],h[j],margin)
        for a,b in self.pairs[hits]:
            if convex_collision(points(a),points(b),margin): return self.fail(self.names[a],self.names[b])
        # The base is fixed to its support. All moving links, including fingers,
        # are checked against the table and shelves.
        for name,center,half in self.environment:
            hits=overlaps(c,r,h,center,np.eye(3),half,margin)&self.moving
            index=confirmed(hits,box_vertices(center,np.eye(3),half))
            if index is not None: return self.fail(self.names[index],name)
        tcp=self.robot.fk(q); cubes=[]
        for block,location in locations.items():
            if block==held:
                center=tcp[:3,3]; rotation=tcp[:3,:3]@held_rotation
            else:
                center=self.layout.xyz(location).copy(); center[2]=self.cfg['table_z']+self.cfg['cube_size']/2
                rotation=np.eye(3)
            half=np.full(3,self.cfg['cube_size']/2)
            cubes.append((block,center,rotation,half))
            hits=overlaps(c,r,h,center,rotation,half,margin)
            # Only contact of fingers with the selected/held cube is allowed.
            if block in (held,touch): hits &= ~self.fingers
            index=confirmed(hits,box_vertices(center,rotation,half))
            if index is not None: return self.fail(self.names[index],block)
        if held:
            block,center,rotation,half=next(cube for cube in cubes if cube[0]==held)
            for name,other,size in self.environment:
                if name=='table':
                    bottom=center[2]-np.abs(rotation[2])@half
                    if bottom>=self.cfg['table_z']-.001: continue
                if overlaps(center,rotation,half,other,np.eye(3),size,margin): return self.fail(block,name)
            for other,position,axes,size in cubes:
                if other!=held and overlaps(center,rotation,half,position,axes,size,margin): return self.fail(block,other)
        return True
