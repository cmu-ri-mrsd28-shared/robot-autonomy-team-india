"""Lab2-style collision-checked IK approaches and RRT-Connect in joint space."""
import numpy as np
from lab3_cspace.kinematics import QMIN,QMAX,transform,rotvec,duration


class JointPlanner:
    def __init__(self,robot,scene,cfg,seed=13):
        self.robot,self.scene,self.cfg=robot,scene,cfg
        self.rng=np.random.default_rng(seed)
        self.rrt_searches=0

    def edge(self,a,b,valid):
        n=max(1,int(np.ceil(np.max(abs(np.asarray(b)-a))/self.cfg['motion']['edge_resolution'])))
        return all(valid(np.asarray(a)+s*(np.asarray(b)-a)) for s in np.linspace(0,1,n+1))

    def transfer(self,start,goal,valid):
        if not valid(start) or not valid(goal): raise RuntimeError(f'Transfer endpoint collides: {self.scene.last_collision}')
        if self.edge(start,goal,valid): return [start,goal]
        self.rrt_searches+=1
        trees=[([np.array(start)],[-1]),([np.array(goal)],[-1])]
        def extend(tree,target):
            vertices,parents=tree
            i=int(np.argmin(np.linalg.norm(np.asarray(vertices)-target,axis=1)))
            diff=target-vertices[i]; length=np.linalg.norm(diff)
            q=vertices[i]+diff*min(1.,.22/max(length,1e-12))
            if not self.edge(vertices[i],q,valid): return None,False
            vertices.append(q); parents.append(i)
            return len(vertices)-1,length<=.22
        def trace(tree,index):
            result=[]
            while index!=-1: result.append(tree[0][index]); index=tree[1][index]
            return result[::-1]
        for iteration in range(3500):
            k=iteration%2
            target=trees[1-k][0][0] if self.rng.random()<.15 else self.rng.uniform(QMIN,QMAX)
            i,_=extend(trees[k],target)
            if i is None: continue
            while True:
                j,reached=extend(trees[1-k],trees[k][0][i])
                if j is None: break
                if reached:
                    paths=[None,None]; paths[k]=trace(trees[k],i); paths[1-k]=trace(trees[1-k],j)
                    result=paths[0]+paths[1][::-1][1:]
                    for _ in range(60):
                        if len(result)<3: break
                        a,b=sorted(self.rng.choice(len(result),2,replace=False))
                        if b>a+1 and self.edge(result[a],result[b],valid): result=result[:a+1]+result[b:]
                    return result
        raise RuntimeError('RRT-Connect exhausted its search; no robot commands were sent')

    def approach(self,pre,grasp,seed,valid):
        seeds=[seed,self.robot.home]
        for _ in range(8): seeds.append(np.clip(self.robot.home+self.rng.normal(0,.55,7),QMIN,QMAX))
        for guess in seeds:
            q=self.robot.ik(pre,guess)
            if q is None or not valid(q): continue
            path=self.cartesian(q,pre,grasp,valid)
            if path is not None: return path
        raise RuntimeError(f'No continuous collision-free approach; last collision={self.scene.last_collision}')

    def cartesian(self,q,start,end,valid):
        n=max(1,int(np.ceil(np.linalg.norm(end[:3,3]-start[:3,3])/self.cfg['motion']['cartesian_step'])))
        path=[np.array(q)]
        for s in np.linspace(0,1,n+1)[1:]:
            target=start.copy(); target[:3,3]=start[:3,3]+s*(end[:3,3]-start[:3,3])
            nxt=self.robot.ik(target,path[-1])
            if nxt is None or np.max(abs(nxt-path[-1]))>.2 or not self.edge(path[-1],nxt,valid): return None
            # Validate TCP deviation along the actual joint-interpolated edge.
            for u in (.25,.5,.75):
                pose=self.robot.fk(path[-1]+u*(nxt-path[-1]))
                a=self.robot.fk(path[-1])[:3,3]
                desired=a+u*(target[:3,3]-a)
                if np.linalg.norm(pose[:3,3]-desired)>.001 or np.linalg.norm(rotvec(pose[:3,:3]@target[:3,:3].T))>.01: return None
            path.append(nxt)
        return path

    def segments(self,path,label,endpoint=None):
        result=[]
        for a,b in zip(path[:-1],path[1:]):
            if np.max(abs(b-a))<1e-6: continue
            result.append({'kind':'move','label':label,'start_q':np.asarray(a).tolist(),
                           'goal_q':np.asarray(b).tolist(),'duration':duration(a,b,self.cfg['motion'])})
        if result and endpoint is not None: result[-1]['target_tcp']=endpoint.tolist()
        return result
