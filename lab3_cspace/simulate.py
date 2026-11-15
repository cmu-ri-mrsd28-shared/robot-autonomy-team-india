"""Execute a saved joint plan in MuJoCo with physical grasps, no IK at runtime."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco as mj
from lab3_cspace.scene import Scene
from lab3_cspace.task import Planner
from lab3_cspace.kinematics import task_config
from lab3_cspace.viewer_lifecycle import close_viewer


class Simulator:
    def __init__(self,plan,gui=False,speed=1.):
        self.plan=plan; self.cfg=plan['config']; self.layout=Planner(task_config(self.cfg))
        self.scene=Scene(self.cfg,self.layout); self.model,self.data=self.scene.model,self.scene.data
        self.data.qpos[:7]=plan['start_q']; self.data.ctrl[self.scene.arm_act]=plan['start_q']
        mj.mj_forward(self.model,self.data)
        self.held=None; self.viewer=None; self.speed=speed; self.checkpoints=[]
        if gui:
            from mujoco import viewer
            self.viewer=viewer.launch_passive(self.model,self.data)
            self.viewer.cam.distance=1.7; self.viewer.cam.azimuth=135; self.viewer.cam.elevation=-40
            self.viewer.cam.lookat[:]=[.45,0,.2]

    def step(self,q):
        self.data.ctrl[self.scene.arm_act]=q
        self.data.qfrc_applied[:7]=self.data.qfrc_bias[:7]
        mj.mj_step(self.model,self.data)
        if not np.isfinite(self.data.qpos).all(): raise RuntimeError('Nonfinite simulation state')

    def advance(self,seconds,start=None,goal=None):
        dt=self.model.opt.timestep; n=int(np.ceil(seconds/dt)); wall=time.monotonic()
        hold=self.data.qpos[:7].copy() if goal is None else None
        for i in range(n+1):
            t=min(i*dt/seconds,1.); blend=10*t**3-15*t**4+6*t**5
            q=hold if goal is None else start+blend*(goal-start)
            if self.viewer:
                if not self.viewer.is_running(): raise RuntimeError('Viewer closed')
                with self.viewer.lock(): self.step(q)
                if i%10==0:
                    self.viewer.sync()
                    if self.speed>0:
                        delay=i*dt/self.speed-(time.monotonic()-wall)
                        if delay>0: time.sleep(delay)
            else: self.step(q)

    def position(self,block): return self.data.xpos[self.scene.block_bodies[block]].copy()

    def execute(self):
        self.advance(.5)
        last_label=None
        for index,action in enumerate(self.plan['actions']):
            kind=action['kind']
            if action['label']!=last_label:
                print(index,action['label'],flush=True); last_label=action['label']
            if kind=='move':
                start=np.asarray(action['start_q']); goal=np.asarray(action['goal_q'])
                if np.max(abs(self.data.qpos[:7]-start))>.03:
                    raise RuntimeError(f'Joint tracking lost before action {index}: {action["label"]}')
                self.advance(action['duration'],start,goal)
                if np.max(abs(self.data.qpos[:7]-goal))>.025:
                    raise RuntimeError(f'Joint endpoint not reached: {action["label"]}')
                if self.held and np.linalg.norm(self.position(self.held)-self.data.site_xpos[self.scene.tcp])>.015:
                    raise RuntimeError(f'Lost physical grasp of {self.held} during {action["label"]}: '
                                       f'cube={self.position(self.held)}, tcp={self.data.site_xpos[self.scene.tcp]}, '
                                       f'jaw_width={sum(self.data.qpos[7:9]):.5f}')
            elif kind=='grip':
                self.model.actuator_forcerange[self.scene.finger_act]=[-self.cfg['gripper']['force'],self.cfg['gripper']['force']]
                self.model.actuator_forcelimited[self.scene.finger_act]=True
                self.data.ctrl[self.scene.finger_act]=action['width']/2
                self.advance(1.)
                width=float(sum(self.data.qpos[7:9]))
                if action['width']==0:
                    if not self.cfg['cube_size']-.012<width<self.cfg['cube_size']+.005:
                        raise RuntimeError(f'Empty or blocked grasp, width={width:.5f}')
                    if np.linalg.norm(self.position(action['block'])-self.data.site_xpos[self.scene.tcp])>.012:
                        raise RuntimeError('Target block is not between finger pads')
                    self.held=action['block']
                else:
                    if width<self.cfg['cube_size']+.01: raise RuntimeError('Gripper failed to open')
                    self.held=None
            elif kind=='checkpoint':
                self.advance(.5)
                errors={}
                for block,location in action['locations'].items():
                    expected=self.layout.xyz(location).copy(); expected[2]=self.cfg['table_z']+self.cfg['cube_size']/2
                    errors[block]=float(np.linalg.norm(self.position(block)-expected))
                record={'label':action['label'],'max_block_error_m':max(errors.values()),'block_errors_m':errors}
                self.checkpoints.append(record); print(json.dumps(record),flush=True)
                if max(errors.values())>.012: raise RuntimeError('Physical arrangement differs from planned arrangement')
        return {'passed':True,'checkpoints':self.checkpoints,'simulated_seconds':float(self.data.time)}

    def close(self):
        # Keep self.model/data and the handle alive until native render cleanup
        # completes; close() alone is an asynchronous shutdown request.
        close_viewer(self.viewer)
        self.viewer=None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',default='results/plan.json')
    parser.add_argument('--gui',action='store_true'); parser.add_argument('--speed',type=float,default=1.)
    parser.add_argument('--report',default='results/simulation_report.json')
    args=parser.parse_args(); plan=json.loads(Path(args.plan).read_text())
    sim=Simulator(plan,args.gui,args.speed)
    dest=Path(args.report)
    try:
        report=sim.execute()
        # Preserve the physics result even if GUI cleanup reports an error.
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(report,indent=2)+'\n')
    finally: sim.close()
    print('PASS: physical simulation completed; report:',dest)


if __name__=='__main__': main()
